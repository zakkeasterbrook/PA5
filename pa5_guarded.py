"""Conservative range-screened PA5 prototype, not interval certification.

The original pa5.py and its benchmarks remain unchanged. This optional wrapper
avoids applying the margin test to out-of-range raw input magnitudes, checks
shapes, owns a read-only cache copy, and rejects nonfinite dense scores.
"""
import numpy as np
from pa5 import Cache

LOW, HIGH = 1e-50, 1e50


def in_range(a):
    m = np.abs(a)
    return bool(np.all((m == 0) | ((m >= LOW) & (m <= HIGH))))


class GuardedCache:
    def __init__(self, K, V, b=128):
        self.K = np.array(K, dtype=np.float64, copy=True, order='C')
        self.V = np.array(V, dtype=np.float64, copy=True, order='C')
        K, V = self.K, self.V
        if (not isinstance(b, (int, np.integer)) or b < 1 or K.ndim != 2
                or V.ndim != 2 or len(K) == 0 or len(K) != len(V)
                or len(K) % b or K.shape[1] == 0 or V.shape[1] == 0):
            raise ValueError('Require nonempty equal-size full blocks')
        if not np.isfinite(K).all() or not np.isfinite(V).all():
            raise ValueError('Nonfinite cache')
        K.flags.writeable = V.flags.writeable = False
        self.b = b
        self.raw = K.nbytes + V.nbytes
        self.cache = Cache(K, V, b) if in_range(K) and in_range(V) else None

    def decide(self, q, W, mode='dual'):
        q, W = np.asarray(q, dtype=np.float64), np.asarray(W, dtype=np.float64)
        if (mode not in ('dual', 'values') or q.shape != (self.K.shape[1],)
                or W.ndim != 2 or W.shape[0] != self.V.shape[1] or W.shape[1] < 2
                or not np.isfinite(q).all() or not np.isfinite(W).all()):
            raise ValueError('Invalid mode, shapes, or nonfinite query/readout')
        def fallback(reason):
            with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                s = self.K @ q
                if not np.isfinite(s).all():
                    raise FloatingPointError('Nonfinite dense attention logits')
                scores = (np.exp(s - s.max()) @ self.V) @ W
            if not np.isfinite(scores).all():
                raise FloatingPointError('Nonfinite dense output scores')
            return dict(token=int(np.argmax(scores)), accepted=False, bytes=self.raw,
                        keys=len(self.K)//self.b, values=len(self.V)//self.b,
                        bound=None, reason=reason)
        if self.cache is None or not in_range(q) or not in_range(W):
            return fallback('range-screened dense fallback')
        # Screening is not a proof of safe intermediate rounding or cancellation.
        # Within the screened range, use the original approximate margin test.
        return dict(self.cache.decide(q, W, mode), reason='original numerical path')
