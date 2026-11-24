"""Reproduce known underflow failure; test the optional guarded wrapper.
New seeds and more extreme scales than the original benchmark. This is a
regression suite, not a proof of finite-precision correctness.
"""
import json
import numpy as np
from pa5 import Cache, dense
from pa5_guarded import GuardedCache


def main():
    K = np.array([[-1.], [1.]])
    V = np.array([[2e-200], [-1e-200]])
    q = np.array([1.])
    W = np.array([[1e150, -1e150]])
    ref = dense(K, V, q, W)
    original = Cache(K, V, 2).decide(q, W)
    fixed = GuardedCache(K, V, 2).decide(q, W)
    assert ref == 1 and original['token'] == 0 and original['accepted']
    assert fixed['token'] == ref and not fixed['accepted']
    print('Confirmed original incorrect acceptance; guarded path falls back.')
    rng = np.random.default_rng(772041)
    scales = [-200, -150, -100, -30, 0, 30, 100, 150, 200]
    count = wrong_original = wrong_guarded = guarded_accept = screened = 0
    for i in range(1800):
        b = int(rng.choice([2, 4, 8, 16])); blocks = int(rng.integers(1, 9))
        d = int(rng.integers(1, 9)); o = int(rng.integers(2, 9))
        a = scales[i % len(scales)]; e = scales[(i//len(scales)) % len(scales)]
        K = rng.normal(size=(blocks, 1, d)) + 10**rng.uniform(-5, 0)*rng.normal(size=(blocks, b, d))
        V = rng.normal(size=(blocks, 1, d)) + rng.normal(size=(blocks, b, d))
        K = K.reshape(-1, d)*10.**a; q = rng.normal(size=d)*10.**(-a)
        V = V.reshape(-1, d)*10.**e; W = rng.normal(size=(d, o))*10.**(-e)
        if i % 7 == 0: W[:] = W[:, :1]
        with np.errstate(all='ignore'):
            ref = dense(K, V, q, W)
            C, G = Cache(K, V, b), GuardedCache(K, V, b)
            for mode in ('dual', 'values'):
                z = C.decide(q, W, mode)
                h = G.decide(q, W, mode)
                count += 1; wrong_original += z['token'] != ref
                wrong_guarded += h['token'] != ref; guarded_accept += h['accepted']
                screened += h['reason'] == 'range-screened dense fallback'
    assert wrong_guarded == 0
    K = np.zeros((4,1)); V = np.ones((4,1)); q = np.ones(1); W = np.array([[1., -1.]])
    G = GuardedCache(K,V,2); V *= -1
    assert G.decide(q,W)['token'] == 0 and dense(K,V,q,W) == 1
    try:
        GuardedCache(np.full((4,1), 1e300), np.ones((4,1)), 2).decide(np.array([1e300]),W)
    except FloatingPointError:
        pass
    else:
        raise AssertionError('nonfinite dense score was not rejected')
    report = dict(fixtures=1800,evaluations=count,original_disagreements=wrong_original,
                  guarded_disagreements=wrong_guarded,guarded_accepted=guarded_accept,
                  range_fallbacks=screened,explicit_counterexample_reproduced=True,
                  mutation_regression_passed=True,overflow_rejected=True)
    print(json.dumps(report,indent=2))
    with open('numerical_audit_results.json','w') as f: json.dump(report,f,indent=2);f.write('\n')


if __name__ == '__main__':
    main()
