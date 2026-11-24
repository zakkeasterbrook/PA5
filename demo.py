"""Minimal deterministic PA5 demonstration: identical keys and forced ties."""
import numpy as np
from pa5 import Cache, dense


def main():
    rng = np.random.default_rng(5)
    K = np.zeros((4096, 32))
    V = rng.normal(size=K.shape)
    q = rng.normal(size=32)
    W = rng.normal(size=(32, 16))
    cache = Cache(K, V, b=128)
    result = cache.decide(q, W)
    assert result["token"] == dense(K, V, q, W)
    assert result["accepted"] and result["keys"] == result["values"] == 0
    print("Identical keys (constructed best case)")
    print("  Winner:", result["token"])
    print("  Margin test accepted:", result["accepted"])
    print("  Logical read fraction:", result["bytes"] / cache.raw)
    tie = np.repeat(W[:, :1], W.shape[1], axis=1)
    result = cache.decide(q, tie)
    assert not result["accepted"]
    assert result["token"] == dense(K, V, q, tie)
    print("Forced tie")
    print("  Dense fallback matched; no unique winner certified.")
    print("  Logical read fraction:", result["bytes"] / cache.raw)


if __name__ == "__main__":
    main()
