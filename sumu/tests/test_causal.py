import numpy as np

import sumu


def test_pairwise_causal_estimation():
    np.random.seed(0)
    n = 6
    bn = sumu.GaussianBNet.random(n, enb=2)
    data = bn.sample(20000)
    dags, meta = sumu.Gadget(data=data, mcmc={"iters": 80000}).sample()
    causal_effects = sumu.Beeps(dags=dags, data=data).sample_pairwise()
    mean_causal_effects = causal_effects.mean(axis=0)
    print("mean ces:", mean_causal_effects)
    true_causal_effects = np.linalg.inv(np.eye(n) - bn.B)
    print("causal_effects", causal_effects)
    print("true ces:", true_causal_effects)
    errors = true_causal_effects - mean_causal_effects
    print("errors:", errors)
    mse = (errors ** 2).sum() / (n * n - n)
    print("mse:", mse)
    assert mse < 0.1


def main():
    test_pairwise_causal_estimation()


if __name__ == "__main__":
    main()
