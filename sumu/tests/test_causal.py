import numpy as np
import sumu


def test_pairwise_causal_estimation():
    np.random.seed(0)
    n = 6
    bn = sumu.GaussianBNet.random(n=n)
    data = bn.sample(20000)
    dags, scores = sumu.Gadget(data=data, mcmc={"iters": 80000}).sample()
    causal_effects = sumu.beeps(dags, data)
    mse = (
        (np.linalg.inv(np.eye(n) - bn.B) - causal_effects.mean(axis=0)) ** 2
    ).mean()
    print(mse)
    assert mse < 0.1


def main():
    test_pairwise_causal_estimation()


if __name__ == "__main__":
    main()
