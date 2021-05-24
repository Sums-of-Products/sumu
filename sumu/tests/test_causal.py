import numpy as np
import sumu


# There is probably a better place for this function
def create_data(n, N, enb=4, lb_e=0.1, ub_e=2, lb_ce=0.5, ub_ce=1.5):
    pedge = min(1, enb/(n-1))
    order = np.random.permutation(n)
    G = np.tril(np.random.choice(range(2), (n,n), p=[1-pedge, pedge]), k=-1)
    G = G[:, order][order, :]
    B = G * np.random.uniform(lb_e, ub_e, (n,n)) * np.random.choice((-1, 1), (n, n))
    Ce = np.diag(np.random.uniform(lb_ce, ub_ce, n))
    iA = np.linalg.inv(np.eye(n)-B)
    X = np.random.normal(size=(N, n))
    X = (iA @ np.sqrt(Ce) @ X.T).T
    return X, {"G": G, "Ce": Ce, "B": B}


def test_pairwise_causal_estimation():
    np.random.seed(0)
    n = 6
    data, M = create_data(n, 20000, enb=2)
    dags, scores = sumu.Gadget(data=data, mcmc={"iters": 80000}).sample()
    causal_effects = sumu.beeps(dags, data)
    mse = ((np.linalg.inv(np.eye(n) - M["B"]) - causal_effects.mean(axis=0))**2).mean()
    print(mse)
    assert mse < 0.1


def main():
    test_pairwise_causal_estimation()


if __name__ == '__main__':
    main()
