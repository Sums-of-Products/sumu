import numpy as np
from _multivariate import multivariate_t as mvt


def BEEBS(dags, data, joint=False):

    n = data.shape[1]
    As = np.zeros((len(dags), n*n - n))
    if joint:
        pairs = np.array([np.array(pair)
                          for pair in zip(range(n), range(1, n+1))])
        As = np.zeros((len(pairs), 2 + 2 * (n - 2)))
        As[:, :2] = pairs
        print(As.shape)

    # Prior parameters.
    nu = np.zeros(n)
    am = 1
    aw = n + am + 1
    Tmat = np.identity(n) * (aw - n - 1) / (am + 1)

    # Sufficient statistics
    xN = np.mean(X, axis=0)
    SN = (X - xN).T @ (X - xN)

    # Parameters for the posterior are.
    nuN = (am * nu + N * xN) / (am + N)
    amN = am + N
    awN = aw + N
    R = Tmat + SN + am * N / (am + N) * (nu - xN) @ (nu - xN).T

    psd = list()

    for i, dag in enumerate(dags):
        Bmcmc = np.zeros((n, n))
        for node in range(n):
            pa = np.where(dag[node])[0]
            if len(pa) == 0:
                continue
            l = len(pa) + 1

            R11 = R[node, node]
            R12 = R[pa, node]
            R21 = R[node, pa]
            R11inv = np.linalg.inv(R[pa[:, None], pa])

            df = awN - n + l
            mb = R11inv @ R12
            divisor = R11 - R11inv @ R12
            covb = divisor / df * R11inv

            # https://stackoverflow.com/questions/41515522/numpy-positive-semi-definite-warning
            covb -= 1e-12 * np.eye(covb.shape[0])

            # psd.append(isPSD(covb))
            b = mvt.rvs(loc=mb, shape=covb, df=df)

            Bmcmc[node, pa] = b
            if not joint:
                A = np.linalg.inv(np.eye(n) - Bmcmc)
                As[i] = A.T[np.eye(n) == 0]
            else:
                for j in range(pairs.shape[1]):
                    Umat = np.eye(n)
                    Umat[pairs[j], :] = 0
                    A = np.linalg.inv(np.eye(n) - Umat @ Bmcmc)
                    As[j, 2:As.shape[1]] += A[~np.in1d(np.arange(n), pairs[j])][:, pairs[j]].flatten() / len(dags)

    return As
