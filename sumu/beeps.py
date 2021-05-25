import numpy as np
from scipy.stats import multivariate_t as mvt
from .gadget import Data
from .bnet import family_sequence_to_adj_mat


class Beeps:

    def __init__(self, *, dags, data):
        self.dags = dags
        if not type(self.dags[0]) == np.ndarray:
            self.dags = [family_sequence_to_adj_mat(d) for d in self.dags]
        self.data = Data(data)
        n = self.data.n
        N = self.data.N
        self.Bs = None

        # Prior parameters
        nu = np.zeros(n)
        am = 1
        aw = n + am + 1
        Tmat = np.identity(n) * (aw - n - 1) / (am + 1)

        # Sufficient statistics
        xN = np.mean(data.data, axis=0)
        SN = (data.data - xN).T @ (data.data - xN)

        # Parameters for the posterior
        nuN = (am * nu + N * xN) / (am + N)
        amN = am + N
        self.awN = aw + N
        self.R = Tmat + SN + ((am * N) / (am + N)) * np.outer((nu - xN), (nu - xN))


    def sample_pairwise(self):
        As = np.ones((len(self.dags), self.data.n, self.data.n))
        Bs = self._sample_Bs()
        n = self.data.n
        for i in range(Bs.shape[0]):
            As[i] = np.linalg.inv(np.eye(self.data.n) - Bs[i])
        return As

    def sample_direct(self):
        return self._sample_Bs() + np.eye(self.data.n)

    def sample_joint(self, *, y, x, resample=False):
        As = np.ones((len(self.dags), len(y), len(x)))
        Bs = self._sample_Bs(resample)
        n = self.data.n
        for i in range(Bs.shape[0]):
            Umat = np.eye(n)
            Umat[x, :] = 0
            A = np.linalg.inv(np.eye(n) - Umat @ Bs[i])
            A = A[y, :][:, x]
            As[i] = A
        return As

    def _sample_Bs(self, resample=True):
        if resample is False and self.Bs is not None:
            return self.Bs
        n = self.data.n
        R = self.R
        awN = self.awN
        Bs = np.zeros((len(self.dags), n, n))
        for i, dag in enumerate(self.dags):
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
                b = mvt.rvs(loc=mb, shape=covb, df=df)
                Bs[i, node, pa] = b
        self.Bs = Bs
        return Bs


def beeps(dags, data, y=None, x=None):

    joint = False
    if y is not None:
        joint = True

    if not type(dags[0]) == np.ndarray:
        dags = [family_sequence_to_adj_mat(d) for d in dags]

    data = Data(data)
    n = data.n
    N = data.N

    A_shape = (len(dags), n, n)
    if joint is True:
        A_shape = (len(dags), len(y), len(x))
    As = np.ones(A_shape)

    # Prior parameters
    nu = np.zeros(n)
    am = 1
    aw = n + am + 1
    Tmat = np.identity(n) * (aw - n - 1) / (am + 1)

    # Sufficient statistics
    xN = np.mean(data.data, axis=0)
    SN = (data.data - xN).T @ (data.data - xN)

    # Parameters for the posterior
    nuN = (am * nu + N * xN) / (am + N)
    amN = am + N
    awN = aw + N
    R = Tmat + SN + ((am * N) / (am + N)) * np.outer((nu - xN), (nu - xN))

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
            b = mvt.rvs(loc=mb, shape=covb, df=df)

            Bmcmc[node, pa] = b

        if joint:
            Umat = np.eye(n)
            Umat[x, :] = 0
            A = np.linalg.inv(np.eye(n) - Umat @ Bmcmc)
            A = A[y, :][:, x]
            As[i] = A
        else:
            A = np.linalg.inv(np.eye(n) - Bmcmc)
            As[i] = A

    return As
