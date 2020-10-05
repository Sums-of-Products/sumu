import numpy as np
from scipy.special import loggamma as lgamma
from scipy.linalg import solve_triangular


class BGe:
    """Ported to Python from the R version by Jack Kuipers and Giusi Moffa
    :cite:`kuipers:2014`.
    """

    def __init__(self, data):
        n = data.shape[1]
        N = data.shape[0]
        mu0 = np.zeros(n)

        # Scoring parameters.
        am = 1
        aw = n + am + 1
        T0scale = am * (aw - n - 1) / (am + 1)
        T0 =  T0scale * np.eye(n)
        TN = T0 + (N - 1) * np.cov(data.T) + ((am * N) / (am + N)) * (mu0 + np.mean(data, axis=0)) @ (mu0 - np.mean(data, axis=0))
        awpN = aw + N
        constscorefact = -(N / 2) * np.log(np.pi) + 0.5 * np.log(am / (am + N))
        scoreconstvec = np.zeros(n)
        for i in range(n):
            awp = aw - n + i + 1
            scoreconstvec[i] = constscorefact - lgamma(awp / 2) + lgamma((awp + N) / 2) + (awp + i) / 2 * np.log(T0scale)

        # Just to keep the above calculations cleaner
        self.data = data
        self.n = n
        self.N = N
        self.mu0 = mu0
        self.am = am
        self.aw = aw
        self.T0scale = T0scale
        self.T0 = T0
        self.TN = TN
        self.awpN = awpN
        self.constscorefact = constscorefact
        self.scoreconstvec = scoreconstvec

    def DAGcorescore(self, v, pset):
        lp = len(pset)
        awpNd2 = (self.awpN - self.n + lp + 1) / 2
        A = self.TN[v, v]

        if lp == 0:
            return self.scoreconstvec[lp] - awpNd2 * np.log(A)
        if lp == 1:
            D = self.TN[pset[:, None], pset]
            logdetD = np.log(D)[0, 0]
            B = self.TN[v, pset]
            logdetpart2 = np.log(A - B**2 / D)[0, 0]
            return self.scoreconstvec[lp] - awpNd2 * logdetpart2 - logdetD / 2

        if lp < 4:  # Note: The limit is a bit arbitrary?
            D = self.TN[pset[:, None], pset]
            logdetD = np.linalg.slogdet(D)[1]
            B = self.TN[v, pset]
            logdetpart2 = np.linalg.slogdet(D - np.outer(B, B) / A)[1] + np.log(A) - logdetD
            return self.scoreconstvec[lp] - awpNd2 * logdetpart2 - logdetD / 2

        else:  # This seems about 2x slower than the above
            D = self.TN[pset[:, None], pset]
            choltemp = np.linalg.cholesky(D).T
            logdetD = 2 * np.log(np.prod(choltemp.flatten()[(lp + 1) * np.arange(lp)]))
            B = self.TN[v, pset]
            logdetpart2 = np.log(A - np.sum(solve_triangular(choltemp, B, trans=1)**2))
            return self.scoreconstvec[lp] - awpNd2 * logdetpart2 - logdetD / 2
