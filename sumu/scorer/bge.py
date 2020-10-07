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
        TN = T0 + (N - 1) * np.cov(data.T) + ((am * N) / (am + N)) * np.outer((mu0 - np.mean(data, axis=0)), (mu0 - np.mean(data, axis=0)))
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

        self._cache = {frozenset(): 0}

    def clear_cache(self):
        self._cache = {frozenset(): 0}

    def precompute_dets(self):  # Function of v and C
        self.det = mat2pm(self.TN)

    def score_component_precomputed(self, nodes):
        try:
            return self._cache[nodes]
        except KeyError:
            component = -0.5*(self.awpN - self.n + len(nodes)) * np.log(self.det[bm(nodes) - 1])
            self._cache[nodes] = component
            return component

    def score_component(self, nodes):
        try:
            return self._cache[nodes]
        except KeyError:
            _nodes = np.array(list(nodes))
            k = len(nodes)
            component = -0.5*(self.awpN - self.n + k)
            if k == 1:
                D = self.TN[_nodes[:, None], _nodes]
                component *= np.log(D)[0, 0]

            else:
                D = self.TN[_nodes[:, None], _nodes]
                component *= np.linalg.slogdet(D)[1]

            self._cache[nodes] = component
            return component

    def score_precomputed(self, v, pset):
        return self.scoreconstvec[len(pset)] \
            + self.score_component_precomputed(frozenset(pset).union({v})) \
            - self.score_component_precomputed(frozenset(pset))

    def score(self, v, pset):
        return self.scoreconstvec[len(pset)] \
            + self.score_component(frozenset(pset).union({v})) \
            - self.score_component(frozenset(pset))


def bm(ints, ix=None):
    if type(ints) not in [set, tuple]:
        ints = {int(ints)}
    if ix is not None:
        ints = [ix.index(i) for i in ints]
    bitmap = 0
    for k in ints:
        bitmap += 2**k
    return int(bitmap)  # without the cast np.int64 might sneak in somehow and break drv


def bitcmp(mask, k):
    return ((1 << k) - 1) ^ mask


def msb(n):
    blen = 0
    while (n > 0):
        n >>= 1
        blen += 1
    return blen


def mat2pm(a, thresh=None):
    n = a.shape[0]
    scale = np.abs(a).mean()
    if scale == 0:
        scale = 1
    ppivot = scale
    if thresh is None:
        thresh = 1.e-5*scale

    zeropivs = set()
    pm = np.zeros(2**n - 1)
    ipm = 0
    q = np.zeros((1, n, n))
    q[0,...] = a
    pivmin = float("inf")

    for level in range(n):
        nq, n1, n1 = q.shape
        qq = np.zeros((nq*2, n1-1, n1-1))
        ipm1 = 0
        for i in range(nq):
            a = q[i]
            pm[ipm] = a[0, 0]
            if n1 > 1:
                abspiv = abs(pm[ipm])
                if abspiv <= thresh:
                    zeropivs.add(ipm)
                    pm[ipm] += ppivot
                    abspiv = abs(pm[ipm])
                if abspiv < pivmin:
                    pivmin = abspiv

                b = a[1:n1+1, 1:n1+1]
                d = a[1:n1+1, 0] / pm[ipm]
                c = b - np.outer(d, a[0, 1:n1+1])

                qq[i] = b
                qq[i+nq] = c

            if i > 0:
                pm[ipm] *= pm[ipm1]
                ipm1 += 1

            ipm += 1
        q = qq

    for mask in zeropivs:
        delta = msb(mask)
        delta2 = 2*delta
        ipm1 = mask & bitcmp(delta, 48)
        if ipm1 == 0:
            pm[mask] -= ppivot
        else:
            pm[mask] = (pm[mask] / pm[imp1] - ppivot) * pm[ipm1]
        for j in range(mask+delta2, 2**n, delta2):
            pm[j] -= ppivot*pm[j - delta]

    return pm
