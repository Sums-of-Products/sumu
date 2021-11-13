import numpy as np
from scipy.special import loggamma as lgamma

from ..utils.math_utils import subsets
from ..utils.bitmap import bm, bm_to_np64


class BGe:
    """Ported to Python from the R version by Jack Kuipers and Giusi Moffa
    :footcite:`kuipers:2014`.
    """

    def __init__(self, *, data, maxid):

        self.maxid = maxid

        n = data.n
        N = data.N
        data = data.data
        mu0 = np.zeros(n)

        # Scoring parameters.
        am = 1
        aw = n + am + 1
        T0scale = am * (aw - n - 1) / (am + 1)
        T0 = T0scale * np.eye(n)
        TN = (
            T0
            + (N - 1) * np.cov(data.T)
            + ((am * N) / (am + N))
            * np.outer(
                (mu0 - np.mean(data, axis=0)), (mu0 - np.mean(data, axis=0))
            )
        )
        awpN = aw + N
        constscorefact = -(N / 2) * np.log(np.pi) + 0.5 * np.log(am / (am + N))
        scoreconstvec = np.zeros(n)
        for i in range(n):
            awp = aw - n + i + 1
            scoreconstvec[i] = (
                constscorefact
                - lgamma(awp / 2)
                + lgamma((awp + N) / 2)
                + (awp + i) / 2 * np.log(T0scale)
            )

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

    def score_component(self, nodes):
        try:
            return self._cache[nodes]
        except KeyError:
            _nodes = np.array(list(nodes))
            k = len(nodes)
            component = -0.5 * (self.awpN - self.n + k)
            if k == 1:
                D = self.TN[_nodes[:, None], _nodes]
                component *= np.log(D)[0, 0]

            else:
                D = self.TN[_nodes[:, None], _nodes]
                component *= np.linalg.slogdet(D)[1]

            self._cache[nodes] = component
            return component

    def local_precomputed(self, v, pset):
        # Both v and pset in the index of sorted( C[v] + [v] )
        return (
            self.scoreconstvec[len(pset)]
            + self.score_component_precomputed(frozenset(pset).union({v}))
            - self.score_component_precomputed(frozenset(pset))
        )

    def local(self, v, pset):
        return (
            self.scoreconstvec[len(pset)]
            + self.score_component(frozenset(pset).union({v}))
            - self.score_component(frozenset(pset))
        )

    def candidate_score_array(self, C_array):
        C = dict({v: tuple(C_array[v]) for v in range(C_array.shape[0])})
        scores = np.full((self.n, 2 ** len(C[0])), -float("inf"))
        for v in range(self.n):
            for pset in subsets(
                C[v], 0, [len(C[v]) if self.maxid == -1 else self.maxid][0]
            ):
                scores[v, bm(pset, ix=C[v])] = self.local(v, pset)
            self.clear_cache()

        return scores

    def complement_psets_and_scores(self, v, C, d):
        n = len(C)
        k = (n - 1) // 64 + 1
        pset_tuple = list(
            filter(
                lambda ss: not set(ss).issubset(C[v]),
                subsets([u for u in C if u != v], 1, d),
            )
        )
        pset_len = np.array(list(map(len, pset_tuple)), dtype=np.int32)
        pset_bm = list(
            map(lambda pset: bm_to_np64(bm(set(pset)), k), pset_tuple)
        )
        scores = np.array([self.local(v, pset) for pset in pset_tuple])
        self.clear_cache()
        return np.array(pset_bm), scores, pset_len
