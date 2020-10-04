import numpy as np
from .weight_sum import weight_sum, weight_sum_contribs
from .zeta_transform import zeta_transform_array_inplace as zeta_transform

from .mcmc import PartitionMCMC, MC3

from .utils.bitmap import bm, bm_to_ints, msb, bm_to_pyint_chunks, bm_to_np64, bms_to_np64, np64_to_bm, fbit, kzon, dkbit, ikbit, subsets_size_k, ssets
from .utils.core_utils import arg
from .utils.io import read_candidates, get_n
from .utils.math_utils import log_minus_exp, close, comb, subsets

from .scoring import DiscreteData, ContinuousData, BDeu, BGe
from .scorer import BDeu as BDeu2
from .CandidateRestrictedScore import CandidateRestrictedScore
from .DAGR import DAGR as DAGR_precompute

import sumu.candidates_no_r as cnd


class Gadget():

    def __init__(self, **kwargs):
        self.datapath = arg("datapath", kwargs)
        self.scoref = arg("scoref", kwargs)
        self.ess = arg("ess", kwargs)
        self.maxid = arg("max_id", kwargs)
        self.K = arg("K", kwargs)
        self.d = arg("d", kwargs)
        self.n = get_n(self.datapath)
        self.cp_algo = arg("cp_algo", kwargs)
        self.cp_path = arg("cp_path", kwargs)
        self.mc3_chains = arg("mc3_chains", kwargs)
        self.burn_in = arg("burn_in", kwargs)
        self.iterations = arg("iterations", kwargs)
        self.thinning = arg("thinning", kwargs)
        self.tolerance = arg("tolerance", kwargs)
        self.stats = dict()

    def sample(self):
        self._find_candidate_parents()
        self._precompute_scores_for_all_candidate_psets()
        self._precompute_candidate_restricted_scoring()
        self._precompute_candidate_complement_scoring()
        self._init_mcmc()
        self._run_mcmc()
        return self._sample_dags()

    def _find_candidate_parents(self):
        self.l_score = LocalScore(self.datapath, scoref=self.scoref,
                                  maxid=self.maxid, ess=self.ess,
                                  stats=self.stats)

        if self.cp_path is None:
            # NOTE: datapath is only used by ges, pc and hc which are
            #       not in the imported candidates_no_r
            self.C = cnd.algo[self.cp_algo](self.K, n=self.n,
                                            scores=self.l_score,
                                            datapath=self.datapath)
        else:
            self.C = read_candidates(self.cp_path)

    def _precompute_scores_for_all_candidate_psets(self):
        self.score_array = self.l_score.as_array(self.C)

    def _precompute_candidate_restricted_scoring(self):
        # TODO: New CandidateRestrictedScore takes candidate parents
        # as np.array. Should adapt to this change everywhere else too.
        C = np.empty((self.n, self.K), dtype=np.int32)
        for v in self.C:
            C[v] = np.array(self.C[v])

        self.c_r_score = CandidateRestrictedScore(self.score_array, C, self.K)

        # self.c_r_score = TOBEREMOVED_CandidateRestrictedScore(self.score_array, self.C,
        #                                                       tolerance=self.tolerance,
        #                                                       stats=self.stats)

    def _precompute_candidate_complement_scoring(self):
        self.c_c_score = None
        if self.K < self.n - 1:
            # NOTE: CandidateComplementScore gives error if K >= n-1.
            self.l_score = LocalScore(self.datapath, scoref=self.scoref,
                                      maxid=self.d, ess=self.ess)
            self.c_c_score = CandidateComplementScore(self.C, self.l_score, self.d)
            del self.l_score

    def _init_mcmc(self):

        self.score = Score(C=self.C,
                           score_array=self.score_array,
                           c_r_score=self.c_r_score,
                           c_c_score=self.c_c_score)

        if self.mc3_chains > 1:
            self.mcmc = MC3([PartitionMCMC(self.C, self.score, self.d,
                                           temperature=i/(self.mc3_chains-1),
                                           stats=self.stats)
                             for i in range(self.mc3_chains)],
                            stats=self.stats)
        else:
            self.mcmc = PartitionMCMC(self.C, self.score, self.d,
                                      stats=self.stats)

    def _run_mcmc(self):
        for i in range(self.burn_in):
            self.mcmc.sample()
        self.Rs = list()
        for i in range(self.iterations):
            if i % self.thinning == 0:
                self.Rs.append(self.mcmc.sample()[0])
            else:
                self.mcmc.sample()

    def _sample_dags(self):
        ds = DAGR(self.score, self.C, tolerance=self.tolerance,
                  stats=self.stats)
        self.dags = [[] for i in range(len(self.Rs))]
        self.dag_scores = [0]*len(self.Rs)
        for v in self.C:
            ds.precompute_pset_sampling(v)
            for i in range(len(self.Rs)):
                family, family_score = ds.sample_pset(v, self.Rs[i],
                                                      score=True)
                self.dags[i].append(family)
                self.dag_scores[i] += family_score
        return self.dags, self.dag_scores


class DAGR:

    def __init__(self, score, C, tolerance=2**(-32), stats=None):
        self.stats = None
        if stats is not None:
            self.stats = stats
            self.stats[type(self).__name__] = dict()
            self.stats[type(self).__name__]["CC"] = 0

        C_m = np.empty((len(C), len(C[0])), dtype=np.int32)
        for v in C:
            C_m[v] = np.array(C[v])

        self.pc = DAGR_precompute(score.score_array, C_m, len(C[0]))

        self.score = score
        self.C = C
        self.tol = tolerance

    def precompute_pset_sampling(self, v):

        self.pc.precompute(v)

        K = len(self.C[v])

        # self._f = [0]*2**K
        # for X in range(2**K):
        #     self._f[X] = np.array([-float("inf")]*2**(K-bin(X).count("1")))
        #     self._f[X][0] = self.score.score_array[v][X]

        # for k in range(1, K+1):
        #     for k_x in range(K-k+1):
        #         for X in subsets_size_k(k_x, K):
        #             for Y in subsets_size_k(k, K-k_x):
        #                 i = fbit(Y)
        #                 self._f[X][Y] = np.logaddexp(self._f[kzon(X, i)][dkbit(Y, i)], self._f[X][Y & ~(Y & -Y)])

    def sample_pset(self, v, R, score=False):

        # TODO: keep track of positions in move functions
        for i in range(len(R)):
            if v in R[i]:
                break

        if i == 0:
            family = (v,)
            family_score = self.score.score_array[v][0]

        else:

            U = set().union(*R[:i])
            T = R[i-1]

            # NOTE: Current CandidateRestrictedScore requires these
            # bitmap representations; probably better to move the
            # logic to the c++ side for a cleaner interface.
            U_bm = bm(U.intersection(self.C[v]), ix=self.C[v])
            T_bm = bm(T.intersection(self.C[v]), ix=self.C[v])

            w_C = -float("inf")
            if len(T.intersection(self.C[v])) > 0:
                w_C = self.score.c_r_score.sum(v, U_bm, T_bm)

            w_compl_sum = -float("inf")
            if self.score.c_c_score is not None:
                w_compl_sum, contribs = self.score.c_c_score.sum(v, U, T, -float("inf"), contribs=True)

            if -np.random.exponential() < w_C - np.logaddexp(w_compl_sum, w_C):
                pset = tuple(sorted(self._sample_pset(v, set().union(*R[:i]), R[i-1])))
                family = (v, pset)
                family_score = self.score.score_array[v][bm(family[1], ix=self.C[v])]
            else:
                p = np.array([self.score.c_c_score.ordered_scores[v][j]
                              for j in contribs])
                p = np.exp(p - w_compl_sum)
                j = np.random.choice(contribs, p=p)
                pset = np64_to_bm(self.score.c_c_score.ordered_psets[v][j])
                family_score = self.score.c_c_score.ordered_scores[v][j]
                family = (v, bm_to_ints(pset))

        if score is True:
            return family, family_score
        return family

    def _sample_pset(self, v, U, T):

        def g(X, E, U, T):

            X_bm = bm(X, ix=self.C[v])
            E_bm = bm(E, ix=sorted(set(self.C[v]).difference(X)))
            U_bm = bm(U.difference(X), ix=sorted(set(self.C[v]).difference(X)))
            T_bm = bm(T.difference(X), ix=sorted(set(self.C[v]).difference(X)))

            # score_1 = [self._f[X_bm][U_bm & ~E_bm] if X.issubset(U.difference(E)) else -float("inf")][0]
            # score_2 = [self._f[X_bm][(U_bm & ~E_bm) & ~T_bm] if X.issubset(U.difference(E.union(T))) else -float("inf")][0]

            score_1 = [self.pc.f(X_bm, U_bm & ~E_bm) if X.issubset(U.difference(E)) else -float("inf")][0]
            score_2 = [self.pc.f(X_bm, (U_bm & ~E_bm) & ~T_bm) if X.issubset(U.difference(E.union(T))) else -float("inf")][0]

            if not close(score_1, score_2, self.tol):
                return log_minus_exp(score_1, score_2)
            else:  # CC
                return None

        U = U.intersection(self.C[v])
        T = T.intersection(self.C[v])

        X = set()
        E = set()
        for i in U:
            try:
                if -np.random.exponential() < g(X.union({i}), E, U, T) - g(X, E, U, T):
                    X.add(i)
                else:
                    E.add(i)
            except TypeError:
                if self.stats is not None:
                    self.stats[type(self).__name__]["CC"] += 1

                return self._sample_pset_brute(v, U, T)
        return X

    def _sample_pset_brute(self, v, U, T):

        U = U.intersection(self.C[v])
        T = T.intersection(self.C[v])

        # TODO: These have known lengths so initialize them to np.arrays
        #       of correct size.
        probs = list()
        psets = list()
        for T_set in subsets(T, 1, len(T)):
            for U_set in subsets(U.difference(T), 0, len(U.difference(T))):
                pset = set(T_set).union(U_set)
                probs.append(self.score.score_array[v][bm(pset, ix=self.C[v])])
                psets.append(pset)
        probs = np.array(probs)
        probs -= np.logaddexp.reduce(probs)
        probs = np.exp(probs)
        return psets[np.random.choice(range(len(psets)), p=probs)]


class LocalScore:
    """Class for computing local scores given input data.
    """

    def __init__(self, datapath, scoref="bdeu", maxid=-1, ess=10, stats=None):
        # NOTE: __init__ creates self.local(node, parents) function.
        self.datapath = datapath
        self.scoref = scoref
        self.maxid = maxid
        self.ess = ess
        self.stats = None
        if stats is not None:
            self.stats = stats
            self.stats[type(self).__name__] = dict()
            self.stats[type(self).__name__]["clear_cache"] = 0

        if self.scoref == "bdeu":

            def local(node, parents):
                # NOTE: Cache is not emptied ever
                node = np.int32(node)
                p_cliq = np.sort(np.array(parents, dtype=np.int32))
                idx = p_cliq.searchsorted(node)
                pi_cliq = np.concatenate((p_cliq[:idx], [node], p_cliq[idx:]))
                spi = self.cliq_cache.get(tuple(pi_cliq))
                sp = self.cliq_cache.get(tuple(p_cliq))
                if spi is None:
                    spi = self.scorer.cliq(pi_cliq)
                    self.cliq_cache[tuple(pi_cliq)] = spi
                if not parents:
                    return spi - np.log(comb(self.n - 1, len(parents)))
                if sp is None:
                    sp = self.scorer.cliq(p_cliq)
                    self.cliq_cache[tuple(p_cliq)] = sp
                return spi - sp - np.log(comb(self.n - 1, len(parents)))

            d = np.loadtxt(self.datapath, dtype=np.int32)
            r = d[1]
            d = d[2:]
            self.n = d.shape[1]

            self.scorer = BDeu2()
            self.scorer.read(d)
            self.scorer.set_ess(self.ess)
            # self.scorer.set_r(r)
            self.cliq_cache = dict()
            self._local = local

        elif self.scoref == "bge":

            def local(node, parents):
                if len(self.score._cache) > 1000000:
                    self.score._cache = dict()
                    if self.stats:
                        self.stats[type(self).__name__]["clear_cache"] += 1
                if self.maxid == -1 or len(parents) <= self.maxid:
                    return self.score.bge_score(node, parents)[0] - np.log(comb(self.n - 1, len(parents)))
                else:
                    return -float("inf")

            d = ContinuousData(self.datapath, header=False)
            d._varidx = {v: v for v in d._varidx.values()}
            self.n = len(d._variables)
            self.score = BGe(d)
            self._local = local

    def local(self, v, pset):
        """Local score for input node v and pset, with score function self.scoref.

        This is the "safe" version, raising error if queried for variables not existing in data.
        The unsafe self._local will just segfault.
        """
        if v in pset:
            return -float("inf")
        if min(v, min(pset)) < 0 or max(v, max(pset)) >= self.n:
            raise IndexError("Attempting to query score for non-existing variables")
        return self._local(v, pset)

    def complementary_scores_dict(self, C, d):
        """C candidates, d indegree for complement psets"""
        cscores = dict()
        for v in C:
            cscores[v] = dict()
            for pset in subsets([u for u in C if u != v], 1, d):
                if not (set(pset)).issubset(C[v]):
                    cscores[v][pset] = self._local(v, pset)
        return cscores

    def all_scores_dict(self, C=None):
        # NOTE: Not used in Gadget pipeline, but useful for example
        # when computing input data for aps.
        scores = dict()
        if C is None:
            C = {v: tuple(sorted(set(range(self.n)).difference({v}))) for v in range(self.n)}
        for v in C:
            tmp = dict()
            for pset in subsets(C[v], 0, [len(C[v]) if self.maxid == -1 else self.maxid][0]):
                tmp[pset] = self._local(v, pset)
            scores[v] = tmp
        return scores

    def as_array(self, C):
        scores = np.full((self.n, 2**len(C[0])), -float('inf'))
        for v in range(self.n):
            for pset in subsets(C[v], 0, [len(C[v]) if self.maxid == -1 else self.maxid][0]):
                scores[v, bm(pset, ix=C[v])] = self._local(v, pset)

        return scores


class Score:

    def __init__(self, **kwargs):

        self.C = arg("C", kwargs)
        self.score_array = arg("score_array", kwargs)
        self.c_r_score = arg("c_r_score", kwargs)
        self.c_c_score = arg("c_c_score", kwargs)

    def sum(self, v, U, T):

        # NOTE: Current CandidateRestrictedScore requires these
        # bitmap representations; probably better to move the
        # logic to the c++ side for a cleaner interface.
        U_bm = bm(U.intersection(self.C[v]), ix=self.C[v])
        T_bm = bm(T.intersection(self.C[v]), ix=self.C[v])

        W_prime = self.c_r_score.sum(v, U_bm, T_bm)
        if self.c_c_score is None:
            return W_prime
        return self.c_c_score.sum(v, U, T, W_prime)[0]


class TOBEREMOVED_CandidateRestrictedScore:
    """For computing the local score sum given root-partition and candidate parents.
    """

    def __init__(self, score_array, C, tolerance=2**(-32), stats=None):

        self.stats = None
        if stats is not None:
            self.stats = stats
            self.stats[type(self).__name__] = dict()
            self.stats[type(self).__name__]["CC"] = 0
            self.stats[type(self).__name__]["CC basecases"] = 0
            self.stats[type(self).__name__]["basecases"] = 0

        self.score_array = score_array
        self.C = C
        self.tol = tolerance
        self._precompute_a()
        self._precompute_basecases()
        self._precompute_psum()

    def _precompute_a(self):
        self._a = [0]*len(self.score_array)
        for v in range(len(self.score_array)):
            self._a[v] = zeta_transform(self.score_array[v])

    def _precompute_basecases(self):
        K = len(self.C[0])
        self._psum = {v: dict() for v in range(len(self.C))}
        for v in self.C:
            for k in range(K):
                x = 1 << k
                U_minus_x = (2**K - 1) & ~x
                tmp = [0]*2**(K-1)
                tmp[0] = self.score_array[v][x]
                self._psum[v][x] = dict()
                for S in ssets(U_minus_x):
                    if S | x not in self._psum[v]:
                        self._psum[v][S | x] = dict()
                    tmp[dkbit(S, k)] = self.score_array[v][S | x]
                tmp = zeta_transform(tmp)
                if self.stats:
                    self.stats[type(self).__name__]["basecases"] += len(tmp)
                for S in range(len(tmp)):
                    # only save basecase if it can't be computed as difference
                    # makes a bit slower, makes require a bit less space
                    if self._cc(v, ikbit(S, k, 1), x):
                        if self.stats:
                            self.stats[type(self).__name__]["CC basecases"] += 1
                        self._psum[v][ikbit(S, k, 1)][x] = tmp[S]

    def _precompute_psum(self):

        K = len(self.C[0])
        n = len(self.C)

        for v in self.C:
            for U in range(1, 2**K):
                for T in ssets(U):
                    if self._cc(v, U, T):
                        if self.stats:
                            self.stats[type(self).__name__]["CC"] += 1
                        T1 = T & -T
                        U1 = U & ~T1
                        T2 = T & ~T1

                        self._psum[v][U][T] = np.logaddexp(self._sum(v, U, T1),
                                                           self._sum(v, U1, T2))
        if self.stats:
            self.stats[type(self).__name__]["relative CC"] = self.stats[type(self).__name__]["CC"] / (n*3**K)

    def _cc(self, v, U, T):
        return close(self._a[v][U], self._a[v][U & ~T], self.tol)

    def _sum(self, v, U, T):
        # NOTE: As opposed to .sum() this functions takes as input
        #       bitmaps in the space of candidate parents.
        #       Presumably the two functions might merge, if the
        #       root-partition representation is unified somehow.
        if U == 0 and T == 0:
            return self.score_array[v][0]
        if T == 0:  # special case for T2 in precompute
            return -float("inf")
        if v in self._psum and U in self._psum[v] and T in self._psum[v][U]:
            return self._psum[v][U][T]
        else:
            return log_minus_exp(self._a[v][U], self._a[v][U & ~T])

    def sum(self, v, U, T):
        """Computes the local score sum of node v with parents from U and
        at least one parent from T.

        Args:
           v (int) node label in part :math:`R_{t}`
           U (set) set of node labels in parts :math:`R_{1,t-1}`
           T (set) set of node labels in part :math:`R_{t-1}`

        Returns:
            double: sum of scores satisfying the constraints.

        """

        if len(T) == 0:
            return self.score_array[v][0]

        if len(T.intersection(self.C[v])) > 0:
            W_prime = self._sum(v, bm(U.intersection(self.C[v]), ix=self.C[v]),
                                bm(T.intersection(self.C[v]), ix=self.C[v]))
        else:
            W_prime = -float("inf")

        return W_prime


class CandidateComplementScore:
    """For computing the local score sum complementary to those obtained from :py:class:`.old_CandidateRestrictedScore` and constrained by maximum indegree.
    """

    def __init__(self, C, scores, d):

        self.C = C
        self.n = len(C)
        self.d = d
        minwidth = 1
        if self.n > 64:
            minwidth = 2

        # object of class LocalScore
        scores = scores.complementary_scores_dict(C, d)

        ordered_psets = dict()
        ordered_scores = dict()

        for v in scores:
            ordered_scores[v] = sorted(scores[v].items(), key=lambda item: item[1], reverse=True)
            ordered_psets[v] = bms_to_np64([bm(item[0]) for item in ordered_scores[v]], minwidth=minwidth)
            ordered_scores[v] = np.array([item[1] for item in ordered_scores[v]], dtype=np.float64)

        self.ordered_psets = ordered_psets
        self.ordered_scores = ordered_scores

        if self.d == 1:
            self.pset_to_idx = dict()
            for v in scores:
                # wrong if over 64 variables?
                # ordered_psets[v] = ordered_psets[v].flatten()
                ordered_psets[v] = [np64_to_bm(pset) for pset in ordered_psets[v]]
                self.pset_to_idx[v] = dict()
                for i, pset in enumerate(ordered_psets[v]):
                    self.pset_to_idx[v][pset] = i

        self.t_ub = np.zeros(shape=(self.n, self.n), dtype=np.int32)
        for u in range(1, self.n+1):
            for t in range(1, u+1):
                self.t_ub[u-1][t-1] = self.n_valids_ub(u, t)

    def n_valids(self, v, U, T):
        n = 0
        for k in range(1, self.d+1):
            n += comb(len(U), k) - comb(len(U.intersection(self.C[v])), k)
            n -= comb(len(U.difference(T)), k) - comb(len(U.difference(T).intersection(self.C[v])), k)
        return n

    def n_valids_ub(self, u, t):
        n = 0
        for k in range(self.d+1):
            n += comb(u, k) - comb(u - t, k)
        return n

    def sum(self, v, U, T, W_prime, debug=False, contribs=False):

        # NOTE: This is the final score calculation.

        if self.d == 1:  # special case
            contribs = list()
            w_contribs = list()
            for u in T:
                if u not in self.C[v]:
                    pset_idx = self.pset_to_idx[v][bm(u)]
                    contribs.append(pset_idx)
                    w_contribs.append(self.ordered_scores[v][pset_idx])
            w_contribs.append(W_prime)
            return np.logaddexp.reduce(w_contribs), contribs

        if self.n <= 64:
            U_bm = bm(U)
            T_bm = bm(T)
        else:  # if 64 < n <= 128
            U_bm = bm_to_pyint_chunks(bm(U), 2)
            T_bm = bm_to_pyint_chunks(bm(T), 2)

        # contribs should be a param to weight_sum()
        if contribs is True:
            return weight_sum_contribs(W_prime,
                                       self.ordered_psets[v],
                                       self.ordered_scores[v],
                                       self.n,
                                       U_bm,
                                       T_bm,
                                       int(self.t_ub[len(U)][len(T)]))

        W_sum = weight_sum(W_prime,
                           self.ordered_psets[v],
                           self.ordered_scores[v],
                           self.n,
                           U_bm,
                           T_bm,
                           int(self.t_ub[len(U)][len(T)]))

        # TODO: replace this with some raise Error
        if W_sum == -float("inf"):
            print("INFFI")
            print("U types {}".format([type(x) for x in U]))
            print("T types {}".format([type(x) for x in T]))
            print("W_prime {}".format(W_prime))
            #np.set_printoptions(threshold=np.inf)
            #print(self.ordered_psets[v])
            np.save("psets.npy", self.ordered_psets[v])
            np.save("weights.npy", self.ordered_scores[v])
            #print(self.ordered_scores[v])
            print("U {}".format(bm(U)))
            print("T {}".format(bm(T)))
            print("t_ub {}".format(int(self.t_ub[len(U)][len(T)])))
            exit()
        return W_sum, None