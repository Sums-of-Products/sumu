import numpy as np

from utils import subsets, bm, bm_to_ints, log_minus_exp, comb
import zeta_transform.zeta_transform as zeta_transform
from scoring import DiscreteData, ContinuousData, BDeu, BGe


def close(a, b, tolerance):
    return 1 - a/b < tolerance


def fbit(mask):
    """get index of first set bit"""
    k = 0
    while 1 & mask == 0:
        k += 1
        mask >>= 1
    return k


def kzon(mask, k):
    """set kth zerobit on"""
    nmask = ~mask
    for i in range(k):
        nmask &= ~(nmask & -nmask)
    return mask | (nmask & -nmask)


def dkbit(mask, k):
    """drop kth bith"""
    if mask == 0:
        return mask
    trunc = mask >> (k+1)
    #trunc <<= k
    trunc *= 2**k
    # return ((1 << k) - 1) & mask | trunc
    return (2**k - 1) & mask | trunc


def ikbit(mask, k, bit):
    """shift all bits >=k to the left and insert bit to k"""
    if k == 0:
        # newmask = mask << 1
        newmask = mask * 2
    else:
        newmask = mask >> k-1
    newmask ^= (-bit ^ newmask) & 1
    #newmask <<= k
    newmask *= 2**k
    #return newmask | ((1 << k) - 1) & mask
    return newmask | (2**k - 1) & mask


def subsets_size_k(k, n):
    if k == 0:
        yield 0
        return
    #S = (1 << k) - 1
    S = 2**k - 1
    # limit = (1 << n)
    limit = 2**n
    while S < limit:
        yield S
        x = S & -S
        r = S + x
        S = (((r ^ S) >> 2) // x) | r


def ssets(mask):
    S = mask
    while S > 0:
        yield S
        S = (S - 1) & mask


class DAGR:

    def __init__(self, scores, C, tolerance=1e-32, stats=None):
        if stats is not None:
            self.stats = stats
            self.stats[type(self).__name__] = dict()
            self.stats[type(self).__name__]["CC"] = 0

        self.scores = scores.scores
        self.C = C
        self.tol = tolerance

    def precompute(self, v):

        K = len(self.C[v])

        self._f = [0]*2**K
        for X in range(2**K):
            self._f[X] = [-float("inf")]*2**(K-bin(X).count("1"))
            self._f[X][0] = self.scores[v][X]

        for k in range(1, K+1):
            for k_x in range(K-k+1):
                for X in subsets_size_k(k_x, K):
                    for Y in subsets_size_k(k, K-k_x):
                        i = fbit(Y)
                        self._f[X][Y] = np.logaddexp(self._f[kzon(X, i)][dkbit(Y, i)], self._f[X][Y & ~(Y & -Y)])

    def sample(self, v, R, score=False):

        if v in R[0]:
            family = (v,)
            family_score = self.scores[v][0]

        else:
            for i in range(1, len(R)):
                if v in R[i]:
                    break
            family = (v, self._sample_pset(v, set().union(*R[:i]), R[i-1]))
            family_score = self.scores[v][bm(family[1], ix=self.C[v])]

        if score is True:
            return family, family_score
        return family

    def _sample_pset(self, v, U, T):

        def g(X, E, U, T):

            X_bm = bm(X, ix=self.C[v])
            E_bm = bm(E, ix=sorted(set(self.C[v]).difference(X)))
            U_bm = bm(U.difference(X), ix=sorted(set(self.C[v]).difference(X)))
            T_bm = bm(T.difference(X), ix=sorted(set(self.C[v]).difference(X)))

            score_1 = [self._f[X_bm][U_bm & ~E_bm] if X.issubset(U.difference(E)) else -float("inf")][0]
            score_2 = [self._f[X_bm][(U_bm & ~E_bm) & ~T_bm] if X.issubset(U.difference(E.union(T))) else -float("inf")][0]

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

        probs = list()
        psets = list()
        for T_set in subsets(T, 1, len(T)):
            for U_set in subsets(U.difference(T), 0, len(U.difference(T))):
                pset = set(T_set).union(U_set)
                probs.append(self.scores[v][bm(pset, ix=self.C[v])])
                psets.append(pset)
        probs = np.array(probs)
        probs -= np.logaddexp.reduce(probs)
        probs = np.exp(probs)
        return psets[np.random.choice(range(len(psets)), p=probs)]


class PartitionMCMC:

    def __init__(self, C, sr, temperature=1):
        self.n = len(C)
        self.C = C
        self.temp = temperature
        self.sr = sr
        self.stay_prob = 0.01
        self._moves = [self._R_basic_move, self._R_swap_any]
        self._moveprobs = [0.5, 0.5]
        self.R = self._random_partition()
        if self.temp == 0:
            self.sample = self._sample_temp0
            self.R_node_scores = self._R_node_scores_temp0
            self.R_score = 0
        else:
            self.R_node_scores = self._pi(self.R)
            self.R_score = self.temp * sum(self.R_node_scores)

    def _valid(self, R):
        if sum(len(R[i]) for i in range(len(R))) != self.n:
            return False
        if len(R) == 1:
            return True
        for i in range(1, len(R)):
            for v in R[i]:
                if len(R[i-1].intersection(self.C[v])) == 0:
                    return False
        return True

    def _random_partition(self):

        def n(R):
            n_nodes = 1
            while np.random.random() < (self.n/2-1)/(self.n-1) and sum(len(R[i]) for i in range(len(R))) + n_nodes < self.n:
                    n_nodes += 1
            return n_nodes

        while(True):
            U = set(range(self.n))
            R = [set(np.random.choice(list(U), n([]), replace=False))]
            U = U.difference(R[0])
            while len(U) > 0:
                pool = list(U.intersection(set().union({v for v in self.C if set(self.C[v]).intersection(R[-1])})))
                if len(pool) == 0:
                    break

                R_i = np.random.choice(pool, min(n(R), len(pool)), replace=False)
                R.append(set(R_i))
                U = U.difference(R[-1])

            if self._valid(R):
                return tuple(R)

    def _R_basic_move(self, **kwargs):

        def valid():
            return True

        if "validate" in kwargs and kwargs["validate"] is True:
            return valid()

        m = len(self.R)
        sum_binoms = [sum([comb(len(self.R[i]), v) for v in range(1, len(self.R[i]))]) for i in range(m)]
        nbd = m - 1 + sum(sum_binoms)
        q = 1/nbd

        j = np.random.randint(1, nbd+1)

        R_prime = list()
        if j < m:
            R_prime = [self.R[i] for i in range(j-1)] + [self.R[j-1].union(self.R[j])] + [self.R[i] for i in range(min(m, j+1), m)]
            return R_prime, q, q, self.R[j].union(self.R[min(m-1, j+1)])

        sum_binoms = [sum(sum_binoms[:i]) for i in range(1, len(sum_binoms)+1)]
        i_star = [m-1 + sum_binoms[i] for i in range(len(sum_binoms)) if m-1 + sum_binoms[i] < j]
        i_star = len(i_star)

        c_star = [comb(len(self.R[i_star]), v) for v in range(1, len(self.R[i_star])+1)]
        c_star = [sum(c_star[:i]) for i in range(1, len(c_star)+1)]

        c_star = [m-1 + sum_binoms[i_star-1] + c_star[i] for i in range(len(c_star))
                  if m-1 + sum_binoms[i_star-1] + c_star[i] < j]
        c_star = len(c_star)+1

        nodes = np.random.choice(list(self.R[i_star]), c_star)

        R_prime = [self.R[i] for i in range(i_star)] + [set(nodes)]
        R_prime += [self.R[i_star].difference(nodes)] + [self.R[i] for i in range(min(m, i_star+1), m)]

        return tuple(R_prime), q, q, self.R[i_star].difference(nodes).union(self.R[min(m-1, i_star+1)])

    def _R_swap_any(self, **kwargs):

        def valid():
            return len(self.R) > 1

        m = len(self.R)

        if "validate" in kwargs and kwargs["validate"] is True:
            return valid()

        j, k = np.random.choice(range(len(self.R)), 2, replace=False)
        v_j = np.random.choice(list(self.R[j]))
        v_k = np.random.choice(list(self.R[k]))
        R_prime = list()
        for i in range(m):
            if i == j:
                R_prime.append(self.R[i].difference({v_j}).union({v_k}))
            elif i == k:
                R_prime.append(self.R[i].difference({v_k}).union({v_j}))
            else:
                R_prime.append(self.R[i])

        q = 1/(comb(m, 2)*len(self.R[j])*len(self.R[k]))

        return tuple(R_prime), q, q, {v_j, v_k}.union(*self.R[min(j, k)+1:min(max(j, k)+2, m+1)])

    def _pi(self, R, R_node_scores=None, rescore=None):

        inpart = [0] * sum(len(R[i]) for i in range(len(R)))
        for i in range(len(R)):
            for v in R[i]:
                inpart[v] = i

        if R_node_scores is None:
            R_node_scores = [0] * len(inpart)
        else:
            # Don't copy whole list, just the nodes to rescore?
            R_node_scores = list(R_node_scores)

        if rescore is None:
            rescore = set().union(*R)

        for v in rescore:

            if inpart[v] == 0:
                R_node_scores[v] = self.sr.psum(v, 0, 0)

            else:

                R_node_scores[v] = self.sr.psum(v,
                                                bm(set().union(*R[:inpart[v]]).intersection(self.C[v]), ix=self.C[v]),
                                                bm(R[inpart[v]-1].intersection(self.C[v]), ix=self.C[v]))

        return R_node_scores

    def sample(self):

        if np.random.rand() > self.stay_prob:
            move = np.random.choice(self._moves, p=self._moveprobs)
            if not move(R=self.R, validate=True):
                return self.R, self.R_score

            R_prime, q, q_rev, rescore = move(R=self.R)

            if not self._valid(R_prime):
                return self.R, self.R_score

            R_prime_node_scores = self._pi(R_prime, R_node_scores=self.R_node_scores, rescore=rescore)

            # make this happen in log space
            #if -np.random.exponential() < self.temp * sum(R_prime_node_scores) - self.R_score + np.log(q_rev) - np.log(q):
            if np.random.rand() < np.exp(self.temp * sum(R_prime_node_scores) - self.R_score)*q_rev/q:
                self.R = R_prime
                self.R_node_scores = R_prime_node_scores
                self.R_score = self.temp * sum(self.R_node_scores)

        return self.R, self.R_score

    def _sample_temp0(self):

        if np.random.rand() > self.stay_prob:
            move = np.random.choice(self._moves, p=self._moveprobs)
            if not move(R=self.R, validate=True):
                return self.R, self.R_score

            R_prime, q, q_rev, rescore = move(R=self.R)

            if not self._valid(R_prime):
                return self.R, self.R_score

            if np.random.rand() < q_rev/q:
                self.R = R_prime

        return self.R, self.R_score

    @property
    def _R_node_scores_temp0(self):
        return self._pi(self.R)


class MC3:

    def __init__(self, chains, stats=None):
        if stats is not None:
            self.stats = stats
            self.stats[type(self).__name__] = dict()
            self.stats[type(self).__name__]["n chains"] = len(chains)
            self.stats[type(self).__name__]["temperatures"] = [round(c.temp, 3) for c in chains]
        self.chains = chains

    def sample(self):
        for c in self.chains:
            c.sample()
        i = np.random.randint(len(self.chains) - 1)
        ap = sum(self.chains[i+1].R_node_scores)*self.chains[i].temp
        ap += sum(self.chains[i].R_node_scores)*self.chains[i+1].temp
        ap -= sum(self.chains[i].R_node_scores)*self.chains[i].temp
        ap -= sum(self.chains[i+1].R_node_scores)*self.chains[i+1].temp
        if -np.random.exponential() < ap:
            R_tmp = self.chains[i].R
            R_node_scores_tmp = self.chains[i].R_node_scores
            self.chains[i].R = self.chains[i+1].R
            self.chains[i].R_node_scores = self.chains[i+1].R_node_scores
            self.chains[i].R_score = self.chains[i].temp * sum(self.chains[i].R_node_scores)
            self.chains[i+1].R = R_tmp
            self.chains[i+1].R_node_scores = R_node_scores_tmp
            self.chains[i+1].R_score = self.chains[i+1].temp * sum(self.chains[i+1].R_node_scores)
        return self.chains[-1].R, self.chains[-1].R_score


class Score:

    def __init__(self, datapath, scoref="bdeu", maxid=-1, ess=10, stats=None):

        self.maxid = maxid
        if stats is not None:
            self.stats = stats
            self.stats[type(self).__name__] = dict()
            self.stats[type(self).__name__]["clear_cache"] = 0

        if scoref == "bdeu":

            def local(node, parents):
                if len(self.score._cache) > 1000000:
                    self.score.clear_cache()
                    if self.stats:
                        self.stats[type(self).__name__]["clear_cache"] += 1

                if self.maxid == -1 or len(parents) <= self.maxid:
                    score = self.score.bdeu_score(node, parents)[0]
                else:
                    return -float("inf")
                # consider putting the prior explicitly somewhere
                return score - np.log(comb(self.n - 1, len(parents)))

            d = DiscreteData(datapath)
            d._varidx = {v: v for v in d._varidx.values()}
            self.n = len(d._variables)
            self.score = BDeu(d, alpha=ess)
            self.local = local

        elif scoref == "bge":

            def local(node, parents):
                if len(self.score._cache) > 1000000:
                    self.score._cache = dict()
                    if self.stats:
                        self.stats[type(self).__name__]["clear_cache"] += 1
                if self.maxid == -1 or len(parents) <= self.maxid:
                    return self.score.bge_score(node, parents)[0] - np.log(comb(self.n - 1, len(parents)))
                else:
                    return -float("inf")

            d = ContinuousData(datapath, header=False)
            d._varidx = {v: v for v in d._varidx.values()}
            self.n = len(d._variables)
            self.score = BGe(d)
            self.local = local

    def all_scores_dict(self, C=None):
        scores = dict()
        if C is None:
            C = {v: tuple(sorted(set(range(self.n)).difference({v}))) for v in range(self.n)}
        for v in C:
            tmp = dict()
            for pset in subsets(C[v], 0, [len(C[v]) if self.maxid == -1 else self.maxid][0]):
                tmp[pset] = self.local(v, pset)
            scores[v] = tmp
        return scores

    def all_scores_list(self, C):
        scores = list()
        for v in C:
            tmp = [-float('inf')]*2**len(C[0])
            for pset in subsets(C[v], 0, [len(C[v]) if self.maxid == -1 else self.maxid][0]):
                tmp[bm(pset, ix=C[v])] = self.local(v, pset)
            scores.append(tmp)
        return scores


class ScoreR:

    def __init__(self, scores, C, tolerance=1e-32, stats=None):

        if stats is not None:
            self.stats = stats
            self.stats[type(self).__name__] = dict()
            self.stats[type(self).__name__]["CC"] = 0
            self.stats[type(self).__name__]["CC basecases"] = 0
            self.stats[type(self).__name__]["basecases"] = 0

        self.scores = scores
        self.C = C
        self.tol = tolerance
        self._precompute_a()
        self._precompute_basecases()
        self._precompute_psum()

    def _precompute_a(self):
        self._a = [0]*len(self.scores)
        for v in range(len(self.scores)):
            self._a[v] = zeta_transform.from_list(self.scores[v])

    def _precompute_basecases(self):
        K = len(self.C[0])
        self._psum = {v: dict() for v in range(len(self.C))}
        for v in self.C:
            for k in range(K):
                x = 1 << k
                U_minus_x = (2**K - 1) & ~x
                tmp = [0]*2**(K-1)
                tmp[0] = self.scores[v][x]
                self._psum[v][x] = dict()
                for S in ssets(U_minus_x):
                    if S | x not in self._psum[v]:
                        self._psum[v][S | x] = dict()
                    tmp[dkbit(S, k)] = self.scores[v][S | x]
                tmp = zeta_transform.from_list(tmp)
                for S in range(len(tmp)):
                    if self.stats:
                            self.stats[type(self).__name__]["basecases"] += len(tmp)
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

                        self._psum[v][U][T] = np.logaddexp(self.psum(v, U, T1),
                                                           self.psum(v, U1, T2))
        if self.stats:
            self.stats[type(self).__name__]["relative CC"] = self.stats[type(self).__name__]["CC"] / (n*3**K)

    def _cc(self, v, U, T):
        return close(self._a[v][U], self._a[v][U & ~T], self.tol)
        # return self._a[v][U] == self._a[v][U & ~T]

    def psum(self, v, U, T):
        if U == 0 and T == 0:
            return self.scores[v][0]
        if T == 0:  # special case for T2 in precompute
            return -float("inf")
        if v in self._psum and U in self._psum[v] and T in self._psum[v][U]:
            return self._psum[v][U][T]
        else:
            return log_minus_exp(self._a[v][U], self._a[v][U & ~T])
