import sys
import math
from itertools import chain, combinations
import argparse
import time

import numpy as np

import zeta_transform.zeta_transform as zeta_transform
from scoring import DiscreteData, ContinuousData, BDeu, BGe

def read_jkl(scorepath):
    scores = dict()
    with open(scorepath, 'r') as jkl_file:
        rows = jkl_file.readlines()
        scores = dict()
        n_scores = 0
        for row in rows[1:]:

            if not n_scores:
                n_scores = int(row.strip().split()[1])
                current_var = int(row.strip().split()[0])
                scores[current_var] = dict()
                continue

            row_list = row.strip().split()
            prob_sum = float(row_list[0])
            n_parents = int(row_list[1])

            parents = frozenset()
            if n_parents > 0:
                parents = frozenset([int(x) for x in row_list[2:]])
            scores[current_var][tuple(sorted(tuple(parents)))] = prob_sum
            n_scores -= 1

    if min(scores.keys()) == 1:
        scores = jkl_to_zero_based_indexing(scores)

    return scores


def jkl_to_zero_based_indexing(jkl):
    for old_node in sorted(jkl.keys()):
        tmp_dict = dict()
        for pset in jkl[old_node]:
                tmp_dict[tuple(np.array(pset) - 1)] = jkl[old_node][pset]
        jkl[old_node - 1] = tmp_dict
        del jkl[old_node]
    return jkl


def write_jkl(scores, fpath):
    """Assumes the psets are iterables, not bitmaps
    """
    with open(fpath, 'w') as f:
        lines = list()
        n_vars = len(scores)
        lines.append(str(n_vars) + "\n")
        for v in sorted(scores):
            lines.append("{} {}\n".format(v, len(scores[v])))
            for pset in sorted(scores[v], key=lambda pset: len(pset)):
                lines.append("{} {} {}\n".format(scores[v][pset], len(pset), ' '.join([str(p) for p in pset])))
        f.writelines(lines)


def subsets(iterable, fromsize, tosize):
    s = list(iterable)
    step = 1 + (fromsize > tosize) * -2
    return chain.from_iterable(combinations(s, i)
                               for i in range(fromsize, tosize + step, step))


def arg(name, kwargs):
    if name not in kwargs:
        return None
    return kwargs[name]


def candidates_rnd(K, **kwargs):

    n = arg("n", kwargs)
    assert n is not None, "nvars (-n) required for algo == rnd"

    C = dict()
    for v in range(n):
        C[v] = tuple(sorted(np.random.choice([u for u in range(n) if u != v], K, replace=False)))
    return C


def candidates_greedy_backward_forward(K, **kwargs):

    scores = arg("scores", kwargs)
    assert scores is not None, "scorepath (-s) required for algo == greedy-backward-forward"

    def min_max(v):
        return min([max([(u, scores.local(v, tuple(set(S + (u,)))))
                         for S in subsets(C[v].difference({u}), 0, len(C[v]) - 1)],
                        key=lambda item: item[1])
                    for u in C[v]], key=lambda item: item[1])[0]

    def highest_uncovered(v, U):
        return max([(u, scores.local(v, tuple(set(S + (u,)))))
                    for S in subsets(C[v], 0, len(C[v]))
                    for u in U],
                   key=lambda item: item[1])[0]

    C = candidates_rnd(K, n=len(scores.score._variables))
    C = {v: set(C[v]) for v in C}
    for v in C:
        C_prev = dict(C)
        while True:
            u_hat = min_max(v)
            C[v] = C[v].difference({u_hat})
            u_hat = highest_uncovered(v, set(C).difference(C[v]).difference({v}))
            C[v].add(u_hat)
            if C == C_prev:
                break
            else:
                C_prev = dict(C)
        C[v] = tuple(sorted(C[v]))

    return C


def prune_scores(C, scores):
    for v in scores:
        tmp = dict()
        for pset in scores[v]:
            if set(pset).issubset(C[v]):
                tmp[pset] = scores[v][pset]
        scores[v] = tmp


def bm(ints, ix=None):
    if type(ints) not in [set, tuple]:
        ints = {int(ints)}
    if ix is not None:
        ints = [ix.index(i) for i in ints]
    bitmap = 0
    for k in ints:
        bitmap += 2**k
    return int(bitmap)  # without the cast np.int64 might sneak in somehow and break drv


def bm_to_ints(bm):
    return tuple(i for i in range(len(format(bm, 'b')[::-1]))
                 if format(bm, 'b')[::-1][i] == "1")


def translate_psets_to_bitmaps(C, scores):
    K = len(C[0])
    scores_list = list()
    for v in sorted(scores):
        tmp = [-float('inf')]*2**K
        for pset in scores[v]:
            tmp[bm(set(pset), ix=C[v])] = scores[v][pset]
        scores_list.append(tmp)
    return scores_list


def comb(n, r):
    if r > n:
        return 0
    f = math.factorial
    return f(n) // f(r) // f(n-r)


def log_minus_exp(p1, p2):
    if p1 == p2 and p1:
        return -float("inf")
    return max(p1, p2) + np.log1p(-np.exp(min(p1, p2)-max(p1, p2)))


def parse_DAG(DAG, C):
    return [(i[0],) + tuple(C[i[0]][u] for u in [bm_to_ints(i[1]) if len(i) > 1 else tuple()][0]) for i in sorted(DAG, key=lambda x: x[0])]


class DAGR:

    def __init__(self, scores, C):
        self.scores = scores
        self.C = C
        self._precompute(scores, C)

    def _precompute(self, scores, C):

        self._f = [[0]*2**len(C[0]) for v in range(len(scores))]
        for v in self.C:
            for X in subsets(self.C[v], 0, len(self.C[v])):
                X_bm = bm(X, ix=self.C[v])
                self._f[v][X_bm] = [-float("inf")]*2**(len(self.C[v])-len(X))
                for S in subsets(set(self.C[v]).difference(X), 0, len(self.C[v]) - len(X)):
                    self._f[v][X_bm][bm(S, ix=sorted(set(self.C[v]).difference(X)))] = scores[v][bm(X + S, ix=self.C[v])]
                self._f[v][X_bm] = zeta_transform.from_list(self._f[v][X_bm])

    def sample(self, R, score=False):
        DAG = [(v,) for v in R[0]]
        for i in range(1, len(R)):
            for v in R[i]:
                DAG.append((v, self._sample_pset(v, set().union(*R[:i]), R[i-1])))
        if score is True:
            return DAG, sum(self.scores[i[0]][0] if len(i) < 2 else self.scores[i[0]][bm(i[1], ix=self.C[i[0]])] for i in DAG)
        return DAG

    def _sample_pset(self, v, U, T):

        def g(X, E, U, T):

            X_bm = bm(X, ix=self.C[v])
            E_bm = bm(E, ix=sorted(set(self.C[v]).difference(X)))
            U_bm = bm(U.difference(X), ix=sorted(set(self.C[v]).difference(X)))
            T_bm = bm(T.difference(X), ix=sorted(set(self.C[v]).difference(X)))

            score_1 = [self._f[v][X_bm][U_bm & ~E_bm] if X.issubset(U.difference(E)) else -float("inf")][0]
            score_2 = [self._f[v][X_bm][(U_bm & ~E_bm) & ~T_bm] if X.issubset(U.difference(E.union(T))) else -float("inf")][0]

            return log_minus_exp(score_1, score_2)

        U = U.intersection(self.C[v])
        T = T.intersection(self.C[v])

        X = set()
        E = set()
        for i in U:
            if -np.random.exponential() < g(X.union({i}), E, U, T) - g(X, E, U, T):
                X.add(i)
            else:
                E.add(i)
        return X


class PartitionMCMC:

    def __init__(self, scores, C, sr, temperature=1):
        self.scores = scores
        self.n = len(scores)
        self.C = C
        self.temp = temperature
        self.sr = sr
        self.stay_prob = 0.01
        self._moves = [self._R_basic_move, self._R_swap_any]
        self._moveprobs = [0.5, 0.5]
        self.score_cache = dict()
        self.score_cache["new"] = 0
        self.score_cache["cache"] = 0
        self._precompute()
        self.R = self._random_partition()
        if self.temp == 0:
            self.sample = self._sample_temp0
            self.R_node_scores = self._R_node_scores_temp0
            self.R_score = 0
        else:
            self.R_node_scores = self._pi(self.R)
            self.R_score = self.temp * sum(self.R_node_scores)

    def _precompute(self):
        self._a = [0]*self.n
        for v in range(self.n):
            self._a[v] = zeta_transform.from_list(self.scores[v])

    def _valid(self, R):
        if sum(len(R[i]) for i in range(len(R))) != len(self.C):
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

        R = kwargs["R"]

        if "validate" in kwargs and kwargs["validate"] is True:
            return valid()

        m = len(R)
        sum_binoms = [sum([comb(len(R[i]), c) for c in range(1, len(R[i]))]) for i in range(m)]
        nbd = m - 1 + sum(sum_binoms)
        q = 1/nbd

        j = np.random.choice(range(1, nbd+1))

        R_prime = list()
        if j < m:
            R_prime = [R[i] for i in range(j-1)] + [R[j-1].union(R[j])] + [R[i] for i in range(min(m, j+1), m)]
            return R_prime, q, q, R[j].union(R[min(m-1, j+1)])

        sum_binoms = [sum(sum_binoms[:i]) for i in range(1, len(sum_binoms)+1)]
        i_star = [m-1 + sum_binoms[i] for i in range(len(sum_binoms)) if m-1 + sum_binoms[i] < j]
        i_star = len(i_star)

        c_star = [comb(len(R[i_star]), c) for c in range(1, len(R[i_star])+1)]
        c_star = [sum(c_star[:i]) for i in range(1, len(c_star)+1)]

        c_star = [m-1 + sum_binoms[i_star-1] + c_star[i] for i in range(len(c_star))
                  if m-1 + sum_binoms[i_star-1] + c_star[i] < j]
        c_star = len(c_star)+1

        nodes = np.random.choice(list(R[i_star]), c_star)

        R_prime = [R[i] for i in range(i_star)] + [set(nodes)]
        R_prime += [R[i_star].difference(nodes)] + [R[i] for i in range(min(m, i_star+1), m)]

        return tuple(R_prime), q, q, R[i_star].difference(nodes).union(R[min(m-1, i_star+1)])

    def _R_swap_any(self, **kwargs):

        def valid():
            return len(R) > 1

        R = kwargs["R"]
        m = len(R)

        if "validate" in kwargs and kwargs["validate"] is True:
            return valid()

        j, k = np.random.choice(range(len(R)), 2, replace=False)
        v_j = np.random.choice(list(R[j]))
        v_k = np.random.choice(list(R[k]))
        R_prime = list()
        for i in range(len(R)):
            if i == j:
                R_prime.append(R[i].difference({v_j}).union({v_k}))
            elif i == k:
                R_prime.append(R[i].difference({v_k}).union({v_j}))
            else:
                R_prime.append(R[i])

        q = 1/(comb(len(R), 2)*len(R[j])*len(R[k]))

        return tuple(R_prime), q, q, {v_j, v_k}.union(*R[min(j, k)+1:min(max(j, k)+2, m+1)])

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
                R_node_scores[v] = self.scores[v][0]

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

    def __init__(self, chains):
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

    def __init__(self, datapath, datatype="discrete"):

        if datatype == "discrete":

            def local(node, parents):
                score = self.score.bdeu_score(node, parents)[0]
                if len(self.score._cache) > 1000000:
                    self.score.clear_cache()
                # consider putting the prior explicitly somewhere
                return score #- np.log(comb(self.n - 1, len(parents)))

            d = DiscreteData(datapath)
            d._varidx = {v: v for v in d._varidx.values()}
            self.n = len(d._variables)
            self.score = BDeu(d)
            self.local = local  # self.score.bdeu_score

        elif datatype == "continuous":

            def local(node, parents):
                return self.score.bge_score(node, parents)[0] - np.log(comb(self.n - 1, len(parents)))

            d = ContinuousData(datapath, header=False)
            d._varidx = {v: v for v in d._varidx.values()}
            self.n = len(d._variables)
            self.score = BGe(d)
            self.local = local

    def all_scores(self, C):
        scores = list()
        for v in C:
            tmp = [-float('inf')]*2**len(C[0])
            for pset in subsets(C[v], 0, len(C[v])):
                tmp[bm(pset, ix=C[v])] = self.local(v, pset)
            scores.append(tmp)
        return scores


class ScoreR:

    def __init__(self, scores, C):
        self.brute_n = 0
        self.cc_n = 0
        self.scores = scores
        self.C = C
        self._precompute_a()
        self._precompute_psum()

    def _precompute_a(self):
        self._a = [0]*len(self.scores)
        for v in range(len(self.scores)):
            self._a[v] = zeta_transform.from_list(self.scores[v])

    def _precompute_psum(self):
        self._psum = dict()
        for v in self.C:
            for U in subsets(self.C[v], 1, len(self.C[v])):
                U0 = bm(set(U), ix=self.C[v])
                for T in subsets(U, 1, len(U)):
                    T0 = bm(set(T), ix=self.C[v])
                    if self._cc(v, U0, T0):
                        self.cc_n += 1
                        if v not in self._psum:
                            self._psum[v] = dict()
                        if U0 not in self._psum[v]:
                            self._psum[v][U0] = dict()
                        U1 = bm({u for u in U if u != T[0]}, ix=self.C[v])
                        T1 = bm({T[0]}, ix=self.C[v])
                        T2 = bm({t for t in T if t != T[0]}, ix=self.C[v])

                        self._psum[v][U0][T0] = np.logaddexp(self.psum(v, U0, T1),
                                                             self.psum(v, U1, T2))

    def _cc(self, v, U, T):
        return self._a[v][U] == self._a[v][U & ~T]

    def psum(self, v, U, T):
        if T == 0:  # special case for T2 in precompute
            return -float("inf")
        if v in self._psum and U in self._psum[v] and T in self._psum[v][U]:
            return self._psum[v][U][T]
        else:
            return self._psum_diff(v, U, T)

    def _psum_diff(self, v, U, T):

        score_U_cap_C = self._a[v][U]
        score_U_minus_T_cap_C = self._a[v][U & ~T]
        if score_U_cap_C == score_U_minus_T_cap_C:
            self.brute_n += 1
            U_minus_T_list = bm_to_ints(U & ~T)
            T_list = bm_to_ints(T)
            s = list()
            for t in subsets(T_list, 1, len(T_list)):
                for u in subsets(U_minus_T_list, 0, len(U_minus_T_list)):
                    s.append(self.scores[v][bm(t) & bm(u)])
            return np.logaddexp.reduce(s)

        return log_minus_exp(score_U_cap_C, score_U_minus_T_cap_C)


def pset_posteriors(DAGs):
    posteriors = dict({v: dict() for v in range(len(DAGs[0]))})
    for DAG in DAGs:
        for v in DAG:
            pset = [tuple(sorted(v[1])) if len(v) > 1 else tuple()][0]
            v = v[0]
            if pset not in posteriors[v]:
                posteriors[v][pset] = 1
            else:
                posteriors[v][pset] += 1
    for v in posteriors:
        for pset in posteriors[v]:
            posteriors[v][pset] /= len(DAGs)
            posteriors[v][pset] = np.log(posteriors[v][pset])
    return posteriors


def main():
    K = 9

    t0 = time.process_time()

    scores = Score(sys.argv[1], datatype="discrete")

    # np.random.seed(2)
    t0 = time.process_time()
    C = candidates_greedy_backward_forward(K, scores=scores)
    print("computing candidate parents {}".format(time.process_time() - t0))

    scores = scores.all_scores(C)

    t0 = time.process_time()
    sr = ScoreR(scores, C)
    print("precompute scoresums {}".format(time.process_time() - t0))
    print("number of brutes {}".format(sr.brute_n))
    print("number of cc {}".format(sr.cc_n))

    print("initiate chain")
    mcmc = MC3([PartitionMCMC(scores, C, sr, temperature=i/15) for i in range(16)])

    print("run chain")
    t0 = time.process_time()
    for i in range(50000):
        #print(i, end="\r")
        mcmc.sample()
    print("50k mcmc steps {}".format(time.process_time() - t0))

    t0 = time.process_time()
    ds = DAGR(scores, C)
    print("compute DAG samplers {}".format(time.process_time() - t0))

    t0 = time.process_time()
    DAGs = list()
    for i in range(1000):
        DAGs.append(ds.sample(mcmc.sample()[0], score=True)[1])
    print("sample 1000 DAGs {}".format(time.process_time() - t0))

    print(DAGs)

    t0 = time.process_time()
    DAGs = list()
    for i in range(1000):
        DAGs.append(ds.sample(mcmc.sample()[0]))
    print("sample 1000 DAGs {}".format(time.process_time() - t0))

    ppost = pset_posteriors(DAGs)
    write_jkl(ppost, "ppost.jkl")


if __name__ == '__main__':
    main()
