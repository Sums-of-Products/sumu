import sys
import math
from itertools import chain, combinations
import argparse
import time

import numpy as np

import zeta_transform.zeta_transform as zeta_transform
import discrete_random_variable.discrete_random_variable as drv


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
        scores = jkl_to_zero_based_ixg(scores)

    return scores


def jkl_to_zero_based_ixg(jkl):
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
        return min([max([(u, scores[v][tuple(sorted(set(S + (u,))))])
                         for S in subsets(C[v].difference({u}), 0, len(C[v]) - 1)
                         if tuple(sorted(set(S + (u,)))) in scores[v]], key=lambda item: item[1])
                    for u in C[v]], key=lambda item: item[1])[0]

    def highest_uncovered(v, U):
        return max([(u, scores[v][tuple(sorted(set(S + (u,))))])
                    for S in subsets(C[v], 0, len(C[v]))
                    for u in U
                    if tuple(sorted(set(S + (u,)))) in scores[v]], key=lambda item: item[1])[0]

    C = candidates_rnd(K, n=len(scores))
    C = {v: set(C[v]) for v in C}
    for v in scores:
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
    if type(ints) == int:
        ints = [ints]
    if ix is not None:
        ints = [ix.index(i) for i in ints]
    bm = 0
    for k in ints:
        bm += 2**k
    return int(bm)  # without the cast np.int64 might sneak in somehow and break drv


def bm_to_ints(bm):
    return tuple(i for i in range(len(format(bm, 'b')[::-1]))
                 if format(bm, 'b')[::-1][i] == "1")


def translate_psets_to_bitmaps(C, scores):
    K = len(C[0])
    scores_list = list()
    for v in sorted(scores):
        tmp = [-float('inf')]*2**K
        for pset in scores[v]:
            tmp[bm([C[v].index(p) for p in pset])] = scores[v][pset]
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

    def __init__(self, scores, C, temperature=1):
        self.scores = scores
        self.n = len(scores)
        self.C = C
        self.temp = temperature
        self.stay_prob = 0.01
        self._moves = [self._R_basic_move, self._R_swap_any]
        self._moveprobs = [0.5, 0.5]
        self.cache_scores = dict()
        self.cache_psets = dict()
        self.cache_scores["new"] = 0
        self.cache_scores["cache"] = 0
        self._precompute()
        self.R = self._random_partition()
        self.R_node_scores = self._pi(self.R)
        self.R_score = self.temp * sum(self.R_node_scores)

    def _precompute(self):
        self.a = [0]*self.n
        for v in range(self.n):
            self.a[v] = zeta_transform.from_list(self.scores[v])

    def _valid(self, R):
        if len(R) == 1:
            return True
        for i in range(1, len(R)):
            for v in R[i]:
                if len(R[i-1].intersection(self.C[v])) == 0:
                    return False
        return True

    def _random_partition(self):
        while(True):
            R = list()
            U = list(range(self.n))
            while sum(R) < self.n:
                n_nodes = 1
                while np.random.random() < (self.n/2-1)/(self.n-1) and sum(R) + n_nodes < self.n:
                    n_nodes += 1
                R.append(n_nodes)
            for i in range(len(R)):
                R_i = np.random.choice(U, R[i], replace=False)
                R[i] = set(R_i)
                U = [u for u in U if u not in R_i]

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

                score_U_cap_C = self.a[v][bm(set().union(*R[:inpart[v]]).intersection(self.C[v]), ix=self.C[v])]
                score_U_minus_T_cap_C = self.a[v][bm(set().union(*R[:inpart[v]-1]).intersection(self.C[v]), ix=self.C[v])]

                if score_U_cap_C == score_U_minus_T_cap_C:  # catastrofic cancellation, need to brute force
                    R_bm = tuple(bm(R_j) for R_j in R[:inpart[v]])
                    if R_bm in self.cache_scores and v in self.cache_scores[R_bm]:
                        # print("brute cache")
                        self.cache_scores["cache"] += 1
                        R_node_scores[v] = self.cache_scores[R_bm][v][-1]
                    else:
                        # print("brute new")
                        self.cache_scores["new"] += 1
                        if R_bm not in self.cache_scores:
                            self.cache_scores[R_bm] = dict()
                            self.cache_psets[R_bm] = dict()

                        v_pset_scores = list()
                        v_psets = list()
                        for T_sub in subsets(R[inpart[v]-1].intersection(self.C[v]), 1, len(R[inpart[v]-1])):
                            for U_minus_T_sub in subsets(set().union(*R[:inpart[v]-1]).intersection(self.C[v]), 0, sum(len(R[j]) for j in range(inpart[v]-1))):
                                v_pset_scores.append(self.scores[v][bm(T_sub + U_minus_T_sub, ix=self.C[v])])
                                v_psets.append(bm(T_sub + U_minus_T_sub, ix=self.C[v]))
                        self.cache_scores[R_bm][v] = v_pset_scores
                        self.cache_psets[R_bm][v] = v_psets
                        # the individual scores are preserved as they might be needed in DAG sampling
                        self.cache_scores[R_bm][v].append(np.logaddexp.reduce(v_pset_scores))
                        R_node_scores[v] = self.cache_scores[R_bm][v][-1]
                else:
                    R_node_scores[v] = log_minus_exp(score_U_cap_C, score_U_minus_T_cap_C)

        return R_node_scores

    def sample(self):

        if np.random.rand() > self.stay_prob:
            move = np.random.choice(self._moves, p=self._moveprobs)
            R_prime, q, q_rev, rescore = move(R=self.R)

            if not self._valid(R_prime):
                return self.R, self.R_score

            R_prime_node_scores = self._pi(R_prime, R_node_scores=self.R_node_scores, rescore=rescore)

            if np.random.rand() < np.exp(self.temp * sum(R_prime_node_scores) - self.R_score)*q_rev/q:
                self.R = R_prime
                self.R_node_scores = R_prime_node_scores
                self.R_score = self.temp * sum(self.R_node_scores)

        return self.R, self.R_score


class MC3:

    def __init__(self, chains):
        self.chains = chains
        self.prop_prob = 0.02

    def sample(self):
        for c in self.chains:
            c.sample()
        if np.random.random() < self.prop_prob:
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


def main():
    K = 12

    scores = read_jkl(sys.argv[1])

    #np.random.seed(1)
    print("Computing candidates")
    C = candidates_greedy_backward_forward(K, scores=scores)
    prune_scores(C, scores)
    scores = translate_psets_to_bitmaps(C, scores)
    mcmc = PartitionMCMC(scores, C, temperature=1)

    t0 = time.process_time()
    for i in range(50000):
        mcmc.sample()
    print(time.process_time() - t0)

    exit()
    t0 = time.process_time()
    ds = DAGR(scores, C)
    print(time.process_time() - t0)

    DAGs = list()
    for i in range(1000):
        DAGs.append(ds.sample(mcmc.sample()[0], score=True)[1])

    print(DAGs)


def main2():
    K = 12

    scores = read_jkl(sys.argv[1])

    #np.random.seed(1)
    print("Computing candidates")
    C = candidates_greedy_backward_forward(K, scores=scores)
    prune_scores(C, scores)
    scores = translate_psets_to_bitmaps(C, scores)

    mcmc = MC3([PartitionMCMC(scores, C, temperature=i/15) for i in range(16)])

    t0 = time.process_time()
    for i in range(50000):
        mcmc.sample()
    print(time.process_time() - t0)

    t0 = time.process_time()
    ds = DAGR(scores, C)
    print(time.process_time() - t0)

    t0 = time.process_time()
    DAGs = list()
    for i in range(1000):
        DAGs.append(ds.sample(mcmc.sample()[0], score=True)[1])
    print(time.process_time() - t0)

    print(DAGs)


if __name__ == '__main__':
    main2()
