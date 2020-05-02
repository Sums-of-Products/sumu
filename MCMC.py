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


def bitmap(ints, indexin=None):
    if type(ints) == int:
        ints = [ints]
    if indexin is not None:
        ints = [indexin.index(i) for i in ints]
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
            tmp[bitmap([C[v].index(p) for p in pset])] = scores[v][pset]
        scores_list.append(tmp)
    return scores_list


def valid(R, C):
    if len(R) == 1:
        return True
    for i in range(1, len(R)):
        for v in R[i]:
            if len(R[i-1].intersection(C[v])) == 0:
                return False
    return True


def random_partition(n, C, seed=None):
    if seed is not None:
        np.random.seed(int(seed))
    while(True):
        R = list()
        U = list(range(n))
        while sum(R) < n:
            n_nodes = 1
            while np.random.random() < (n/2-1)/(n-1) and sum(R) + n_nodes < n:
                n_nodes += 1
            R.append(n_nodes)
        for i in range(len(R)):
            R_i = np.random.choice(U, R[i], replace=False)
            R[i] = set(R_i)
            U = [u for u in U if u not in R_i]

        if valid(R, C):
            return tuple(R)


def comb(n, r):
    if r > n:
        return 0
    f = math.factorial
    return f(n) // f(r) // f(n-r)


def R_basic_move(**kwargs):

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
        return R_prime, q, q

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

    return tuple(R_prime), q, q


def R_swap_any(**kwargs):

    def valid():
        return len(R) > 1

    R = kwargs["R"]

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

    return tuple(R_prime), q, q


def log_minus_exp(p1, p2):
    if p1 == p2 and p1:
        return -float("inf")
    return max(p1, p2) + np.log1p(-np.exp(min(p1, p2)-max(p1, p2)))


def parse_DAG(DAG, C):
    return [(i[0],) + tuple(C[i[0]][u] for u in [bm_to_ints(i[1]) if len(i) > 1 else tuple()][0]) for i in sorted(DAG, key=lambda x: x[0])]


def sample_DAGs(Rs, C, scores):
    """Sample DAGs with alias-method random variables
    """
    rv = {v: dict() for v in C}
    for v in C:
        print(v)
        for Q in subsets(C[v], 0, len(C[v])):
            psets = [bitmap([C[v].index(p) for p in pset]) for pset in subsets(Q, 0, len(Q))]
            probs = np.array([scores[v][pset] for pset in psets])
            probs -= np.logaddexp.reduce(probs)
            probs = list(np.exp(probs))  # this will make some very small probabilities disappear
            rv[v][bitmap(Q)] = drv.DRVInt(psets, probs)

    DAG = list()
    DAG_prob = list()
    iterations = dict()
    for ri, R in enumerate(Rs):
        DAG.append(list())
        for i in range(len(R)):
            print("    sampling DAG {}".format(ri), end="\r")
            for v in R[i]:
                if i == 0:
                    DAG[-1].append((v,))
                else:
                    rv_pset = rv[v][bitmap(set().union(*R[:i]).intersection(C[v]))]
                    mask = bitmap(C[v].index(u) for u in R[i-1].intersection(C[v]))

                    R_bm = tuple(bitmap(R_j) for R_j in R[:i])
                    if (R_bm in cache_scores and v in cache_scores[R_bm]):
                        pset = np.random.choice(cache_psets[R_bm][v],
                                                p=np.exp(cache_scores[R_bm][v][:-1] - cache_scores[R_bm][v][-1]))
                    else:
                        pset = rv_pset()

                    it = 1
                    brute = False
                    while not pset & mask:
                        pset = rv_pset()
                        it += 1
                        if it == 100:
                            brute = True

                            if not (R_bm in cache_scores and v in cache_scores[R_bm]):

                                cache_scores["new"] += 1

                                if R_bm not in cache_scores:
                                    cache_scores[R_bm] = dict()
                                    cache_psets[R_bm] = dict()

                                v_pset_scores = list()
                                v_psets = list()
                                for T_sub in subsets([u for u in R[i-1] if u in C[v]], 1, len(R[i-1])):
                                    for U_minus_T_sub in subsets([u for u in R[:i-1] if u in C[v]], 0, sum(len(R[j]) for j in range(i-1))):
                                        v_pset_scores.append(scores[v][bitmap(C[v].index(u) for u in T_sub + U_minus_T_sub)])
                                        v_psets.append(bitmap(C[v].index(u) for u in T_sub + U_minus_T_sub))
                                cache_scores[R_bm][v] = v_pset_scores
                                cache_scores[R_bm][v].append(np.logaddexp.reduce(v_pset_scores))
                                cache_psets[R_bm][v] = v_psets

                            else:
                                cache_scores["cache"] += 1

                            pset = np.random.choice(cache_psets[R_bm][v],
                                                    p=np.exp(cache_scores[R_bm][v][:-1] - cache_scores[R_bm][v][-1]))
                            break
                    if brute:
                        brute = False

                    if it not in iterations:
                        iterations[it] = 1
                    else:
                        iterations[it] += 1
                    DAG[-1].append((v, pset))
        DAG_prob.append(sum(scores[item[0]][item[1]] if len(item) > 1 else scores[item[0]][0] for item in DAG[-1]))
    return DAG, DAG_prob, iterations


def MCMC(iterations, scores, C, a, seed=None):

    if seed is not None:
        np.random.seed(seed)

    R_score_dict = dict()

    cache_scores = dict()
    cache_psets = dict()
    cache_scores["new"] = 0
    cache_scores["cache"] = 0

    def pi(R):
        prob_sum = sum([scores[v][0] for v in R[0]])
        for i in range(1, len(R)):
            for v in R[i]:

                score_U_cap_C = a[v][bitmap([C[v].index(u) for u in [u for R_j in R[:i] for u in R_j if u in C[v]]])]
                score_U_minus_T_cap_C = a[v][bitmap([C[v].index(u) for u in [u for R_j in R[:i-1] for u in R_j if u in C[v]]])]

                if score_U_cap_C == score_U_minus_T_cap_C:  # catastrofic cancellation, need to brute forc
                    # R_bm = bitmap(set().union(*R[:i]))
                    R_bm = tuple(bitmap(R_j) for R_j in R[:i])
                    if R_bm in cache_scores and v in cache_scores[R_bm]:
                        # print("brute cache")
                        cache_scores["cache"] += 1
                        prob_sum += cache_scores[R_bm][v][-1]
                    else:
                        # print("brute new")
                        cache_scores["new"] += 1
                        if R_bm not in cache_scores:
                            cache_scores[R_bm] = dict()
                            cache_psets[R_bm] = dict()

                        v_pset_scores = list()
                        v_psets = list()
                        for T_sub in subsets([u for u in R[i-1] if u in C[v]], 1, len(R[i-1])):
                            for U_minus_T_sub in subsets([u for u in R[:i-1] if u in C[v]], 0, sum(len(R[j]) for j in range(i-1))):
                                v_pset_scores.append(scores[v][bitmap(C[v].index(u) for u in T_sub + U_minus_T_sub)])
                                v_psets.append(bitmap(C[v].index(u) for u in T_sub + U_minus_T_sub))
                        cache_scores[R_bm][v] = v_pset_scores
                        cache_psets[R_bm][v] = v_psets
                        # the individual scores are preserved as they might be needed in DAG sampling
                        cache_scores[R_bm][v].append(np.logaddexp.reduce(v_pset_scores))
                        prob_sum += cache_scores[R_bm][v][-1]
                else:
                    prob_sum += log_minus_exp(score_U_cap_C, score_U_minus_T_cap_C)
        R_score_dict[tuple(bitmap(Ri) for Ri in R)] = prob_sum
        return prob_sum


    stay_prob = 0.01

    R_moves = [R_basic_move, R_swap_any]
    moves = R_moves
    moveprob_counts = np.array([10, 10])

    #print("    Creating random start partition")
    R = random_partition(len(C), C, seed)
    Rs = [R]
    #print("    Computing score for the start partition")
    R_probs = [pi(R)]

    #print("    Running MCMC in the root-partition space")
    for i in range(iterations-1):
        #print("    iteration {}".format(i), end="\r")
        if np.random.rand() < stay_prob:
            Rs.append(Rs[-1])
            R_probs.append(R_probs[-1])

        else:

            move = np.random.choice(moves, p=moveprob_counts/sum(moveprob_counts))

            R_prime, q, q_rev = move(R=Rs[-1])

            if not valid(R_prime, C):
                Rs.append(Rs[-1])
                R_probs.append(R_probs[-1])
                continue

            R_prob = R_probs[-1]
            R_prime_prob = pi(R_prime)

            acc_prob = np.exp(R_prime_prob - R_prob)*q_rev/q

            if np.random.rand() < acc_prob:
                Rs.append(R_prime)
                R_probs.append(R_prime_prob)

            else:
                Rs.append(Rs[-1])
                R_probs.append(R_probs[-1])

    #print("brute new {}, brute cache {}".format(cache_scores["new"], cache_scores["cache"]))
    # DAG, DAG_prob, iterations = sample_DAGs(Rs, C, scores)
    # return Rs, R_probs, DAG, DAG_prob, cache_scores
    return Rs, R_probs

### JUST FOR CHECKING RESULTS AGAINST BRUTE FORCE COMPUTATIONS ###
def possible_psets(U, T, max_indegree):
    if not U:
        yield tuple()
    for required in subsets(T, 1, max(1, max_indegree)):
        for additional in subsets(U.difference(T), 0, max_indegree - len(required)):
            yield tuple(sorted(set(required).union(additional)))


def score_v(v, pset, scores):
    return scores[v][pset]


def hat_pi(v, U, T, scores, max_indegree):
    return np.logaddexp.reduce([score_v(v, pset, scores) for pset in possible_psets(U, T, max_indegree)])


def f(U, T, S, scores, max_indegree):
    hat_pi_sum = 0
    for v in S:
        hat_pi_sum += hat_pi(v, U, T, scores, max_indegree)
    return hat_pi_sum


def pi_R(R, scores, max_indegree):
    f_sum = 0
    for i in range(len(R)):
        f_sum += f(set().union(*R[:i]),
                    [R[i-1] if i-1>-1 else set()][0],
                    R[i], scores, max_indegree)
    return f_sum
###################################################################


def main():

    K = 12

    scores = read_jkl(sys.argv[1])
    #scores_old = read_jkl(sys.argv[1])

    # np.random.seed(1)
    print("Computing candidates")
    C = candidates_greedy_backward_forward(K, scores=scores)
    prune_scores(C, scores)
    scores = translate_psets_to_bitmaps(C, scores)

    print("Computing zeta transform")
    a = [0]*len(scores)
    for v in range(len(scores)):
        print("    variable {}".format(v))
        a[v] = zeta_transform.from_list(scores[v])

    print("Running MCMC chain")
    t0 = time.process_time()
    Rs, R_probs = MCMC(50000, scores, C, a)
    t1 = time.process_time() - t0
    print("time {}".format(t1))
    print(Rs[-1])
    exit()

    # Tutki miksi nämä eroaa, precision juttuja?
    # print("ensimmäinen R", R_probs[0])
    # print("viimeinen R", R_probs[-1])
    # print("eka vanhalla metodilla", pi_R(R[0], scores_old, 5))
    # print("vipa vanhalla metodilla", pi_R(R[-1], scores_old, 5))

    print("Sampling DAGs")
    print("    Constructing data structure for sampling")

    f = [[0]*2**K for v in range(len(C))]
    for v in C:
        print("        for variable {}".format(v))
        for X in subsets(C[v], 0, K):
            X_bm = bitmap(C[v].index(u) for u in X)
            f[v][X_bm] = [-float("inf")]*2**(K-len(X))
            for S in subsets(set(C[v]).difference(X), 0, K - len(X)):
                f[v][X_bm][bitmap(sorted(set(C[v]).difference(X)).index(u) for u in S)] = scores[v][bitmap(C[v].index(u) for u in X + S)]
            f[v][X_bm] = zeta_transform.from_list(f[v][X_bm])


    def sample_pset(v, U, T, f):

        def g(X, E, U, T):

            X_bm = bitmap(X, indexin=C[v])
            E_bm = bitmap(E, indexin=sorted(set(C[v]).difference(X)))
            U_bm = bitmap(U.difference(X), indexin=sorted(set(C[v]).difference(X)))
            T_bm = bitmap(T.difference(X), indexin=sorted(set(C[v]).difference(X)))

            score_1 = [f[X_bm][U_bm & ~E_bm] if X.issubset(U.difference(E)) else -float("inf")][0]
            score_2 = [f[X_bm][(U_bm & ~E_bm) & ~T_bm] if X.issubset(U.difference(E.union(T))) else -float("inf")][0]

            return log_minus_exp(score_1, score_2)

        X = set()
        E = set()
        for i in U:
            if -np.random.exponential() < g(X.union({i}), E, U, T) - g(X, E, U, T):
                X.add(i)
            else:
                E.add(i)
        return X


    DAGs = list()
    for R in Rs:
        DAGs.append([(v,) for v in R[0]])

    for j, R in enumerate(Rs):
        print("    Sampling DAG {}".format(j), end="\r")
        for i in range(1, len(R)):
            for v in R[i]:
                DAGs[j].append((v, sample_pset(v,
                                               set().union(*R[:i]).intersection(C[v]),
                                               R[i-1].intersection(C[v]),
                                               f[v])))

    # print(Rs)
    # print(DAGs)
    # exit()

    DAG_probs = list()
    for DAG in DAGs:
        DAG_probs.append(sum(scores[v[0]][0] if len(v) < 2 else scores[v[0]][bitmap(v[1], indexin=C[v[0]])] for v in DAG))

    # print(DAG_probs)
    # exit()

    # print(DAG_prob)
    with open("R.trace", "w") as f:
        f.write(str(R_probs))

    with open("DAG.trace", "w") as f:
        f.write(str(DAG_probs))

    # print("brute new {}, brute cache {}".format(cache_scores["new"], cache_scores["cache"]))

if __name__ == '__main__':
    main()
