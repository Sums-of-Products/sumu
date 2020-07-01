import math
from itertools import chain, combinations

import numpy as np


def pretty_dict(d, n=1):
    for k in d:
        if type(d[k]) == dict:
            print("{}{}".format(" "*n, k))
        else:
            print("{}{}: {}".format(" "*n, k, d[k]))
        if type(d[k]) == dict:
            pretty_dict(d[k], n=n+4)


def close(a, b, tolerance):
    return max(a, b) - min(a, b) < tolerance


def read_candidates(candidate_path):
    C = dict()
    with open(candidate_path, 'r') as f:
        f = f.readlines()
        for v, row in enumerate(f):
            C[v] = tuple([int(x) for x in row.split()])
    return C


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


def comb(n, r):
    if r > n:
        return 0
    f = math.factorial
    return f(n) // f(r) // f(n-r)


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


def log_minus_exp(p1, p2):
    if np.exp(min(p1, p2)-max(p1, p2)) == 1:
        return -float("inf")
    return max(p1, p2) + np.log1p(-np.exp(min(p1, p2)-max(p1, p2)))


def parse_DAG(DAG, C):
    return [(i[0],) + tuple(C[i[0]][u] for u in [bm_to_ints(i[1]) if len(i) > 1 else tuple()][0]) for i in sorted(DAG, key=lambda x: x[0])]


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


def DAG_to_str(DAG):
    return "|".join([str(f[0]) if len(f) == 1 else " ".join([str(f[0]), *[str(v) for v in sorted(f[1])]]) for f in DAG])
