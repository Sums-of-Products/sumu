import numpy as np

from .bitmap import bm_to_ints


def prune_scores(C, scores):
    """Prune input local scores to those conforming to input candidate parents.

    Is this used anywhere?
    """
    for v in scores:
        tmp = dict()
        for pset in scores[v]:
            if set(pset).issubset(C[v]):
                tmp[pset] = scores[v][pset]
        scores[v] = tmp


def parse_DAG(DAG, C):
    """Translates family sequence representation of a DAG
    where each family is a tuple (v, pset) with pset a integer bitmap 
    in the space of v's candidate parents, into format (v, p_1, ..., p_n)
    where p_1 are the labels of the parents in the space of all nodes.

    Is this used anywhere?"""
    return [(i[0],) + tuple(C[i[0]][u]
                            for u in [bm_to_ints(i[1])
                                      if len(i) > 1
                                      else tuple()][0])
            for i in sorted(DAG, key=lambda x: x[0])]


def pset_posteriors(DAGs):
    """"Compute the empirical pset probabilities from input DAGs"""
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


