import numpy as np
import re

from .bitmap import bm_to_ints
from ..bnet import family_sequence_to_adj_mat


def cite(this):
    ks = [k for kl in [k.split(",")
                       for k in re.findall(r':cite:`(.+?(?=`))', this.__doc__)]
          for k in kl]
    bibtex = list()
    inentry = False
    with open("../sources.bib", "r") as f:
        for row in f.readlines():
            if inentry:
                if row[0] == "@":
                    bibtex.append("".join(entry).strip())
                    inentry = False
                    entry = list()
                else:
                    entry.append(row)
            for k in ks:
                if k in row:
                    inentry = True
                    entry = list()
                    entry.append(row)

    for entry in bibtex:
        print(entry)


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


def edge_probs_from_pset_probs(unp_probs):
    """Compute normalized edge probabilities from unnormalized parent set
       log-probabilities.

    Args:
       unp_probs (dict) two level dictionary, where the first level is a node
       and the second level is a parent set. The value is the unnormalized
       log-probability of the parent set for the node.

    Returns:
        matrix with normalized edge probabilities

    """

    # First the pset probabilities are normalized
    for node in unp_probs:
        normalizer = np.logaddexp.reduce(list(unp_probs[node].values()))
        for pset in unp_probs[node]:
            unp_probs[node][pset] -= normalizer

    # Number of variables
    n = len(unp_probs.keys())

    # For each edge the corresponding pset probabilities are collected
    # into a list
    edge_probs = dict({v: dict({v: list() for v in range(n)})
                       for v in range(n)})
    for c in unp_probs:
        for pset in unp_probs[c]:
            for p in pset:
                edge_probs[p][c].append(unp_probs[c][pset])

    # Then the collected probabilities for each edge are summed
    for p in range(n):
        for c in range(n):
            edge_probs[p][c] = np.logaddexp.reduce(edge_probs[p][c])

    # The return value is formatted as 2d numpy array and the
    # logarithmic values are exponentiated
    edge_probs_matrix = np.zeros((n, n))
    for p in edge_probs:
        for c in edge_probs[p]:
            edge_probs_matrix[p, c] = edge_probs[p][c]

    return np.exp(edge_probs_matrix)


def edge_empirical_prob_max_error(dags, exact_probs):

    max_abs_errors = list()

    n = len(dags[0])
    n_dags = 0
    dag_sum = np.zeros((n, n))
    for dag in dags:
        dag = family_sequence_to_adj_mat(dag)
        n_dags += 1
        dag_sum += dag
        max_abs_errors.append(np.max(np.abs(dag_sum/n_dags - exact_probs)))
    return max_abs_errors
