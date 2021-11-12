"""Functions for reading/writing jkl score files, and something else? 
Score file reading/writing should be made irrelevant by faster scoring
allowing to only work directly with data files.
"""
import warnings

import numpy as np

from ..bnet import DiscreteBNet, DiscreteNode
from ..stats import Stats


def pretty_dict(d, n=0, string=""):
    for k in d:
        if type(d[k]) in (dict, Stats):
            string += "{}{}\n".format(" "*n, k)
        else:
            string += "{}{}: {}\n".format(" "*n, k, d[k])
        if type(d[k]) in (dict, Stats):
            string += pretty_dict(d[k], n=n+2)
    return string


def pretty_title(string, n=0, width=80):
    end = "."*((width-len(string)) - 1)
    return "{}{} {}\n".format("\n"*n, string, end)


def dag_to_str(dag):
    return "|".join([str(f[0]) if len(f) == 1 else " ".join([str(f[0]), *[str(v) for v in sorted(f[1])]]) for f in dag])


def str_to_dag(dag_str):

    def parse_family_str(fstr):
        fstr = fstr.split()
        if len(fstr) > 1:
            return (int(fstr[0]), set(map(int, fstr[1:])))
        else:
            return (int(fstr[0]), set())

    return list(map(parse_family_str, dag_str.split("|")))


def write_data(data, datapath, bn=None):
    """Write data to datapath with variable names and arities if provided.

    Args:
       data numpy array
       datapath datapath
       bn Bayesian network (default None)
    """
    if bn is not None:
        names = np.array([node.name for node in bn.nodes])
        arities = np.array([node.r for node in bn.nodes])
        data = np.vstack((names, arities, data))
    np.savetxt(datapath, data, delimiter=" ", fmt='%i')


def get_n(datapath):
    """Get number of space separated columns in datafile"""
    with open(datapath) as f:
        return len(f.readline().split())


def decrement_dict_labels_by_one(jkl):
    """Decrement all integer keys in a two level dict by one.

    This is something to deal with some score files starting indexing from 1,
    as opposed to 0.
    """
    for old_node in sorted(jkl.keys()):
        tmp_dict = dict()
        for pset in jkl[old_node]:
                tmp_dict[tuple(np.array(pset) - 1)] = jkl[old_node][pset]
        jkl[old_node - 1] = tmp_dict
        del jkl[old_node]
    return jkl


def read_candidates(candidate_path):
    """Read parent candidates from file.

    Row number identifies the node and space separated numbers on each row
    identify the candidate parents.
    """
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
            score = float(row_list[0])
            n_parents = int(row_list[1])

            parents = frozenset()
            if n_parents > 0:
                parents = frozenset([int(x) for x in row_list[2:]])
            scores[current_var][frozenset(parents)] = score
            n_scores -= 1

    return scores


def _read_jkl(scorepath):
    """I think this is still needed in
    loading the aps produced parent set posteriors.
    Therefore I will not delete this yet, although
    there is another version using frozensets above."""
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
        scores = decrement_dict_labels_by_one(scores)

    return scores


def write_jkl(scores, fpath):
    """Assumes the psets are iterables, not bitmaps
    """
    with open(fpath, 'w') as f:
        lines = list()
        n = len(scores)
        lines.append(str(n) + "\n")
        for v in sorted(scores):
            lines.append("{} {}\n".format(v, len(scores[v])))
            for pset in sorted(scores[v], key=lambda pset: len(pset)):
                lines.append("{} {} {}\n".format(scores[v][pset], len(pset), ' '.join([str(p) for p in pset])))
        f.writelines(lines)
