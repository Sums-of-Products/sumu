"""Functions for reading/writing jkl score files, and something else? 
Score file reading/writing should be made irrelevant by faster scoring
allowing to only work directly with data files.
"""
import warnings
import numpy as np

from ..bnet import BNet, Node


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
        n_vars = len(scores)
        lines.append(str(n_vars) + "\n")
        for v in sorted(scores):
            lines.append("{} {}\n".format(v, len(scores[v])))
            for pset in sorted(scores[v], key=lambda pset: len(pset)):
                lines.append("{} {} {}\n".format(scores[v][pset], len(pset), ' '.join([str(p) for p in pset])))
        f.writelines(lines)


def read_dsc(filepath, zerolabels = True):
    """Read and parse a .dsc file in the input path into a object of type :py:class:`.BNet`.

    Args:
       filepath path to the .dsc file to load.

    Returns:
        BNet: a fully specified Bayesian network.

    """

    def custom_formatwarning(msg, *args, **kwargs):
        # ignore everything except the message
        return str(msg) + '\n'

    def normalize(name, key, probs):
        # Numpy apparently uses tolerance of 1e-8
        if abs(probs.sum() - 1.) > 1e-8:
            if abs(probs.sum() - 1) > maxerror:
                maxerror = abs(probs.sum() - 1)
                maxerror_node = name + ' ' + str(key)
            return probs / probs.sum()
        return probs

    warnings.formatwarning = custom_formatwarning
    maxerror = -float('inf')
    maxerror_node = ''

    names = list()
    nodes = dict()

    with open(filepath, 'r') as dsc_file:

        rows = dsc_file.readlines()
        current_node_name = None
        for i, row in enumerate(rows):
            try:
                items = row.split()
                if items[0] == 'node':
                    current_node_name = items[1]
                    names.append(current_node_name)
                    nodes[current_node_name] = Node(current_node_name)
                if current_node_name and items[0] == 'type':
                    arity = int(items[4])
                    nodes[current_node_name].r = arity
                if items[0] == 'probability':
                    current_node_name = items[2]
                    if items[3] == '|':
                        pnames = ''.join(items[4:-2]).split(',')
                        nodes[current_node_name].parents = [nodes[name] for name in pnames]
                if current_node_name and items[0][0] == '(':
                    items = ''.join(items).split(':')
                    config = tuple([int(x) for x in items[0][1:-1].split(',')])
                    probs = np.array([float(x) for x in items[1][:-1].split(',')])
                    probs = normalize(current_node_name, config, probs)
                    nodes[current_node_name].cpt[config] = probs
                if current_node_name and items[0][0] in {str(x) for x in range(10)}:
                    probs = np.array([float(x) for x in ''.join(items)[:-1].split(',')])
                    probs = normalize(current_node_name, (), probs)
                    nodes[current_node_name].cpt[()] = probs
            except ValueError as e:
                raise ValueError("Something wrong on row no. " + str(i)) from e
            except KeyError as e:
                raise KeyError("Something wrong on row no. " + str(i)) from e

    if maxerror_node:
        warnings.warn("Some probabilities didn't sum to 1 and were normalized.\n" +
                      "Maximum error: " + maxerror_node + ' ' + str(maxerror))

    bn = BNet([nodes[name] for name in names])
    if zerolabels:
        for i in range(len(bn.nodes)):
            bn.nodes[i].name = i
    return bn
