# -*- coding: utf-8 -*-
"""Module for representation and basic functionalities for Bayesian networks.

The module now includes both module level functions for input DAGs
and some overlapping Structure class functions. Is it ok to have both?
"""

import numpy as np
import itertools
import copy


def family_sequence_to_adj_mat(dag):
    """Format a sequence of families representing a DAG into an adjacency matrix.

    Args:
       dag (iterable): iterable like [(0), (1, 2), (2, 0, 1), ...] where first
                       int is the node and the following, if any, the parents.

    Returns:
        adjacency matrix

    """
    adj_mat = np.zeros((len(dag), len(dag)))
    for f in dag:
        if len(f) > 1:
            adj_mat[f[1:], f[0]] = 1
    return adj_mat


def partition(dag):

    adj_mat = family_sequence_to_adj_mat(dag)

    partitions = list()
    matrix_pruned = adj_mat
    indices = np.arange(adj_mat.shape[0])
    while True:
        pmask = np.sum(matrix_pruned, axis=0) == 0
        if not any(pmask):
            break
        matrix_pruned = matrix_pruned[:, pmask == False][pmask == False, :]
        partitions.append(frozenset(indices[pmask]))
        indices = indices[pmask == False]
    return partitions


def topological_sort(dag):
    """Sort the nodeds in a DAG in a topological order.

    The algorithm is from :cite:`kahn:1962`.
    """
    names = set(range(len(dag)))
    edges = set((f[i], f[0]) for f in dag for i in range(1, len(f)) if len(f) > 1)

    l = list()
    s = set(names)
    for pnode in names:
        for cnode in names:
            if (pnode, cnode) in edges:
                s.discard(cnode)
    while s:
        n = s.pop()
        l.append(n)
        for m in names:
            if (n, m) in edges:
                edges.discard((n, m))
                s.add(m)
                for pm in names:
                    if (pm, m) in edges:
                        s.discard(m)
    if edges:
        return False
    return l


def transitive_closure(dag, mat=False):

    edgeto = {f[0]: [set(f[1:]) if len(f) > 1 else set()][0] for f in dag}
    tclosure = {v: {v} for v in range(len(dag))}
    for v in topological_sort(dag)[::-1]:
        for u in edgeto[v]:
            tclosure[u].update(tclosure[v])
    if mat is False:
        return tclosure
    tclomat = np.zeros((len(dag), len(dag)))
    for v in tclosure:
        tclomat[v, list(tclosure[v])] = 1
    return tclomat


class Structure:
    """Represents Bayesian network structure as used by :py:class:`.BNet`."""

    def __init__(self, names, edges):
        self.names = names
        self.edges = edges

    @classmethod
    def from_nodes(cls, nodes):
        names = [node.name for node in nodes]
        edges = set((parent.name, node.name)
                    for node in nodes
                    for parent in node.parents)
        return cls(names, edges)

    @classmethod
    def from_matrix(cls, mat):
        names = [str(x) for x in range(len(mat))]
        edges = set([(names[i], names[x]) for i in range(len(mat)) for x in np.where(mat[i, :] == 1)[0]])
        return cls(names, edges)

    @property
    def matrix(self):
        mat = np.zeros(shape=(len(self.names), len(self.names)))
        for edge in self.edges:
            mat[self.names.index(edge[0]), self.names.index(edge[1])] = 1
        return mat
    
    @classmethod
    def from_family_list(cls, family_list):
        """Initialize a Structure object from list of families.

        Args:
           list: List of iterables where first item of each  is the node and the following, if any, are the parents.

        Returns:
            Structure

        """
        raise NotImplementedError

    @property
    def family_list(self):
        """Returns a family list representation of Structure.

        Returns:
            list: List of iterables where first item of each is the node and the following, if  any, are the parents.

        """

    @property
    def map(self):
        p = dict([(n, set()) for n in self.names])
        c = copy.deepcopy(p)
        for edge in self.edges:
            p[edge[1]].add(edge[0])
            c[edge[0]].add(edge[1])
        return {'p': p, 'c': c}

    @property
    def partition(self):
        partitions = list()
        matrix_pruned = self.matrix
        indices = np.arange(0, len(self.names))
        while True:
            pmask = np.sum(matrix_pruned, axis=0) == 0
            if not any(pmask):
                break
            matrix_pruned = matrix_pruned[:, pmask == False][pmask == False, :]
            partitions.append(indices[pmask])
            indices = indices[pmask == False]
        return partitions


class BNet:
    """Represents a Bayesian network."""

    def __init__(self, nodes):
        self.nodes = nodes
        self.structure = Structure.from_nodes(nodes)

    def sample(self, n, **kwargs):
        if 'seed' in kwargs and isinstance(kwargs['seed'], int):
            np.random.seed(kwargs['seed'] * n)
        data = np.zeros(shape=(n, len(self.nodes)), dtype=np.int)
        for i in range(n):
            for node in self.nodes:
                node.clear()
            data[i] = tuple(node.sample() for node in self.nodes)
        return data

    def p(self, sample, log=False):
        p = 1
        for node in self.nodes:
            p = p*node.p(sample[self.nodes.index(node)],
                         tuple([sample[self.nodes.index(parent)] for parent in node.parents]))
        return np.log(p) if log else p


class Node:

    def __init__(self, name, r=None, cpt=None, parents=[]):
        self.name = name
        self.r = r
        self.value = None
        self.cpt = dict() if not cpt else cpt
        self.configs = ()
        self.q = 0
        self.parents = list()
        self.set_parents(parents)

    def clear(self):
        self.value = None

    def sample(self):
        if self.value is None:
            self.value = np.nonzero(np.random.multinomial(1, self.cpt[tuple([parent.sample() for parent in self.parents])]))[0][0]
        return self.value

    def p(self, val, config):
        return self.cpt[config][val]

    def set_parents(self, parents):
        self.configs = list(itertools.product(*[range(p.r) for p in parents]))
        self.q = len(self.configs)
        self.parents = parents
