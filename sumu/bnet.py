"""Module for representation and basic functionalities for Bayesian networks.
"""
import logging
import numpy as np
import itertools
from . import validate
from scipy.stats import dirichlet
from scipy.stats import multivariate_t as mvt
from .data import Data


def family_sequence_to_adj_mat(dag, row_parents=False):
    """Format a sequence of families representing a DAG into an adjacency matrix.

    Args:
       dag (iterable): iterable like [(0, {}), (1, {2}), (2, {0, 1}), ...] where first
                       int is the node and the second item is a set of the parents.
       row_parents (bool): If true A[i,j] == 1 if :math:`i` is parent of :math:`j`,
                           otherwise a transpose.

    Returns:
        adjacency matrix

    """
    adj_mat = np.zeros((len(dag), len(dag)), dtype=np.int8)
    for f in dag:
        adj_mat[tuple(f[1]), f[0]] = 1

    if row_parents is False:
        adj_mat = adj_mat.T
    return adj_mat


def adj_mat_to_family_sequence(adj_mat, row_parents=False):
    if row_parents:
        adj_mat = adj_mat.T
    dag = [(i, set(np.where(adj_mat[i])[0])) for i in range(len(adj_mat))]
    return dag


def partition(dag):
    adj_mat = family_sequence_to_adj_mat(dag, row_parents=True)
    partitions = list()
    matrix_pruned = adj_mat
    indices = np.arange(adj_mat.shape[0])
    while True:
        pmask = np.sum(matrix_pruned, axis=0) == 0
        if not any(pmask):
            break
        matrix_pruned = matrix_pruned[:, pmask == False][pmask == False, :]
        partitions.append(set(indices[pmask]))
        indices = indices[pmask == False]
    return partitions


def topological_sort(dag):
    """Sort the nodes in a DAG in a topological order.

    The algorithm is from :footcite:`kahn:1962`.
    """
    names = set(range(len(dag)))
    edges = set((u, f[0]) for f in dag for u in f[1])

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


def transitive_closure(dag, R=None, mat=False):
    tclosure = {v: {v} for v in range(len(dag))}
    if R is not None:
        toposort = [i for part in R for i in part][::-1]
    else:
        toposort = topological_sort(dag)[::-1]
    for v in toposort:
        for u in dag[v][1]:
            tclosure[u].update(tclosure[v])
    if mat is False:
        return tclosure
    tclomat = np.zeros((len(dag), len(dag)))
    for v in tclosure:
        tclomat[v, list(tclosure[v])] = 1
    return tclomat


def nodes_to_family_list(nodes):
    return [
        (u, set([nodes.index(v) for v in node.parents]))
        for u, node in enumerate(nodes)
    ]


def random_dag_with_expected_neighbourhood_size(n, *, enb=4):
    pedge = min(1, enb / (n - 1))
    order = np.random.permutation(n)
    dag = np.tril(
        np.random.choice(range(2), (n, n), p=[1 - pedge, pedge]), k=-1
    )
    dag = dag[:, order][order, :]
    return adj_mat_to_family_sequence(dag)


class GaussianBNet:
    def __init__(self, dag, *, data=None):
        self.n = len(validate.dag(dag))
        self.dag = family_sequence_to_adj_mat(dag)
        self.sample_params(data=data)

    def sample_params(self, data=None, lb_e=0.1, ub_e=2, lb_ce=0.5, ub_ce=1.5):
        nu = np.zeros(self.n)
        am = 1
        aw = self.n + am + 1
        Tmat = np.identity(self.n) * (aw - self.n - 1) / (am + 1)

        N = 0
        if data is not None:
            data = Data(data)
            N = data.N

            # Sufficient statistics
            xN = np.mean(data.data, axis=0)
            SN = (data.data - xN).T @ (data.data - xN)

            # Parameters for the posterior
            R = (
                Tmat
                + SN
                + ((am * N) / (am + N)) * np.outer((nu - xN), (nu - xN))
            )
        else:
            R = Tmat

        if True:  # params == "random":
            self.Ce = np.diag(np.random.uniform(lb_ce, ub_ce, self.n))
            self.B = np.zeros((self.n, self.n))
            for node in range(self.n):
                pa = np.where(self.dag[node])[0]
                if len(pa) == 0:
                    continue
                l = len(pa) + 1
                R11 = R[node, node]
                R12 = R[pa, node]
                R11inv = np.linalg.inv(R[pa[:, None], pa])
                df = aw + N - self.n + l
                mb = R11inv @ R12
                divisor = R11 - R11inv @ R12
                covb = divisor / df * R11inv
                b = mvt.rvs(loc=mb, shape=covb, df=df)
                self.B[node, pa] = b

    def sample(self, N=1):
        iA = np.linalg.inv(np.eye(self.n) - self.B)
        data = np.random.normal(size=(N, self.n))
        data = (iA @ np.sqrt(self.Ce) @ data.T).T
        return data

    @classmethod
    def random(cls, *, n, enb=4):
        return cls(random_dag_with_expected_neighbourhood_size(n, enb=enb))


class BNet:
    """Represents a Bayesian network."""

    def __init__(self, nodes):
        self.nodes = nodes
        self.topo_sort = topological_sort(nodes_to_family_list(nodes))
        self.index_to_pset_indices = {
            u: [self.nodes.index(p) for p in node.parents]
            for u, node in enumerate(self.nodes)
        }

    @classmethod
    def from_dag(cls, dag, *, data=None, arity=2, ess=0.5, params="MP"):
        # Reordering just in case the nodes are not in order.
        nodes = {f[0]: Node() for f in dag}
        for f in dag:
            nodes[f[0]].parents = [nodes[p] for p in sorted(f[1])]
        nodes = [nodes[u] for u in sorted(nodes)]

        if data is None:
            data = np.array([], dtype=np.int32).reshape(0, len(nodes))

        for i, node in enumerate(nodes):
            if type(arity) == list:
                node.arity = arity[i]
            else:
                node.arity = arity

        for i, node in enumerate(nodes):
            p_indices = [nodes.index(p) for p in node.parents]
            p_configs = list(itertools.product(*[range(p.arity) for p in node.parents]))
            r = node.arity
            q = len(p_configs)
            for p_config in p_configs:
                p_config_counts = np.array(
                    [
                        np.all(
                            data[:, p_indices + [i]] == p_config + (i_val,), axis=1
                        ).sum()
                        for i_val in range(node.arity)
                    ]
                )
                if params == "MLE":
                    if p_config_counts.sum() == 0:
                        p_config_counts += 1
                    probs = p_config_counts / p_config_counts.sum()
                if params == "random":
                    # see https://github.com/numpy/numpy/issues/5851
                    probs = np.array([np.nan] * r)
                    while np.isnan(probs).any():
                        probs = dirichlet.rvs(p_config_counts + ess / (r * q)).squeeze()
                if params == "MP":
                    probs = (p_config_counts + ess / (r * q)) / (
                        p_config_counts + ess / (r * q)
                    ).sum()
                if params == "MAP":
                    probs = p_config_counts + ess / (r * q) - 1
                    # The -1 may yield negative counts. Correct way to handle?
                    probs[probs < 0] = 0
                    probs /= probs.sum()
                node.cpt[p_config] = probs

        return cls(nodes)

    @classmethod
    def read_file(cls, path_to_dsc_file):
        """Read and parse a .dsc file in the input path into a object of type :py:class:`.BNet`.

        Args:
           filepath path to the .dsc file to load.

        Returns:
            BNet: a fully specified Bayesian network.

        """

        def normalize(node_name, pset_config, probs):
            if abs(probs.sum() - 1.0) > 1e-10:
                logging.info(
                    f"Probs for {node_name} with pset config {pset_config}"
                    + f"need to be normalized.\n"
                    + f"Probs: {probs}"
                )
                return probs / probs.sum()
            return probs

        names = list()
        nodes = dict()

        with open(path_to_dsc_file, "r") as dsc_file:

            rows = dsc_file.readlines()
            current_node_name = None
            for i, row in enumerate(rows):
                try:
                    items = row.split()
                    if items[0] == "node":
                        current_node_name = items[1]
                        names.append(current_node_name)
                        nodes[current_node_name] = Node(name=current_node_name)
                    if current_node_name and items[0] == "type":
                        arity = int(items[4])
                        nodes[current_node_name].arity = arity
                    if items[0] == "probability":
                        current_node_name = items[2]
                        if items[3] == "|":
                            pnames = "".join(items[4:-2]).split(",")
                            nodes[current_node_name].parents = [
                                nodes[name] for name in pnames
                            ]
                    if current_node_name and items[0][0] == "(":
                        items = "".join(items).split(":")
                        config = tuple([int(x) for x in items[0][1:-1].split(",")])
                        probs = np.array([float(x) for x in items[1][:-1].split(",")])
                        probs = normalize(current_node_name, config, probs)
                        nodes[current_node_name].cpt[config] = probs
                    if current_node_name and items[0][0] in {str(x) for x in range(10)}:
                        probs = np.array(
                            [float(x) for x in "".join(items)[:-1].split(",")]
                        )
                        probs = normalize(current_node_name, (), probs)
                        nodes[current_node_name].cpt[()] = probs
                except ValueError as e:
                    raise ValueError("Something wrong on row no. " + str(i)) from e
                except KeyError as e:
                    raise KeyError("Something wrong on row no. " + str(i)) from e

        nodes = [nodes[name] for name in names]
        return cls(nodes)

    def sample(self, n=1):
        data = np.zeros(shape=(n, len(self.nodes)), dtype=np.int32)
        for i in range(n):
            for i_node in self.topo_sort:
                pset = self.index_to_pset_indices[i_node]
                pset_config = tuple(data[i, pset])
                data[i, i_node] = self.nodes[i_node].sample(config=pset_config)
        return data

    def __getitem__(self, node_name):
        try:
            return [n for n in self.nodes if n.name == node_name][0]
        except IndexError as e:
            raise (
                # Should be KeyError but it doesn't allow
                # nice formatting of the message
                IndexError(
                    f"No node by the name {node_name}.\n"
                    + f"Available nodes: {' '.join([n.name for n in self.nodes])}",
                )
            ) from e


class Node:
    def __init__(self, *, name=None, arity=None, cpt=None, parents=list()):
        # parents are in a list because it specifies the order in cpt
        self.name = name
        self.arity = arity
        self.cpt = dict() if not cpt else cpt
        self.parents = parents

    def sample(self, config=None):
        if config is None:
            value = np.random.choice(
                range(self.arity),
                p=self.cpt[tuple([parent.sample() for parent in self.parents])],
            )
        else:
            value = np.random.choice(range(self.arity), p=self.cpt[config])
        return value
