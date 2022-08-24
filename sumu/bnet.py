"""Module for representation and basic functionalities for Bayesian networks.
"""
import itertools
import logging

import numpy as np
from scipy.stats import dirichlet, multivariate_normal, wishart

from . import validate
from .data import Data


def family_sequence_to_adj_mat(dag, row_parents=False):
    """Format a sequence of families representing a DAG into an adjacency
    matrix.

    Args:
       dag (iterable): iterable like [(0, {}), (1, {2}), (2, {0, 1}), ...]
                       where first int is the node and the second item is a set
                       of the parents.
       row_parents (bool): If true A[i,j] == 1 if :math:`i` is parent of
                           :math:`j`, otherwise a transpose.

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
        matrix_pruned = matrix_pruned[:, ~pmask][~pmask, :]
        partitions.append(set(indices[pmask]))
        indices = indices[~pmask]
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
        self.dag = dag
        self.adj_mat = family_sequence_to_adj_mat(dag)
        self.sample_params(data=data)

    def sample_params(self, data=None):
        # this parameters define the prior used
        nu = np.zeros(self.n)
        am = 1
        aw = self.n + am + 1
        Tmat = np.identity(self.n) * (aw - self.n - 1) / (am + 1)

        N = 0
        if data is not None:  # posterior, need to update params
            data = Data(data)
            N = data.N

            # Sufficient statistics
            xN = np.mean(data.data, axis=0)
            SN = (data.data - xN).T @ (data.data - xN)

            # Updates
            nu = (am * nu + N * xN) / (am + N)  # nu'
            Tmat = (
                Tmat
                + SN
                + ((am * N) / (am + N)) * np.outer((nu - xN), (nu - xN))
            )  # Rmat
            am = am + N  # am'
            aw = aw + N  # aw'
            # rest is the same

        # this is sampling parameters from the prior/posterior
        # according to the formulas in the Gadget papers.

        self.Ce = np.zeros((self.n, self.n))
        self.B = np.zeros((self.n, self.n))
        self.mu = np.zeros(self.n)

        for node in range(self.n):
            pa = np.where(self.adj_mat[node])[0]
            l = len(pa) + 1
            # here l is the number of parents for node i plus 1
            T11 = Tmat[pa[:, None], pa]
            T12 = Tmat[pa, node]
            T21 = Tmat[node, pa]
            T22 = Tmat[node, node]
            T11inv = np.linalg.inv(T11)
            scale = T22 - T21 @ T11inv @ T12
            # q_i ~ W_1(T22-T21*inv(T11)*T12,aw-n+l)
            # the first parameter is not inverted as
            # numpy takes the scale matrix
            q = wishart.rvs(aw - self.n + l, scale, size=1)
            self.Ce[node, node] = 1 / q

            if l == 1:
                continue  # no need to sample parameters if no parents

            # b_i | q_i ~ N( inv(T11)*T12, q_i*T11)
            mb = T11inv @ T12
            vb = np.linalg.inv(q * T11)  # scipy takes variance not precision
            b = multivariate_normal.rvs(mb, vb, size=1)
            self.B[node, pa] = b

        # now the overall covariance, Winv is:
        A = np.linalg.inv(np.eye(self.n) - self.B)
        # mu_t ~ N( nu,am*W)
        Winv = (1 / am) * (A @ self.Ce @ A.transpose())
        self.mu = multivariate_normal.rvs(nu, Winv, size=1)
        # note that this is added to a zero mean data,
        # it is not multiplied by B or inv(I-B)!
        # the model is x = mu + B*e.

    def sample(self, N=1):
        A = np.linalg.inv(np.eye(self.n) - self.B)  # not iA but A
        Winv = A @ self.Ce @ A.transpose()
        data = multivariate_normal.rvs(self.mu, Winv, size=N)
        return Data(data)

    @classmethod
    def random(cls, n, *, enb=4):
        # warning: this is not the prior that was mainly used in the paper
        return cls(random_dag_with_expected_neighbourhood_size(n, enb=enb))


class DiscreteBNet:
    """Represents a Bayesian network."""

    def __init__(self, nodes):
        self.nodes = nodes
        self.topo_sort = topological_sort(nodes_to_family_list(nodes))
        # Parents in the CPTs are not in order so this is needed. Otherwise
        # self.dag would do. Maybe it could be arranged better.
        self.index_to_pset_indices = {
            u: [self.nodes.index(p) for p in node.parents]
            for u, node in enumerate(self.nodes)
        }
        self.dag = [
            (u, {self.nodes.index(p) for p in node.parents})
            for u, node in enumerate(self.nodes)
        ]

    @classmethod
    def from_dag(cls, dag, *, data=None, arity=2, ess=0.5, params="MP"):
        # Reordering just in case the nodes are not in order.
        nodes = {f[0]: DiscreteNode() for f in dag}
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
            p_configs = list(
                itertools.product(*[range(p.arity) for p in node.parents])
            )
            r = node.arity
            q = len(p_configs)
            for p_config in p_configs:
                p_config_counts = np.array(
                    [
                        np.all(
                            data[:, p_indices + [i]] == p_config + (i_val,),
                            axis=1,
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
                        probs = dirichlet.rvs(
                            p_config_counts + ess / (r * q)
                        ).squeeze()
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
        """Read and parse a .dsc file in the input path into a object of type
        :py:class:`.DiscreteBNet`.

        Args:
           filepath path to the .dsc file to load.

        Returns:
            DiscreteBNet: a fully specified Bayesian network.

        """

        def normalize(node_name, pset_config, probs):
            if abs(probs.sum() - 1.0) > 1e-10:
                logging.info(
                    f"Probs for {node_name} with pset config {pset_config} "
                    "need to be normalized.\n "
                    f"Probs: {probs}"
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
                        nodes[current_node_name] = DiscreteNode(
                            name=current_node_name
                        )
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
                        config = tuple(
                            [int(x) for x in items[0][1:-1].split(",")]
                        )
                        probs = np.array(
                            [float(x) for x in items[1][:-1].split(",")]
                        )
                        probs = normalize(current_node_name, config, probs)
                        nodes[current_node_name].cpt[config] = probs
                    if current_node_name and items[0][0] in {
                        str(x) for x in range(10)
                    }:
                        probs = np.array(
                            [float(x) for x in "".join(items)[:-1].split(",")]
                        )

                        probs = normalize(current_node_name, (), probs)
                        nodes[current_node_name].cpt[()] = probs
                except ValueError as e:
                    raise ValueError(
                        "Something wrong on row no. " + str(i)
                    ) from e
                except ValueError as e:
                    raise ValueError(
                        "Something wrong on row no. " + str(i)
                    ) from e

        nodes = [nodes[name] for name in names]
        return cls(nodes)

    def sample(self, N=1):
        data = np.zeros(shape=(N, len(self.nodes)), dtype=np.int32)
        for i in range(N):
            for i_node in self.topo_sort:
                pset = self.index_to_pset_indices[i_node]
                pset_config = tuple(data[i, pset])
                data[i, i_node] = self.nodes[i_node].sample(config=pset_config)
        return Data(data)

    def __getitem__(self, node_name_or_index):

        # TODO: bn["node1", "node2"].sample()
        is_name = type(node_name_or_index) == str

        try:
            if is_name:
                return [n for n in self.nodes if n.name == node_name_or_index][
                    0
                ]
            else:
                return self.nodes[node_name_or_index]
        except IndexError as e:
            raise (
                # Should be KeyError but it doesn't allow
                # nice formatting of the message
                IndexError(
                    str(  # doesn't print newline without the explicit str
                        f"No node by the name or index {node_name_or_index}.\n"
                        + "Available nodes names:"
                        f" {' '.join([str(n.name) for n in self.nodes])}\n"
                        + f"Available node indices: 0-{len(self.nodes)-1}",
                    )
                )
            ) from e


class DiscreteNode:
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
                p=self.cpt[
                    tuple([parent.sample() for parent in self.parents])
                ],
            )
        else:
            value = np.random.choice(range(self.arity), p=self.cpt[config])
        return value
