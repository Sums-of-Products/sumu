import numpy as np
cimport numpy as np
from .utils.math_utils import comb, subsets
from .bnet import transitive_closure


def DAG_edgerev(**kwargs):

    DAG = kwargs["DAG"]
    R = kwargs["R"]
    score = kwargs["score"]
    n = len(DAG)
    # C is needed to make sure the move can be done if d=0
    C = kwargs["C"]
    d = kwargs["d"]

    edges = list((u, f[0]) for f in DAG for u in f[1])
    if d == 0:
        edges = [e for e in edges if e[1] in C[e[0]]]

    if len(edges) == 0:
        return False

    Ndagger = len(edges)
    Ndaggertilde = len(edges)

    edge = edges[np.random.randint(len(edges))]

    for f in DAG:
        if f[0] == edge[0] and len(f) == 2:
            Ndaggertilde -= len(f[1])
        if f[0] == edge[1] and len(f) == 2:
            Ndaggertilde -= len(f[1])

    orphan = [(f[0], set()) if f[0] in edge else f for f in DAG]
    descendorph = transitive_closure(orphan, R)

    # NOTE: old parent and its descendants
    edgeparentbannednodes = descendorph[edge[0]]

    U = set(range(n)).difference(edgeparentbannednodes)
    T = {edge[1]}
    newparents = score.sample_pset(edge[0], U, T)[0][1]

    DAG[edge[0]] = (edge[0], newparents)
    Ndaggertilde += len(newparents)

    Zstarparent = score.sum(edge[0], U, T)

    # NOTE: old parent and its descendants + old child and its descendants
    edgechildbannednodes = descendorph[edge[1]].union(descendorph[edge[0]])

    U = set(range(n)).difference(edgechildbannednodes)
    T = set()
    newparents = score.sample_pset(edge[1], U, T)[0][1]
    DAG[edge[1]] = (edge[1], newparents)
    Ndaggertilde += len(newparents)
    Zpluschild = score.sum(edge[1], U, T)

    # NOTE: old child and its descendants
    edgechildbannednodes = descendorph[edge[1]]

    U = set(range(n)).difference(edgechildbannednodes)
    T = {edge[0]}

    Zstarchild = score.sum(edge[1], U, T)

    # NOTE: old parent and its descendants
    edgeparentbannednodes = descendorph[edge[1]].union(descendorph[edge[0]])

    U = set(range(n)).difference(edgeparentbannednodes)
    T = set()
    Zplusparent = score.sum(edge[0], U, T)
    scoreratio = Ndagger/Ndaggertilde * np.exp(Zstarparent + Zpluschild - Zplusparent - Zstarchild)
    return DAG, scoreratio, edge


def R_basic_move(**kwargs):
    """Splits or merges a root-partition :footcite:`kuipers:2017`.

    Args:
       **kwargs: {"R": root-partition, "validate": boolean for whether to just validate input root-partition}

    Returns:
        tuple: proposed root-partition, proposal probability, inverse proposal probability, set of nodes that need to be rescored

    """

    def nbd_size(R):
        """Size of split/merge neighbourhood of input root-partition

        Args:
           R: root-partition

        Returns:
           The size of the neighbourhood and sum of binomial coefficients needed later
        """
        m = len(R)
        sum_binoms = [sum([comb(len(R[i]), v) for v in range(1, len(R[i]))]) for i in range(m)]
        return m - 1 + sum(sum_binoms), sum_binoms

    R = kwargs["R"]
    m = len(R)

    nbd, sum_binoms = nbd_size(R)
    q = 1/nbd

    j = int(np.random.rand()*nbd) + 1

    R_prime = list()
    if j < m:
        R_prime = [R[i] for i in range(j-1)] + [R[j-1].union(R[j])] + [R[i] for i in range(min(m, j+1), m)]
        q_prime = 1/nbd_size(R_prime)[0]
        rescore = R[j].union(R[min(m-1, j+1)])

    else:
        sum_binoms = [sum(sum_binoms[:i]) for i in range(1, len(sum_binoms)+1)]
        i_star = [m-1 + sum_binoms[i] for i in range(len(sum_binoms)) if m-1 + sum_binoms[i] < j]
        i_star = len(i_star)

        c_star = [comb(len(R[i_star]), v) for v in range(1, len(R[i_star])+1)]
        c_star = [sum(c_star[:i]) for i in range(1, len(c_star)+1)]

        c_star = [m-1 + sum_binoms[i_star-1] + c_star[i] for i in range(len(c_star))
                  if m-1 + sum_binoms[i_star-1] + c_star[i] < j]
        c_star = len(c_star)+1

        nodes = {int(v) for v in np.random.choice(list(R[i_star]), c_star)}

        R_prime = [R[i] for i in range(i_star)] + [nodes]
        R_prime += [R[i_star].difference(nodes)] + [R[i] for i in range(min(m, i_star+1), m)]

        q_prime = 1/nbd_size(R_prime)[0]
        rescore = R[i_star].difference(nodes).union(R[min(m-1, i_star+1)])

    return R_prime, q, q_prime, rescore


def R_swap_any(**kwargs):

    R = kwargs["R"]
    m = len(R)

    if len(R) < 2:
        return False

    if m == 2:
        j = 0
        k = 1
        q = 1/(len(R[j])*len(R[k]))
    else:
        if np.random.random() <= 0.9:  # adjacent
            j = np.random.randint(len(R)-1)
            k = j+1
            q = 0.9 * 1/((m-1) * len(R[j]) * len(R[k]))
        else:
            if m == 3:
                j = 0
                k = 2
                q = 1/(len(R[j])*len(R[k]))
            else:
                j = np.random.randint(m)
                n = list(set(range(m)).difference({j-1, j, j+1}))
                k = np.random.choice(n)
                q = 0.1 * 1/((m*len(n)) * len(R[j]) * len(R[k]))

    v_j = int(np.random.choice(list(R[j])))
    v_k = int(np.random.choice(list(R[k])))
    R_prime = list()
    for i in range(m):
        if i == j:
            R_prime.append(R[i].difference({v_j}).union({v_k}))
        elif i == k:
            R_prime.append(R[i].difference({v_k}).union({v_j}))
        else:
            R_prime.append(R[i])

    return tuple(R_prime), q, q, {v_j, v_k}.union(*R[min(j, k)+1:min(max(j, k)+2, m+1)])
