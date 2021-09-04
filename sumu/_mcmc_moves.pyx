import numpy as np
cimport numpy as np
from .utils.math_utils import comb, subsets
from .bnet import transitive_closure


def DAG_edgerev(**kwargs):

    def valid(edges):
        # Check that there's at least one edge to reverse
        return len(edges) > 0

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

    if "validate" in kwargs and kwargs["validate"] is True:
        return valid(edges)

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

    def valid():
        return True

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

    if "validate" in kwargs and kwargs["validate"] is True:
        return valid()

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

    def valid(R):
        return len(R) > 1

    R = kwargs["R"]

    m = len(R)

    if "validate" in kwargs and kwargs["validate"] is True:
        return valid(R)

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


def B_swap_nonadjacent(**kwargs):
    """Swaps the layer of two nodes in different layers by sampling uniformly at random:

    1. :math:`j \in V` and :math:`k \in V \setminus \{ j-1, j, j+1 \}`
    2. nodes in :math:`B_{j}` and :math:`B_{k}`

    and finally swapping the layers of the chosen nodes.

    The proposed layering is valid and cannot be reached by the relocate function with one step.
    The proposal probability is symmetric.

    Args:
       B (list): Initial state of the layering for the relocation transition

    Returns:
        Proposed new M-layering, proposal probability and reverse proposal probability
    """

    def valid():
        return len(B) > 2

    B = kwargs["B"]

    if "validate" in kwargs and kwargs["validate"] is True:
        return valid()

    if len(B) == 3:
        j = np.random.choice([0,2])
    else:
        j = np.random.choice(range(len(B)))
    k = np.random.choice(np.setdiff1d(np.array(range(len(B))), [max(0, j-1), j, min(len(B)-1, j+1)]))
    v_j= np.random.choice(list(B[j]))
    v_k = np.random.choice(list(B[k]))
    B_prime = list()
    for i in range(len(B)):
        if i == j:
            B_prime.append(B[i].difference({v_j}).union({v_k}))
        elif i == k:
            B_prime.append(B[i].difference({v_k}).union({v_j}))
        else:
            B_prime.append(B[i])

    n_opts = 2*(len(B)-2) + (len(B)-2)*(len(B)-3)/2

    q = 1/(n_opts*len(B[j])*len(B[k]))
    return B_prime, q, q


def B_swap_adjacent(**kwargs):
    """Swaps the layer of two nodes in adjacent layers by sampling uniformly at random:

    1. :math:`j` between 1 and :math:`l-1`
    2. nodes in :math:`B_{j}` and :math:`B_{j+1}`

    and finally swapping the layers of the chosen nodes.

    The proposed layering is valid and cannot be reached by the relocate function with one step.
    The proposal probability is symmetric.

    Args:
       B (list): Initial state of the layering for the relocation transition

    Returns:
        Proposed new M-layering, proposal probability and reverse proposal probability
    """
    def valid():
        return len(B) > 1

    B = kwargs["B"]

    if "validate" in kwargs and kwargs["validate"] is True:
        return valid()

    j = np.random.randint(len(B) - 1)
    v_1 = np.random.choice(list(B[j]))
    v_2 = np.random.choice(list(B[j+1]))
    B_prime = list()
    for i in range(len(B)):
        if i == j:
            B_prime.append(B[i].difference({v_1}).union({v_2}))
        elif i == j+1:
            B_prime.append(B[i].difference({v_2}).union({v_1}))
        else:
            B_prime.append(B[i])
    q = 1/((len(B)-1)*len(B[j])*len(B[j+1]))
    return B_prime, q, q


def B_relocate_many(**kwargs):
    """Relocates :math:`n > 1` nodes in the input :math:`M`-layering :math:`B` by choosing uniformly at random:

    1. a source layer :math:`j` from among the valid possibilities,
    2. number :math:`n` between 2 and :math:`|B_{j}|`
    3. :math:`n` nodes from within the source layer,
    4. a target layer, including any possible new layer, where to move the nodes.

    In step (1), any layer with more than one node is valid, as the problem of invalid
    sources described in :py:func:`B_relocate_one` can be bypassed by choosing :math:`n` appropriately.
    At the moment :math:`n` however is chosen uniformly from :math:`\{ 2,\ldots,|B_{j}| \}` so the move can
    produce an invalid output (which is later discarded in MCMC function).

    In step (4) only a new layer right after :math:`j` is discarded as it is clearly invalid.
    However, as explained, the proposed M-layering can still be invalid.

    Args:
       B (list): Initial state of the layering for the relocation transition
       M (int):  Specifies the space of Bs

    Returns:
        Proposed new M-layering, proposal probability and reverse proposal probability
    """

    def valid():
        return len(B) > 1 and len(valid_sources) > 0

    B = kwargs["B"]
    M = kwargs["M"]

    valid_sources = [i for i in range(len(B)) if len(B[i]) > 1]

    if "validate" in kwargs and kwargs["validate"] is True:
        return valid()


    B_prime = [frozenset()]

    source = np.random.choice(valid_sources)
    target = np.random.choice(list(set(range(2*len(B)+1)).difference({source*2+1})))
    size = np.random.randint(2, len(B[source])+1)
    nodes = np.random.choice(list(B[source]), size)

    B_prime = [layer.difference({*nodes}) for layer in B]

    # Add nodes to target
    rev_source = 0
    if target % 2 == 1:  # add to existing part
        B_prime[target//2] = B_prime[target//2].union({*nodes})
        rev_source = target//2
    else:  # add to new part
        if target == 0:
            B_prime = [frozenset({*nodes})] + B_prime
            rev_source = 0
        else:  # new part after target//2
            B_prime = B_prime[0:target//2+1] + [frozenset({*nodes})] + B_prime[target//2+1:]
            rev_source = target//2

    # delete possible 0 layers and adjust rev_source accordingly
    for i in range(len(B_prime)):
        if len(B_prime[i]) == 0:
            del B_prime[i]
            if i < rev_source:
                rev_source -= 1
            break  # there can be max 1 0-part

    valid_rev_sources = [i for i in range(len(B_prime)) if len(B_prime[i]) > 1]
    q = 1/(len(valid_sources)*len(B[source])*(2*len(B)+1))
    q_rev = 1/(len(valid_rev_sources)*len(B_prime[rev_source])*(2*len(B_prime)+1))

    return B_prime, q, q_rev


def B_relocate_one(**kwargs):
    """Relocates a single node in the input :math:`M`-layering :math:`B` by choosing uniformly at random:

    1. a source layer from among the valid possibilities,
    2. a node within the source layer,
    3. a valid target layer, including any possible new layer, where to move the node.

    In step (1), only such layers are valid, from which it is possible to draw a node,
    and create a valid new :math:`M`-layering by moving it to another part. If the sizes of three
    consequetive layers are :math:`b_1`, :math:`b_2`, :math:`b_3`, such that :math:`b_1 + (b_2 - 1) = M = (b_2 - 1) + b_3`,
    then it is impossible to create a valid layering by relocating one node from the middle layer
    as to get a valid layering one should also merge the left or the right layer
    with the remaining middle layer.

    In step (3) the possible target layers depend on the source part sampled in step (1).
    The proposed :math:`M`-layering is thus guaranteed to be valid and different from the input :math:`M`-layering.

    Args:
       B (list): Initial state of the layering for the relocation transition
       M (int):  Specifies the space of Bs

    Returns:
        Proposed new :math:`M`-layering, proposal probability and reverse proposal probability
    """
    def valid():
        return len(B) > 1

    def valid_sources(b, M):
        # undefined if b, M is invalid
        valids = np.array([False]*len(b))
        for j in range(len(b)):
            if j == 0 or j == len(b)-1:
                valids[j] = True
                continue
            if b[j] > 1:
                if (b[j]-1 + b[j+1] > M and b[j]-1 + b[j-1] > M-1) or (b[j]-1 + b[j+1] > M-1 and b[j]-1 + b[j-1] > M):
                    valids[j] = True
            else:
                # if len(b[j]) == 1 part disappears
                valids[j] = True

        return valids

    def valid_targets(b, j, M):

        # undefined if j is invalid source

        valids = np.array([False]*(2*len(b)+1))
        b_stripped = [b[i] if i != j else b[i]-1 for i in range(len(b))]

        # 1. forced to place v in certain part
        for i in range(len(b_stripped)-1):
            if b_stripped[i] + b_stripped[i+1] <= M and b_stripped[i] != 0 and b_stripped[i+1] != 0:
                if i == j:
                    valids[(i+2)*2-1] = True
                else:
                    valids[(i+1)*2-1] = True
                return valids

        # 2. possibly multiple options where to place v
        sizes = [0]
        for s in b_stripped:
            sizes.append(s)
            sizes.append(0)
        for i in range(len(sizes)):
            if i+1 == (j+1)*2:
                continue
            if i == 0:  # first possible new part
                if sizes[i] + sizes[i+1] >= M:  # new layer
                    valids[i] = True
            elif i == len(sizes) - 1:  # last possible new part
                if sizes[i] + sizes[i-1] >= M:
                    valids[i] = True
            elif i % 2 == 0:  # middle new layers
                if sizes[i] + sizes[i-1] >= M and sizes[i] + sizes[i+1] >= M:
                    valids[i] = True
            elif i % 2 != 0:  # existing layers
                if i == 1:  # first existing part
                    if sizes[i] + sizes[i+2] >= M:
                        valids[i] = True
                elif i == len(sizes)-2:  # last existing part
                    if sizes[i] + sizes[i-2] >= M:
                        valids[i] = True
                else:  # middle layers
                    if sizes[i] + sizes[i-2] >= M and sizes[i] + sizes[i+2]:
                        valids[i] = True

        return valids

    B = kwargs["B"]
    M = kwargs["M"]

    if "validate" in kwargs and kwargs["validate"] is True:
        return valid()

    b = [len(part) for part in B]
    possible_sources = valid_sources(b, M)
    source = np.random.choice(np.nonzero(possible_sources)[0])
    possible_targets = valid_targets(b, source, M)
    target = np.random.choice(np.nonzero(possible_targets)[0])
    v = np.random.choice(list(B[source]))
    B_prime = [layer.difference({v}) for layer in B]

    # Add v to target
    rev_source = 0
    if target % 2 == 1:  # add to existing part
        B_prime[target//2] = B_prime[target//2].union({v})
        rev_source = target//2
    else:  # add to new part
        if target == 0:
            B_prime = [frozenset({v})] + B_prime
            rev_source = 0
        else:  # new part after target//2
            B_prime = B_prime[0:target//2+1] + [frozenset({v})] + B_prime[target//2+1:]
            rev_source = target//2

    # delete possible 0 layers and adjust rev_source accordingly
    for i in range(len(B_prime)):
        if len(B_prime[i]) == 0:
            del B_prime[i]
            if i < rev_source:
                rev_source -= 1
            break  # there can be max 1 0-part

    b_prime = [len(part) for part in B_prime]

    q = 1/(sum(possible_sources)*b[source]*sum(possible_targets))
    q_rev = 1/(sum(valid_sources(b_prime, M))*b_prime[rev_source]*sum(valid_targets(b_prime, rev_source, M)))

    return B_prime, q, q_rev
