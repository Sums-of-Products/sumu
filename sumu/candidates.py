import numpy as np
from .utils.math_utils import subsets
from .aps import aps

r_is_initialized = False


class GlobalImport:

    # https://stackoverflow.com/a/53255802
    # This doesn't seem to like to be imported from elsewhere, e.g.,
    # from utils. Maybe with some work it might be possible too.

    def __enter__(self):
        return self

    def __exit__(self, *args):
        import inspect
        self.collector = inspect.getargvalues(inspect.getouterframes(inspect.currentframe())[1].frame).locals
        globals().update(self.collector)


def init_r():

    global r_is_initialized
    if r_is_initialized:
        return

    with GlobalImport() as gi:
        try:
            from rpy2.robjects import r
            from rpy2.robjects import numpy2ri
            from rpy2.robjects.packages import importr
        except ImportError as e:
            msg = ["To use the candidate parent algorithms pc, mb or ges you",
                   "need to have R installed. Pc and mb require the R-package",
                   "bnlearn; ges requires pcalg. Finally, you also need to",
                   "have the Python package rpy2 installed to interface with R."]
            raise Exception(' '.join(msg)) from e

    load_funcs = """
    datapath_or_matrix_to_numeric_dataframe <- function(data_path_or_matrix,
                                                        discrete=TRUE,
                                                        arities=FALSE)
    {
      if (typeof(data_path_or_matrix) == "character") {
        data <- read.table(data_path_or_matrix, header = FALSE)
      }
      else {
        data <- data_path_or_matrix
        mode(data) = "numeric"
        data <- data.frame(data)
      }
      if (discrete) {
        if (arities) {
          arities <- data[1,]
          data <- data[2:nrow(data),]
        } else {
          arities <- lapply(data, function(x) length(unique(x)))
        }
        data[] <- lapply(data, as.factor)
        for(i in 1:length(arities)) {
          levels(data[, i]) <- as.character(0:(arities[[i]] - 1))
        }
      }

      colnames(data) <- 0:(ncol(data)-1)
      return(data)
    }
    """

    r(load_funcs)
    r_is_initialized = True


def convert_to_r_data(data):
    # Input is sumu.Data

    init_r()

    numpy2ri.activate()
    datar = r.matrix(data.all().flatten(),
                     nrow=data.N,
                     ncol=data.n,
                     byrow=True)
    numpy2ri.deactivate()

    discrete = data.discrete
    arities = True if data.arities is not False else False

    datar = r['datapath_or_matrix_to_numeric_dataframe'](datar,
                                                         discrete=discrete,
                                                         arities=arities)
    return datar


def candidates_to_str(C):
    return '|'.join([' '.join([str(node) for node in C[v]]) for v in sorted(C)])


def parse_candidates(C_str):
    return {v[0]: v[1] for v in zip(range(C_str.count("|") + 1), [tuple(int(c) for c in C.split()) for C in C_str.split("|")])}


def _adjust_number_candidates(K, C, method, scores=None):

    assert method in ['random', 'top'], "fill method should be random or top"
    if method == 'top':
        assert scores is not None, "scorepath (-s) required for fill == top"

    C = dict(C.items())
    n = len(C)

    for v in C:
        add_n = max(0, K - len(C[v]))
        add_from = [add_node for add_node in range(n) if add_node not in C[v] + (v,)]
        if method == 'random':
            if len(C[v]) < K:
                C[v] = C[v] + tuple(np.random.choice(add_from, add_n, replace=False))
            elif len(C[v]) > K:
                C[v] = np.random.choice(C[v], K, replace=False)
        if method == 'top':
            if len(C[v]) < K:
                C_v_top = sorted([(parent, scores.local(v, np.array([parent])))
                                  for parent in add_from],
                                 key=lambda item: item[1], reverse=True)[:add_n]
                C_v_top = tuple([c[0] for c in C_v_top])
                C[v] = C[v] + C_v_top
            elif len(C[v]) > K:
                C[v] = sorted([(parent, scores.local(v, np.array([parent])))
                               for parent in C[v]],
                              key=lambda item: item[1], reverse=True)[:K]
                C[v] = [c[0] for c in C[v]]

    for v in C:
        C[v] = tuple(sorted(C[v]))

    return C


def _most_freq_candidates(K, Cs):

    C = {v: list() for v in range(len(Cs[0]))}
    for C_i in Cs:
        for v in C_i:
            C[v] += C_i[v]

    for v in C:
        C[v] = [i[0] for i in sorted([(u, C[v].count(u)) for u in C if C[v].count(u) > 0], key=lambda i: i[1], reverse=True)][:K]
        C[v] = tuple(sorted(C[v]))

    return C


def hybrid(K, **kwargs):

    algos = kwargs.get("halgos")
    fill = kwargs.get("hfill")
    assert not [algos, fill].count(None), "list of algorithms to use (-ha) and tie breaking method (-hf) required for algo == hybrid"

    if fill == "top":
        scores = kwargs["scores"]

    def vote(Cs):
        C = {v: set() for v in Cs[0][0]}
        for v in C:
            k = 0
            while len(C[v]) < K:
                to_add = tuple()
                for i in range(len(algos)):
                    to_add += tuple(Cs[i][k][v])
                k += 1
                if len(C[v].union(set(to_add))) <= K:
                    C[v] = C[v].union(set(to_add))
                else:
                    to_add = {0: to_add}
                    for u in set(to_add[0]).difference(C[v]):
                        if to_add[0].count(u) in to_add:
                            to_add[to_add[0].count(u)] = to_add[to_add[0].count(u)].union({u})
                        else:
                            to_add[to_add[0].count(u)] = {u}
                    del to_add[0]

                    for count in sorted(to_add.keys(), reverse=True):
                        if len(C[v].union(to_add[count])) <= K:
                            C[v] = C[v].union(to_add[count])
                        else:
                            if fill == 'random':
                                C[v] = C[v].union(np.random.choice(list(to_add[count]), K - len(C[v]), replace=False))
                            elif fill == 'top':
                                C_v_top = sorted([(parent, scores[v][(parent,)])
                                                  for parent in to_add[count]],
                                                 key=lambda item: item[1], reverse=True)[:K - len(C[v])]
                                C_v_top = set([c[0] for c in C_v_top])
                                C[v] = C[v].union(C_v_top)

                            break
        return C

    C = [tuple(algo[a](k, **kwargs) for k in range(1, K+1)) for a in algos]
    C = vote(C)
    for v in C:
        C[v] = tuple(sorted(C[v]))

    return C


def rnd(K, **kwargs):

    n = kwargs.get("n")
    assert n is not None, "nvars (-n) required for algo == rnd"

    C = dict()
    for v in range(n):
        C[v] = tuple(sorted(np.random.choice([u for u in range(n) if u != v], K, replace=False)))
    return C


def ges(K, **kwargs):
    """Greedy equivalence search :cite:`chickering:2002`.

    GES is implemented in the R package pcalg :cite:`hauser:2012,kalisch:2012`,
    for which the function :py:func:`pcalg` provides a Python wrapping.
    """

    init_r()

    data = kwargs["data"]
    data = convert_to_r_data(data)

    B = kwargs.get("B", 20)
    fill = kwargs.get("fill", "top")
    scores = kwargs.get("scores")

    if "B" in kwargs:
        Cs = list()
        for i in range(kwargs["B"]):
            bsample = data.rx(r["sample"](data.nrow, data.nrow, replace=True), True)
            Cs.append(pcalg("ges", K, bsample))
        C = _most_freq_candidates(K, Cs)
    else:
        C = pcalg("ges", K, data)
    if fill:
        C = _adjust_number_candidates(K, C, fill, scores=scores)
    return C


def pcalg(method, K, data):

    init_r()

    base = importr("base")
    importr('pcalg')
    dollar = base.__dict__["$"]

    n = data.ncol
    C = dict({node: list() for node in range(n)})

    data = r["data.matrix"](data)

    if method == 'ges':
        score = r["new"]("GaussL0penObsScore", data)
        cpdag = r["ges"](score).rx2("essgraph")
        for v in range(n):
            # NOTE: undirected edges are represented as bidirected!
            # See pcalg documentation at
            # https://cran.r-project.org/web/packages/pcalg/pcalg.pdf
            # for ges and EssGraph.
            # Also running the ges example confirms this.
            C[v] = [v-1 for v in sorted(dollar(cpdag, ".in.edges").rx2(v+1))]

    for v in C:
        C[v] = tuple(sorted(C[v]))

    return C


def pc(K, **kwargs):

    init_r()

    data = kwargs.get("data")
    data = convert_to_r_data(data)
    alpha = kwargs.get("alpha", 0.1)
    max_sx = kwargs.get("max_sx", 1)

    B = kwargs.get("B", 20)
    fill = kwargs.get("fill", "top")
    scores = kwargs.get("scores")

    if B is not None:
        Cs = list()
        for i in range(B):
            bsample = data.rx(r["sample"](data.nrow,
                                          data.nrow,
                                          replace=True), True)
            Cs.append(bnlearn("pc", K, bsample, alpha=alpha, max_sx=max_sx))
        C = _most_freq_candidates(K, Cs)
    else:
        C = bnlearn("pc", K, data, alpha=alpha, max_sx=max_sx)
    if fill is not None:
        C = _adjust_number_candidates(K, C, fill, scores=scores)
    return C


def mb(K, **kwargs):

    init_r()

    data = kwargs.get("data")
    data = convert_to_r_data(data)
    alpha = kwargs.get("alpha", 0.1)
    max_sx = kwargs.get("max_sx", 1)

    B = kwargs.get("B", 20)
    fill = kwargs.get("fill", "top")
    scores = kwargs.get("scores")

    if B is not None:
        Cs = list()
        for i in range(B):
            bsample = data.rx(r["sample"](data.nrow,data.nrow,replace=True), True)
            Cs.append(bnlearn("mb", K, bsample, alpha=alpha, max_sx=max_sx))
        C = _most_freq_candidates(K, Cs)
    else:
        C = bnlearn("mb", K, data, alpha=alpha, max_sx=max_sx)
    if fill is not None:
        C = _adjust_number_candidates(K, C, fill, scores=scores)
    return C


def hc(K, **kwargs):

    init_r()

    datapath = kwargs.get("datapath")
    assert datapath is not None, "datapath (-d) required for algo == hc"
    B = kwargs.get("B")
    if B is None:
        B = 20
    fill = kwargs.get("fill")
    if fill is None:
        fill = "top"
    scores = kwargs.get("scores")

    data = r['load_dat'](datapath)

    if B is not "none":
        Cs = list()
        for i in range(B):
            bsample = data.rx(r["sample"](data.nrow, data.nrow, replace=True), True)
            Cs.append(bnlearn("hc", K, bsample))
        C = _most_freq_candidates(K, Cs)
    else:
        C = bnlearn("hc", K, data)
    if fill != "none":
        C = _adjust_number_candidates(K, C, fill, scores=scores)
    return C


def bnlearn(method, K, data, **kwargs):

    init_r()

    R_bnlearn = importr('bnlearn')

    n = data.ncol
    C = dict({v: list() for v in range(n)})

    if method == 'mb':
        bn = R_bnlearn.iamb(data, alpha=kwargs["alpha"], max_sx=kwargs["max_sx"])
    if method == 'pc':
        bn = R_bnlearn.pc_stable(data, alpha=kwargs["alpha"], max_sx=kwargs["max_sx"])
    if method == 'hc':
        # Uses BIC by default
        bn = R_bnlearn.hc(data)

    for v in range(n):
        if method == 'mb':
            mb = [int(u) for u in bn.rx2('nodes').rx2(str(v)).rx2('mb')]
            for u in mb:
                if u not in C[v]:
                    C[v].append(u)
        if method == 'pc':
            nbr = [int(u) for u in bn.rx2('nodes').rx2(str(v)).rx2('nbr')]
            children = [int(u) for u in bn.rx2('nodes').rx2(str(v)).rx2('children')]
            for u in nbr:
                if u in children:
                    continue
                if u not in C[v]:
                    C[v].append(u)
        if method == 'hc':
            pset = [int(u) for u in bn.rx2('nodes').rx2(str(v)).rx2('parents')]
            for u in pset:
                if u not in C[v]:
                    C[v].append(u)

    for v in C:
        C[v] = tuple(sorted(C[v]))

    return C


def opt(K, **kwargs):

    scores = kwargs.get("scores")
    n = kwargs.get("n")

    C = np.array([[v for v in range(n) if v != u] for u in range(n)], dtype=np.int32)
    pset_posteriors = aps(scores.all_candidate_restricted_scores(C),
                          as_dict=True, normalize=True)

    C = dict()
    for v in pset_posteriors:
        postsums = dict()
        for candidate_set in subsets(set(pset_posteriors).difference({v}), K, K):
            postsums[candidate_set] = np.logaddexp.reduce([pset_posteriors[v][pset]
                                                           for pset in subsets(candidate_set, 0, K)])
        C[v] = max(postsums, key=lambda candidate_set: postsums[candidate_set])
    return C


def top(K, **kwargs):

    scores = kwargs["scores"]
    assert scores is not None, "scorepath (-s) required for algo == top"

    C = dict()
    for v in range(scores.n):
        top_candidates = sorted([(parent, scores.local(v, (parent,)))
                                 for parent in range(scores.n) if parent != v],
                                key=lambda item: item[1], reverse=True)[:K]
        top_candidates = tuple(sorted(c[0] for c in top_candidates))
        C[v] = top_candidates
    return C


def greedy(K, **kwargs):

    s = kwargs.get("s")
    scores = kwargs.get("scores")
    assert not [s, scores].count(None), "s (-gs) and scorepath (-s) required for algo == greedy"

    def unimportance(v, u, U):
        pi_v = [scores.local(v, S) for S in subsets([m for m in U if m != u], 0, s)]
        return max(pi_v)

    C = dict({v: list() for v in range(scores.n)})
    for v in range(scores.n):
        U = [u for u in range(scores.n) if u != v]
        while len(C[v]) < K:
            least_unimportant = min([(u, unimportance(v, u, U)) for u in U], key=lambda item: item[1])[0]
            C[v].append(least_unimportant)
            U = [u for u in U if u != least_unimportant]
        C[v] = tuple(sorted(C[v]))

    return C


def greedy_1(K, **kwargs):

    scores = kwargs.get("scores")
    assert scores is not None, "scorepath (-s) required for algo == greedy-1"

    def highest_uncovered(v, U):
        return max([(u, scores.local(v, np.array(S + (u,))))
                    for S in subsets(C[v], 0, [len(C[v]) if scores.maxid == -1 else min(len(C[v]), scores.maxid-1)][0])
                    for u in U], key=lambda item: item[1])[0]

    C = dict({int(v): list() for v in range(scores.n)})
    for v in C:
        U = [u for u in C if u != v]
        while len(C[v]) < K:
            u_hat = highest_uncovered(v, U)
            C[v].append(u_hat)
            U = [u for u in U if u != u_hat]
        C[v] = tuple(sorted(C[v]))

    return C


def greedy_lite(K, **kwargs):

    scores = kwargs.get("scores")
    k = kwargs.get("k")
    assert scores is not None
    if k is None:
        k = min(6, K)
    assert k <= K

    def k_highest_uncovered(v, U, k):

        uncovereds = {(u, scores._local(v, np.array(S + (u,))))
                      for S in subsets(C[v], 0, [len(C[v]) if scores.maxid == -1 else min(len(C[v]), scores.maxid-1)][0])
                      for u in U}
        k_highest = set()
        while len(k_highest) < k:
            u_hat = max(uncovereds, key=lambda pair: pair[1])
            k_highest.add(u_hat[0])
            uncovereds.remove(u_hat)
        return k_highest

    C = dict({int(v): set() for v in range(scores.n)})
    for v in C:
        U = [u for u in C if u != v]
        while len(C[v]) < K - k:
            u_hat = k_highest_uncovered(v, U, 1)
            C[v].update(u_hat)
            U = [u for u in U if u not in u_hat]
        C[v].update(k_highest_uncovered(v, U, k))
        C[v] = tuple(sorted(C[v]))
        scores.clear_cache()
    return C


def greedy_1_sum(K, **kwargs):

    scores = kwargs.get("scores")
    assert scores is not None, "scorepath (-s) required for algo == greedy-1"

    def highest_uncovered_sum(v, U):
        sums = list()
        for u in U:
            sums.append((u, np.logaddexp.reduce([scores[v][tuple(sorted(set(S + (u,))))]
                                                      for S in subsets(C[v], 0, len(C[v]))])))
        return max(sums, key=lambda item: item[1])[0]

    C = dict({v: list() for v in scores})
    for v in scores:
        U = [u for u in scores if u != v]
        while len(C[v]) < K:
            u_hat = highest_uncovered_sum(v, U)
            C[v].append(u_hat)
            U = [u for u in U if u != u_hat]
        C[v] = tuple(sorted(C[v]))

    return C


def greedy_1_double(K, **kwargs):

    scores = kwargs.get("scores")
    assert scores is not None, "scorepath (-s) required for algo == greedy-1-double"

    def highest_uncovered(v, U):
        return max([(u, scores[v][tuple(sorted(set(S + (u,))))])
                    for S in subsets(C[v], 0, len(C[v]))
                    for u in U], key=lambda item: item[1])[0]

    def highest_double_uncovered(v, U):
        return max([((u, m), scores[v][tuple(sorted(set(S + (u, m))))])
                    for S in subsets(C[v], 0, len(C[v]))
                    for u in U for m in set(U).difference({u})], key=lambda item: item[1])[0]

    C = dict({v: list() for v in scores})
    for v in scores:
        U = [u for u in scores if u != v]
        while len(C[v]) < K:
            if len(C[v]) == K - 1:
                u_hat = highest_uncovered(v, U)
                C[v].append(u_hat)
                U = [u for u in U if u != u_hat]
            else:
                u_hat, m_hat = highest_double_uncovered(v, U)
                C[v].append(u_hat)
                C[v].append(m_hat)
                U = [u for u in U if u not in [u_hat, m_hat]]
        C[v] = tuple(sorted(C[v]))

    return C


def greedy_1_double_sum(K, **kwargs):

    scores = kwargs.get("scores")
    assert scores is not None, "scorepath (-s) required for algo == greedy-1"

    def highest_uncovered_sum(v, U):
        sums = list()
        for u in U:
            sums.append((u, np.logaddexp.reduce([scores[v][tuple(sorted(set(S + (u,))))]
                                                      for S in subsets(C[v], 0, len(C[v]))])))
        return max(sums, key=lambda item: item[1])[0]

    def highest_double_uncovered_sum(v, U):
        sums = list()
        for u in U:
            for m in set(U).difference({u}):
                sums.append(((u, m), np.logaddexp.reduce([scores[v][tuple(sorted(set(S + (u, m))))]
                                                          for S in subsets(C[v], 0, len(C[v]))])))
        return max(sums, key=lambda item: item[1])[0]

    C = dict({v: list() for v in scores})
    for v in scores:
        U = [u for u in scores if u != v]
        while len(C[v]) < K:
            if len(C[v]) == K - 1:
                u_hat = highest_uncovered_sum(v, U)
                C[v].append(u_hat)
                U = [u for u in U if u != u_hat]
            else:
                u_hat, m_hat = highest_double_uncovered_sum(v, U)
                C[v].append(u_hat)
                C[v].append(m_hat)
                U = [u for u in U if u not in [u_hat, m_hat]]
        C[v] = tuple(sorted(C[v]))

    return C


def greedy_2(K, **kwargs):

    scores = kwargs.get("scores")
    assert scores is not None, "scorepath (-s) required for algo == greedy-2"

    C = dict({v: set() for v in scores})
    for v in scores:
        psets_leq_K = sorted([(pset, scores[v][pset])
                               for pset in scores[v] if len(pset) <= K],
                              key=lambda item: item[1],
                              reverse=True)
        psets_leq_K = [item[0] for item in psets_leq_K]
        i = 0
        while len(C[v]) < K:
            if len(C[v].union(psets_leq_K[i])) <= K:
                C[v] = C[v].union(psets_leq_K[i])
            else:
                pset_diff = list(set(psets_leq_K[i]).difference(C[v]))
                C[v] = C[v].union(np.random.choice(pset_diff, K - len(C[v]), replace=False))
            i += 1
        C[v] = tuple(sorted(C[v]))

    return C


def greedy_2_s(K, **kwargs):

    s = kwargs.get("s")
    scores = kwargs.get("scores")
    assert not [s, scores].count(None), "s (-gs) and scorepath (-s) required for algo == greedy-2-s"

    C = dict({v: set() for v in scores})
    for v in scores:
        psets_leq_s = sorted([(pset, scores[v][pset])
                               for pset in scores[v] if len(pset) <= s],
                              key=lambda item: item[1],
                              reverse=True)
        psets_leq_s = [item[0] for item in psets_leq_s]
        i = 0
        while len(C[v]) < K:
            if len(C[v].union(psets_leq_s[i])) <= K:
                C[v] = C[v].union(psets_leq_s[i])
            else:
                pset_diff = list(set(psets_leq_s[i]).difference(C[v]))
                C[v] = C[v].union(np.random.choice(pset_diff, K - len(C[v]), replace=False))
            i += 1
        C[v] = tuple(sorted(C[v]))

    return C


def greedy_3(K, **kwargs):
    pset_posteriors = kwargs.get("pset_posteriors")
    assert pset_posteriors is not None, "pset posteriors path (-p) required for algo == greedy-3"
    return greedy_2(K, scores=pset_posteriors)


def greedy_2_inverse(K, **kwargs):

    scores = kwargs.get("scores")
    assert scores is not None, "scorepath (-s) required for algo == greedy-2-inverse"

    n = len(scores)

    C = dict({v: set(scores).difference({v}) for v in scores})
    for v in scores:
        psets_leq_n_minus_K = sorted([(pset, scores[v][pset])
                                      for pset in scores[v] if len(pset) <= n - K],
                                     key=lambda item: item[1])
        psets_leq_n_minus_K = [item[0] for item in psets_leq_n_minus_K]
        i = 0
        while len(C[v]) > K:
            if len(C[v].difference(psets_leq_n_minus_K[i])) >= K:
                C[v] = C[v].difference(psets_leq_n_minus_K[i])
            else:
                pset_intersection = list(set(psets_leq_n_minus_K[i]).intersection(C[v]))
                C[v] = C[v].difference(np.random.choice(pset_intersection, len(C[v]) - K, replace=False))
            i += 1
        C[v] = tuple(sorted(C[v]))

    return C


def greedy_2_s_inverse(K, **kwargs):

    s = kwargs.get("s")
    scores = kwargs.get("scores")
    assert not [s, scores].count(None), "s (-gs) and scorepath (-s) required for algo == greedy-2-s-inverse"

    n = len(scores)

    C = dict({v: set(scores).difference({v}) for v in scores})
    for v in scores:
        psets_leq_s = sorted([(pset, scores[v][pset])
                              for pset in scores[v] if len(pset) <= s],
                             key=lambda item: item[1])
        psets_leq_s = [item[0] for item in psets_leq_s]
        i = 0
        while len(C[v]) > K:
            if len(C[v].difference(psets_leq_s[i])) >= K:
                C[v] = C[v].difference(psets_leq_s[i])
            else:
                pset_intersection = list(set(psets_leq_s[i]).intersection(C[v]))
                C[v] = C[v].difference(np.random.choice(pset_intersection, len(C[v]) - K, replace=False))
            i += 1
        C[v] = tuple(sorted(C[v]))

    return C


def greedy_backward_forward(K, **kwargs):

    scores = kwargs.get("scores")
    assert scores is not None, "scorepath (-s) required for algo == greedy-backward-forward"

    def min_max(v):
        return min([max([(u, scores.local(v, tuple(set(S + (u,)))))
                         for S in subsets(C[v].difference({u}), 0, [len(C[v]) - 1 if scores.maxid == -1 else min(len(C[v]) - 1, scores.maxid-1)][0])],
                        key=lambda item: item[1])
                    for u in C[v]], key=lambda item: item[1])[0]

    def highest_uncovered(v, U):
        return max([(u, scores.local(v, tuple(set(S + (u,)))))
                    for S in subsets(C[v], 0, [len(C[v]) if scores.maxid == -1 else min(len(C[v]), scores.maxid-1)][0])
                    for u in U],
                   key=lambda item: item[1])[0]

    C = rnd(K, n=scores.n)
    C = {v: set(C[v]) for v in C}
    for v in C:
        C_prev = dict(C)
        while True:
            u_hat = min_max(v)
            C[v] = C[v].difference({u_hat})
            u_hat = highest_uncovered(v, set(C).difference(C[v]).difference({v}))
            C[v].add(u_hat)
            if C == C_prev:
                break
            else:
                C_prev = dict(C)
        C[v] = tuple(sorted(C[v]))

    return C


def pessy(K, **kwargs):

    s = kwargs.get("s")
    scores = kwargs.get("scores")
    assert not [s, scores].count(None), "s (-gs) and scorepath (-s) required for algo == pessy"

    def sum_scores(v, u):
        sums = list()
        for Y in subsets(sorted(C[v] + [u]),
                 min(len(C[v]) + 1, max(0, K - s)),
                 min(len(C[v]) + 1, max(0, K - s))):
            sums.append(np.logaddexp.reduce([scores.local(v, S) for S in subsets(Y, 0, len(Y))]))
        return min(sums)

    C = dict({v: list() for v in range(scores.n)})

    for v in range(scores.n):
        U = [u for u in range(scores.n) if u != v]

        while len(C[v]) < K:
            max_u = max([(u, sum_scores(v, u)) for u in U], key=lambda item: item[1])[0]
            C[v].append(max_u)
            U = [u for u in U if u != max_u]

        C[v] = tuple(sorted(C[v]))

    return C


candidate_parent_algorithm = {
    "opt": opt,
    # "rnd": rnd,
    "top": top,
    "pc": pc,
    "mb": mb,
    "ges": ges,
    # "hc": hc,
    # "greedy": greedy,
    # "pessy": pessy,
    "greedy": greedy_1,
    "greedy-lite": greedy_lite,
    # "greedy-2": greedy_2,
    # "greedy-3": greedy_3,
    # "greedy-1-double": greedy_1_double,
    # "greedy-1-sum": greedy_1_sum,
    # "greedy-1-double-sum": greedy_1_double_sum,
    # "greedy-2-inverse": greedy_2_inverse,
    # "greedy-2-s": greedy_2_s,
    # "greedy-2-s-inverse": greedy_2_s_inverse,
    "back-forth": greedy_backward_forward,
    # "hybrid": hybrid,

}


def eval_candidates(C, pset_posteriors):
    v_cover = dict()
    for v in C:
        v_cover[v] = np.exp(np.logaddexp.reduce([pset_posteriors[v][pset] for pset in subsets(C[v], 0, len(C[v]))]))
    return v_cover


def eval_candidates_gmean(pset_covers):
    return np.exp(np.log(list(pset_covers.values())).mean())


def eval_candidates_amean(pset_covers):
    return np.mean(list(pset_covers.values()))
