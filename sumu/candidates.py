import time
import numpy as np

from .utils.math_utils import subsets
from .aps import aps


def rnd(K, *, data, **kwargs):

    n = data.n
    C = dict()
    for v in range(n):
        C[v] = tuple(
            sorted(
                np.random.choice(
                    [u for u in range(n) if u != v], K, replace=False
                )
            )
        )
    return C, None


def opt(K, **kwargs):

    scores = kwargs.get("scores")
    n = kwargs.get("n")

    C = np.array(
        [[v for v in range(n) if v != u] for u in range(n)], dtype=np.int32
    )
    pset_posteriors = aps(
        scores.all_candidate_restricted_scores(C), as_dict=True, normalize=True
    )

    C = dict()
    for v in pset_posteriors:
        postsums = dict()
        for candidate_set in subsets(
            set(pset_posteriors).difference({v}), K, K
        ):
            postsums[candidate_set] = np.logaddexp.reduce(
                [
                    pset_posteriors[v][pset]
                    for pset in subsets(candidate_set, 0, K)
                ]
            )
        C[v] = max(postsums, key=lambda candidate_set: postsums[candidate_set])
    return C, None


def greedy(
    K,
    *,
    scores,
    params={"k": 6, "t_budget": None, "criterion": "score"},
    **kwargs,
):
    k = params.get("k")
    if k is not None:
        k = min(k, K)
    t_budget = params.get("t_budget")
    criterion = params.get("criterion")
    assert criterion in ("score", "gain")
    if criterion == "score":
        goodness = lambda v, S, u: scores._local(v, np.array(S + (u,)))
    elif criterion == "gain":
        goodness = lambda v, S, u: scores._local(
            v, np.array(S + (u,))
        ) - scores._local(v, np.array(S))

    def k_highest_uncovered(v, U, k):

        uncovereds = {
            u: max(
                {
                    goodness(v, S, u)
                    for S in subsets(
                        C[v],
                        0,
                        [
                            len(C[v])
                            if scores.maxid == -1
                            else min(len(C[v]), scores.maxid - 1)
                        ][0],
                    )
                }
            )
            for u in U
        }

        k_highest = list()
        while len(k_highest) < k:
            u_hat = max(uncovereds, key=lambda u: uncovereds[u])
            k_highest.append(u_hat)
            del uncovereds[u_hat]
        return k_highest

    def get_k(t_budget):
        if t_budget < 0:
            return K
        U = set(range(1, len(C)))
        t_used = list()
        t = time.time()
        t_pred_next_add = 0
        i = 0
        while len(C[0]) < K and sum(t_used) + t_pred_next_add < t_budget:
            u_hat = k_highest_uncovered(0, U, 1)
            C[0] += u_hat
            U -= set(u_hat)
            t_prev = time.time() - t - sum(t_used)
            t_used.append(t_prev)
            t_pred_next_add = 2 * t_prev
            i += 1
        C[0] = [v for v in C[0] if v not in u_hat]
        U.update(set(u_hat))
        k = K - len(C[0])
        C[0] += k_highest_uncovered(0, U, k)
        C[0] = tuple(C[0])
        scores.clear_cache()
        return k

    C = dict({int(v): list() for v in range(scores.data.n)})

    if t_budget is not None:
        t_budget /= scores.data.n
        k = get_k(t_budget)

    for v in [v for v in C if len(C[v]) == 0]:
        U = [u for u in C if u != v]
        while len(C[v]) < K - k:
            u_hat = k_highest_uncovered(v, U, 1)
            C[v] += u_hat
            U = [u for u in U if u not in u_hat]
        C[v] += k_highest_uncovered(v, U, k)
        C[v] = tuple(C[v])
        scores.clear_cache()

    return C, {"k": k}


candidate_parent_algorithm = {"opt": opt, "rnd": rnd, "greedy": greedy}
