from collections import defaultdict
import numpy as np
from .utils.math_utils import log_minus_exp, subsets
from .bnet import partition
from .mcmc_moves import R_basic_move, R_swap_any, B_relocate_one, B_relocate_many, B_swap_adjacent, B_swap_nonadjacent, DAG_edgerev
from .stats import stats


class LayeringMCMC:

    def __init__(self, M, max_indegree, scores):
        self.M = M
        self.max_indegree = max_indegree
        self.scores = scores
        self.n = len(scores)
        self.stay_prob = 0.01
        if len(scores.keys()) == M:
            self.B = [frozenset(scores.keys())]
        else:
            self.B = self.map_names(list(scores.keys()),
                                    np.random.choice(list(self.gen_M_layerings(self.n, M)), 1)[0])

        self.tau_hat = self.parentsums(self.B, self.M, self.max_indegree, self.scores)
        self.g = self.posterior(self.B, self.M, self.max_indegree, self.scores, self.tau_hat, return_all=True)
        self.B_prob = self.g[0][frozenset()][frozenset()]
        self.R = self.generate_partition(self.B, self.M, self.tau_hat, self.g, self.scores)

        self.B_moves = [B_relocate_one, B_relocate_many, B_swap_adjacent, B_swap_nonadjacent]
        self.R_moves = [R_basic_move, R_swap_any]
        self.DAG_moves = [DAG_edgereversal]
        self.moves = self.B_moves + self.R_moves + self.DAG_moves
        self.move_probs = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05])

    def DAG_edgereversal(self, **kwargs):
        return DAG_edgereversal(**kwargs)

    def R_basic_move(self, **kwargs):
        return R_basic_move(**kwargs)

    def R_swap_any(self, **kwargs):
        return R_swap_any(**kwargs)

    def B_relocate_one(self, **kwargs):
        return B_relocate_one(**kwargs)

    def B_relocate_many(self, **kwargs):
        return B_relocate_many(**kwargs)

    def B_swap_adjacent(self, **kwargs):
        return B_swap_adjacent(**kwargs)

    def B_swap_nonadjacent(self, **kwargs):
        return B_swap_nonadjacent(**kwargs)

    def partitions(self, n):
        div_sets = list(subsets(range(1, n), 0, n-1))
        for divs in div_sets:
            partition = list()
            start = 0
            if divs:
                for end in divs:
                    partition.append(list(range(n))[start:end])
                    start = end
                partition.append(list(range(n))[end:])
            else:
                partition.append(list(range(n)))
            yield partition

    def gen_M_layerings(self, n, M):
        for B in self.partitions(n):
            if len(B) == 1:
                yield B
                continue
            psums = list()
            for i in range(1, len(B)):
                psums.append(len(B[i-1]) + len(B[i]))
            if min(psums) <= M:
                continue
            else:
                yield self.map_names(np.random.permutation(n), B)

    def map_names(self, names, B):
        """Just something to deal with
        node names not being 0-indexed ints"""
        new_B = list()
        for part in B:
            new_layer = set()
            for v in part:
                new_layer.add(names[v])
            new_B.append(frozenset(new_layer))
        return new_B

    def valid_layering(self, B, M):
        if min([len(B[j]) for j in range(len(B))]) == 0:
            return False
        if len(B) == 1:
            return True
        psums = list()
        for j in range(1, len(B)):
            psums.append(len(B[j-1]) + len(B[j]))
        if min(psums) <= M:
            return False
        return True

    def R_to_B(self, R, M):
        B = list()
        B_layer = R[0]
        for i in range(1, len(R)):
            if len(B_layer) + len(R[i]) <= M:
                B_layer = B_layer.union(R[i])
            else:
                B.append(B_layer)
                B_layer = R[i]
        B.append(B_layer)
        return B

    def generate_partition(self, B, M, tau_hat, g, pi_v):

        l = len(B)
        B = [frozenset()] + B + [frozenset()]
        R = list()

        j = 0
        D = frozenset()
        T = frozenset()
        k = 0

        p = dict()
        f = dict()

        while j <= l:
            if D == B[j]:
                j_prime = j+1
                D_prime = frozenset()
            else:
                j_prime = j
                D_prime = D

            if (D.issubset(B[j]) and D != B[j]) or j == l or len(B[j+1]) <= M:
                S = [frozenset(Si) for Si in subsets(B[j_prime].difference(D_prime), 1, max(1, len(B[j_prime].difference(D_prime))))]
                A = frozenset()
            else:
                S = [B[j+1]]
                A = B[j+1].difference({min(B[j+1])})

            for v in B[j_prime].difference(D_prime):
                if not T:
                    p[v] = pi_v[v][frozenset()]
                else:
                    if T == B[j]:
                        p[v] = tau_hat[v][frozenset({v})]
                    else:
                        p[v] = log_minus_exp(tau_hat[v][D], tau_hat[v][D.difference(T)])

            f[A] = 0
            for v in A:
                f[A] += p[v]

            r = -1*(np.random.exponential() - g[j][D][T])
            s = -float('inf')
            for Si in S:
                v = min(Si)
                f[Si] = f[Si.difference({v})] + p[v]
                if D != B[j] or j == 0 or len(Si) > M - len(B[j]):
                    s = np.logaddexp(s, f[Si] + g[j_prime][D_prime.union(Si)][Si])
                if s > r:
                    k = k + 1
                    R.append(Si)
                    T = Si
                    D = D_prime.union(Si)
                    break
            j = j_prime

        return R

    def parentsums(self, B, M, max_indegree, pi_v):

        tau = dict({v: defaultdict(lambda: -float("inf")) for v in set().union(*B)})
        tau_hat = dict({v: dict() for v in set().union(*B)})

        B.append(set())
        for j in range(len(B)-1):
            B_to_j = set().union(*B[:j+1])

            for v in B[j+1]:

                tmp_dict = defaultdict(list)
                for G_v in subsets(B_to_j, 1, max_indegree):
                    if frozenset(G_v).intersection(B[j]):
                        tmp_dict[v].append(pi_v[v][frozenset(G_v)])
                tau_hat[v].update((frozenset({item[0]}), np.logaddexp.reduce(item[1])) for item in tmp_dict.items())

            if len(B[j]) <= M:
                for v in B[j].union(B[j+1]):

                    tmp_dict = defaultdict(list)
                    for G_v in subsets(B_to_j.difference({v}), 0, max_indegree):
                        tmp_dict[frozenset(G_v).intersection(B[j])].append(pi_v[v][frozenset(G_v)])

                    tau[v].update((item[0], np.logaddexp.reduce(item[1])) for item in tmp_dict.items())

                    for D in subsets(B[j].difference({v}), 0, len(B[j].difference({v}))):
                        tau_hat[v][frozenset(D)] = np.logaddexp.reduce(np.array([tau[v][frozenset(C)] for C in subsets(D, 0, len(D))]))

        B.pop()  # drop the added empty set

        return tau_hat

    def posterior(self, B, M, max_indegree, pi_v, tau_hat, return_all=False):

        l = len(B)  # l is last proper index
        B = [frozenset()] + B + [frozenset()]
        g = {i: dict() for i in range(0, l+2)}
        p = dict()
        f = dict()

        for j in range(l, -1, -1):

            if len(B[j]) > M or j == 0:
                P = [(B[j], B[j])]
            else:
                P = list()
                for D in subsets(B[j], len(B[j]), 1):
                    for T in subsets(D, 1, len(D)):
                        P.append((frozenset(D), frozenset(T)))

            for DT in P:
                D, T = DT
                if D == B[j]:
                    j_prime = j + 1
                    D_prime = frozenset()
                else:
                    j_prime = j
                    D_prime = D

                if D == B[j] and j < l and len(B[j+1]) > M:
                    S = [B[j+1]]
                    A = B[j+1].difference({min(B[j+1])})
                else:
                    S = [frozenset(Si) for Si in subsets(B[j_prime].difference(D_prime), 1, max(1, len(B[j_prime].difference(D_prime))))]
                    A = frozenset()

                for v in B[j_prime].difference(D_prime):
                    if not T:
                        p[v] = pi_v[v][frozenset()]
                    else:
                        if T == B[j]:
                            p[v] = tau_hat[v][frozenset({v})]
                        else:
                            p[v] = log_minus_exp(tau_hat[v][D], tau_hat[v][D.difference(T)])

                f[A] = 0
                for v in A:
                    f[A] += p[v]

                if D not in g[j]:
                    g[j][D] = dict()
                if not S:
                    g[j][D][T] = 0.0
                else:
                    g[j][D][T] = -float('inf')

                tmp_list = list([g[j][D][T]])
                for Si in S:
                    v = min(Si)
                    f[Si] = f[Si.difference({v})] + p[v]
                    if D != B[j] or j == 0 or len(Si) > M - len(B[j]):
                        tmp_list.append(f[Si] + g[j_prime][D_prime.union(Si)][Si])

                g[j][D][T] = np.logaddexp.reduce(tmp_list)

        if return_all:
            return g
        return g[0][frozenset()][frozenset()]

    def posterior_R(self, R, scores, max_indegree):
        """Brute force R score"""

        def possible_psets(U, T, max_indegree):
            if not U:
                yield frozenset()
            for required in subsets(T, 1, max(1, max_indegree)):
                for additional in subsets(U.difference(T), 0, max_indegree - len(required)):
                    yield frozenset(required).union(additional)

        def score_v(v, pset, scores):
            return scores[v][pset]

        def hat_pi(v, U, T, scores, max_indegree):
            return np.logaddexp.reduce([score_v(v, pset, scores) for pset in possible_psets(U, T, max_indegree)])

        def f(U, T, S, scores, max_indegree):
            hat_pi_sum = 0
            for v in S:
                hat_pi_sum += hat_pi(v, U, T, scores, max_indegree)
            return hat_pi_sum

        f_sum = 0
        for i in range(len(R)):
            f_sum += f(set().union(*R[:i]),
                       [R[i-1] if i-1>-1 else set()][0],
                       R[i], scores, max_indegree)
        return f_sum

    def sample_DAG(self, R, scores, max_indegree):
        DAG = list((root,) for root in R[0])
        tmp_scores = [scores[root][frozenset()] for root in R[0]]
        for j in range(1, len(R)):
            ban = set().union(*R[min(len(R)-1, j):])
            req = R[max(0, j-1)]
            for i in R[j]:
                psets = list()
                pset_scores = list()
                for pset in scores[i].keys():
                    if not pset.intersection(ban) and pset.intersection(req) and len(pset) <= max_indegree:
                        psets.append(pset)
                        pset_scores.append(scores[i][pset])
                normalizer = sum(np.exp(pset_scores - np.logaddexp.reduce(pset_scores)))
                k = np.where(np.random.multinomial(1, np.exp(pset_scores - np.logaddexp.reduce(pset_scores))/normalizer))[0][0]
                DAG.append((i,) + tuple(psets[k]))
                tmp_scores.append(pset_scores[k])

        return DAG, sum(tmp_scores)

    def sample(self):

        # NOTE: The outernmost if-else is strange? Should return B
        # even if we stay in place.

        if np.random.rand() < self.stay_prob:
            R = self.generate_partition(self.B, self.M, self.tau_hat, self.g, self.scores)
            DAG, DAG_prob = self.sample_DAG(R, self.scores, self.max_indegree)
            # update_stats(B_prob, DAG_prob, B, DAG, None, None, "stay", None, None)

        else:

            move = np.random.choice(self.moves, p=self.move_probs)

            if not move(B=self.B, M=self.M, R=self.R, validate=True):
                self.R = self.generate_partition(self.B, self.M, self.tau_hat, self.g, self.scores)
                DAG, DAG_prob = self.sample_DAG(self.R, self.scores, self.max_indegree)
                # update_stats(B_prob, DAG_prob, B, DAG, None, None, "invalid_input_" + move.__name__, None, None)
                # if print_steps:
                #     print_step()
                # continue
                #B_prob, DAG_prob, B, DAG, None, None, "invalid_input_" + move.__name__, None, None
                return self.B, self.B_prob, DAG, DAG_prob

            if move in self.B_moves:

                B_prime, q, q_rev = move(B=self.B, M=self.M)

                # B_prob = stats["B_prob"][-1]

                if not self.valid_layering(B_prime, self.M):
                    R = self.generate_partition(self.B, self.M, self.tau_hat, self.g, self.scores)
                    DAG, DAG_prob = self.sample_DAG(self.R, self.scores, self.max_indegree)
                    # update_stats(B_prob, DAG_prob, B, DAG, None, None, "invalid_output_" + move.__name__, None, None)
                    # if print_steps:
                    #    print_step()
                    # continue
                    return self.B, self.B_prob, DAG, DAG_prob

                # t_psum = time.process_time()
                tau_hat_prime = self.parentsums(B_prime, self.M, self.max_indegree, self.scores)
                # t_psum = time.process_time() - t_psum

                # t_pos = time.process_time()
                g_prime = self.posterior(B_prime, self.M, self.max_indegree, self.scores, tau_hat_prime, return_all=True)
                B_prime_prob = g_prime[0][frozenset()][frozenset()]
                # t_pos = time.process_time() - t_pos

                acc_prob = np.exp(B_prime_prob - self.B_prob)*q_rev/q

            elif move in self.R_moves:

                R_prime, q, q_rev, rescore = move(R=self.R)
                B_prime = self.R_to_B(R_prime, self.M)

                if B_prime == self.B:
                    R = self.generate_partition(self.B, self.M, self.tau_hat, self.g, self.scores)
                    DAG, DAG_prob = self.sample_DAG(self.R, self.scores, self.max_indegree)
                    # update_stats(B_prob, DAG_prob, B, DAG, None, None, "identical_" + move.__name__, None, None)
                    # if print_steps:
                    #     print_step()
                    # continue
                    return self.B, self.B_prob, DAG, DAG_prob

                # WHY IS THIS CALCULATED ALREADY HERE
                # t_psum = time.process_time()
                tau_hat_prime = self.parentsums(B_prime, self.M, self.max_indegree, self.scores)
                # t_psum = time.process_time() - t_psum

                R_prob = self.posterior_R(self.R, self.scores, self.max_indegree)
                R_prime_prob = self.posterior_R(R_prime, self.scores, self.max_indegree)

                acc_prob = np.exp(R_prime_prob - R_prob)*q_rev/q

            elif move in self.DAG_moves:

                R = self.generate_partition(self.B, self.M, self.tau_hat, self.g, self.scores)
                DAG, DAG_prob = self.sample_DAG(self.R, self.scores, self.max_indegree)
                DAG_prime, acc_prob = move(DAG=DAG, scores=self.scores, max_indegree=self.max_indegree)
                R_prime = partition(DAG_prime)
                R_prime_prob = self.posterior_R(R_prime, self.scores, self.max_indegree)
                B_prime = self.R_to_B(R_prime, self.M)
                if B_prime == self.B:
                    return self.B, self.B_prob, DAG, DAG_prob
                # WHY IS THIS CALCULATED ALREADY HERE
                tau_hat_prime = self.parentsums(B_prime, self.M, self.max_indegree, self.scores)

            if np.random.rand() < acc_prob:

                # t_pos = time.process_time()
                g_prime = self.posterior(B_prime, self.M, self.max_indegree, self.scores, tau_hat_prime, return_all=True)
                B_prime_prob = g_prime[0][frozenset()][frozenset()]
                # t_pos = time.process_time() - t_pos

                self.B = B_prime
                self.tau_hat = tau_hat_prime
                self.g = g_prime
                self.B_prob = self.g[0][frozenset()][frozenset()]
                if move not in self.DAG_moves:
                    self.R = self.generate_partition(self.B, self.M, self.tau_hat, self.g, self.scores)
                    DAG, DAG_prob = self.sample_DAG(self.R, self.scores, self.max_indegree)
                    # update_stats(B_prime_prob, DAG_prob, B_prime, DAG, acc_prob, 1, move.__name__, t_psum, t_pos)
                    return self.B, self.B_prob, DAG, DAG_prob
                else:
                    self.R = R_prime
                    #print("-------")
                    #print(DAG)
                    #print("-------")
                    DAG_prob = sum(self.scores[f[0]][frozenset(f[1:])] if len(f) > 1 else self.scores[f[0]][frozenset()] for f in DAG)
                    return self.B, self.B_prob, DAG, DAG_prob

            else:
                self.R = self.generate_partition(self.B, self.M, self.tau_hat, self.g, self.scores)
                DAG, DAG_prob = self.sample_DAG(self.R, self.scores, self.max_indegree)
                # update_stats(B_prob, DAG_prob, B, DAG, acc_prob, 0, move.__name__, t_psum, t_pos)
                return self.B, self.B_prob, DAG, DAG_prob


        # if print_steps:
        #    print_step()

    #return stats


class PartitionMCMC:
    """Partition-MCMC sampler :footcite:`kuipers:2017` with efficient scoring.
    """

    def __init__(self, C, score, d, temperature=1.0, move_weights=[1, 1, 2]):

        self.n = len(C)
        self.C = C
        self.temp = temperature
        self.score = score
        self.d = d
        self.stay_prob = 0.01
        self._moves = [self.R_basic_move, self.R_swap_any, self.DAG_edgerev]

        for move in self._moves:
            stats["mcmc"][self.temp][move.__name__]["proposed"] = 0
            stats["mcmc"][self.temp][move.__name__]["accepted"] = 0
            stats["mcmc"][self.temp][move.__name__]["accept_ratio"] = 0

        if self.temp != 1:
            move_weights = move_weights[:-1]
        # Each move is repeated weights[move] times to allow uniform sampling
        # from the list (np.random.choice can be very slow).
        self._moves = [m for m, w in zip(self._moves, move_weights) for _ in range(w)]


        self.R = self._random_partition()
        self.R_node_scores = self._pi(self.R)
        self.R_score = self.temp * sum(self.R_node_scores)

    def R_basic_move(self, **kwargs):
        # NOTE: Is there value in having these as methods?
        return R_basic_move(**kwargs)

    def R_swap_any(self, **kwargs):
        return R_swap_any(**kwargs)

    def DAG_edgerev(self, **kwargs):
        return DAG_edgerev(**kwargs)

    def _valid(self, R):
        if sum(len(R[i]) for i in range(len(R))) != self.n:
            return False
        if len(R) == 1:
            return True
        for i in range(1, len(R)):
            for v in R[i]:
                if len(R[i-1].intersection(self.C[v])) == 0:
                    return False
        return True

    def _random_partition(self):

        def rp_d_gt0(n):
            R = list()
            U = list(range(n))
            while sum(R) < n:
                n_nodes = 1
                while np.random.random() < (n/2-1)/(n-1) and sum(R) + n_nodes < n:
                    n_nodes += 1
                R.append(n_nodes)
            for i in range(len(R)):
                # node labels need to be kept as Python ints
                # for all the bitmap operations to work as expected
                R_i = set(int(v) for v in np.random.choice(U, R[i], replace=False))
                R[i] = R_i
                U = [u for u in U if u not in R_i]
            return tuple(R)

        if self.d > 0:
            return rp_d_gt0(self.n)

        def n(R):
            n_nodes = 1
            while np.random.random() < (self.n/2-1)/(self.n-1) and sum(len(R[i]) for i in range(len(R))) + n_nodes < self.n:
                    n_nodes += 1
            return n_nodes

        while(True):
            U = set(range(self.n))
            R = [set(np.random.choice(list(U), n([]), replace=False))]
            U = U.difference(R[0])
            while len(U) > 0:
                pool = list(U.intersection(set().union({v for v in self.C if set(self.C[v]).intersection(R[-1])})))
                if len(pool) == 0:
                    break

                R_i = np.random.choice(pool, min(n(R), len(pool)), replace=False)
                R.append(set(R_i))
                U = U.difference(R[-1])
            if self.d > 0:
                return tuple(R)
            if self._valid(R):
                return tuple(R)

    def _pi(self, R, R_node_scores=None, rescore=None):

        inpart = [0] * sum(len(R[i]) for i in range(len(R)))
        for i in range(len(R)):
            for v in R[i]:
                inpart[v] = i

        if R_node_scores is None:
            R_node_scores = [0] * len(inpart)
        else:
            # Don't copy whole list, just the nodes to rescore?
            R_node_scores = list(R_node_scores)

        if rescore is None:
            rescore = set().union(*R)

        for v in rescore:
            if inpart[v] == 0:
                R_node_scores[v] = self.score.sum(v, set(), set())

            else:
                R_node_scores[v] = self.score.sum(v,
                                                  set().union(*R[:inpart[v]]),
                                                  R[inpart[v]-1])

        # if -float("inf") in R_node_scores:
        #     print("Something is wrong")
        #     u = R_node_scores.index(-float("inf"))
        #     print("R {}".format(R))
        #     print("R_node_scores {}".format(R_node_scores))
        #     print("u = {},\tC[{}] = {}".format(u, u, self.C[u]))
        #     print("u = {},\tU = {},\tT = {}".format(u, set().union(*R[:inpart[u]]), R[inpart[u]-1]))
        #     print("")
        #     self.score.sum(u, set().union(*R[:inpart[u]]), R[inpart[u]-1], debug=True)
        #     exit()

        return R_node_scores


    def _rescore(self, R, R_prime):
        rescore = list()
        UT = dict()
        U = set()
        T = set()
        for i in range(len(R)):
            for u in R[i]:
                UT[u] = (U, T)
            U = U.union(R[i])
            T = R[i]
        U = set()
        T = set()
        for i in range(len(R_prime)):
            for u in R_prime[i]:
                if UT[u] != (U, T):
                    rescore.append(u)
            U = U.union(R_prime[i])
            T = R_prime[i]
        return rescore
    

    def sample(self):

        if np.random.rand() > self.stay_prob:
            move = self._moves[np.random.randint(len(self._moves))]
            stats["mcmc"][self.temp][move.__name__]["proposed"] += 1
            if move.__name__ == 'DAG_edgerev':
                DAG, _ = self.score.sample_DAG(self.R)
                # NOTE: DAG equals DAG_prime after this, since no copy
                #       is made. If necessary, make one.
                return_value = move(DAG=DAG, score=self.score, R=self.R, C=self.C, d=self.d)
                if return_value is False:
                    return self.R, self.R_score
                DAG_prime, ap, edge = return_value
                R_prime = partition(DAG_prime)

                R_prime_node_scores = self._pi(R_prime,
                                               R_node_scores=self.R_node_scores,
                                               rescore=self._rescore(self.R, R_prime))

            elif move.__name__[0] == 'R':
                return_value = move(R=self.R)
                if return_value is False:
                    return self.R, self.R_score
                R_prime, q, q_rev, rescore = return_value
                R_prime_node_scores = self._pi(R_prime, R_node_scores=self.R_node_scores, rescore=rescore)
                ap = np.exp(self.temp * sum(R_prime_node_scores) - self.R_score)*q_rev/q

            R_prime_valid = self._valid(R_prime)

            if self.d == 0 and not R_prime_valid:
                return self.R, self.R_score

            # make this happen in log space?
            # if -np.random.exponential() < self.temp * sum(R_prime_node_scores) - self.R_score + np.log(q_rev) - np.log(q):
            if np.random.rand() < ap:
                stats["mcmc"][self.temp][move.__name__]["accepted"] += 1
                a = stats["mcmc"][self.temp][move.__name__]["accepted"]
                p = stats["mcmc"][self.temp][move.__name__]["proposed"]
                stats["mcmc"][self.temp][move.__name__]["accept_ratio"] = a/p
                self.R = R_prime
                self.R_node_scores = R_prime_node_scores
                self.R_score = self.temp * sum(self.R_node_scores)

        return self.R, self.R_score


class MC3:

    def __init__(self, chains):

        stats["mc3"]["proposed"] = np.zeros(len(chains)-1)
        stats["mc3"]["accepted"] = np.zeros(len(chains)-1)
        stats["mc3"]["accept_ratio"] = np.array([np.nan]*(len(chains)-1))
        self.chains = chains

    def sample(self):
        for c in self.chains:
            c.sample()
        i = np.random.randint(len(self.chains) - 1)
        stats["mc3"]["proposed"][i] += 1
        ap = sum(self.chains[i+1].R_node_scores)*self.chains[i].temp
        ap += sum(self.chains[i].R_node_scores)*self.chains[i+1].temp
        ap -= sum(self.chains[i].R_node_scores)*self.chains[i].temp
        ap -= sum(self.chains[i+1].R_node_scores)*self.chains[i+1].temp
        if -np.random.exponential() < ap:
            stats["mc3"]["accepted"][i] += 1
            R_tmp = self.chains[i].R
            R_node_scores_tmp = self.chains[i].R_node_scores
            self.chains[i].R = self.chains[i+1].R
            self.chains[i].R_node_scores = self.chains[i+1].R_node_scores
            self.chains[i].R_score = self.chains[i].temp * sum(self.chains[i].R_node_scores)
            self.chains[i+1].R = R_tmp
            self.chains[i+1].R_node_scores = R_node_scores_tmp
            self.chains[i+1].R_score = self.chains[i+1].temp * sum(self.chains[i+1].R_node_scores)
        stats["mc3"]["accept_ratio"][i] = stats["mc3"]["accepted"][i] / stats["mc3"]["proposed"][i]
        return self.chains[-1].R, self.chains[-1].R_score

