from collections import defaultdict
import numpy as np

from ..utils import subsets, bm, bm_to_ints, log_minus_exp, comb, close
from ..exact import zeta_transform
from ..scoring import DiscreteData, ContinuousData, BDeu, BGe
from . import gadget
from .MCMC_moves import R_basic_move, R_swap_any, B_relocate_one, B_relocate_many, B_swap_adjacent, B_swap_nonadjacent


def msb(n):
    blen = 0
    while (n > 0):
        n >>= 1
        blen += 1
    return blen


def bm_to_pyint_chunks(bitmap, minwidth=1):
    chunk = [0]*max(minwidth, (msb(bitmap)-1)//64+1)
    if len(chunk) == 1:
        return bitmap
    mask = (1 << 64) - 1
    for j in range(len(chunk)):
        chunk[j] = (bitmap & mask) >> 64*j
        mask *= 2**64
    return chunk


def bm_to_np64(bitmap):
    chunk = np.zeros(max(1, (msb(bitmap)-1)//64+1), dtype=np.uint64)
    mask = (1 << 64) - 1
    for j in range(len(chunk)):
        chunk[j] = (bitmap & mask) >> 64*j
        mask *= 2**64
    return chunk


def bms_to_np64(bitmaps, minwidth=1):
    blen = np.array([msb(x) for x in bitmaps])
    dim1 = len(bitmaps)
    dim2 = max(minwidth, max((blen - 1) // 64) + 1)
    if dim2 == 1:
        return np.array(bitmaps, dtype=np.uint64)
    chunks = np.zeros(shape=(dim1, dim2), dtype=np.uint64)
    for i in range(dim1):
        n_c = (blen[i] - 1)//64
        mask = (1 << 64) - 1
        for j in range(n_c + 1):
            chunks[i][j] = (bitmaps[i] & mask) >> 64*j
            mask *= 2**64
    return chunks


def np64_to_bm(chunk):
    if type(chunk) in {np.uint64, int}:
        return int(chunk)
    bm = 0
    for part in chunk[::-1]:
        bm |= int(part)
        bm *= 2**64
    bm >>= 64
    return bm


def fbit(mask):
    """get index of first set bit"""
    k = 0
    while 1 & mask == 0:
        k += 1
        mask >>= 1
    return k


def kzon(mask, k):
    """set kth zerobit on"""
    nmask = ~mask
    for i in range(k):
        nmask &= ~(nmask & -nmask)
    return mask | (nmask & -nmask)


def dkbit(mask, k):
    """drop kth bith"""
    if mask == 0:
        return mask
    trunc = mask >> (k+1)
    #trunc <<= k
    trunc *= 2**k
    # return ((1 << k) - 1) & mask | trunc
    return (2**k - 1) & mask | trunc


def ikbit(mask, k, bit):
    """shift all bits >=k to the left and insert bit to k"""
    if k == 0:
        # newmask = mask << 1
        newmask = mask * 2
    else:
        newmask = mask >> k-1
    newmask ^= (-bit ^ newmask) & 1
    #newmask <<= k
    newmask *= 2**k
    #return newmask | ((1 << k) - 1) & mask
    return newmask | (2**k - 1) & mask


def subsets_size_k(k, n):
    if k == 0:
        yield 0
        return
    #S = (1 << k) - 1
    S = 2**k - 1
    # limit = (1 << n)
    limit = 2**n
    while S < limit:
        yield S
        x = S & -S
        r = S + x
        S = (((r ^ S) >> 2) // x) | r


def ssets(mask):
    S = mask
    while S > 0:
        yield S
        S = (S - 1) & mask


class DAGR:

    def __init__(self, scores, C, complementary_scores, tolerance=2**(-32), stats=None):
        if stats is not None:
            self.stats = stats
            self.stats[type(self).__name__] = dict()
            self.stats[type(self).__name__]["CC"] = 0

        self.scores = scores
        self.C = C
        self.cscores = complementary_scores
        self.tol = tolerance

    def precompute(self, v):

        K = len(self.C[v])

        self._f = [0]*2**K
        for X in range(2**K):
            self._f[X] = [-float("inf")]*2**(K-bin(X).count("1"))
            self._f[X][0] = self.scores.scores[v][X]

        for k in range(1, K+1):
            for k_x in range(K-k+1):
                for X in subsets_size_k(k_x, K):
                    for Y in subsets_size_k(k, K-k_x):
                        i = fbit(Y)
                        self._f[X][Y] = np.logaddexp(self._f[kzon(X, i)][dkbit(Y, i)], self._f[X][Y & ~(Y & -Y)])

    def sample_pset(self, v, R, score=False):

        # TODO: keep track of positions in move functions
        for i in range(len(R)):
            if v in R[i]:
                break

        if i == 0:
            family = (v,)
            family_score = self.scores.scores[v][0]

        else:

            U = set().union(*R[:i])
            T = R[i-1]

            w_C = -float("inf")
            if len(T.intersection(self.C[v])) > 0:
                w_C = self.scores.scoresum(v, U, T)

            w_compl_sum, contribs = self.cscores.scoresum(v, U, T, -float("inf"), contribs=True)

            if -np.random.exponential() < w_C - np.logaddexp(w_compl_sum, w_C):
                family = (v, self._sample_pset(v, set().union(*R[:i]), R[i-1]))
                family_score = self.scores.scores[v][bm(family[1], ix=self.C[v])]
            else:
                pset, family_score = self.cscores.sample_pset(v, contribs, w_compl_sum)
                family = (v, bm_to_ints(pset))

        if score is True:
            return family, family_score
        return family

    def _sample_pset(self, v, U, T):

        def g(X, E, U, T):

            X_bm = bm(X, ix=self.C[v])
            E_bm = bm(E, ix=sorted(set(self.C[v]).difference(X)))
            U_bm = bm(U.difference(X), ix=sorted(set(self.C[v]).difference(X)))
            T_bm = bm(T.difference(X), ix=sorted(set(self.C[v]).difference(X)))

            score_1 = [self._f[X_bm][U_bm & ~E_bm] if X.issubset(U.difference(E)) else -float("inf")][0]
            score_2 = [self._f[X_bm][(U_bm & ~E_bm) & ~T_bm] if X.issubset(U.difference(E.union(T))) else -float("inf")][0]

            if not close(score_1, score_2, self.tol):
                return log_minus_exp(score_1, score_2)
            else:  # CC
                return None

        U = U.intersection(self.C[v])
        T = T.intersection(self.C[v])

        X = set()
        E = set()
        for i in U:
            try:
                if -np.random.exponential() < g(X.union({i}), E, U, T) - g(X, E, U, T):
                    X.add(i)
                else:
                    E.add(i)
            except TypeError:
                if self.stats is not None:
                    self.stats[type(self).__name__]["CC"] += 1

                return self._sample_pset_brute(v, U, T)
        return X

    def _sample_pset_brute(self, v, U, T):

        U = U.intersection(self.C[v])
        T = T.intersection(self.C[v])

        probs = list()
        psets = list()
        for T_set in subsets(T, 1, len(T)):
            for U_set in subsets(U.difference(T), 0, len(U.difference(T))):
                pset = set(T_set).union(U_set)
                probs.append(self.scores.scores[v][bm(pset, ix=self.C[v])])
                psets.append(pset)
        probs = np.array(probs)
        probs -= np.logaddexp.reduce(probs)
        probs = np.exp(probs)
        return psets[np.random.choice(range(len(psets)), p=probs)]


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
        self.moves = self.B_moves + self.R_moves
        self.move_probs = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

    def R_basic_move(**kwargs):
        return R_basic_move(**kwargs)

    def R_swap_any(**kwargs):
        return R_swap_any(**kwargs)

    def B_relocate_one(**kwargs):
        return B_relocate_one(**kwargs)

    def B_relocate_many(**kwargs):
        return B_relocate_many(**kwargs)

    def B_swap_adjacent(**kwargs):
        return B_swap_adjacent(**kwargs)

    def B_swap_nonadjacent(**kwargs):
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

                # t_psum = time.process_time()
                tau_hat_prime = self.parentsums(B_prime, self.M, self.max_indegree, self.scores)
                # t_psum = time.process_time() - t_psum

                R_prob = self.posterior_R(self.R, self.scores, self.max_indegree)
                R_prime_prob = self.posterior_R(R_prime, self.scores, self.max_indegree)

                acc_prob = np.exp(R_prime_prob - R_prob)*q_rev/q

            if np.random.rand() < acc_prob:

                # t_pos = time.process_time()
                g_prime = self.posterior(B_prime, self.M, self.max_indegree, self.scores, tau_hat_prime, return_all=True)
                B_prime_prob = g_prime[0][frozenset()][frozenset()]
                # t_pos = time.process_time() - t_pos

                self.B = B_prime
                self.tau_hat = tau_hat_prime
                self.g = g_prime
                self.B_prob = self.g[0][frozenset()][frozenset()]
                self.R = self.generate_partition(self.B, self.M, self.tau_hat, self.g, self.scores)
                DAG, DAG_prob = self.sample_DAG(self.R, self.scores, self.max_indegree)
                # update_stats(B_prime_prob, DAG_prob, B_prime, DAG, acc_prob, 1, move.__name__, t_psum, t_pos)
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

    def __init__(self, C, sr, cscores, temperature=1, stats=None):

        if stats is not None:
            self.stats = stats
            self.key = type(self).__name__
            if self.key not in stats:
                self.stats[self.key] = dict()
            self.stats[self.key][temperature] = dict()
            self.stats[self.key][temperature][self.R_basic_move.__name__] = dict()
            self.stats[self.key][temperature][self.R_basic_move.__name__]["candidate-valid"] = dict()
            self.stats[self.key][temperature][self.R_basic_move.__name__]["candidate-valid"]["n"] = 0
            self.stats[self.key][temperature][self.R_basic_move.__name__]["candidate-valid"]["ratio"] = 0
            self.stats[self.key][temperature][self.R_basic_move.__name__]["candidate-valid"]["accepted"] = 0
            self.stats[self.key][temperature][self.R_basic_move.__name__]["candidate-invalid"] = dict()
            self.stats[self.key][temperature][self.R_basic_move.__name__]["candidate-invalid"]["n"] = 0
            self.stats[self.key][temperature][self.R_basic_move.__name__]["candidate-invalid"]["ratio"] = 0
            self.stats[self.key][temperature][self.R_basic_move.__name__]["candidate-invalid"]["accepted"] = 0
            self.stats[self.key][temperature][self.R_swap_any.__name__] = dict()
            self.stats[self.key][temperature][self.R_swap_any.__name__]["candidate-valid"] = dict()
            self.stats[self.key][temperature][self.R_swap_any.__name__]["candidate-valid"]["n"] = 0
            self.stats[self.key][temperature][self.R_swap_any.__name__]["candidate-valid"]["ratio"] = 0
            self.stats[self.key][temperature][self.R_swap_any.__name__]["candidate-valid"]["accepted"] = 0
            self.stats[self.key][temperature][self.R_swap_any.__name__]["candidate-invalid"] = dict()
            self.stats[self.key][temperature][self.R_swap_any.__name__]["candidate-invalid"]["n"] = 0
            self.stats[self.key][temperature][self.R_swap_any.__name__]["candidate-invalid"]["ratio"] = 0
            self.stats[self.key][temperature][self.R_swap_any.__name__]["candidate-invalid"]["accepted"] = 0

        self.n = len(C)
        self.C = C
        self.temp = temperature
        self.sr = sr
        self.cscores = cscores
        self.stay_prob = 0.01
        self._moves = [R_basic_move, R_swap_any]
        self._moveprobs = [0.5, 0.5]
        self.R = self._random_partition()
        self.R_node_scores = self._pi(self.R)
        self.R_score = self.temp * sum(self.R_node_scores)

    def R_basic_move(**kwargs):
        return R_basic_move(**kwargs)

    def R_swap_any(**kwargs):
        return R_swap_any(**kwargs)

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

        if self.cscores.d > 0:
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
            if self.cscores.d > 0:
                return tuple(R)
            if self._valid(R):
                return tuple(R)

    def _R_basic_move(self, **kwargs):

        def valid():
            return True

        if "validate" in kwargs and kwargs["validate"] is True:
            return valid()

        m = len(self.R)
        sum_binoms = [sum([comb(len(self.R[i]), v) for v in range(1, len(self.R[i]))]) for i in range(m)]
        nbd = m - 1 + sum(sum_binoms)
        q = 1/nbd

        j = np.random.randint(1, nbd+1)

        R_prime = list()
        if j < m:
            R_prime = [self.R[i] for i in range(j-1)] + [self.R[j-1].union(self.R[j])] + [self.R[i] for i in range(min(m, j+1), m)]
            return R_prime, q, q, self.R[j].union(self.R[min(m-1, j+1)])

        sum_binoms = [sum(sum_binoms[:i]) for i in range(1, len(sum_binoms)+1)]
        i_star = [m-1 + sum_binoms[i] for i in range(len(sum_binoms)) if m-1 + sum_binoms[i] < j]
        i_star = len(i_star)

        c_star = [comb(len(self.R[i_star]), v) for v in range(1, len(self.R[i_star])+1)]
        c_star = [sum(c_star[:i]) for i in range(1, len(c_star)+1)]

        c_star = [m-1 + sum_binoms[i_star-1] + c_star[i] for i in range(len(c_star))
                  if m-1 + sum_binoms[i_star-1] + c_star[i] < j]
        c_star = len(c_star)+1

        nodes = {int(v) for v in np.random.choice(list(self.R[i_star]), c_star)}

        R_prime = [self.R[i] for i in range(i_star)] + [nodes]
        R_prime += [self.R[i_star].difference(nodes)] + [self.R[i] for i in range(min(m, i_star+1), m)]

        return tuple(R_prime), q, q, self.R[i_star].difference(nodes).union(self.R[min(m-1, i_star+1)])

    def _R_swap_any(self, **kwargs):

        def valid():
            return len(self.R) > 1

        m = len(self.R)

        if "validate" in kwargs and kwargs["validate"] is True:
            return valid()

        if m == 2:
            j = 0
            k = 1
            q = 1/(len(self.R[j])*len(self.R[k]))
        else:
            if np.random.random() <= 0.9:  # adjacent
                j = np.random.randint(len(self.R)-1)
                k = j+1
                q = 0.9 * 1/((m-1) * len(self.R[j]) * len(self.R[k]))
            else:
                if m == 3:
                    j = 0
                    k = 2
                    q = 1/(len(self.R[j])*len(self.R[k]))
                else:
                    j = np.random.randint(m)
                    n = list(set(range(m)).difference({j-1, j, j+1}))
                    k = np.random.choice(n)
                    q = 0.1 * 1/((m*len(n)) * len(self.R[j]) * len(self.R[k]))

        v_j = int(np.random.choice(list(self.R[j])))
        v_k = int(np.random.choice(list(self.R[k])))
        R_prime = list()
        for i in range(m):
            if i == j:
                R_prime.append(self.R[i].difference({v_j}).union({v_k}))
            elif i == k:
                R_prime.append(self.R[i].difference({v_k}).union({v_j}))
            else:
                R_prime.append(self.R[i])

        return tuple(R_prime), q, q, {v_j, v_k}.union(*self.R[min(j, k)+1:min(max(j, k)+2, m+1)])

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
                R_node_scores[v] = self.sr.scoresum(v, set(), set())

            else:

                R_node_scores[v] = self.sr.scoresum(v,
                                                    set().union(*R[:inpart[v]]),
                                                    R[inpart[v]-1])
        if -float("inf") in R_node_scores:
            print("Something is wrong")
            u = R_node_scores.index(-float("inf"))
            print("R {}".format(R))
            print("R_node_scores {}".format(R_node_scores))
            print("u = {},\tC[{}] = {}".format(u, u, self.C[u]))
            print("u = {},\tU = {},\tT = {}".format(u, set().union(*R[:inpart[u]]), R[inpart[u]-1]))
            print("")
            self.sr.scoresum(u, set().union(*R[:inpart[u]]), R[inpart[u]-1], debug=True)
            exit()
        return R_node_scores

    def sample(self):

        if np.random.rand() > self.stay_prob:
            move = np.random.choice(self._moves, p=self._moveprobs)
            if not move(R=self.R, validate=True):
                return self.R, self.R_score

            R_prime, q, q_rev, rescore = move(R=self.R)

            R_prime_valid = self._valid(R_prime)
            if self.stats:
                if R_prime_valid:
                    self.stats[self.key][self.temp][move.__name__]["candidate-valid"]["n"] += 1
                else:
                    self.stats[self.key][self.temp][move.__name__]["candidate-invalid"]["n"] += 1

            if self.cscores.d == 0 and not R_prime_valid:
                return self.R, self.R_score

            R_prime_node_scores = self._pi(R_prime, R_node_scores=self.R_node_scores, rescore=rescore)

            # make this happen in log space?
            # if -np.random.exponential() < self.temp * sum(R_prime_node_scores) - self.R_score + np.log(q_rev) - np.log(q):
            if np.random.rand() < np.exp(self.temp * sum(R_prime_node_scores) - self.R_score)*q_rev/q:
                if self.stats:
                    if R_prime_valid:
                        self.stats[self.key][self.temp][move.__name__]["candidate-valid"]["accepted"] += 1
                    else:
                        self.stats[self.key][self.temp][move.__name__]["candidate-invalid"]["accepted"] += 1
                self.R = R_prime
                self.R_node_scores = R_prime_node_scores
                self.R_score = self.temp * sum(self.R_node_scores)

        return self.R, self.R_score


class MC3:

    def __init__(self, chains, stats=None):
        if stats is not None:
            self.stats = stats
            self.stats[type(self).__name__] = dict()
            self.stats[type(self).__name__]["n chains"] = len(chains)
            self.stats[type(self).__name__]["temperatures"] = [round(c.temp, 3) for c in chains]
            self.stats[type(self).__name__]["swaps proposed"] = [0]*len(chains)
            self.stats[type(self).__name__]["swaps accepted"] = [0]*len(chains)
        self.chains = chains

    def sample(self):
        for c in self.chains:
            c.sample()
        i = np.random.randint(len(self.chains) - 1)
        if self.stats:
            self.stats[type(self).__name__]["swaps proposed"][i] += 1
        ap = sum(self.chains[i+1].R_node_scores)*self.chains[i].temp
        ap += sum(self.chains[i].R_node_scores)*self.chains[i+1].temp
        ap -= sum(self.chains[i].R_node_scores)*self.chains[i].temp
        ap -= sum(self.chains[i+1].R_node_scores)*self.chains[i+1].temp
        if -np.random.exponential() < ap:
            if self.stats:
                self.stats[type(self).__name__]["swaps accepted"][i] += 1
            R_tmp = self.chains[i].R
            R_node_scores_tmp = self.chains[i].R_node_scores
            self.chains[i].R = self.chains[i+1].R
            self.chains[i].R_node_scores = self.chains[i+1].R_node_scores
            self.chains[i].R_score = self.chains[i].temp * sum(self.chains[i].R_node_scores)
            self.chains[i+1].R = R_tmp
            self.chains[i+1].R_node_scores = R_node_scores_tmp
            self.chains[i+1].R_score = self.chains[i+1].temp * sum(self.chains[i+1].R_node_scores)
        return self.chains[-1].R, self.chains[-1].R_score


class Score:

    def __init__(self, datapath, scoref="bdeu", maxid=-1, ess=10, stats=None):

        self.maxid = maxid
        self.stats = None
        if stats is not None:
            self.stats = stats
            self.stats[type(self).__name__] = dict()
            self.stats[type(self).__name__]["clear_cache"] = 0

        if scoref == "bdeu":

            def local(node, parents):
                if len(self.score._cache) > 1000000:
                    self.score.clear_cache()
                    if self.stats:
                        self.stats[type(self).__name__]["clear_cache"] += 1

                if self.maxid == -1 or len(parents) <= self.maxid:
                    score = self.score.bdeu_score(node, parents)[0]
                else:
                    return -float("inf")
                # consider putting the prior explicitly somewhere
                return score - np.log(comb(self.n - 1, len(parents)))

            d = DiscreteData(datapath)
            d._varidx = {v: v for v in d._varidx.values()}
            self.n = len(d._variables)
            self.score = BDeu(d, alpha=ess)
            self.local = local

        elif scoref == "bge":

            def local(node, parents):
                if len(self.score._cache) > 1000000:
                    self.score._cache = dict()
                    if self.stats:
                        self.stats[type(self).__name__]["clear_cache"] += 1
                if self.maxid == -1 or len(parents) <= self.maxid:
                    return self.score.bge_score(node, parents)[0] - np.log(comb(self.n - 1, len(parents)))
                else:
                    return -float("inf")

            d = ContinuousData(datapath, header=False)
            d._varidx = {v: v for v in d._varidx.values()}
            self.n = len(d._variables)
            self.score = BGe(d)
            self.local = local

    def complementary_scores_dict(self, C, d):
        """C candidates, d indegree for complement psets"""
        cscores = dict()
        for v in C:
            cscores[v] = dict()
            for pset in subsets([u for u in C if u != v], 1, d):
                if not (set(pset)).issubset(C[v]):
                    cscores[v][pset] = self.local(v, pset)
        return cscores

    def all_scores_dict(self, C=None):
        scores = dict()
        if C is None:
            C = {v: tuple(sorted(set(range(self.n)).difference({v}))) for v in range(self.n)}
        for v in C:
            tmp = dict()
            for pset in subsets(C[v], 0, [len(C[v]) if self.maxid == -1 else self.maxid][0]):
                tmp[pset] = self.local(v, pset)
            scores[v] = tmp
        return scores

    def all_scores_list(self, C):
        scores = list()
        for v in C:
            tmp = [-float('inf')]*2**len(C[0])
            for pset in subsets(C[v], 0, [len(C[v]) if self.maxid == -1 else self.maxid][0]):
                tmp[bm(pset, ix=C[v])] = self.local(v, pset)
            scores.append(tmp)
        return scores


class ScoreR:

    def __init__(self, scores, C, tolerance=2**(-32), D=2, cscores=None, stats=None):

        if stats is not None:
            self.stats = stats
            self.stats[type(self).__name__] = dict()
            self.stats[type(self).__name__]["CC"] = 0
            self.stats[type(self).__name__]["CC basecases"] = 0
            self.stats[type(self).__name__]["basecases"] = 0

        self.scores = scores
        self.C = C
        self.tol = tolerance
        self.cscores = cscores
        self._precompute_a()
        self._precompute_basecases()
        self._precompute_psum()

    def _precompute_a(self):
        self._a = [0]*len(self.scores)
        for v in range(len(self.scores)):
            self._a[v] = zeta_transform.solve(self.scores[v])

    def _precompute_basecases(self):
        K = len(self.C[0])
        self._psum = {v: dict() for v in range(len(self.C))}
        for v in self.C:
            for k in range(K):
                x = 1 << k
                U_minus_x = (2**K - 1) & ~x
                tmp = [0]*2**(K-1)
                tmp[0] = self.scores[v][x]
                self._psum[v][x] = dict()
                for S in ssets(U_minus_x):
                    if S | x not in self._psum[v]:
                        self._psum[v][S | x] = dict()
                    tmp[dkbit(S, k)] = self.scores[v][S | x]
                tmp = zeta_transform.solve(tmp)
                if self.stats:
                    self.stats[type(self).__name__]["basecases"] += len(tmp)
                for S in range(len(tmp)):
                    # only save basecase if it can't be computed as difference
                    # makes a bit slower, makes require a bit less space
                    if self._cc(v, ikbit(S, k, 1), x):
                        if self.stats:
                            self.stats[type(self).__name__]["CC basecases"] += 1
                        self._psum[v][ikbit(S, k, 1)][x] = tmp[S]

    def _precompute_psum(self):

        K = len(self.C[0])
        n = len(self.C)

        for v in self.C:
            for U in range(1, 2**K):
                for T in ssets(U):
                    if self._cc(v, U, T):
                        if self.stats:
                            self.stats[type(self).__name__]["CC"] += 1
                        T1 = T & -T
                        U1 = U & ~T1
                        T2 = T & ~T1

                        self._psum[v][U][T] = np.logaddexp(self.psum(v, U, T1),
                                                           self.psum(v, U1, T2))
        if self.stats:
            self.stats[type(self).__name__]["relative CC"] = self.stats[type(self).__name__]["CC"] / (n*3**K)

    def _cc(self, v, U, T):
        return close(self._a[v][U], self._a[v][U & ~T], self.tol)
        # return self._a[v][U] == self._a[v][U & ~T]

    def _scoresum(self, v, U, T):
        return self.psum(v, bm(U, ix=self.C[v]), bm(T, ix=self.C[v]))

    def scoresum(self, v, U, T, debug=False):

        if len(T) == 0:
            return self.scores[v][0]

        if len(T.intersection(self.C[v])) > 0:
            W_prime = self.psum(v, bm(U.intersection(self.C[v]), ix=self.C[v]), bm(T.intersection(self.C[v]), ix=self.C[v]))
        else:
            W_prime = -float("inf")

        #print("Y", W_prime)
        # This does not have to check whether CScoreR.d == 0
        # because if it is 0 the proposal is already rejected in
        # PartitionMCMC.sample if it is invalid
        return self.cscores.scoresum(v, U, T, W_prime, debug=debug)[0]

    def psum(self, v, U, T):
        if U == 0 and T == 0:
            return self.scores[v][0]
        if T == 0:  # special case for T2 in precompute
            return -float("inf")
        if v in self._psum and U in self._psum[v] and T in self._psum[v][U]:
            return self._psum[v][U][T]
        else:
            return log_minus_exp(self._a[v][U], self._a[v][U & ~T])


class CScoreR:

    """Complementary scores

    Scores complementary to those constrained
    by the candidate parent sets"""

    def __init__(self, C, scores, d):

        self.C = C
        self.n = len(C)
        self.d = d
        minwidth = 1
        if self.n > 64:
            minwidth = 2

        scores = scores.complementary_scores_dict(C, d)

        ordered_psets = dict()
        ordered_scores = dict()

        for v in scores:
            ordered_scores[v] = sorted(scores[v].items(), key=lambda item: item[1], reverse=True)
            ordered_psets[v] = bms_to_np64([bm(item[0]) for item in ordered_scores[v]], minwidth=minwidth)
            ordered_scores[v] = np.array([item[1] for item in ordered_scores[v]], dtype=np.float64)

        self.ordered_psets = ordered_psets
        self.ordered_scores = ordered_scores

        if self.d == 1:
            self.pset_to_idx = dict()
            for v in scores:
                # wrong if over 64 variables?
                # ordered_psets[v] = ordered_psets[v].flatten()
                ordered_psets[v] = [np64_to_bm(pset) for pset in ordered_psets[v]]
                self.pset_to_idx[v] = dict()
                for i, pset in enumerate(ordered_psets[v]):
                    self.pset_to_idx[v][pset] = i

        self.t_ub = np.zeros(shape=(len(C), len(C)), dtype=np.int32)
        for u in range(1, len(C)+1):
            for t in range(1, u+1):
                self.t_ub[u-1][t-1] = self.n_valids_ub(u, t)

    def sample_pset(self, v, pset_indices, w_sum):
        i = np.random.choice(pset_indices, p=np.exp(self._scores(v, pset_indices)-w_sum))
        return np64_to_bm(self.ordered_psets[v][i]), self.ordered_scores[v][i]

    def _scores(self, v, indices):
        return np.array([self.ordered_scores[v][i] for i in indices])
        #return np.array([self.ordered_scores[v][i][1] for i in indices])

    def _valids(self, v, U, T):
        """This is used just for debugging I think, delete when unnecessary"""
        return [i for i, pset_score in enumerate(self.ordered_scores[v])
                if set(pset_score[0]).issubset(U)
                and len(set(pset_score[0]).intersection(T)) > 0]

    def n_valids(self, v, U, T):
        n = 0
        for k in range(1, self.d+1):
            n += comb(len(U), k) - comb(len(U.intersection(self.C[v])), k)
            n -= comb(len(U.difference(T)), k) - comb(len(U.difference(T).intersection(self.C[v])), k)
        return n

    def n_valids_ub(self, u, t):
        n = 0
        for k in range(self.d+1):
            n += comb(u, k) - comb(u - t, k)
        return n

    def scoresum(self, v, U, T, W_prime, debug=False, contribs=False):

        if self.d == 1:  # special case
            contribs = list()
            w_contribs = list()
            for u in T:
                if u not in self.C[v]:
                    pset_idx = self.pset_to_idx[v][bm(u)]
                    contribs.append(pset_idx)
                    w_contribs.append(self.ordered_scores[v][pset_idx])
            w_contribs.append(W_prime)
            return np.logaddexp.reduce(w_contribs), contribs

        if self.n <= 64:
            U_bm = bm(U)
            T_bm = bm(T)
        else:  # if 64 < n <= 128
            U_bm = bm_to_pyint_chunks(bm(U), 2)
            T_bm = bm_to_pyint_chunks(bm(T), 2)

        # contribs should be a param to weight_sum()
        if contribs is True:
            return gadget.weight_sum_contribs(W_prime,
                                              self.ordered_psets[v],
                                              self.ordered_scores[v],
                                              self.n,
                                              U_bm,
                                              T_bm,
                                              int(self.t_ub[len(U)][len(T)]))

        W_sum = gadget.weight_sum(W_prime,
                                  self.ordered_psets[v],
                                  self.ordered_scores[v],
                                  self.n,
                                  U_bm,
                                  T_bm,
                                  int(self.t_ub[len(U)][len(T)]))

        """
        if contribs is True:
            if self.n <= 64:
                return gadget.weight_sum_contribs_64(W_prime,
                                                     self.ordered_psets[v],
                                                     self.ordered_scores[v],
                                                     bm(U),
                                                     bm(T),
                                                     int(self.t_ub[len(U)][len(T)]))
            else:
                U_bm = bm_to_pyint_chunks(bm(U), 2)
                T_bm = bm_to_pyint_chunks(bm(T), 2)
                return gadget.weight_sum_contribs_128(W_prime,
                                                      self.ordered_psets[v],
                                                      self.ordered_scores[v],
                                                      U_bm[0],
                                                      U_bm[1],
                                                      T_bm[0],
                                                      T_bm[1],
                                                      int(self.t_ub[len(U)][len(T)]))
        if self.n <= 64:
            W_sum = gadget.weight_sum_64(W_prime,
                                         self.ordered_psets[v],
                                         self.ordered_scores[v],
                                         bm(U),
                                         bm(T),
                                         int(self.t_ub[len(U)][len(T)]))
        else:
            U_bm = bm_to_pyint_chunks(bm(U), 2)
            T_bm = bm_to_pyint_chunks(bm(T), 2)
            W_sum = gadget.weight_sum_128(W_prime,
                                          self.ordered_psets[v],
                                          self.ordered_scores[v],
                                          U_bm[0],
                                          U_bm[1],
                                          T_bm[0],
                                          T_bm[1],
                                          int(self.t_ub[len(U)][len(T)]))
        """

        # print("XXXXXXXXXXXXXXXXXXXXX", W_sum)
        if W_sum == -float("inf"):
            print("INFFI")
            print("U types {}".format([type(x) for x in U]))
            print("T types {}".format([type(x) for x in T]))
            print("W_prime {}".format(W_prime))
            #np.set_printoptions(threshold=np.inf)
            #print(self.ordered_psets[v])
            np.save("psets.npy", self.ordered_psets[v])
            np.save("weights.npy", self.ordered_scores[v])
            #print(self.ordered_scores[v])
            print("U {}".format(bm(U)))
            print("T {}".format(bm(T)))
            print("t_ub {}".format(int(self.t_ub[len(U)][len(T)])))
            exit()
        return W_sum, None
