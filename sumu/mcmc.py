import copy

import numpy as np

from .bnet import partition
from .mcmc_moves import DAG_edgerev, R_basic_move, R_swap_any


class PartitionMCMC:
    """Partition-MCMC sampler :footcite:`kuipers:2017` with efficient
    scoring."""

    def __init__(
        self,
        C,
        score,
        d,
        temperature=1.0,
        move_weights=[1, 1, 2],
        stats=None,
        R=None,
    ):

        self.n = len(C)
        self.C = C
        self.temp = temperature
        self.score = score
        self.d = d
        self.stay_prob = 0.01
        self._all_moves = [
            self.R_basic_move,
            self.R_swap_any,
            self.DAG_edgerev,
        ]
        self._move_weights = list(move_weights)
        self.stats = stats

        if self.stats is not None:
            for move in self._all_moves:
                self.stats["mcmc"][self.temp][move.__name__]["proposed"] = 0
                self.stats["mcmc"][self.temp][move.__name__]["accepted"] = 0
                self.stats["mcmc"][self.temp][move.__name__][
                    "accept_ratio"
                ] = 0

        self._init_moves()

        self.R = R
        if self.R is None:
            self.R = self._random_partition()
        self.R_node_scores = self._pi(self.R)
        self.R_score = self.temp * sum(self.R_node_scores)

    def _init_moves(self):
        # This needs to be called if self.temp changes from/to 1.0
        move_weights = self._move_weights
        if self.temp != 1:
            move_weights = self._move_weights[:-1]
        # Each move is repeated weights[move] times to allow uniform sampling
        # from the list (np.random.choice can be very slow).
        self._moves = [
            m for m, w in zip(self._all_moves, move_weights) for _ in range(w)
        ]

    def __copy__(self):
        return PartitionMCMC(
            self.C,
            self.score,
            self.d,
            temperature=self.temp,
            move_weights=self._move_weights,
            stats=self.stats,
            R=self.R,
        )

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
                if len(R[i - 1].intersection(self.C[v])) == 0:
                    return False
        return True

    def _random_partition(self):
        def rp_d_gt0(n):
            R = list()
            U = list(range(n))
            while sum(R) < n:
                n_nodes = 1
                while (
                    np.random.random() < (n / 2 - 1) / (n - 1)
                    and sum(R) + n_nodes < n
                ):
                    n_nodes += 1
                R.append(n_nodes)
            for i in range(len(R)):
                # node labels need to be kept as Python ints
                # for all the bitmap operations to work as expected
                R_i = set(
                    int(v) for v in np.random.choice(U, R[i], replace=False)
                )
                R[i] = R_i
                U = [u for u in U if u not in R_i]
            return tuple(R)

        if self.d > 0:
            return rp_d_gt0(self.n)

        def n(R):
            n_nodes = 1
            while (
                np.random.random() < (self.n / 2 - 1) / (self.n - 1)
                and sum(len(R[i]) for i in range(len(R))) + n_nodes < self.n
            ):
                n_nodes += 1
            return n_nodes

        while True:
            U = set(range(self.n))
            R = [set(np.random.choice(list(U), n([]), replace=False))]
            U = U.difference(R[0])
            while len(U) > 0:
                pool = list(
                    U.intersection(
                        set().union(
                            {
                                v
                                for v in self.C
                                if set(self.C[v]).intersection(R[-1])
                            }
                        )
                    )
                )
                if len(pool) == 0:
                    break

                R_i = np.random.choice(
                    pool, min(n(R), len(pool)), replace=False
                )
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
                R_node_scores[v] = self.score.sum(
                    v, set().union(*R[: inpart[v]]), R[inpart[v] - 1]
                )

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
        def update_stats(accepted):
            self.stats["mcmc"][self.temp][move.__name__]["proposed"] += 1
            if accepted:
                self.stats["mcmc"][self.temp][move.__name__]["accepted"] += 1
            a = self.stats["mcmc"][self.temp][move.__name__]["accepted"]
            if type(a) != int:
                a = 0
            p = self.stats["mcmc"][self.temp][move.__name__]["proposed"]
            try:
                ap = a / p
            except ZeroDivisionError:
                ap = 0.0
            self.stats["mcmc"][self.temp][move.__name__]["accept_ratio"] = ap

        if np.random.rand() > self.stay_prob:
            move = self._moves[np.random.randint(len(self._moves))]
            if move.__name__ == "DAG_edgerev":
                DAG, _ = self.score.sample_DAG(self.R)
                # NOTE: DAG equals DAG_prime after this, since no copy
                #       is made. If necessary, make one.
                return_value = move(
                    DAG=DAG, score=self.score, R=self.R, C=self.C, d=self.d
                )
                if return_value is False:
                    return self.R, self.R_score
                DAG_prime, ap, edge = return_value
                R_prime = partition(DAG_prime)

                R_prime_node_scores = self._pi(
                    R_prime,
                    R_node_scores=self.R_node_scores,
                    rescore=self._rescore(self.R, R_prime),
                )

            elif move.__name__[0] == "R":
                return_value = move(R=self.R)
                if return_value is False:
                    return self.R, self.R_score
                R_prime, q, q_rev, rescore = return_value
                R_prime_node_scores = self._pi(
                    R_prime, R_node_scores=self.R_node_scores, rescore=rescore
                )
                ap = (
                    np.exp(self.temp * sum(R_prime_node_scores) - self.R_score)
                    * q_rev
                    / q
                )

            R_prime_valid = self._valid(R_prime)

            if self.d == 0 and not R_prime_valid:
                return self.R, self.R_score

            # make this happen in log space?
            # if -np.random.exponential() < self.temp*sum(R_prime_node_scores)
            # - self.R_score + np.log(q_rev) - np.log(q):
            accepted = False
            if np.random.rand() < ap:
                accepted = True
                self.R = R_prime
                self.R_node_scores = R_prime_node_scores
                self.R_score = self.temp * sum(self.R_node_scores)

            if self.stats is not None:
                update_stats(accepted)

        return self.R, self.R_score


class MC3:
    def __init__(self, chains, stats=None):

        self.stats = stats
        if self.stats is not None:
            self.stats["mc3"]["proposed"] = np.zeros(len(chains) - 1)
            self.stats["mc3"]["accepted"] = np.zeros(len(chains) - 1)
            self.stats["mc3"]["accept_ratio"] = np.array(
                [np.nan] * (len(chains) - 1)
            )
        self.chains = chains

    @staticmethod
    def get_inv_temperatures(scheme, M):
        linear = [i / (M - 1) for i in range(M)]
        quadratic = [1 - ((M - 1 - i) / (M - 1)) ** 2 for i in range(M)]
        sigmoid = [
            1 / (1 + np.exp((M - 1) * (0.5 - (i / (M - 1))))) for i in range(M)
        ]
        sigmoid[0] = 0.0
        sigmoid[-1] = 1.0
        return locals()[scheme]

    @classmethod
    def adaptive(cls, mcmc, stats=None, target=0.25):
        mcmc0 = copy.copy(mcmc)
        mcmc0.temp = 0.0
        mcmc0._init_moves()
        chains = [mcmc0, mcmc]

        def acceptance_prob(temp):
            chains[-1].temp = temp
            chains[-1]._init_moves()
            proposed = 0
            accepted = 0
            while proposed < 1000:
                i = np.random.randint(len(chains) - 1)
                # i = -2
                if i == len(chains) - 2:
                    proposed += 1
                for c in chains[-2:]:
                    c.sample()
                ap = sum(chains[i + 1].R_node_scores) * chains[i].temp
                ap += sum(chains[i].R_node_scores) * chains[i + 1].temp
                ap -= sum(chains[i].R_node_scores) * chains[i].temp
                ap -= sum(chains[i + 1].R_node_scores) * chains[i + 1].temp
                if -np.random.exponential() < ap:
                    if i == len(chains) - 2:
                        accepted += 1
                    R_tmp = chains[i].R
                    R_node_scores_tmp = chains[i].R_node_scores
                    chains[i].R = chains[i + 1].R
                    chains[i].R_node_scores = chains[i + 1].R_node_scores
                    chains[i].R_score = chains[i].temp * sum(
                        chains[i].R_node_scores
                    )
                    chains[i + 1].R = R_tmp
                    chains[i + 1].R_node_scores = R_node_scores_tmp
                    chains[i + 1].R_score = chains[i + 1].temp * sum(
                        chains[i + 1].R_node_scores
                    )
            return accepted / proposed

        done = False
        while not done:
            print("add chain")
            ub = 1.0
            lb = chains[-2].temp
            temp = 1.0
            acc_prob = acceptance_prob(temp)
            # print(temp, acc_prob)
            heat = acc_prob < target
            while abs(target - acc_prob) > 0.05:
                if heat:
                    ub = temp
                    temp = temp - (temp - lb) / 2
                else:
                    if temp == 1.0:
                        break
                    lb = temp
                    temp = temp + (ub - temp) / 2
                acc_prob = acceptance_prob(temp)
                print(temp, acc_prob)
                heat = acc_prob < target
            if temp == 1.0:
                done = True
            else:
                chains.append(copy.copy(chains[-1]))
        chains[-1].temp = 1.0
        for c in chains:
            c.stats = stats
        chains = [copy.copy(c) for c in chains]

        return cls(chains, stats=stats)

    def sample(self):
        for c in self.chains:
            c.sample()
        i = np.random.randint(len(self.chains) - 1)
        if self.stats is not None:
            self.stats["mc3"]["proposed"][i] += 1
        ap = sum(self.chains[i + 1].R_node_scores) * self.chains[i].temp
        ap += sum(self.chains[i].R_node_scores) * self.chains[i + 1].temp
        ap -= sum(self.chains[i].R_node_scores) * self.chains[i].temp
        ap -= sum(self.chains[i + 1].R_node_scores) * self.chains[i + 1].temp
        if -np.random.exponential() < ap:
            if self.stats is not None:
                self.stats["mc3"]["accepted"][i] += 1
            R_tmp = self.chains[i].R
            R_node_scores_tmp = self.chains[i].R_node_scores
            self.chains[i].R = self.chains[i + 1].R
            self.chains[i].R_node_scores = self.chains[i + 1].R_node_scores
            self.chains[i].R_score = self.chains[i].temp * sum(
                self.chains[i].R_node_scores
            )
            self.chains[i + 1].R = R_tmp
            self.chains[i + 1].R_node_scores = R_node_scores_tmp
            self.chains[i + 1].R_score = self.chains[i + 1].temp * sum(
                self.chains[i + 1].R_node_scores
            )
        if self.stats is not None:
            self.stats["mc3"]["accept_ratio"][i] = (
                self.stats["mc3"]["accepted"][i]
                / self.stats["mc3"]["proposed"][i]
            )
        return self.chains[-1].R, self.chains[-1].R_score
