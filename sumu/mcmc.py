import copy
import time
import warnings

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
        inv_temp=1.0,
        move_weights=[1, 1, 2],
        stats=None,
        R=None,
    ):

        self.n = len(C)
        self.C = C
        self.inv_temp = inv_temp
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
                self.stats["mcmc"][self.inv_temp][move.__name__][
                    "proposed"
                ] = 0
                self.stats["mcmc"][self.inv_temp][move.__name__][
                    "accepted"
                ] = 0
                self.stats["mcmc"][self.inv_temp][move.__name__][
                    "accept_ratio"
                ] = 0

        self._init_moves()

        self.R = R
        if self.R is None:
            self.R = self._random_partition()
        self.R_node_scores = self._pi(self.R)
        self.R_score = self.inv_temp * sum(self.R_node_scores)

    def _init_moves(self):
        # This needs to be called if self.inv_temp changes from/to 1.0
        move_weights = self._move_weights
        if self.inv_temp != 1:
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
            inv_temp=self.inv_temp,
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
            self.stats["mcmc"][self.inv_temp][move.__name__]["proposed"] += 1
            if accepted:
                self.stats["mcmc"][self.inv_temp][move.__name__][
                    "accepted"
                ] += 1
            a = self.stats["mcmc"][self.inv_temp][move.__name__]["accepted"]
            if type(a) != int:
                a = 0
            p = self.stats["mcmc"][self.inv_temp][move.__name__]["proposed"]
            try:
                ap = a / p
            except ZeroDivisionError:
                ap = 0.0
            self.stats["mcmc"][self.inv_temp][move.__name__][
                "accept_ratio"
            ] = ap

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
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    ap = (
                        np.exp(
                            self.inv_temp * sum(R_prime_node_scores)
                            - self.R_score
                        )
                        * q_rev
                        / q
                    )

            R_prime_valid = self._valid(R_prime)

            if self.d == 0 and not R_prime_valid:
                return self.R, self.R_score

            # make this happen in log space?
            # if -np.random.exponential() < self.inv_temp * sum(
            #     R_prime_node_scores
            # ) - self.R_score + np.log(q_rev) - np.log(q):
            #     pass
            accepted = False
            if np.random.rand() < ap:
                accepted = True
                self.R = R_prime
                self.R_node_scores = R_prime_node_scores
                self.R_score = self.inv_temp * sum(self.R_node_scores)

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

    @staticmethod
    def get_swap_acceptance_prob(chains, i, j):
        """Compute the swap acceptance probability between chains in indices i
        and j."""
        ap = sum(chains[i].R_node_scores) * chains[j].inv_temp
        ap += sum(chains[j].R_node_scores) * chains[i].inv_temp
        ap -= sum(chains[j].R_node_scores) * chains[j].inv_temp
        ap -= sum(chains[i].R_node_scores) * chains[i].inv_temp
        return ap

    @staticmethod
    def make_swap(chains, i, j):
        R_tmp = chains[i].R
        R_node_scores_tmp = chains[i].R_node_scores
        chains[i].R = chains[j].R
        chains[i].R_node_scores = chains[j].R_node_scores
        chains[i].R_score = chains[i].inv_temp * sum(chains[i].R_node_scores)
        chains[j].R = R_tmp
        chains[j].R_node_scores = R_node_scores_tmp
        chains[j].R_score = chains[j].inv_temp * sum(chains[j].R_node_scores)

    @classmethod
    def adaptive(cls, mcmc, t_budget=None, stats=None, target=0.25, log=None):

        if t_budget is not None:
            t0 = time.time()

        mcmc0 = copy.copy(mcmc)
        mcmc0.inv_temp = 0.0
        mcmc0._init_moves()
        chains = [mcmc0, mcmc]

        if stats is not None:
            stats["iters"]["adaptive tempering"] = 0

        msg_tmpl = "{:<8}" + "{:<9}" * 2
        if log is not None:
            log(msg_tmpl.format("chain", "temp^-1", "swap_prob"))
            log(msg_tmpl.format("1", "0.0", "-"))

        def acceptance_prob(i_target, inv_temp):
            chains[i_target].inv_temp = inv_temp
            chains[i_target].R = chains[i_target - 1].R
            chains[i_target].R_node_scores = chains[i_target - 1].R_node_scores
            chains[i_target]._init_moves()
            proposed = 0
            accepted = 0
            while proposed < 1000:
                if t_budget is not None:
                    if time.time() - t0 > t_budget:
                        log.br()
                        log("Time budget exceeded, terminating.")
                        exit(1)
                # Only the target chain and one hotter than it are sampled
                for c in chains[i_target - 1 : i_target + 1]:
                    if stats is not None:
                        stats["iters"]["adaptive tempering"] += 1
                    c.sample()
                # j = np.random.randint(len(chains) - 1)
                j = i_target - 1
                if j == i_target - 1:
                    proposed += 1
                ap = MC3.get_swap_acceptance_prob(chains, j, j + 1)
                if -np.random.exponential() < ap:
                    if j == i_target - 1:
                        accepted += 1
                    MC3.make_swap(chains, j, j + 1)
            return accepted / proposed

        # Commented out option to rerun the temperature estimations
        # until the number of chains equals the previous run +/- 1.

        # all_done = False
        # while not all_done:
        for l in range(1):
            # start_len = len(chains)
            done = False
            i = 0
            while not done:
                i += 1
                ub = 1.0
                lb = chains[i - 1].inv_temp
                chains[i].inv_temp = max(lb, chains[i].inv_temp)
                acc_prob = acceptance_prob(i, chains[i].inv_temp)
                if log is not None:
                    log(
                        msg_tmpl.format(
                            i + 1,
                            round(chains[i].inv_temp, 3),
                            round(acc_prob, 3),
                        )
                    )
                heat = acc_prob < target
                search_steps = 0
                while abs(target - acc_prob) > 0.05:
                    search_steps += 1
                    if search_steps > 20:
                        break

                    # the ub/lb is set half way between the previous ub/lb and
                    # current temperature, to avoid getting trapped in wrong
                    # region.

                    if heat:
                        ub = chains[i].inv_temp + 0.5 * (
                            ub - chains[i].inv_temp
                        )
                        chains[i].inv_temp = (
                            chains[i].inv_temp - (chains[i].inv_temp - lb) / 2
                        )
                    else:
                        if abs(chains[i].inv_temp - 1.0) < 1e-4:
                            break
                        lb = chains[i].inv_temp - 0.5 * (
                            chains[i].inv_temp - lb
                        )
                        chains[i].inv_temp = (
                            chains[i].inv_temp + (ub - chains[i].inv_temp) / 2
                        )
                    acc_prob = acceptance_prob(i, chains[i].inv_temp)
                    if log is not None:
                        log(
                            msg_tmpl.format(
                                "",
                                round(chains[i].inv_temp, 3),
                                round(acc_prob, 3),
                            )
                        )
                    heat = acc_prob < target
                if abs(chains[i].inv_temp - 1.0) < 1e-4:
                    chains = chains[: i + 1]
                    done = True
                else:
                    chain = copy.copy(chains[i])
                    chain.inv_temp = 1.0
                    chain._init_moves()
                    chains.append(chain)
            chains[-1].inv_temp = 1.0
            chains[-1] = copy.copy(chains[-1])

            # all_done = len(chains) in range(start_len - 1, start_len + 2)

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
        ap = MC3.get_swap_acceptance_prob(self.chains, i, i + 1)
        if -np.random.exponential() < ap:
            if self.stats is not None:
                self.stats["mc3"]["accepted"][i] += 1
            MC3.make_swap(self.chains, i, i + 1)
        if self.stats is not None:
            self.stats["mc3"]["accept_ratio"][i] = (
                self.stats["mc3"]["accepted"][i]
                / self.stats["mc3"]["proposed"][i]
            )
        return self.chains[-1].R, self.chains[-1].R_score
