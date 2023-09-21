import copy
import warnings

import numpy as np

from .bnet import partition
from .mcmc_moves import DAG_edge_reversal, R_split_merge, R_swap_node_pair
from .stats import Describable


class PartitionMCMC(Describable):
    """Partition-MCMC sampler :footcite:`kuipers:2017` with efficient
    scoring."""

    def __init__(
        self,
        C,
        score,
        d,
        inv_temp=1.0,
        move_weights={
            "R_split_merge": 1,
            "R_swap_node_pair": 1,
            "DAG_edge_reversal": 2,
        },
        R=None,
    ):

        self.n = len(C)
        self.C = C
        self.inv_temp = inv_temp
        self.score = score
        self.d = d
        self.stay_prob = 0.01
        self._all_moves = [
            self.R_split_merge,
            self.R_swap_node_pair,
            self.DAG_edge_reversal,
        ]
        self._move_weights = move_weights

        # These should be in self._stats
        self.proposed = {move.__name__: 0 for move in self._all_moves}
        self.accepted = {move.__name__: 0 for move in self._all_moves}
        self.iter_count = 0

        self._init_moves()

        self.R = R
        if self.R is None:
            self.R = self._random_partition()
        self.R_node_scores = self._pi(self.R)
        self.R_score = self.inv_temp * sum(self.R_node_scores)

    def describe(self):
        description = dict(accept_prob=dict(), iter_count=self.iter_count)
        for move in self._all_moves:
            move = move.__name__
            if self.proposed[move] == 0:
                ap = np.nan
            else:
                ap = self.accepted[move] / self.proposed[move]
            # Lot of this just to conform to MC3.describe()
            # unifying their handling in Gadget
            description["accept_prob"][move] = np.array([ap])
            description["inv_temp"] = np.array([self.inv_temp])
            description["target_chain_iter_count"] = self.iter_count
            description["M"] = 1
        return description

    def _init_moves(self):
        # This needs to be called if self.inv_temp changes from/to 1.0
        move_weights = [
            self._move_weights[k.__name__] for k in self._all_moves
        ]
        if self.inv_temp != 1:
            # NOTE: dropping edge reversal. This now depends on the order in
            #       self._all_moves .
            # TODO: refactor to something less fragile.
            move_weights = move_weights[:-1]
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
            R=self.R,
        )

    def R_split_merge(self, **kwargs):
        # NOTE: Is there value in having these as methods?
        return R_split_merge(**kwargs)

    def R_swap_node_pair(self, **kwargs):
        return R_swap_node_pair(**kwargs)

    def DAG_edge_reversal(self, **kwargs):
        return DAG_edge_reversal(**kwargs)

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

        self.iter_count += 1

        # NOTE: Multiple points of return, consider refactoring.
        if np.random.rand() > self.stay_prob:
            move = self._moves[np.random.randint(len(self._moves))]
            if move == self.DAG_edge_reversal:
                DAG, _ = self.score.sample_DAG(self.R)
                # NOTE: DAG equals DAG_prime after this, since no copy
                #       is made. If necessary, make one.
                return_value = move(
                    DAG=DAG, score=self.score, R=self.R, C=self.C, d=self.d
                )
                if return_value is False:
                    return [self.R], np.array([self.R_score])
                DAG_prime, ap, edge = return_value
                R_prime = partition(DAG_prime)

                R_prime_node_scores = self._pi(
                    R_prime,
                    R_node_scores=self.R_node_scores,
                    rescore=self._rescore(self.R, R_prime),
                )

            elif move in {self.R_split_merge, self.R_swap_node_pair}:
                return_value = move(R=self.R)
                if return_value is False:
                    return [self.R], np.array([self.R_score])
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
                return [self.R], np.array([self.R_score])

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

            self.proposed[move.__name__] += 1
            self.accepted[move.__name__] += 1 * accepted

        return [self.R], np.array([self.R_score])


class MC3(Describable):
    def __init__(self, chains, mode, **params):

        self._stats = {
            "target_chain_iter_count": 0,
            "iter_count": 0,
            "proposed": np.array([0 for c in chains[:-1]]),
            "accepted": np.array([0 for c in chains[:-1]]),
            "local_accept_history": [
                np.zeros(params["sliding_window"], dtype=np.int8)
                for c in chains[:-1]
            ],
        }

        self.chains = chains
        self.params = params
        self.mode = mode
        self.__dict__.update(params)

    def describe(self):

        # TODO: Enforce some schema for these.
        # {"accept_prob": {"mc3": [0.2, ..., 0.3], "local_mc3": [...], "move1":
        # [...]}, "inv_temp": [...]}

        component_stats = [c.describe() for c in self.chains]
        component_accept_probs = [
            {k: v[0] for k, v in cs["accept_prob"].items()}
            for cs in component_stats
        ]
        moves = set().union(
            *list(map(lambda c: set(c.keys()), component_accept_probs))
        )
        component_merged_accept_probs = {
            k: np.zeros(len(self.chains)) for k in moves
        }

        for i, c in enumerate(component_accept_probs):
            for k in c:
                component_merged_accept_probs[k][i] = c[k]

        with np.errstate(divide="ignore", invalid="ignore"):  # dangerous?
            accept_probs = dict(
                mc3=np.concatenate(
                    (
                        self._stats["accepted"] / self._stats["proposed"],
                        np.array([np.nan]),
                    )
                ),
                local_mc3=np.concatenate(
                    (
                        np.array(
                            (
                                list(
                                    map(
                                        np.sum,
                                        self._stats["local_accept_history"],
                                    )
                                )
                            )
                            / np.minimum(
                                self.params["sliding_window"],
                                self._stats["proposed"],
                            )
                        ),
                        np.array([np.nan]),
                    )
                ),
            )

        # This way to keep desired key order (for logging)
        component_merged_accept_probs.update(accept_probs)

        description = dict(
            target_chain_iter_count=self._stats["target_chain_iter_count"],
            iter_count=self._stats["iter_count"],
            accept_prob=component_merged_accept_probs,
            # TODO: replace with output from PartitionMCMC.describe()
            inv_temp=self.inverse_temperatures(),
        )
        description["M"] = len(description["inv_temp"])
        return description

    def inverse_temperatures(self):
        return np.array([c.inv_temp for c in self.chains])

    @staticmethod
    def get_inv_temperatures(heating, M, step=1):
        """Returns the inverse temperatures in descending order."""
        linear = [i / (M - 1) for i in range(M)]
        quadratic = [1 - ((M - 1 - i) / (M - 1)) ** 2 for i in range(M)]
        sigmoid = (
            [0.0]
            + [
                1 / (1 + np.exp((M - 1) * (0.5 - (i / (M - 1)))))
                for i in range(1, M - 1)
            ]
            + [1.0]
        )
        inv_linear = [1 / (1 + (M - 1 - i) * step) for i in range(M)]
        return locals()[heating][::-1]

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

    def _increment_chains(self):
        self.chains.append(copy.copy(self.chains[-1]))
        self.chains[-1].inv_temp = 1 / (
            1 / self.chains[-1].inv_temp + 2 * len(self.chains)
        )
        self._stats["proposed"] = np.append(self._stats["proposed"], 0)
        self._stats["accepted"] = np.append(self._stats["accepted"], 0)
        self._stats["local_accept_history"].append(
            np.zeros(self.sliding_window, dtype=np.int8)
        )

    def _decrement_chains(self):
        self.chains.pop()
        self._stats["proposed"] = self._stats["proposed"][:-1]
        self._stats["accepted"] = self._stats["accepted"][:-1]
        self._stats["local_accept_history"] = self._stats[
            "local_accept_history"
        ][:-1]

    def adapt_temperatures(self):
        stats = self.describe()
        acc_probs = stats["accept_prob"]["mc3"]
        local_acc_probs = stats["accept_prob"]["local_mc3"]
        temps = 1 / stats["inv_temp"]
        delta_temps = np.array(
            [temps[i] - temps[i - 1] for i in range(1, len(temps))]
        )
        diff = (
            np.nan_to_num(
                local_acc_probs, copy=True, nan=0.0, posinf=None, neginf=None
            )
            - self.p_target
        )[:-1]

        delta_new = delta_temps * (1 + diff / self.smoothing)

        inv_temps_new = 1 / np.array(
            [1 + delta_new[:i].sum() for i in range(len(delta_new) + 1)]
        )

        for i, c in enumerate(self.chains):
            c.inv_temp = inv_temps_new[i]

        if (len(self.chains) > 2 and (acc_probs > 0.9).sum() > 1) or np.isnan(
            self.inverse_temperatures()
        ).any():
            self._decrement_chains()
        if (acc_probs > 0.9).sum() == 0:
            self._increment_chains()

    def sample(self):
        local_history_index = (
            self._stats["target_chain_iter_count"] % self.sliding_window
        )
        self._stats["target_chain_iter_count"] += 1
        for c in self.chains:
            self._stats["iter_count"] += 1
            c.sample()
        i = np.random.randint(len(self.chains) - 1)
        self._stats["proposed"][i] += 1
        ap = MC3.get_swap_acceptance_prob(self.chains, i, i + 1)
        if -np.random.exponential() < ap:
            MC3.make_swap(self.chains, i, i + 1)
            self._stats["accepted"][i] += 1
            self._stats["local_accept_history"][i][local_history_index] = 1
        else:
            self._stats["local_accept_history"][i][local_history_index] = 0

        if (
            self.mode == "adaptive"
            and self._stats["target_chain_iter_count"] % self.update_n == 0
        ):
            self.adapt_temperatures()
            self.update_n = round(self.slowdown * self.update_n)

        return [c.R for c in self.chains], np.array(
            [sum(c.R_node_scores) for c in self.chains]
        )
