from collections import defaultdict
import numpy as np
from .utils.math_utils import log_minus_exp, subsets
from .bnet import partition
from .mcmc_moves import R_basic_move, R_swap_any, DAG_edgerev
from .stats import stats


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

