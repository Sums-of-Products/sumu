"""The module implements the algorithm Gadget as first detailed in
:footcite:`viinikka:2020a`.

Limitations:
  The computations rely heavily on bitwise operations, which for
  reasons of efficiency have been implemented using primitive data
  types (i.e., uint64_t). In the current version this sets a hard
  limit on the maximum number of variables in the data at 256.


"""

import sys
import os
import time
import pprint

import numpy as np

from .weight_sum import weight_sum
from .mcmc import PartitionMCMC, MC3
from .utils.bitmap import bm, bm_to_ints, bm_to_pyint_chunks, bms_to_np64, np64_to_bm
from .utils.io import read_candidates, get_n, pretty_dict, pretty_title
from .utils.math_utils import log_minus_exp, close, comb, subsets
from .scorer import BDeu, BGe
from .CandidateRestrictedScore import CandidateRestrictedScore
from .DAGR import DAGR as DAGR_precompute
from .candidates import candidate_parent_algorithm
from .stats import stats

# default parameter values used by multiple classes
default = {
    "score": lambda discrete:
      {"name": "bdeu", "ess": 10} if discrete else {"name": "bge"},
    "prior": {"name": "fair"},
    "max_id": -1,
    "K": lambda n: min(n-1, 16),
    "d": lambda n: min(n-1, 3),
    "cp_algo": "greedy-lite",
    "mc3": 16,
    "cc_tolerance": 2**-32,
    "cc_cache_size": 10**7,
    "pruning_eps": 0.001,
    "logfile": sys.stdout,
    "silent": False,
    "stats_period": 15
}


class Data:
    """Class for holding data.

    Assumes the input data is either discrete or continuous.

    The data can be input as either a path to a space delimited csv
    file, a numpy array or a object of type Data (in which case a new
    object is created pointing to same data).
    """

    def __init__(self, data_or_path):

        # Copying existing Data object
        if type(data_or_path) == Data:
            self.data = data_or_path.data
            self.discrete = data_or_path.discrete
            return

        # Initializing from np.array
        if type(data_or_path) == np.ndarray:
            self.data = data_or_path
            self.discrete = self.data.dtype != np.float64
            return

        # Initializing from path
        if type(data_or_path) == str:
            with open(data_or_path) as f:
                # . is assumed to be a decimal separator
                if '.' in f.read():
                    self.discrete = False
            if self.discrete:
                self.data = np.loadtxt(data_or_path, dtype=np.int32, delimiter=' ')
            else:  # continuous
                self.data = np.loadtxt(data_or_path, dtype=np.float64, delimiter=' ')
            return

        else:
            raise TypeError("Unknown type for Data: {}.".format(type(data_or_path)))

    @property
    def n(self):
        return self.data.shape[1]

    @property
    def N(self):
        return self.data.shape[0]

    @property
    def arities(self):
        return np.count_nonzero(np.diff(np.sort(self.data.T)), axis=1)+1

    def all(self):
        # TODO: NEED TO GET RID OF THIS?
        # This is to simplify passing data to R
        data = self.data
        if self.arities is not False:
            arities = np.reshape(self.arities, (-1, len(self.n)))
            data = np.append(arities, data, axis=0)
        return data

    @property
    def info(self):
        info = {
            "no. variables": self.n,
            "no. data points": self.N,
            "type of data": ["continuous", "discrete"][1*self.discrete]
        }
        if self.discrete:
            info["arities [min, max]"] = "[{}, {}]".format(
                min(self.arities), max(self.arities))
        return info


class Gadget():

    def __init__(self, *, data, score=None, prior=default["prior"],
                 max_id=default["max_id"], K=None, d=None,
                 cp_algo=default["cp_algo"], cp_path=None,
                 mc3=default["mc3"],
                 burn_in, iterations, thinning,
                 cc_tolerance=default["cc_tolerance"],
                 cc_cache_size=default["cc_cache_size"],
                 pruning_eps=default["pruning_eps"],
                 logfile=default["logfile"],
                 stats_period=default["stats_period"]):
        self.data = Data(data)
        if score is None:
            score = default["score"](self.data.discrete)
        if K is None:
            K = default["K"](self.data.n)
        if d is None:
            d = default["d"](self.data.n)
        self.params = {
            "score": score,
            "prior": prior,
            "maxid": max_id,
            "K": K,
            "d": d,
            "cp_algo": cp_algo,
            "cp_path": cp_path,
            "mc3": mc3,
            "burn_in": burn_in,
            "iterations": iterations,
            "thinning": thinning,
            "cc_tolerance": cc_tolerance,
            "cc_cache_size": cc_cache_size,
            "pruning_eps": pruning_eps,
            "stats_period": stats_period
        }

        self._silent = default["silent"]
        # No output.
        if logfile is None:
            self._silent = True
            self._logfile = open(os.devnull, "w")
            self._logfilename = ""
        # Output to file.
        elif type(logfile) == str:
            self._logfile = open(logfile, "a")
            self._logfilename = self._logfile.name
        # Output to sdout.
        else:
            self._logfile = logfile
            self._logfilename = ""
        self._outputwidth = max(80, 6+12+6*mc3-1)

    def _param(self, *params):
        # Utility to simplify passing parameters
        return {k: self.params[k] for k in params}

    def sample(self):

        if self._logfile:
            print(pretty_title("1. RUN PARAMETERS", 0,
                               self._outputwidth), file=self._logfile)
            print(pretty_dict(self.params), file=self._logfile)
            print(pretty_title("2. FINDING CANDIDATE PARENTS", 2,
                               self._outputwidth), file=self._logfile)
            self._logfile.flush()

        stats["t"]["C"] = time.time()
        self._find_candidate_parents()
        stats["t"]["C"] = time.time() - stats["t"]["C"]
        if self._logfile:
            #print(pretty_dict(self.C), file=self._logfile)
            print(pprint.pformat(self.C) + "\n", file=self._logfile)
            print("time used: {}s\n".format(round(stats["t"]["C"])), file=self._logfile)
            print(pretty_title("3. PRECOMPUTING SCORING STRUCTURES FOR CANDIDATE PARENT SETS", 2,
                               self._outputwidth), file=self._logfile)
            self._logfile.flush()

        stats["t"]["crscore"] = time.time()
        self._precompute_scores_for_all_candidate_psets()
        self._precompute_candidate_restricted_scoring()
        stats["t"]["crscore"] = time.time() - stats["t"]["crscore"]
        if self._logfile:
            print("time used: {}s\n".format(round(stats["t"]["crscore"])),
                  file=self._logfile)
            print(pretty_title("4. PRECOMPUTING SCORING STRUCTURES FOR COMPLEMENTARY PARENT SETS", 2,
                               self._outputwidth), file=self._logfile)
            self._logfile.flush()

        stats["t"]["ccscore"] = time.time()
        self._precompute_candidate_complement_scoring()
        stats["t"]["ccscore"] = time.time() - stats["t"]["ccscore"]
        if self._logfile:
            print("time used: {}s\n".format(round(stats["t"]["ccscore"])),
                  file=self._logfile)
            print(pretty_title("5. RUNNING MCMC", 2, self._outputwidth),
                  file=self._logfile)
            self._logfile.flush()

        stats["t"]["mcmc"] = time.time()
        self._init_mcmc()
        self._run_mcmc()
        stats["t"]["mcmc"] = time.time() - stats["t"]["mcmc"]
        if self._logfile:
            print("time used: {}s\n".format(round(stats["t"]["mcmc"])),
                  file=self._logfile)

        return self.dags, self.dag_scores


    def _find_candidate_parents(self):
        self.l_score = LocalScore(data=self.data,
                                  **self._param("score", "maxid"))

        if self.params["cp_path"] is None:
            self.C = candidate_parent_algorithm[self.params["cp_algo"]](self.params["K"],
                                                                        n=self.data.n,
                                                                        scores=self.l_score,
                                                                        data=self.data)

        else:
            self.C = read_candidates(self.params["cp_path"])

        # TODO: Use this everywhere instead of the dict
        self.C_array = np.empty((self.data.n, self.params["K"]), dtype=np.int32)
        for v in self.C:
            self.C_array[v] = np.array(self.C[v])

    def _precompute_scores_for_all_candidate_psets(self):
        self.score_array = self.l_score.all_candidate_restricted_scores(self.C_array)

    def _precompute_candidate_restricted_scoring(self):
        self.c_r_score = CandidateRestrictedScore(score_array=self.score_array,
                                                  C=self.C_array,
                                                  **self._param("K",
                                                                "cc_tolerance",
                                                                "cc_cache_size",
                                                                "pruning_eps"),
                                                  logfile=self._logfilename,
                                                  silent=self._silent)

    def _precompute_candidate_complement_scoring(self):
        self.c_c_score = None
        if self.params["K"] < self.data.n - 1:
            # NOTE: CandidateComplementScore gives error if K >= n-1.
            self.l_score = LocalScore(data=self.data,
                                      score=self.params["score"],
                                      maxid=self.params["d"])
            self.c_c_score = CandidateComplementScore(self.C, self.l_score, self.params["d"])
            del self.l_score

    def _init_mcmc(self):

        self.score = Score(C=self.C,
                           score_array=self.score_array,
                           c_r_score=self.c_r_score,
                           c_c_score=self.c_c_score)

        if self.params["mc3"] > 1:
            self.mcmc = MC3([PartitionMCMC(self.C, self.score, self.params["d"],
                                           temperature=i/(self.params["mc3"]-1))
                             for i in range(self.params["mc3"])])

        else:
            self.mcmc = PartitionMCMC(self.C, self.score, self.params["d"])

    def _run_mcmc(self):

        msg_tmpl = "{:<5.5} {:<12.12}" + " {:<5.5}"*self.params["mc3"]
        temps = list(stats["mcmc"].keys())[::-1]
        temps_labels = [round(t, 2) for t in temps]
        moves = stats["mcmc"][1.0].keys()

        def print_stats_title():
            msg = "Cumulative acceptance probability by move and inverse temperature.\n\n"
            msg += msg_tmpl.format("%", "move", *temps_labels)
            msg += "\n"+"-"*self._outputwidth
            print(msg, file=self._logfile)
            self._logfile.flush()

        def print_stats(i, header=False):
            if header:
                print_stats_title()
            p = round(100*i/(self.params["burn_in"] + self.params["iterations"]))
            p = str(p)
            for m in moves:
                ar = [stats["mcmc"][t][m]["accep_ratio"] for t in temps]
                ar = [round(r,2) if type(r) == float else "" for r in ar]
                msg = msg_tmpl.format(p, m, *ar)
                print(msg, file=self._logfile)
            if self.params["mc3"] > 1:
                ar = stats["mc3"]["accepted"] / stats["mc3"]["proposed"]
                ar = [round(r, 2) for r in ar] + [0.0]
                msg = msg_tmpl.format(p, "MC^3", *ar)
                print(msg, file=self._logfile)
            print(file=self._logfile)
            self._logfile.flush()

        timer = time.time()
        header = True
        for i in range(self.params["burn_in"]):
            self.mcmc.sample()
            if self._logfile and time.time() - timer > self.params["stats_period"]:
                timer = time.time()
                print_stats(i, header)
                header = False
                self._logfile.flush()

        if self._logfile:
            print("Sampling DAGs...\n", file=self._logfile)
            self._logfile.flush()

        self.dags = list()
        self.dag_scores = list()
        for i in range(self.params["iterations"]):
            if self._logfile:
                if time.time() - timer > self.params["stats_period"]:
                    timer = time.time()
                    print_stats(i + self.params["burn_in"])
                    self._logfile.flush()
            if i % self.params["thinning"] == 0:
                dag, score = self.score.sample_DAG(self.mcmc.sample()[0])
                self.dags.append(dag)
                self.dag_scores.append(score)
            else:
                self.mcmc.sample()

        return self.dags, self.dag_scores


class LocalScore:
    """Class for computing local scores given input data.

    Implemented scores are BDeu and BGe. The scores by default use the "fair"
    modular structure prior :cite:`eggeling:2019`.

    """

    def __init__(self, *, data, score=None, prior=default["prior"], maxid=default["max_id"]):
        self.data = Data(data)
        self.score = score
        if score is None:
            self.score = default["score"](self.data.discrete)
        self.prior = prior
        self.priorf = {"fair": self._prior_fair,
                       "unif": self._prior_unif}
        self.maxid = maxid
        self._precompute_prior()

        if self.score["name"] == "bdeu":
            self.scorer = BDeu(data=self.data.data,
                               maxid=self.maxid,
                               ess=self.score["ess"])

        elif self.score["name"] == "bge":
            self.scorer = BGe(data=self.data,
                              maxid=self.maxid)

    def _prior_fair(self, indegree):
        return self._prior[indegree]

    def _prior_unif(self, indegree):
        return 0

    def _precompute_prior(self):
        if self.prior["name"] == "fair":
            self._prior = np.zeros(self.data.n)
            self._prior = -np.array(list(map(np.log, [float(comb(self.data.n - 1, k))
                                                      for k in range(self.data.n)])))

    def local(self, v, pset):
        """Local score for input node v and pset, with score function self.scoref.

        This is the "safe" version, raising error if queried with invalid input.
        The unsafe self._local will just segfault.
        """
        if v in pset:
            raise IndexError("Attempting to query score for (v, pset) where v \in pset")
        # Because min() will raise error with empty pset
        if v in range(self.data.n) and len(pset) == 0:
            return self._local(v, pset)
        if min(v, min(pset)) < 0 or max(v, max(pset)) >= self.data.n:
            raise IndexError("Attempting to query score for (v, pset) where some variables don't exist in data")
        return self._local(v, pset)

    def _local(self, v, pset):
        # NOTE: How expensive are nested function calls?
        return self.scorer.local(v, pset) + self.priorf[self.prior["name"]](len(pset))

    def clear_cache(self):
        self.scorer.clear_cache()

    def complementary_scores_dict(self, C, d):
        """C candidates, d indegree for complement psets"""
        cscores = dict()
        for v in C:
            cscores[v] = dict()
            for pset in subsets([u for u in C if u != v], 1, d):
                if not (set(pset)).issubset(C[v]):
                    cscores[v][pset] = self._local(v, np.array(pset))
        return cscores

    def all_candidate_restricted_scores(self, C=None):
        if C is None:
            C = np.array([np.array([j for j in range(self.data.n) if j != i])
                          for i in range(self.data.n)], dtype=np.int32)
        prior = np.array([bin(i).count("1") for i in range(2**len(C[0]))])
        prior = np.array(list(map(lambda k: self.priorf[self.prior["name"]](k), prior)))
        return self.scorer.all_candidate_restricted_scores(C) + prior

    def all_scores_dict(self, C=None):
        # NOTE: Not used in Gadget pipeline, but useful for example
        #       when computing input data for aps.
        scores = dict()
        if C is None:
            C = {v: tuple(sorted(set(range(self.data.n)).difference({v}))) for v in range(self.data.n)}
        for v in C:
            tmp = dict()
            for pset in subsets(C[v], 0, [len(C[v]) if self.maxid == -1 else self.maxid][0]):
                tmp[pset] = self._local(v, np.array(pset))
            scores[v] = tmp
        return scores


class Score:  # should be renamed to e.g. ScoreHandler

    def __init__(self, *, C, score_array,
                 c_r_score, c_c_score):

        self.C = C
        self.n = len(self.C)
        self.score_array = score_array
        self.c_r_score = c_r_score
        self.c_c_score = c_c_score

    def sum(self, v, U, T=set()):
        """Returns the sum of scores for node v over the parent sets that
        1. are subsets of U;
        2. and, if T is not empty, have at least one member in T.

        The sum is computed over first the scores restricted to candidate
        parents (self.C), and then the result is augmented by scores
        complementary to those restricted to the candidate parents, until
        some predefined level of error.

        Args:
           v (int): Label of the node whose local scores are summed.
           U (set): Parent sets of scores to be summed are the subsets of U.
           T (set): Parent sets must have at least one member in T (if T is not empty).

        Returns:
            Sum of scores (float).

        """

        U_bm = bm(U.intersection(self.C[v]), ix=self.C[v])
        # T_bm can be 0 if T is empty or does not intersect C[v]
        T_bm = bm(T.intersection(self.C[v]), ix=self.C[v])
        if len(T) > 0:
            if T_bm == 0:
                W_prime = -float("inf")
            else:
                W_prime = self.c_r_score.sum(v, U_bm, T_bm)
        else:
            W_prime = self.c_r_score.sum(v, U_bm)
        if self.c_c_score is None or U.issubset(self.C[v]):
            return W_prime
        if len(T) > 0:
            return self.c_c_score.sum(v, U, T, W_prime)#[0]
        else:
            # empty pset handled in c_r_score
            return self.c_c_score.sum(v, U, U, W_prime)#[0]


    def sample_pset(self, v, U, T=set()):
        U_bm = bm(U.intersection(self.C[v]), ix=self.C[v])
        T_bm = bm(T.intersection(self.C[v]), ix=self.C[v])

        if len(T) > 0:
            if T_bm == 0:
                w_crs = -float("inf")
            else:
                w_crs = self.c_r_score.sum(v, U_bm, T_bm)
        else:
            w_crs = self.c_r_score.sum(v, U_bm)

        w_ccs = -float("inf")
        if self.c_c_score is not None and not U.issubset(self.C[v]):
            if len(T) > 0:
                w_ccs, contribs = self.c_c_score.sum(v, U, T, -float("inf"), contribs=True)
            else:
                # Empty pset is handled in c_r_score
                w_ccs, contribs = self.c_c_score.sum(v, U, U, -float("inf"), contribs=True)

        if -np.random.exponential() < w_crs - np.logaddexp(w_ccs, w_crs):
            # Sampling from candidate psets.
            pset = self.c_r_score.sample_pset(v, U_bm, T_bm, -np.random.exponential())
            family_score = self.score_array[v][pset]
            family = (v, set(self.C[v][i] for i in bm_to_ints(pset)))

        else:
            # Sampling from complement psets.
            p = np.array([self.c_c_score.ordered_scores[v][j]
                          for j in contribs])
            p = np.exp(p - w_ccs)
            j = np.random.choice(contribs, p=p)
            pset = np64_to_bm(self.c_c_score.ordered_psets[v][j])
            family_score = self.c_c_score.ordered_scores[v][j]
            family = (v, set(bm_to_ints(pset)))

        return family, family_score

    def sample_DAG(self, R):
        DAG = list()
        DAG_score = 0
        for v in range(self.n):
            for i in range(len(R)):
                if v in R[i]:
                    break
            if i == 0:
                family = (v, set())
                family_score = self.score_array[v][0]
            else:
                U = set().union(*R[:i])
                T = R[i-1]
                family, family_score = self.sample_pset(v, U, T)
            DAG.append(family)
            DAG_score += family_score
        return DAG, DAG_score


class CandidateComplementScore:
    """For computing the local score sum complementary to those obtained from :py:class:`.old_CandidateRestrictedScore` and constrained by maximum indegree.
    """

    def __init__(self, C, scores, d):

        self.C = C
        self.n = len(C)
        self.d = d
        self.minwidth = (self.n - 1) // 64 + 1

        # object of class LocalScore
        scores = scores.complementary_scores_dict(C, d)

        ordered_psets = dict()
        ordered_scores = dict()

        for v in scores:
            ordered_scores[v] = sorted(scores[v].items(), key=lambda item: item[1], reverse=True)
            ordered_psets[v] = bms_to_np64([bm(item[0]) for item in ordered_scores[v]], minwidth=self.minwidth)
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

        self.t_ub = np.zeros(shape=(self.n, self.n), dtype=np.int32)
        for u in range(1, self.n+1):
            for t in range(1, u+1):
                self.t_ub[u-1][t-1] = self.n_valids_ub(u, t)

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

    def sum(self, v, U, T, W_prime, debug=False, contribs=False):

        # NOTE: This is the final score calculation.

        if self.d == 1:  # special case
            pset_idxs = list()
            w_contribs = list()
            for u in T:
                if u not in self.C[v]:
                    pset_idx = self.pset_to_idx[v][bm(u)]
                    pset_idxs.append(pset_idx)
                    w_contribs.append(self.ordered_scores[v][pset_idx])
            w_contribs.append(W_prime)
            W_sum = np.logaddexp.reduce(w_contribs)

        else:
            if self.n <= 64:
                U_bm = bm(U)
                T_bm = bm(T)
            else:
                U_bm = bm_to_pyint_chunks(bm(U), self.minwidth)
                T_bm = bm_to_pyint_chunks(bm(T), self.minwidth)

            W_sum = weight_sum(w=W_prime,
                               psets=self.ordered_psets[v],
                               weights=self.ordered_scores[v],
                               n=self.n,
                               U=U_bm,
                               T=T_bm,
                               t_ub=int(self.t_ub[len(U)][len(T)]),
                               contribs=contribs)
            if contribs is True:
                W_sum, pset_idxs = W_sum

        if contribs is True:
            return W_sum, pset_idxs
        return W_sum
