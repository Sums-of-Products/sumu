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

import numpy as np

from .weight_sum import CandidateRestrictedScore, CandidateComplementScore
from .mcmc import PartitionMCMC, MC3
from .utils.bitmap import bm, bm_to_ints, bm_to_np64
from .utils.io import read_candidates, get_n, pretty_dict, pretty_title
from .utils.math_utils import log_minus_exp, close, comb, subsets
from .scorer import BDeu, BGe
from .candidates import candidate_parent_algorithm as cpa
from .stats import stats

# default parameter values used by multiple classes
default = {
    "mcmc": {
        "iters": 320000,
        "mc3": 16,
        "burn_in": 0.5,
        "n_dags": 10000},
    "score": lambda discrete: {
        "name": "bdeu",
        "params": {"ess": 10}} if discrete else {"name": "bge"},
    "prior": {
        "name": "fair"
    },
    "cons": {
        "max_id": -1,
        "K": lambda n: min(n-1, 16),
        "d": lambda n: min(n-1, 3),
        "pruning_eps": 0.001,
        "score_sum_eps": 0.1
    },
    "candp": {
        "name": "greedy-lite",
        "params": {"k": 6}},
    "catc": {
        "tolerance": 2**-32,
        "cache_size": 10**7
    },
    "logging": {
        "logfile": sys.stdout,
        "stats_period": 15
    },
    "silent": False
}


class Data:
    """Class for holding data.

    The data can be input as either a path to a space delimited csv
    file, a numpy array or an object of type :py:class:`.Data` (in which case a new
    object is created pointing to the same underlying data).

    Assumes the input data is either discrete or continuous. The type
    is either read directly from the input (numpy array or Data), or
    it is inferred from the file the input path points to: "." is considered
    a decimal separator, i.e., it indicates continuous data.
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
                else:
                    self.discrete = True
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
    """Class implementing the Gadget pipeline for MCMC sampling from
    the structure posterior of DAG models. The user interface consists
    of:

    1. The constructor for setting all the parameters.
    2. :py:meth:`.sample()` method which runs the MCMC chain and
       returns the sampled DAGs and their scores.

    All the constructor arguments are keyword arguments, i.e., the
    **data** argument should be given as ``data=data``, etc. Only the
    data argument is required; other arguments have some more or less
    sensible defaults.

    There is a lot of parameters that can be adjusted. To make
    managing the parameters easier, they are grouped into dict-objects
    around some common theme, except the **data** argument which
    accepts any valid constructor argument for a :py:class:`.Data`
    object.

    The (nested) lists in the following description reflect the
    structure of the dict objects. For example, to set the equivalent
    sample size for BDeu score to some value :math:`a`, you should
    construct the object as

    >>> Gadget(data=data, score={"name": "bdeu", "params": {"ess": a}}).

    To only adjust some parameter within a dict-argument while keeping
    the others at default, it suffices to set the one parameter. For
    example, to set the number of candidate parents :math:`K` to some
    value :math:`k`, you should construct the object as

    >>> Gadget(data=data, cons={"K": k}).

    - **score**: The score to use.

      - **name**: Name of the score.

        **Default**: ``bdeu`` (i.e., Bayesian Dirichlet equivalent
        uniform) for discrete data, and ``bge`` (i.e., Bayesian
        Gaussian equivalent) for continuous data.

      - **params**: A dict of parameters for the score.

        **Default**: ``{"ess": 10}`` for ``bdeu``.

    - **prior**: Modular structure prior to use.

      - **name**: Name of the prior, either *fair* or *unif* :footcite:`eggeling:2019`.

        **Default**: fair.

    - **mcmc**: General Markov Chain Monte Carlo arguments.

      - **iters**: The total number of iterations across all the Metropolis
        coupled chains, i.e., if the number of coupled chains is :math:`k` then
        each runs for **iters/k** iterations. If the given **iters** is not a multiple of
        the number of chains it is adjusted downwards.

        **Default**: 320000.

      - **mc3**: The number of of Metropolis coupled chains. The
        temperatures of the chains are spread evenly between uniform
        and the target distribution.

        **Default**: 16.

      - **burn_in**: Ratio of how much of the iterations to use for burn-in (0.5 is 50%).

        **Default**: 0.5.

      - **n_dags**: Number of DAGs to sample. The maximum number of
        DAGs that can be sampled is **iters/mc3*(1-burn_in)**; if the given
        **n_dags** is higher than the maximum, it is adjusted
        downwards.

        **Default**: 10000.

    - **cons**: Constraints on the explored DAG space.

      - **K**: Number of candidate parents per node.

        **Default**: :math:`\min(n-1, 16)`, where :math:`n` is the number of nodes.

      - **d**: Maximum size of parent sets that are not subsets of the candidate parents.

        **Default**: :math:`\min(n-1, 3)`, where :math:`n` is the number of nodes.

      - **max_id**: Maximum size of parent sets that are subsets of
        candidates. There should be no reason to change this from
        the default.

        **Default**: -1, i.e., unlimited.

      - **pruning_eps**: Allowed relative error for a root-partition
        node score sum. Setting this to some value :math:`>0` allows
        some candidate parent sets to be pruned, expediting parent
        set sampling.

        **Default**: 0.001.

      - **score_sum_eps**: Tolerated relative error when computing
        score sums from parent sets that are not subsets of the
        candidate parents.

        **Default**: 0.1.

    - **candp**: Algorithm to use for finding candidate parents.

      - **name**: Name of the algorithm.

        **Default**: ``greedy-lite``.

      - **params**: A dict of parameters for the algorithm.

        **Default**: ``{"k": 6}``. The default algorithm
        :py:func:`~sumu.candidates.greedy_lite` has one parameter,
        :math:`k`, determining the number of parents to add during
        the last iteration of the algorithm. The candidate selection
        phase can be made faster by incrementing this value.

      - **path**: Path to precomputed file storing the candidate
        parents. The format is such that the row number determines the
        node in question, and on each row there are the :math:`K`
        space separated candidate parents. If path is given no
        computations are done.

        **Default**: ``None``.

    - **catc**: Parameters determining how catastrofic cancellations are
      handled. Catastrofic cancellation occurs when a score sum
      :math:`\\tau_i(U,T)` computed as :math:`\\tau_i(U) - \\tau_i(U
      \setminus T)` evaluates to zero due to numerical reasons.

      - **tolerance**: how small should the absolute difference
        between two log score sums be in order for the subtraction
        to be determined to lead to catastrofic cancellation.

        **Default**: :math:`2^{-32}`.

      - **cache_size**: Maximum amount of score sums that cannot be
        computed through subtraction to be stored separately. If there is a
        lot of catastrofic cancellations, setting this value high can
        have a big impact on memory use.

        **Default**: :math:`10^7`

    - **logging**: Parameters determining the logging output during
      running of the sampler.

      - **stats_period**: Interval in seconds for printing more statistics.

        **Default**: 15.

      - **logfile**: File path to print the output to. To suppress all
        output set this to ``None``.

        **Default**: ``sys.stdout``.

    """

    def __init__(self, *,
                 data,
                 mcmc=default["mcmc"],
                 score=None,
                 prior=default["prior"],
                 cons=None,
                 candp=default["candp"],
                 catc=default["catc"],
                 logging=default["logging"]
                 ):

        self.data = Data(data)
        if score is None:
            score = default["score"](self.data.discrete)
        defcons = dict()
        defcons["K"] = default["cons"]["K"](self.data.n)
        defcons["d"] = default["cons"]["d"](self.data.n)
        defcons["max_id"] = default["cons"]["max_id"]
        defcons["pruning_eps"] = default["cons"]["pruning_eps"]
        defcons["score_sum_eps"] = default["cons"]["score_sum_eps"]
        if cons is None:
            cons = defcons
        else:
            cons = dict(defcons, **cons)

        p = {
            "mcmc": dict(default["mcmc"], **mcmc),
            "score": score,
            "prior": prior,
            "cons": cons,
            "candp": candp,
            "catc": dict(default["catc"], **catc),
            "logging": dict(default["logging"], **logging)
        }

        self._silent = default["silent"]
        # No output.
        if p["logging"]["logfile"] is None:
            self._silent = True
            self._logfile = open(os.devnull, "w")
            self._logfilename = ""
        # Output to file.
        elif type(p["logging"]["logfile"]) == str:
            self._logfile = open(p["logging"]["logfile"], "a")
            self._logfilename = self._logfile.name
        # Output to stdout.
        else:
            self._logfile = p["logging"]["logfile"]
            self._logfilename = ""
        self._outputwidth = max(80, 6+12+6*p["mcmc"]["mc3"]-1)
        # To prevent ugly print
        del p["logging"]["logfile"]

        # Adjust "mcmc" parameters if inconsistent
        iters = p["mcmc"]["iters"]
        mc3 = p["mcmc"]["mc3"]
        burn_in = p["mcmc"]["burn_in"]
        n_dags = p["mcmc"]["n_dags"]
        p["mcmc"]["iters"] = iters // mc3 * mc3
        p["mcmc"]["n_dags"] = min((iters - int(iters*burn_in)) // mc3, n_dags)
        adjusted = (p["mcmc"]["iters"] != iters, p["mcmc"]["n_dags"] != n_dags)

        self.p = p

        if self._logfile:
            print(pretty_title("1. PROBLEM INSTANCE", 0, self._outputwidth),
                  file=self._logfile)
            print(pretty_dict(self.data.info), file=self._logfile)
            print(pretty_title("2. RUN PARAMETERS", 2,
                               self._outputwidth), file=self._logfile)
            print(pretty_dict(self.p), file=self._logfile)
            if any(adjusted):
                print("WARNING", file = self._logfile)
            if adjusted[0]:
                print("iters adjusted downwards: needs to be multiple of mc3.",
                      file=self._logfile)
            if adjusted[1]:
                print("n_dags adjusted downwards: max is (iters * (1 - burn_in)) / mc3.",
                      file=self._logfile)

    def sample(self):

        if self._logfile:
            print(pretty_title("3. FINDING CANDIDATE PARENTS", 2,
                               self._outputwidth), file=self._logfile)
            self._logfile.flush()

        stats["t"]["C"] = time.time()
        self._find_candidate_parents()
        stats["t"]["C"] = time.time() - stats["t"]["C"]
        if self._logfile:
            np.savetxt(self._logfile, self.C_array, fmt="%i")
            print("\ntime used: {}s\n".format(round(stats["t"]["C"])), file=self._logfile)
            print(pretty_title("4. PRECOMPUTING SCORING STRUCTURES FOR CANDIDATE PARENT SETS", 2,
                               self._outputwidth), file=self._logfile)
            self._logfile.flush()

        stats["t"]["crscore"] = time.time()
        self._precompute_scores_for_all_candidate_psets()
        self._precompute_candidate_restricted_scoring()
        stats["t"]["crscore"] = time.time() - stats["t"]["crscore"]
        if self._logfile:
            print("time used: {}s\n".format(round(stats["t"]["crscore"])),
                  file=self._logfile)
            print(pretty_title("5. PRECOMPUTING SCORING STRUCTURES FOR COMPLEMENTARY PARENT SETS", 2,
                               self._outputwidth), file=self._logfile)
            self._logfile.flush()

        stats["t"]["ccscore"] = time.time()
        self._precompute_candidate_complement_scoring()
        stats["t"]["ccscore"] = time.time() - stats["t"]["ccscore"]
        if self._logfile:
            print("time used: {}s\n".format(round(stats["t"]["ccscore"])),
                  file=self._logfile)
            print(pretty_title("6. RUNNING MCMC", 2, self._outputwidth),
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
                                  score=self.p["score"],
                                  maxid=self.p["cons"]["max_id"])

        if self.p["candp"].get("path") is None:
            self.C = cpa[self.p["candp"]["name"]](self.p["cons"]["K"],
                                                  scores=self.l_score,
                                                  data=self.data,
                                                  params=self.p["candp"].get("params"))

        else:
            self.C = read_candidates(self.p["candp"]["path"])

        # TODO: Use this everywhere instead of the dict
        self.C_array = np.empty((self.data.n, self.p["cons"]["K"]), dtype=np.int32)
        for v in self.C:
            self.C_array[v] = np.array(self.C[v])

    def _precompute_scores_for_all_candidate_psets(self):
        self.score_array = self.l_score.all_candidate_restricted_scores(self.C_array)

    def _precompute_candidate_restricted_scoring(self):
        self.c_r_score = CandidateRestrictedScore(score_array=self.score_array,
                                                  C=self.C_array,
                                                  K=self.p["cons"]["K"],
                                                  cc_tolerance=self.p["catc"]["tolerance"],
                                                  cc_cache_size=self.p["catc"]["cache_size"],
                                                  pruning_eps=self.p["cons"]["pruning_eps"],
                                                  logfile=self._logfilename,
                                                  silent=self._silent)
        del self.score_array

    def _precompute_candidate_complement_scoring(self):
        self.c_c_score = None
        if self.p["cons"]["K"] < self.data.n - 1:
            # NOTE: CandidateComplementScore gives error if K >= n-1.
            # NOTE: Does this really need to be reinitialized?
            self.l_score = LocalScore(data=self.data,
                                      score=self.p["score"],
                                      maxid=self.p["cons"]["d"])
            self.c_c_score = CandidateComplementScore(localscore=self.l_score,
                                                      C=self.C,
                                                      d=self.p["cons"]["d"],
                                                      eps=self.p["cons"]["score_sum_eps"])
            del self.l_score

    def _init_mcmc(self):

        self.score = Score(C=self.C,
                           c_r_score=self.c_r_score,
                           c_c_score=self.c_c_score)

        if self.p["mcmc"]["mc3"] > 1:
            self.mcmc = MC3([PartitionMCMC(self.C, self.score, self.p["cons"]["d"],
                                           temperature=i/(self.p["mcmc"]["mc3"]-1))
                             for i in range(self.p["mcmc"]["mc3"])])

        else:
            self.mcmc = PartitionMCMC(self.C, self.score, self.p["cons"]["d"])

    def _run_mcmc(self):

        p = self.p["mcmc"]  # Just to shorten rows

        self.dags = list()
        self.dag_scores = list()

        msg_tmpl = "{:<5.5} {:<12.12}" + " {:<5.5}"*p["mc3"]
        temps = sorted(list(stats["mcmc"].keys()), reverse=True)
        temps_labels = [round(t, 2) for t in temps]
        moves = stats["mcmc"][1.0].keys()

        iters_burn_in = int(p["iters"] / p["mc3"] * p["burn_in"])
        iters_dag_sampling = p["iters"] // p["mc3"] - iters_burn_in

        def print_stats_title():
            msg = "Cumulative acceptance probability by move and inverse temperature.\n\n"
            msg += msg_tmpl.format("%", "move", *temps_labels)
            msg += "\n"+"-"*self._outputwidth
            print(msg, file=self._logfile)
            self._logfile.flush()

        def print_stats(i, header=False):
            if header:
                print_stats_title()
            progress = round(100*i/(p["iters"] // p["mc3"]))
            progress = str(progress)
            for m in moves:
                ar = [stats["mcmc"][t][m]["accep_ratio"] for t in temps]
                ar = [round(r,2) if type(r) == float else "" for r in ar]
                msg = msg_tmpl.format(progress, m, *ar)
                print(msg, file=self._logfile)
            if p["mc3"] > 1:
                ar = stats["mc3"]["accepted"] / stats["mc3"]["proposed"]
                ar = [round(r, 2) for r in ar] + [""]
                msg = msg_tmpl.format(progress, "MC^3", *ar)
                print(msg, file=self._logfile)
            print(file=self._logfile)
            self._logfile.flush()

        timer = time.time()
        first = True
        for i in range(iters_burn_in):
            self.mcmc.sample()
            if self._logfile and time.time() - timer > self.p["logging"]["stats_period"]:
                timer = time.time()
                print_stats(i, first)
                first = False
                self._logfile.flush()

        if self._logfile and not first:
            print("Sampling DAGs...\n", file=self._logfile)
            self._logfile.flush()

        dag_count = 0
        for i in range(iters_dag_sampling):
            if self._logfile:
                if time.time() - timer > self.p["logging"]["stats_period"]:
                    timer = time.time()
                    print_stats(i + iters_burn_in, first)
                    first = False
                    self._logfile.flush()
            if i >= iters_dag_sampling / p["n_dags"] * dag_count:
                dag_count += 1
                dag, score = self.score.sample_DAG(self.mcmc.sample()[0])
                self.dags.append(dag)
                self.dag_scores.append(score)
            else:
                self.mcmc.sample()

        if self._logfile and first:
            print_stats(iters_burn_in + iters_dag_sampling, first)
            self._logfile.flush()

        return self.dags, self.dag_scores


class LocalScore:
    """Class for computing local scores given input data.

    Implemented scores are BDeu and BGe. The scores by default use the "fair"
    modular structure prior :footcite:`eggeling:2019`.

    """

    def __init__(self, *, data, score=None, prior=default["prior"],
                 maxid=default["cons"]["max_id"]):
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
                               ess=self.score["params"]["ess"])

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

    def complementary_scores(self, v, C, d):
        K = len(C[0])
        k = (self.data.n - 1) // 64 + 1
        psets = np.empty((sum(comb(self.data.n-1, k) - comb(K, k) for k in range(d+1)), k), dtype=np.uint64)
        scores = np.empty((sum(comb(self.data.n-1, k) - comb(K, k) for k in range(d+1))))
        i = 0
        for pset in subsets([u for u in C if u != v], 1, d):
            if not (set(pset)).issubset(C[v]):
                scores[i] = self._local(v, np.array(pset))
                psets[i] = bm_to_np64(bm(set(pset)), k=k)
                i += 1
        return psets, scores

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

    def __init__(self, *, C, c_r_score, c_c_score):

        self.C = C
        self.n = len(self.C)
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

        U_bm = bm(U.intersection(self.C[v]), idx=self.C[v])
        # T_bm can be 0 if T is empty or does not intersect C[v]
        T_bm = bm(T.intersection(self.C[v]), idx=self.C[v])
        if len(T) > 0:
            if T_bm == 0:
                W_prime = -float("inf")
            else:
                W_prime = self.c_r_score.sum(v, U_bm, T_bm)
        else:
            W_prime = self.c_r_score.sum(v, U_bm)
        if self.c_c_score is None or U.issubset(self.C[v]):
            # This also handles the case U=T={}
            return W_prime
        if len(T) > 0:
            return self.c_c_score.sum(v, U, T, W_prime)#[0]
        else:
            # empty pset handled in c_r_score
            return self.c_c_score.sum(v, U, U, W_prime)#[0]


    def sample_pset(self, v, U, T=set()):

        U_bm = bm(U.intersection(self.C[v]), idx=self.C[v])
        T_bm = bm(T.intersection(self.C[v]), idx=self.C[v])

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
                w_ccs = self.c_c_score.sum(v, U, T)
            else:
                # Empty pset is handled in c_r_score
                w_ccs = self.c_c_score.sum(v, U, U)

        if -np.random.exponential() < w_crs - np.logaddexp(w_ccs, w_crs):
            # Sampling from candidate psets.
            pset, family_score = self.c_r_score.sample_pset(v, U_bm, T_bm, w_crs - np.random.exponential())
            family = (v, set(self.C[v][i] for i in bm_to_ints(pset)))

        else:
            # Sampling from complement psets.
            if len(T) > 0:
                pset, family_score = self.c_c_score.sample_pset(v, U, T, w_ccs - np.random.exponential())
            else:
                pset, family_score = self.c_c_score.sample_pset(v, U, U, w_ccs - np.random.exponential())

            family = (v, set(pset))

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
                family_score = self.sum(v, set(), set())
            else:
                U = set().union(*R[:i])
                T = R[i-1]
                family, family_score = self.sample_pset(v, U, T)
            DAG.append(family)
            DAG_score += family_score
        return DAG, DAG_score
