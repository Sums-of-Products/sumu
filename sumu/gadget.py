"""The module implements the algorithm Gadget as first detailed in
:footcite:`viinikka:2020a`.
"""

import copy
import datetime
import gc
import os
import pathlib
import sys
import time
from pathlib import Path

import numpy as np
import psutil

try:
    import plotext as plt

    plot_trace = plt.__version__ == "4.1.3"
except ImportError:
    plot_trace = False

from sumu import __version__

from . import validate
from .candidates import candidate_parent_algorithm as cpa
from .data import Data
from .mcmc import MC3, PartitionMCMC
from .scorer import BDeu, BGe
from .stats import Stats, stats
from .utils.bitmap import bm, bm_to_ints, bm_to_np64
from .utils.io import read_candidates
from .utils.math_utils import comb, fit_linreg, subsets
from .weight_sum import CandidateComplementScore, CandidateRestrictedScore

# Debugging level. 0 = no debug prints.
DEBUG = 0


def DBUG(msg):
    if DEBUG:
        print("DEBUG: " + msg)


class Defaults:

    # default parameter values for Gadget

    def __init__(self):

        default = dict()

        default["run_mode"] = lambda name: {
            name
            == "normal": {
                "name": "normal",
                "params": {"n_target_chain_iters": 20000},
            },
            name == "anytime": {"name": name},
            name
            in {"budget", None}: {
                "name": "budget",
                "params": {
                    "t": lambda data: int(
                        min(1.0, 0.01 * data.n)
                        * np.multiply(*data.shape)
                        * 3300
                        # cpu_freq.max sometimes returns 0
                        / [psutil.cpu_freq().max, 3300][
                            psutil.cpu_freq().max == 0
                        ]
                    ),
                    "mem": int(
                        psutil.virtual_memory().available / 1024.0 ** 2
                    ),
                    "t_share": {"C": 1 / 9, "K": 1 / 9, "d": 1 / 9},
                },
            },
        }.get(True)

        default["mcmc"] = {
            "initial_rootpartition": None,
            "n_independent": 1,
            "burnin": 0.5,
            "n_dags": 10000,
            "move_weights": {
                "R_split_merge": 1,
                "R_swap_node_pair": 1,
                "DAG_edge_reversal": 2,
            },
        }

        default["metropolis_coupling"] = lambda name: {
            name
            == "adaptive": {
                "name": name,
                "params": {
                    "M": 2,
                    "p_target": 0.234,
                    "delta_init": 0.5,
                    "sliding_window": 1000,
                    "update_n": 100,
                    "smoothing": 2.0,
                    "slowdown": 1.0,
                },
            },
            name
            != "adaptive": {
                "name": "static",
                "params": {
                    "M": 16,
                    "heating": "linear",
                    "sliding_window": 100,
                },
            },
        }.get(True)

        default["score"] = (
            lambda data: {"name": "bdeu", "params": {"ess": 10}}
            if data.discrete
            else {"name": "bge"}
        )

        default["structure_prior"] = {"name": "fair"}

        default["constraints"] = lambda runmode: {
            runmode
            in ["normal", "anytime"]: {
                "max_id": -1,
                "K": lambda data: min(data.n - 1, 16),
                "d": lambda data: min(data.n - 1, 3),
            },
            runmode
            == "budget": {
                "max_id": -1,
                "K_min": 1,
                "d_min": 2,
            },
        }.get(True)

        default["candidate_parent_algorithm"] = lambda runmode: {
            runmode
            == "budget": {
                "name": "greedy",
                "params": {"association_measure": "gain"},
            },
            runmode
            in ["normal", "anytime"]: {
                "name": "greedy",
                "params": {"K_f": 6, "association_measure": "gain"},
            },
        }.get(True)

        default["catastrophic_cancellation"] = {
            "tolerance": 2 ** -32,
            "cache_size": 10 ** 7,
        }

        default["pruning_tolerance"] = 0.001

        default["scoresum_tolerance"] = 0.01

        default["logging"] = {
            "silent": False,
            "verbose_prefix": None,
            "period": 15,
            "overwrite": False,
        }

        self.default = default

    def __call__(self):
        return self.default

    def __getitem__(self, key):
        return self.default[key]


class GadgetParameters:
    def __init__(
        self,
        *,
        data,
        validate_params=True,
        is_prerun=False,
        run_mode=dict(),
        mcmc=dict(),
        score=dict(),
        structure_prior=dict(),
        constraints=dict(),
        candidate_parent_algorithm=dict(),
        metropolis_coupling=dict(),
        catastrophic_cancellation=dict(),
        pruning_tolerance=None,
        scoresum_tolerance=None,
        candidate_parents_path=None,
        candidate_parents=None,
        logging=dict(),
    ):
        # Save parameters initially given by user.
        # locals() has to be the first thing called in __init__.
        self.p_user = dict(**locals())
        del self.p_user["self"]
        del self.p_user["data"]
        del self.p_user["is_prerun"]

        self.t0 = time.time()
        self.data = Data(data)

        # useful to set this to False when developing new features
        if validate.is_boolean(
            validate_params, msg="'validate_params' should be boolean"
        ):
            self._validate_parameters()
        del self.p_user["validate_params"]

        self.default = Defaults()()
        self.p = copy.deepcopy(self.p_user)

        self._populate_default_parameters()
        self._complete_user_given_parameters()

        # This is run regardless of run_mode, to estimate and log
        # - time use
        # - memory footprint (requires estimating number of cc)
        if not is_prerun:

            self.time_use_estimate = dict(K=dict(), d=dict())

            K_prerun = min(15, self.data.n - 3)
            if "K" in self.p_user["constraints"]:
                K_prerun = max(
                    2, min(self.p_user["constraints"]["K"] - 2, K_prerun)
                )
            self.prerun(K_prerun)

            if self.p["run_mode"]["name"] == "budget":

                # TODO: get rid of this
                self.budget = dict()

                if "K" in self.p_user["constraints"]:
                    self.time_use_estimate["K"][
                        self.p_user["constraints"]["K"]
                    ] = self.pred_time_use_K(self.p_user["constraints"]["K"])
                else:
                    if "t" in self.p["run_mode"]["params"]:
                        K, pred = self.adjust_to_time_budget_K(
                            self.p["run_mode"]["params"]["t_share"]["K"]
                            * self.p["run_mode"]["params"]["t"],
                            self.p["constraints"]["K_min"],
                        )
                        self.p["constraints"]["K"] = K
                        self.time_use_estimate["K"] = pred
                if "d" in self.p_user["constraints"]:
                    self.time_use_estimate["d"][
                        self.p_user["constraints"]["d"]
                    ] = self.pred_time_use_d(self.p_user["constraints"]["d"])
                else:
                    if "t" in self.p["run_mode"]["params"]:
                        d, pred = self.adjust_to_time_budget_d(
                            self.p["run_mode"]["params"]["t_share"]["d"]
                            * self.p["run_mode"]["params"]["t"],
                            self.p["constraints"]["d_min"],
                        )
                        self.p["constraints"]["d"] = d
                        self.time_use_estimate["d"] = pred

                if "mem" in self.p["run_mode"]["params"]:
                    DBUG("adjusting to mem_budget")
                    self._adjust_to_mem_budget(
                        self.p["run_mode"]["params"]["mem"]
                    )

                # set budget for candidate parents search if k not set
                if "t" in self.p["run_mode"]["params"]:
                    budget = (
                        self.p["run_mode"]["params"]["t_share"]["C"]
                        * self.p["run_mode"]["params"]["t"]
                    )
                    try:
                        self.p_user["candidate_parent_algorithm"]["params"][
                            "K_f"
                        ]
                    except KeyError:
                        try:
                            self.p["candidate_parent_algorithm"]["params"][
                                "t_budget"
                            ] = budget
                        except KeyError:
                            self.p["candidate_parent_algorithm"]["params"] = {
                                "t_budget": budget
                            }

            else:
                self.time_use_estimate["K"][
                    self.p["constraints"]["K"]
                ] = self.pred_time_use_K(self.p["constraints"]["K"])
                self.time_use_estimate["d"][
                    self.p["constraints"]["d"]
                ] = self.pred_time_use_d(self.p["constraints"]["d"])

            # Now all params, except k if run_mode == budget, are fixed
            # Estimate candidate parent search time use.
            # NOTE: The parameters are validated before so we can assume
            #       this does what its supposed to.
            if (
                "candidate_parents" in self.p
                or "candidate_parents_path" in self.p
                or self.p["candidate_parent_algorithm"]["name"] == "random"
            ):
                self.time_use_estimate["C"] = 0
                return
            if self.p["candidate_parent_algorithm"]["name"] == "optimal":
                # TODO: something about this.
                pass
            if self.p["run_mode"]["name"] == "budget":
                assert "t" in self.p["run_mode"]["params"]
                self.time_use_estimate["C"] = (
                    self.p["run_mode"]["params"]["t_share"]["C"]
                    * self.p["run_mode"]["params"]["t"]
                )

            estimate_candidate_search_time_use = False
            try:
                # checking if k is given by user
                self.p_user["candidate_parent_algorithm"]["params"]["K_f"]
                estimate_candidate_search_time_use = True
            except KeyError:
                try:
                    self.p["candidate_parent_algorithm"]["params"]["K_f"]
                    if self.p["run_mode"]["name"] != "budget":
                        estimate_candidate_search_time_use = True
                except KeyError:
                    pass

            if estimate_candidate_search_time_use:
                assert self.p["candidate_parent_algorithm"]["name"] == "greedy"
                ls = LocalScore(
                    data=self.data,
                    score=self.p["score"],
                    prior=self.p["structure_prior"],
                    maxid=self.p["constraints"]["max_id"],
                )
                t0 = time.time()
                params = dict(self.p["candidate_parent_algorithm"]["params"])
                params["K_f"] = self.p["constraints"]["K"] - 2
                cpa["greedy"](
                    self.p["constraints"]["K"],
                    scores=ls,
                    data=self.data,
                    params=params,
                )
                self.time_use_estimate["C"] = (time.time() - t0) * 1.9 ** (
                    self.p["constraints"]["K"]
                    - 2
                    - self.p["candidate_parent_algorithm"]["params"]["K_f"]
                )

    def _validate_parameters(self):
        try:
            self.p_user["mcmc"]["initial_rootpartition"]
            validate.rootpartition(
                self.p_user["mcmc"]["initial_rootpartition"]
            )
        except KeyError:
            pass
        validate.run_mode_args(self.p_user["run_mode"])
        validate.mcmc_args(self.p_user["mcmc"])
        validate.metropolis_coupling_args(self.p_user["metropolis_coupling"])
        validate.score_args(self.p_user["score"])
        validate.structure_prior_args(self.p_user["structure_prior"])
        validate.constraints_args(self.p_user["constraints"])
        validate.catastrophic_cancellation_args(
            self.p_user["catastrophic_cancellation"]
        )
        validate.logging_args(self.p_user["logging"])
        # Ensure candidate parents are set only by one of the three ways and
        # remove all except the used param from self.p_user.
        # If none of the ways are set use the defaults for
        # candidate_parent_algorithm.
        # Finally, validate the used way.
        alt_candp_params = [
            "candidate_parent_algorithm",
            "candidate_parents_path",
            "candidate_parents",
        ]
        validate.max_n_truthy(
            1,
            [self.p_user[k] for k in alt_candp_params],
            msg=f"only one of {alt_candp_params} can be set",
        )
        removed = list()
        for k in alt_candp_params:
            if not bool(self.p_user[k]):
                del self.p_user[k]
                removed.append(k)
        if len(removed) == 3:
            self.p_user[alt_candp_params[0]] = dict()
        if alt_candp_params[0] in self.p_user:
            validate.candidate_parent_algorithm_args(
                self.p_user[alt_candp_params[0]]
            )
        if alt_candp_params[1] in self.p_user:
            validate.is_string(
                self.p_user[alt_candp_params[1]],
                msg=f"'{alt_candp_params[1]}' should be string",
            )
            self.p_user["constraints"]["K"] = len(
                read_candidates(self.p_user[alt_candp_params[1]])[0]
            )
        if alt_candp_params[2] in self.p_user:
            validate.candidates(self.p_user[alt_candp_params[2]])
            self.p_user["constraints"]["K"] = len(
                self.p_user[alt_candp_params[2]][0]
            )

    def _populate_default_parameters(self):
        # Some defaults are defined as functions of data or other parameters.
        # Evaluate the functions here.
        DBUG(str(psutil.cpu_freq()))
        DBUG(str(psutil.virtual_memory()))
        self.default["run_mode"] = self.default["run_mode"](
            self.p["run_mode"].get("name")
        )
        if self.default["run_mode"]["name"] == "budget":
            self.default["run_mode"]["params"]["t"] = self.default["run_mode"][
                "params"
            ]["t"](self.data)
        self.default["constraints"] = self.default["constraints"](
            self.default["run_mode"]["name"]
        )
        if self.default["run_mode"]["name"] == "normal":
            self.default["constraints"]["K"] = self.default["constraints"][
                "K"
            ](self.data)
            self.default["constraints"]["d"] = self.default["constraints"][
                "d"
            ](self.data)
        self.default["score"] = self.default["score"](self.data)
        self.default["metropolis_coupling"] = self.default[
            "metropolis_coupling"
        ](self.p["metropolis_coupling"].get("name"))
        self.default["candidate_parent_algorithm"] = self.default[
            "candidate_parent_algorithm"
        ](self.default["run_mode"]["name"])

    def _complete_user_given_parameters(self):
        def complete(default, p):
            if all(type(p[k]) != dict for k in p):
                return dict(default, **p)
            for k in p:
                if type(p[k]) == dict:
                    p[k] = complete(default[k], p[k])
            return dict(default, **p)

        for k in self.p:
            if k not in self.default:
                continue
            if (
                type(self.p[k]) == dict
                and "name" in self.p[k]
                and self.p[k]["name"] != self.default[k]["name"]
            ):
                continue
            if type(self.p[k]) == dict:
                self.p[k] = complete(self.default[k], self.p[k])
            elif self.p[k] is None:
                self.p[k] = self.default[k]

    # def _adjust_inconsistent_parameters(self):
    #     iters = self.p["mcmc"]["iters"]
    #     M = self.p["mc3"].get("M", 1)
    #     burnin = self.p["mcmc"]["burnin"]
    #     n_dags = self.p["mcmc"]["n_dags"]
    #     self.p["mcmc"]["iters"] = iters // M * M
    #     self.p["mcmc"]["n_dags"] = min(
    #         (iters - int(iters * burnin)) // M, n_dags
    #     )
    #     self.adjusted = (
    #         self.p["mcmc"]["iters"] != iters,
    #         self.p["mcmc"]["n_dags"] != n_dags,
    #     )

    @staticmethod
    def mem_estimate(
        n, K, d, n_cc, a=4.43449129e02, b=3.22309337e-05, c=4.98862512e-04
    ):
        def n_psets(n, K, d):
            return sum(comb(n - 1, i) - comb(K, i) for i in range(1, d + 1))

        return a + b * n * (n_psets(n, K, d) + 2 ** K) + c * n_cc

    def _adjust_to_mem_budget(self, budget):
        def n_cc(K):
            return sum(self.est["n_cc_v"][v](K) for v in range(self.data.n))

        n = self.data.n
        K = self.p["constraints"]["K"]
        d = self.p["constraints"]["d"]

        memtable = (
            np.dstack(
                np.meshgrid(np.arange(1, K + 1), np.arange(d + 1), 0)
            ).reshape(-1, 3)
            # .astype(np.float32)
        )
        for i in range(len(memtable)):
            K, d, _ = memtable[i]
            memtable[i, 2] = budget - self.mem_estimate(n, K, d, n_cc(K))

        conditions = list()
        if "d" in self.p_user["constraints"]:
            conditions.append(
                memtable[:, 1] == self.p_user["constraints"]["d"]
            )
        elif "d_min" in self.p["constraints"]:
            conditions.append(memtable[:, 1] >= self.p["constraints"]["d_min"])
        if "K" in self.p_user["constraints"]:
            conditions.append(
                memtable[:, 0] == self.p_user["constraints"]["K"]
            )
        elif "K_min" in self.p["constraints"]:
            conditions.append(memtable[:, 0] >= self.p["constraints"]["K_min"])

        memtable = memtable[np.logical_and.reduce(conditions)]
        DBUG(str(memtable))

        if sum(memtable[:, 2] > 0) > 0:
            memtable = memtable[memtable[:, 2] > 0]
            memtable = memtable[memtable[:, 2].argsort()]
            K, d, _ = memtable[0]
        else:
            memtable = memtable[memtable[:, 2].argsort()[::-1]]
            K, d, _ = memtable[0]

        self.p["constraints"]["K"] = K
        self.p["constraints"]["d"] = d

    def prerun(self, K_max):
        self.prstats = dict()
        K_max = min(K_max, self.data.n - 3)
        for K in range(1, K_max + 1):
            params = {
                "constraints": {"K": K},
                "candidate_parent_algorithm": {"name": "random"},
                "logging": {"silent": True},
            }
            g = Gadget(data=self.data, **params, is_prerun=True)
            g._find_candidate_parents()
            t1 = time.perf_counter_ns()
            g._precompute_scores_for_all_candidate_psets()
            t1 = (time.perf_counter_ns() - t1) / 10 ** 9
            t2 = time.perf_counter_ns()
            g._init_crs()
            t2 = (time.perf_counter_ns() - t2) / 10 ** 9
            t3 = time.perf_counter_ns()
            g.c_r_score.precompute_tau_simple()
            t3 = (time.perf_counter_ns() - t3) / 10 ** 9
            t4 = g.c_r_score.precompute_tau_cc_basecases()
            cc_basecases = g.c_r_score.number_of_scoresums_in_cache()
            t5 = g.c_r_score.precompute_tau_cc()
            cc = g.c_r_score.number_of_scoresums_in_cache()
            self.prstats[K] = dict()
            self.prstats[K]["cc_basecases"] = cc_basecases
            self.prstats[K]["cc_total"] = cc
            self.prstats[K]["t_score_psets"] = t1
            self.prstats[K]["t_prune_scores"] = t2
            self.prstats[K]["t_score_sums"] = t3
            self.prstats[K]["t_cc_basecases"] = t4
            self.prstats[K]["t_cc"] = t5
        # NOTE: This is an attempt to fix some sort of a memory/cython related
        # bug that causes self.c_r_score.sum(v, U_bm, T_bm) to sometimes output
        # strange values (e.g. 90278873608348.86) presumably due to some kind
        # of memory overflow problem. The problem at least in some runs also
        # seemed to disappaear from sight after adding prints that really
        # should not solve anything (e.g. print(self.score_array) or prints to
        # see whether Cython and/or C++ constructors/destructors are
        # called). I'm guessing it might have something to do with memory not
        # freed properly, so trying explicit garbage collection. Also this
        # seemed to fix the issue, but whether it really did is unsure.
        gc.collect()
        DBUG(f"prerun : self.prstats = {self.prstats}")
        self._init_pred(1, K_max)

    def _init_pred(self, K_low, K_high):

        exp_base = dict(
            t_prune_scores=2.35,
            t_cc_basecases=2.2,
            cc_basecases=2.2,
            t_cc=2.2,
            cc_total=2.2,
        )
        n = self.data.n
        cc_max = np.floor(
            self.p["catastrophic_cancellation"]["cache_size"] / n
        )
        Ks = np.arange(K_low, K_high + 1)
        prerun = self.prstats

        est = dict()

        y = 1 / (
            np.array([prerun[K]["t_score_psets"] for K in Ks]) / (2 ** Ks * n)
        )
        f = fit_linreg(Ks[-2:], y[-2:])[0]
        est["t_score_psets"] = lambda K: (1 / f(K)) * 2 ** K * n

        y = np.array([prerun[K]["t_prune_scores"] for K in Ks])
        est["t_prune_scores"] = fit_linreg(
            Ks, y, lambda K: exp_base["t_prune_scores"] ** K
        )[0]

        y = np.array([prerun[K]["t_score_sums"] for K in Ks])
        est["t_score_sums"] = fit_linreg(Ks, y, lambda K: 2 ** K * K ** 2)[0]

        est["t_cc_basecases_v"] = dict()
        est["t_cc_v"] = dict()
        est["n_cc_v"] = dict()
        for v in range(n):
            y = np.array([prerun[K]["t_cc_basecases"][v] for K in Ks])
            f_t_cc_bc = fit_linreg(
                Ks, y, lambda K: exp_base["t_cc_basecases"] ** K
            )[0]

            y = np.array([prerun[K]["cc_basecases"][v] for K in Ks])
            f_tmp, a, b = fit_linreg(
                Ks, y, lambda K: exp_base["cc_basecases"] ** K
            )[:-1]
            f_n_cc_bc = (
                lambda f_tmp: lambda K: max(0.0, min(cc_max, f_tmp(K)))
            )(f_tmp)

            # The lowest K for which it is estimated the number of cc basecases
            # will hit the cache size limit
            K_ceil = np.inf
            if b > 0:
                K_ceil = np.ceil(
                    np.log((cc_max - a) / b) / np.log(exp_base["cc_basecases"])
                )
            est["t_cc_basecases_v"][v] = (
                lambda f_t_cc_bc, K_ceil: lambda K: max(
                    0.0, f_t_cc_bc(min(K, K_ceil))
                )
            )(f_t_cc_bc, K_ceil)

            y = np.array([prerun[K]["cc_total"][v] for K in Ks])
            f_tmp = fit_linreg(Ks, y, lambda K: exp_base["cc_total"] ** K)[0]
            f_n_cc = (lambda f_tmp: lambda K: max(0.0, min(cc_max, f_tmp(K))))(
                f_tmp
            )
            est["n_cc_v"][v] = f_n_cc

            f_n_cc_nonbc = (
                lambda f_n_cc, f_n_cc_bc: lambda K: f_n_cc(K) - f_n_cc_bc(K)
            )(f_n_cc, f_n_cc_bc)

            y = np.array([prerun[K]["t_cc"][v] for K in Ks])
            pred_time = fit_linreg(Ks, y, lambda K: exp_base["t_cc"] ** K)[0]
            est["t_cc_v"][v] = (
                lambda f_n_cc_bc, f_n_cc_nonbc, pred_time: lambda K: 0
                if f_n_cc_bc(K) == cc_max and f_n_cc_nonbc(K) == 0
                else pred_time(K)
            )(f_n_cc_bc, f_n_cc_nonbc, pred_time)

        est["t_cc_basecases"] = lambda K: sum(
            [est["t_cc_basecases_v"][v](K) for v in range(n)]
        )
        est["t_cc"] = lambda K: sum([est["t_cc_v"][v](K) for v in range(n)])
        self.est = est

    def __getitem__(self, key):
        return self.p[key]

    def __contains__(self, key):
        return key in self.p

    def pred_time_use_d(self, d):
        # predicts time use for d by extrapolating from d-1
        if d == 0:
            return 0

        K_max = 25
        C = {
            v: tuple([u for u in range(self.data.n) if u != v])
            for v in range(self.data.n)
        }
        C = {v: C[v][:K_max] for v in C}
        ls = LocalScore(
            data=self.data,
            score=self.p["score"],
            prior=self.p["structure_prior"],
            maxid=self.p["constraints"]["max_id"],
        )

        t0 = time.time()

        if d == 1:
            ls.complement_psets_and_scores(0, C, d)
            return (time.time() - t0) * self.data.n

        ls.complement_psets_and_scores(0, C, d - 1)
        return (
            (time.time() - t0)
            * comb(self.data.n, d)
            / comb(self.data.n, d - 1)
            * self.data.n
        )

    def adjust_to_time_budget_d(self, budget, d_min=0):
        pred = dict()
        d = max(0, d_min)
        pred[d] = self.pred_time_use_d(d)
        pred_next = self.pred_time_use_d(d + 1)
        while d + 1 < self.data.n and pred_next < budget:
            d += 1
            pred[d] = pred_next
            pred_next = self.pred_time_use_d(d + 1)
        return d, pred

    def pred_time_use_K(self, K):
        t = [
            self.est[k](K)
            for k in [
                "t_score_psets",
                "t_prune_scores",
                "t_score_sums",
                "t_cc_basecases",
                "t_cc",
            ]
        ]
        DBUG(f"_pred_K_time_use : K = {K}, t = {t}")
        return sum(t)

    def adjust_to_time_budget_K(self, budget, K_min=1):
        pred = dict()
        K = max(1, K_min)
        pred[K] = self.pred_time_use_K(K)
        pred_next = self.pred_time_use_K(K + 1)
        while K + 1 < self.data.n and pred_next < budget:
            K += 1
            pred[K] = pred_next
            pred_next = self.pred_time_use_K(K + 1)
        return K, pred

    def left(self):
        return self.p["run_mode"]["params"]["t"] - (time.time() - self.t0)


class Logger:
    def __init__(self, *, logfile, mode="a", overwrite=False):
        self._mode = mode
        # No output.
        if logfile is None:
            self._logfile = os.devnull
        # Output to file.
        elif type(logfile) in {pathlib.PosixPath, pathlib.WindowsPath}:
            if logfile.is_file():
                if not overwrite:
                    raise FileExistsError(f"{logfile} exists.")
                else:
                    logfile.unlink()
            else:
                logfile.parent.mkdir(parents=True, exist_ok=True)
            self._logfile = logfile
        # Output to stdout.
        elif logfile == sys.stdout:
            self._logfile = logfile
        else:
            raise TypeError(
                "logfile should be either None, PosixPath, pathlib.WindowsPath"
                f" or sys.stdout: {logfile} of type {type(logfile)} given."
            )

    def __call__(self, string):
        if self._logfile == sys.stdout:
            print(string, file=self._logfile, flush=True)
        else:
            with open(self._logfile, self._mode) as f:
                print(string, file=f, flush=True)

    def unlink(self):
        if type(self._logfile) == pathlib.PosixPath:
            self._logfile.unlink()

    def silent(self):
        return self._logfile == os.devnull

    def dict(self, data):
        def pretty_dict(d, n=0, string=""):
            for k in d:
                if type(d[k]) in (dict, Stats):
                    string += f"{' '*n}{k}\n"
                else:
                    string += f"{' '*n}{k}: {d[k]}\n"
                if type(d[k]) in (dict, Stats):
                    string += pretty_dict(d[k], n=n + 2)
            return string

        self(pretty_dict(data))

    def numpy(self, array, fmt="%.2f"):
        if self._logfile == sys.stdout:
            np.savetxt(self._logfile, array, fmt=fmt)
        else:
            with open(self._logfile, self._mode) as f:
                np.savetxt(f, array, fmt=fmt)

    def br(self, n=1):
        self("\n" * (n - 1))

    def fraction(self, n, d):
        if d == 0:
            return ""
        return n / d


class GadgetLogger(Logger):
    """Stuff for printing stuff."""

    def __init__(self, gadget):

        super().__init__(
            logfile=None if gadget.p["logging"]["silent"] else sys.stdout,
        )

        log_params = gadget.p["logging"]
        self.verbose_logger = dict()
        # if log_params["verbose_prefix"] is not None:

        for verbose_output in [
            "score",
            "inv_temp",
            "mc3_swap_prob",
            "mc3_local_swap_prob",
        ]:
            self.verbose_logger[verbose_output] = dict()
            for i in range(gadget.p["mcmc"]["n_independent"]):
                if log_params["verbose_prefix"] is not None:
                    verbose_log_path = Path(
                        str(Path(log_params["verbose_prefix"]).absolute())
                        + f".{i}.{verbose_output}.tmp"
                    )
                    # Just to raise error if the final log files
                    # without .tmp suffix exist when overwrite=False
                    Logger(
                        logfile=verbose_log_path.with_suffix(""),
                        overwrite=log_params["overwrite"],
                    )
                    self.verbose_logger[verbose_output][i] = Logger(
                        logfile=verbose_log_path,
                        overwrite=log_params["overwrite"],
                    )
                else:
                    self.verbose_logger[verbose_output][i] = Logger(
                        logfile=None,
                    )

        self._time_last_periodic_stats = time.time()
        self._running_sec_num = 0
        self._linewidth = 80
        self.g = gadget

    def finalize(self):
        if self.g.p["logging"]["verbose_prefix"] is None:
            return
        for i in range(self.g.p["mcmc"]["n_independent"]):
            M_max = 0

            with open(self.verbose_logger["score"][i]._logfile, "r") as f:
                for line in f:
                    M_max = max(M_max, line.count(" ") + 1)

            for verbose_output in self.verbose_logger:
                with open(
                    self.verbose_logger[verbose_output][
                        i
                    ]._logfile.with_suffix(""),
                    "w",
                ) as f_output:
                    f_output.write(" ".join(map(str, range(M_max))) + "\n")
                    with open(
                        self.verbose_logger[verbose_output][i]._logfile, "r"
                    ) as f_input:
                        for line in f_input:
                            f_output.write(line)
                    self.verbose_logger[verbose_output][i].unlink()

    def h(self, title, secnum=True):
        if secnum:
            self._running_sec_num += 1
            end = "." * (self._linewidth - len(title) - 4)
            title = f"{self._running_sec_num}. {title} {end}"
        else:
            end = "." * (self._linewidth - len(title) - 1)
            title = f"{title} {end}"
        self(title)
        self.br()

    def periodic_stats(self):

        if (
            time.time() - self._time_last_periodic_stats
            < self.g.p["logging"]["period"]
        ):
            return False

        self._time_last_periodic_stats = time.time()

        chain_stats = [indep_chain.describe() for indep_chain in self.g.mcmc]

        self.progress(chain_stats)
        if plot_trace:
            self.br()
            self.plot_score_trace()
        else:
            self.last_root_partition_scores()
        self.move_probs(chain_stats)
        self.br()
        return True

    def move_probs(self, stats):
        self.h("Move acceptance probabilities", secnum=False)
        for i, s in enumerate(stats):
            self(f"Chain {i+1}:")
            self(" " * 20 + "inv_temp")
            msg_tmpl = "{:<17.17} |" + " {:<5.5}" * s["M"]
            temps = [1.0]
            temps_labels = [1.0]
            temps = sorted(s["inv_temp"], reverse=True)
            temps_labels = [round(t, 2) for t in temps]
            moves = s["accept_prob"].keys()
            msg = msg_tmpl.format("move", *temps_labels) + "\n"
            msg = msg.replace("|", " ")
            hr = ["-"] * self._linewidth
            hr[18] = "+"
            hr = "".join(hr)
            msg += hr
            self(msg)
            for m in moves:
                ar = [
                    round(r, 2) if not np.isnan(r) else ""
                    for r in s["accept_prob"][m]
                ]
                msg = msg_tmpl.format(m, *ar)
                self(msg)

            self.br()

    def run_stats(self):

        stats = self.g._stats
        w_target_iters = str(
            max(
                len("1.0_iters"),
                len(str(stats["mcmc"]["target_chain_iter_count"])),
            )
            + 2
        )
        w_iters = str(
            max(
                len("iters"),
                len(str(stats["mcmc"]["iter_count"])),
            )
            + 2
        )
        w_seconds = str(len(str(int(stats["mcmc"]["time_used"]))) + 2)
        msg_title_tmpl = (
            "{:<12}   {:<"
            + w_target_iters
            + "}{:<5}   {:<"
            + w_iters
            + "}{:<6}   {:<"
            + w_seconds
            + "}{:<8}{:<8} {:<8}"
        )
        msg_tmpl = (
            "{:<12} | {:<"
            + w_target_iters
            + "}{:<6.2} | {:<"
            + w_iters
            + "}{:<6.2} | {:<"
            + w_seconds
            + "}{:<8.2}{:<8.2} {:<8.2}"
        )

        msg = (
            msg_title_tmpl.format(
                "phase",
                "1.0_iters",
                "/total",
                "iters",
                "/total",
                "s",
                "/total",
                "/iter",
                "/1.0_iter",
            )
            + "\n"
        )

        hr = ["-"] * self._linewidth
        hr[13] = "+"
        hr[33] = "+"
        hr[50] = "+"
        msg += "".join(hr) + "\n"

        for phase in ["burnin", "after_burnin"]:
            msg += (
                msg_tmpl.format(
                    phase,
                    stats[phase]["target_chain_iter_count"],
                    self.fraction(
                        stats[phase]["target_chain_iter_count"],
                        stats["mcmc"]["target_chain_iter_count"],
                    ),
                    stats[phase]["iter_count"],
                    self.fraction(
                        stats[phase]["iter_count"], stats["mcmc"]["iter_count"]
                    ),
                    round(stats[phase]["time_used"]),
                    self.fraction(
                        stats[phase]["time_used"], stats[phase]["time_used"]
                    ),
                    self.fraction(
                        stats[phase]["time_used"], stats[phase]["iter_count"]
                    ),
                    self.fraction(
                        stats[phase]["time_used"],
                        stats[phase]["target_chain_iter_count"],
                    ),
                )
                + "\n"
            )
        msg += msg_tmpl.format(
            "mcmc total",
            stats["mcmc"]["target_chain_iter_count"],
            1.0,
            stats["mcmc"]["iter_count"],
            1.0,
            round(stats["mcmc"]["time_used"]),
            1.0,
            self.fraction(
                stats["mcmc"]["time_used"], stats["mcmc"]["iter_count"]
            ),
            self.fraction(
                stats["mcmc"]["time_used"],
                stats["mcmc"]["target_chain_iter_count"],
            ),
        )
        self(msg)
        self.br()

    def progress(self, stats):

        target_chain_iter_count = sum(
            s["target_chain_iter_count"] for s in stats
        )
        iter_count = sum(s["iter_count"] for s in stats)
        n_independent = self.g.p["mcmc"]["n_independent"]

        percentage = ""
        # stats = self.g._stats
        if self.g.p["run_mode"]["name"] == "normal":
            percentage = round(
                100
                * target_chain_iter_count
                / (
                    self.g.p["run_mode"]["params"]["n_target_chain_iters"]
                    * n_independent
                )
            )
        elif self.g.p["run_mode"]["name"] == "budget":
            percentage = round(
                100
                * (time.time() - self.g._stats["mcmc"]["time_start"])
                / self.g.p.budget["mcmc"]
            )

        self.h(
            f"PROGRESS: {percentage}{'%' if percentage else ''}", secnum=False
        )
        self(
            f"- target temperature:  {target_chain_iter_count} "
            f"iters (in {len(stats)} independent chain(s))"
        )
        self(f"- all temperatures:    {iter_count} iters")
        self.br()

    def last_root_partition_scores(self):
        self.h("Last root-partition scores", secnum=False)
        for i in range(self.g.p["mcmc"]["n_independent"]):
            score = self.g._verbose["score"][i][
                self.g._stats["mcmc"]["target_chain_iter_count"]
                % self.g._verbose_len
            ]
            if len(score) > 0:
                score = round(score[0], 2)
            else:
                score = "-"
            msg = f"Chain {i+1}: {score}"
            self(msg)
        self.br()

    def plot_score_trace(self):
        self.h("Root-partition score traces", secnum=False)
        t = self.g._stats["mcmc"]["target_chain_iter_count"]
        r = self.g._verbose_len
        R_scores = self.g._verbose["score"]
        plt.clear_plot()
        for i in range(self.g.p["mcmc"]["n_independent"]):
            if t < r:
                to_plot = np.array([s[0] for s in R_scores[i][:t]])
                plt.scatter(
                    to_plot,
                    label=str(i),
                    color=i + 1,
                    marker="•",
                )
            else:
                to_plot = np.array([s[0] for s in R_scores[i]])
                to_plot = to_plot[np.r_[(t % r) : r, 0 : (t % r)]]
                plt.scatter(
                    to_plot,
                    label=str(i),
                    color=i + 1,
                    marker="dot",
                )
        plt.plotsize(80, 20)
        plt.yfrequency(4)
        if t < r:
            xticks = [int(w * t) for w in np.arange(0, 1 + 1 / 3, 1 / 3)]
            xlabels = [str(round(x / r, 1)) + "k" for x in xticks]
        else:
            xticks = np.array([r // 3 * i for i in range(4)])
            xlabels = [
                str(round((x + t) / r, 1)) + "k" for x in -1 * xticks[::-1]
            ]
        plt.xticks(xticks, xlabels)
        plt.canvas_color("default")
        plt.axes_color("default")
        plt.ticks_color("default")
        self(plt.build())
        self.br()


class Gadget:
    """Class implementing the Gadget pipeline for MCMC sampling from
    the structure posterior of DAG models. The user interface consists
    of:

    1. Constructor for setting all the parameters.
    2. :py:meth:`.sample()` method which runs the MCMC chain and
       returns the sampled DAGs along with meta data.

    All the constructor arguments are keyword arguments, i.e., the
    **data** argument should be given as ``data=data``, etc. Only the
    data argument is required; other arguments have some more or less
    sensible defaults.

    There are many parameters that can be adjusted. To make managing the
    parameters easier, most of them are grouped into dictionary objects around
    some common theme.

    In this documentation nested parameters within the dictionaries are
    referenced as **outer:inner** (e.g., ``silent`` in ``logging={'silent':
    True}``, corresponds to **logging:silent**) or as:

    - **outer**

      - **inner**

    Some of the parameters have the keys **name** and **params** (i.e.,
    ``foo={'name': 'bar', 'params': {'baz': 'qux'}}``), **params** being a
    dictionary the structure of which depends on the value of
    **name**. When describing such a parameter the following style is used:

    - **foo**

      *Default*: ``bar``

      - **name**: ``bar``

        General description of ``bar``.

        **params**

        - **baz**: Description of the parameter **baz**.

    To only adjust some parameter within a dictionary argument while
    keeping the others at default, it suffices to set the one
    parameter. For example, to set the number of candidate parents
    **constraints:K** to some value :math:`K` but let the maximum size of
    arbitrary parent sets **constraints:d** be determined automatically you
    should construct the **constraints** parameter as

    >>> Gadget(data=data, constraints={"K": K}).

    The following describes all Gadget constructor parameters.

    - **data**

      Data to run the MCMC simulation on. Accepts any valid constructor
      argument for a :py:class:`.Data` object.

    - **run_mode**

      Which mode to run Gadget in.

      *Default*: ``budget``.

      - **name**: ``normal``

        MCMC simulation is run for a predetermined number of iteration steps.

        **params**

        - **n_target_chain_iters**: For how many iterations to run the
          target chain (i.e., the unheated chain; see
          **metropolis_coupling**).

      - **name**: ``budget``

        MCMC simulation is run until a given time budget is used up. A
        fraction of the time budget is allocated for precomputations, with
        the remainder being used on the Markov chain simulation itself.

        There are three precomputation phases:

        1. Selecting candidate parents.
        2. Building score sum data structures given the candidate parents.
        3. Computing scores of the parent sets for which we allow nodes
           outside of the candidates.

        The parameters principally governing the time use for each part
        are, correspondingly, the number of nodes to add at the final step
        of the greedy candidate selection algorithm
        (**candidate_parent_algorithm:params:K_f**), the number of
        candidate parents (**constraints:K**), and the maximum size of the
        parent sets for which nodes outside the chosen candidates are
        permitted (**constraints:d**). In budget mode, these parameters are
        set automatically, so as to use a predetermined fraction of the
        budget on each precomputation phase.

        With big time budgets the automatically selected parameters might
        result in infeasibly large memory requirements. To avoid this there is
        an additional memory budget parameter to cap **constraints:K** and
        **constraints:d**.

        If any of the parameters **constraints:K**, **constraints:d** or
        **candidate_parent_algorithm:params:K_f** are explicitly set, that
        parameter will not be programmatically adjusted. The parameters
        **constraints:min_K** and **constraints:min_d** can be used to set
        the minimum values for **constraints:K** and **constraints:d**,
        respectively.

        **params**

        - **t**: Time budget in seconds.

          *Default*: Roughly sensible amount of time as a function of data
          shape.

        - **mem**: Memory budget in megabytes.

          *Default*: Amount of memory available.

        - **t_share**: Dictionary with the keys ``C``, ``K``, and ``d``
          corresponding to the above precomputation phases 1-3,
          respectively, with float values determining the fraction of the
          overall budget to use on the particular phase.

          *Default*: ``{'C': 1 / 9, 'K': 1 / 9, 'd': 1 / 9}``

      - **name**: ``anytime``

        If ran in this mode the first ``CTRL-C`` after calling
        :py:meth:`.sample()` stops the burn-in phase and starts sampling
        DAGs. The second ``CTRL-C`` stops the sampling. DAG sampling first
        accumulates up to 2 :math:`\\cdot` **mcmc**:**n_dags** - 1 DAGs with
        thinning 1 (i.e., a DAG is sampled for each sampled root-partition),
        then each time the number of DAGs reaches 2 :math:`\\cdot`
        **mcmc**:**n_dags** the thinning is doubled and every 2nd already
        sampled DAG is deleted.

    - **mcmc**

      General Markov Chain Monte Carlo arguments.

      - **initial_rootpartition**

        The root-partition to initialize the MCMC chain(s) with. If not set
        the chains will start from a random state.

        The root-partition format is a list partitioning integers 0..n to
        sets.

        *Default*: Not set.

      - **n_independent**: Number of independent chains to run. For each
        independent chain there will be **metropolis_coupling:M**
        coupled chains run in logical parallel. The DAGs are sampled evenly
        from each unheated chain (see **metropolis_coupling**).

        Values greater than 1 are mostly useful for analyzing mixing.

        *Default*: 1

      - **burnin**: The fraction of **run_mode:params:n_target_chain_iters**
        (**run_mode** ``normal``), or the fraction of time budget
        remaining after precomputations (**run_mode** ``budget``) to use
        on burn-in.

        *Default*: 0.5

      - **n_dags**: The (approximate) number of DAGs to sample.

        *Default*: 10 000

      - **move_weights**: Dictionary with the keys ``R_split_merge``,
        ``R_swap_node_pair`` and ``DAG_edge_reversal``. The values are
        integer weights specifying the unnormalized probability
        distribution from which the type of each proposed move is sampled.

        Note that when employing Metropolis coupling the edge reversal move
        is only available for the unheated chain (in heated chains its
        weight will be 0), and that the state swap move is proposed
        uniformly at random for any two adjacent chains always after each
        chain has first progressed by one step (i.e., its proposal
        probability cannot be adjusted).

        *Default*: ``{'R_split_merge': 1, 'R_swap_node_pair': 1,
        'DAG_edge_reversal': 2}``

    - **score**

      The score to use.

      *Default*: ``bdeu`` for discrete and ``bge`` for continuous data.

      - **name**: ``bdeu``

        Bayesian Dirichlet equivalent uniform.

        **params**

        - **ess**: Equivalent sample size.

          *Default*: 10

      - **name**: ``bge``

        Bayesian Gaussian equivalent.

    - **structure_prior**

      The modular structure prior to use. Modular structure priors are
      composed as a product of local factors, i.e., the prior for graph
      :math:`G` factorizes as

      .. math::

         P(G) \\propto \\prod_{i=1}^{n}\\rho_i(\\mathit{pa}(i)),

      where :math:`\\rho_i` are node specific factors, and
      :math:`\\mathit{pa}(i)` are the parents of node :math:`i` in :math:`G`.

      Two types of factors are implemented, dubbed ``fair`` and ``unif``.
      See :footcite:`eggeling:2019`.

      *Default*: ``fair``.

      - **name**: ``fair``

        Balances the probabilities of different indegrees (i.e., of
        :math:`|\\mathit{pa}(i)|`), with the factors taking the form

        .. math::

           \\rho_i(S) = 1\\big/\\binom{n-1}{|S|}.

      - **name**: ``unif``

        Uniform over different graphs, i.e., the factors are simply

        .. math::

           \\rho_i(S) = 1.

    - **constraints**

      Constraints on the explored DAG space.

      - **K**: Number of candidate parents per node.

        *Default*: :math:`\min(n-1,16)`, where :math:`n` is the number of
        nodes (**run_mode** ``normal`` or ``anytime``), or parameter not
        set (**run_mode** ``budget``).

      - **K_min**: Sets the minimum level for **K** if using **run_mode**
        ``budget``.

        *Default*: 1 (**run_mode** ``budget``), or parameter not
        set (**run_mode** ``normal`` or ``anytime``).

      - **d**: Maximum size of parent sets for which nodes outside of the
        candidates are allowed.

        *Default*: :math:`\min(n-1, 3)`, where :math:`n` is the number of
        nodes (**run_mode** ``normal`` or ``anytime``) or parameter not
        set (**run_mode** ``budget``).

      - **d_min**: Sets the minimum level for **d** if using **run_mode**
        ``budget``.

        *Default*: 2 (**run_mode** ``budget``), or parameter not
        set (**run_mode** ``normal`` or ``anytime``).

      - **max_id**: Maximum size of parent sets that are subsets of
        the candidate parents. Set to -1 for unlimited. There should
        be no reason to limit the size.

        *Default*: -1

    - **candidate_parent_algorithm**

      Algorithm to use for finding the candidate parents :math:`C = C_1C_2
      \\ldots C_n`, where :math:`C_i` is the set of candidates for node
      :math:`i`.

      Implemented algorithms are ``greedy``, ``random`` and ``optimal``.

      *Default*: ``greedy``

      - **name**: ``greedy``

        Iteratively, add a best node to the initially empty :math:`C_i`
        (given the already added ones), until :math:`|C_i|=K-K_f`. Finally
        add the :math:`K_f` best nodes.

        **params**

        - **association_measure**: The measure for *goodness* of a
          candidate parent :math:`j`. One of

          - ``score``: :math:`\\max_{S \\subseteq C_i}
            \\pi_i(S\\cup\{j\})`

          - ``gain``: :math:`\\max_{S \\subseteq C_i} \\pi_i(S\\cup\{j\})
            - \\pi_i(S)`,

          where :math:`\\pi_i(S)` is the local score of the parent set
          :math:`S` for node :math:`i`.

          *Default*: ``gain``.

        - **K_f**: The number of nodes to add at the final step of the
          algorithm. Higher values result in faster computation.

      - **name**: ``random``

        Select the candidates uniformly at random.

      - **name**: ``optimal``

        Select the candidates so as to maximize the posterior probability
        that :math:`\\mathit{pa}(i) \\subseteq C_i`. Only feasible up to a
        couple of dozen variables.

    - **metropolis_coupling**

      Metropolis coupling is implemented by running :math:`M` "heated" chains
      in parallel with their stationary distributions proportional to the
      :math:`\\beta_i\\text{th}` power of the posterior, with some appropriate
      *inverse temperatures* :math:`1 = \\beta_1 > \\beta_2 > \\ldots >
      \\beta_M \\geq 0`. Periodically, a swap of states between two adjacent
      chains is proposed and accepted with a certain probability.

      Two modes available: ``static`` and ``adaptive``. To *not use*
      Metropolis coupling use ``static`` with **params:M** set to 1.

      *Default*: ``adaptive``.

      - **name**: ``static``

        Metropolis coupling with fixed number of chains and temperatures. The
        lowest inverse temperature is exactly zero, corresponding to the
        uniform distribution.

        **params**

        - **M**: The number of coupled chains to run. Set to 1 to disable
          Metropolis coupling.

          *Default*: 16

        - **heating**: The functional form of the sequence of inverse
          temperatures. One of:

          - ``linear``: :math:`\\beta_i = (M - i) \\big/ (M - 1)`

          - ``quadratic``: :math:`\\beta_i = 1- ((i-1) \\big/ (M - 1))^2`

          - ``sigmoid``: :math:`\\beta_1=1,\\beta_{i:i\\not\\in \{1,M\}} =
            \\frac{1}{1+e^{(M-1)/2-(M-i)}},\\beta_M=0`

          *Default*: ``linear``

        - **sliding_window**: The number of the most recent swap proposals from
          which a local swap probability is computed.

          *Default*: 1000

      - **name**: ``adaptive``

        The number of chains is initialized to :math:`M=2` (**params:M**). The
        inverse temperatures are defined as

        .. math::

               \\beta_k = 1 / (1 + \\sum_{k`=1}^{k-1}{\\delta_{k`}}),

        where :math:`\\delta_k, k = 1,...,M-1` are auxiliary temperature delta
        variables with an initial value :math:`\\delta_1 = 0.5`
        (**params:delta_init**) yielding :math:`\\beta_1 = 1` and
        :math:`\\beta_2 \\approx 0.67`. Then, the following process to adjust
        :math:`M` and :math:`\\beta_1,\\beta_2,\\ldots,\\beta_M` is repeated
        indefinitely:

        1. Each of the chains are sampled for a specific number of iterations
        (**params:update_n**), with the number multiplied on each cycle by a
        *slowdown factor* (**params:slowdown**). Then the empirical probability
        :math:`p_{s}(k)` for a proposed swap between chains :math:`k` and
        :math:`k+1` to have been accepted is computed. In addition to computing
        the probability given the full history of the chain it is also computed
        for a specific number of the most recent proposals
        (**params:sliding_window**), with the latter probablity
        denoted by :math:`p_{sw}(k)`.

        2. The temperature delta variables for :math:`k=1,...,M-1` are updated
        as

        .. math::

               \\delta_k := \\delta_k (1 + \\frac{p_{sw}(k) - p_{t}}{s}),

        where :math:`p_{t}` (**params:p_target**) is the target
        swap probability, and :math:`s` is a *smoothing factor*
        (**params:smoothing**).

        3. Finally, denoting the number of chains for which :math:`p_{s}(k) >
        0.9` by :math:`r`, either

        - a new chain is added (if :math:`r = 0`),
        - the number of chains remains unchanged (if :math:`r = 1`),
        - a chain is deleted (if :math:`r > 1` and :math:`M` > 2).

        On addition the new chain takes the index position :math:`M+1`, its
        state is initialized to that of chain :math:`M` and :math:`t_M` is set
        to value :math:`2(M+1)`. On deletion the chain at the index position
        :math:`M` is removed. Here :math:`M` refers to the number of chains
        prior to the addition or removal.

        **params**

        - **M**: The initial number of coupled chains to run.

          *Default*: 2

        - **p_target**: Target local swap probability

          *Default*: 0.234 :footcite:`roberts:1997`

        - **delta_init**: The initial temperature difference between adjacent
          chains.

          *Default*: 0.5

        - **sliding_window**: The number of most recent swap proposals from
          which the local swap probability is computed.

         *Default*: 1000

        - **update_n**: Temperatures are updated every nth iteration, with
          n set by this parameter. **update_n** is multiplied by **slowdown**
          on every temperature update event.

          *Default*: 100

        - **smoothing**: A parameter with effect on the magnitude of change in
          temperature update. See description above.

          *Default*: 1

        - **slowdown**: A parameter with effect on the frequency of temperature
          updates. See description above and **update_n**.

          *Default*: 1

    - **catastrophic_cancellation**: Catastrophic cancellation occurs when a
      score sum :math:`\\tau_i(U,T)` computed as :math:`\\tau_i(U) -
      \\tau_i(U \\setminus T)` evaluates to zero due to numerical reasons.

      - **tolerance**: how small should the absolute difference
        between two log score sums be in order for the subtraction
        to be determined to lead to catastrofic cancellation.

        *Default*: :math:`2^{-32}`.

      - **cache_size**: Maximum amount of score sums that cannot be
        computed through subtraction to be stored separately. If there is a
        lot of catastrofic cancellations, setting this value high can
        have a big impact on memory use.

        *Default*: :math:`10^7`

    - **pruning_tolerance**: Allowed relative error for a root-partition
      node score sum. Setting this to a positive value allows
      some candidate parent sets to be pruned, expediting parent
      set sampling.

      *Default*: 0.001.

    - **scoresum_tolerance**: Tolerated relative error when computing
      score sums from individual parent set local scores.

      *Default*: 0.01.

    - **candidate_parents_path**

      Alternatively to **candidate_parent_algorithm** the candidate parents
      can be precomputed and read from a file, with the path given under
      this parameter.

      In the expected format row numbers determine the nodes, while
      :math:`K` space separated numbers on the rows identify the candidate
      parents.

    - **candidate_parents**

      Alternatively to **candidate_parent_algorithm** the candidate parents
      can be precomputed and read from a Python dictionary given under this
      parameter.

      In the expected format integer keys determine the nodes, while the
      values are tuples with :math:`K` integers identifying the parents.

    - **logging**: Parameters determining the logging output.

      - **silent**: Whether to suppress logging output or not.

        *Default*: ``False``.

      - **period**: Interval in seconds for printing more statistics.

        *Default*: 15.

      - **verbose_prefix**: If set, more verbose output is created in files at
        ``<working directory>/prefix``. For example prefix ``x/y`` would create
        files with names starting with ``y`` in directory ``x``.

        *Default*: ``None``.

      - **overwrite**: If ``True`` files in ``verbose_prefix`` are
        overwritten if they exist.

        *Default*: ``False``.
    """

    def __init__(
        self,
        *,
        data,
        validate_params=True,
        is_prerun=False,
        run_mode=dict(),
        mcmc=dict(),
        score=dict(),
        structure_prior=dict(),
        constraints=dict(),
        candidate_parent_algorithm=dict(),
        metropolis_coupling=dict(),
        catastrophic_cancellation=dict(),
        pruning_tolerance=None,
        scoresum_tolerance=None,
        candidate_parents_path=None,
        candidate_parents=None,
        logging=dict(),
    ):

        # locals() has to be the first thing called in __init__.
        user_given_parameters = locals()
        del user_given_parameters["self"]

        self.t0 = time.time()

        self.p = GadgetParameters(**user_given_parameters)
        self.data = self.p.data

        self.log = GadgetLogger(self)
        log = self.log

        # Things collected along the running of the chain used e.g. in control
        # structures. Here to make the overall structure explicit.
        self._stats = dict(
            highest_scoring_rootpartition=None,
            candp=dict(time_used=0),
            crscore=dict(time_used=0),
            ccscore=dict(time_used=0),
            mcmc=dict(
                target_chain_iter_count=0,
                iter_count=0,
                deadline=None,
                time_start=None,
                time_used=0,
                time_per_dag=0,
                thinning=1,
            ),
            burnin=dict(
                target_chain_iter_count=0,
                iter_count=0,
                deadline=None,
                time_start=None,
                time_used=0,
            ),
            after_burnin=dict(
                target_chain_iter_count=0,
                iter_count=0,
                time_start=None,
                time_used=0,
            ),
        )

        # Number of iterations
        # 1. to plot in score trace
        # 2. specifying the size of in-memory verbose output arrays
        #    which are periodically stored to file
        self._verbose_len = 1000
        self._verbose = dict()
        for verbose_output in [
            "score",
            "inv_temp",
            "mc3_swap_prob",
            "mc3_local_swap_prob",
        ]:
            self._verbose[verbose_output] = {
                c: [np.array([]) for i in range(self._verbose_len)]
                for c in range(self.p["mcmc"]["n_independent"])
            }

        self.dags = list()
        self.dag_scores = list()

        log(f"sumu version: {__version__}")
        log(f"run started: {datetime.datetime.now()}")
        log.br()

        log.h("PROBLEM INSTANCE")
        log.dict(self.data.info)
        log.br()

        log.h("RUN PARAMETERS")
        log.dict(self.p.p)

        if not is_prerun:  # p.est does not exist for prerun
            precomp_time_estimate = round(
                (time.time() - self.p.t0)
                + self.p.time_use_estimate["K"][self.p["constraints"]["K"]]
                + self.p.time_use_estimate["d"][self.p["constraints"]["d"]]
                + self.p.time_use_estimate["C"]
            )
            cc_estimate = sum(
                self.p.est["n_cc_v"][v](self.p["constraints"]["K"])
                for v in range(self.data.n)
            )
            mem_use_estimate = round(
                self.p.mem_estimate(
                    self.data.n,
                    self.p["constraints"]["K"],
                    self.p["constraints"]["d"],
                    cc_estimate,
                )
            )

            if self.p["run_mode"]["name"] == "budget":
                budget_exceeded_msg = list()
                if "t" in self.p["run_mode"]["params"]:
                    if (
                        precomp_time_estimate
                        > self.p["run_mode"]["params"]["t"]
                    ):
                        budget_exceeded_msg.append(
                            "estimated time use "
                            "for precomputations exceeds budget: "
                            f"{precomp_time_estimate} > "
                            f"{self.p['run_mode']['params']['t']}"
                        )

                if "mem" in self.p["run_mode"]["params"]:
                    if mem_use_estimate > self.p["run_mode"]["params"]["mem"]:
                        budget_exceeded_msg.append(
                            "estimated memory use exceeds budget: "
                            f"{mem_use_estimate} > "
                            f"{self.p['run_mode']['params']['mem']}"
                        )

                for msg in budget_exceeded_msg:
                    log(msg)
                if len(budget_exceeded_msg) > 0:
                    log.br()
                    log("terminating ...")
                    exit()

            log(
                "estimated time use for precomputations: "
                f"{precomp_time_estimate} s"
            )
            log(f"estimated memory use: {mem_use_estimate} MB")
            log.br()

        self.precomputations_done = False

    def precompute(self):

        log = self.log
        K = self.p["constraints"]["K"]
        d = self.p["constraints"]["d"]

        log.h("FINDING CANDIDATE PARENTS")
        self._stats["candp"]["time_start"] = time.time()
        self._find_candidate_parents()
        self._stats["candp"]["time_used"] = (
            time.time() - self._stats["candp"]["time_start"]
        )
        log.numpy(self.C_array, "%i")
        log.br()
        try:
            # trying if all nested keys set
            self.p.p_user["candidate_parent_algorithm"]["params"]["K_f"]
            log(f"time predicted: {round(self.p.time_use_estimate['C'])}s")
        except KeyError:
            c = "candidate_parent_algorithm"  # to shorten next rows
            if (
                self.p["run_mode"]["name"] == "budget"
                and self.p[c]["name"] == "greedy"
            ):
                log.br()
                log(f"Adjusted for time budget: k = {stats['C']['K_f']}")
                log(
                    f"time budgeted: {round(self.p[c]['params']['t_budget'])}s"
                )
        log(f"time used: {round(self._stats['candp']['time_used'])}s")
        log.br(2)

        log.h("PRECOMPUTING SCORING STRUCTURES FOR CANDIDATE PARENT SETS")
        self._stats["crscore"]["time_start"] = time.time()
        self._precompute_scores_for_all_candidate_psets()
        self._precompute_candidate_restricted_scoring()
        n_cc = sum(self.c_r_score.number_of_scoresums_in_cache())
        log.br()
        log(f"Number of score sums stored in cc cache: {n_cc}")
        log.br()
        self._stats["crscore"]["time_used"] = (
            time.time() - self._stats["crscore"]["time_start"]
        )
        log(f"time predicted: {round(self.p.time_use_estimate['K'][K])}s")
        log(f"time used: {round(self._stats['crscore']['time_used'])}s")
        log.br(2)

        log.h("PRECOMPUTING SCORING STRUCTURES FOR COMPLEMENTARY PARENT SETS")
        self._stats["ccscore"]["time_start"] = time.time()
        self._precompute_candidate_complement_scoring()
        self._stats["ccscore"]["time_used"] = (
            time.time() - self._stats["ccscore"]["time_start"]
        )
        log(f"time predicted: {round(self.p.time_use_estimate['d'][d])}s")
        log(f"time used: {round(self._stats['ccscore']['time_used'])}s")
        log.br(2)

        self.precomputations_done = True

    def sample(self):

        if not self.precomputations_done:
            self.precompute()

        log = self.log

        log.h("RUNNING MCMC")
        self._stats["mcmc"]["time_start"] = time.time()
        self._mcmc_init()
        if self.p["run_mode"]["name"] == "normal" or (
            self.p["run_mode"]["name"] == "budget"
            and "t" not in self.p["run_mode"]["params"]
        ):
            self._mcmc_run_normal()
        if (
            self.p["run_mode"]["name"] == "budget"
            and "t" in self.p["run_mode"]["params"]
        ):
            self._mcmc_run_time_budget()
        if self.p["run_mode"]["name"] == "anytime":
            self._mcmc_run_anytime()
        self._stats["mcmc"]["time_used"] = (
            time.time() - self._stats["mcmc"]["time_start"]
        )
        log(f"time used: {round(self._stats['mcmc']['time_used'])}s")
        log.br(2)

        log.h("RUN STATISTICS")
        log.run_stats()
        log(f"no. dags sampled: {len(self.dags)}")

        log.finalize()

        return self.dags, dict(
            parameters=self.p.p,
            scores=self.dag_scores,
            candidates=self.C,
            mcmc=self.mcmc[0].describe(),
            highest_scoring_rootpartition=self._stats[
                "highest_scoring_rootpartition"
            ],
            stats=stats,  # TODO: Get rid of this?
        )

    def _find_candidate_parents(self):
        self.l_score = LocalScore(
            data=self.data,
            score=self.p["score"],
            prior=self.p["structure_prior"],
            maxid=self.p["constraints"]["max_id"],
        )

        if "candidate_parents_path" in self.p:
            self.C = validate.candidates(
                read_candidates(self.p["candidate_parents_path"])
            )
        elif "candidate_parents" in self.p:
            self.C = self.p["candidate_parents"]

        else:
            self.C, stats["C"] = cpa[
                self.p["candidate_parent_algorithm"]["name"]
            ](
                self.p["constraints"]["K"],
                scores=self.l_score,
                data=self.data,
                params=self.p["candidate_parent_algorithm"].get("params"),
            )

        self.C_array = np.empty(
            (self.data.n, self.p["constraints"]["K"]), dtype=np.int32
        )
        for v in self.C:
            self.C_array[v] = np.array(self.C[v])

    def _precompute_scores_for_all_candidate_psets(self):
        self.score_array = self.l_score.candidate_scores(self.C_array)

    def _precompute_candidate_restricted_scoring(self):
        self._init_crs()
        self.c_r_score.precompute_tau_simple()
        self.c_r_score.precompute_tau_cc_basecases()
        self.c_r_score.precompute_tau_cc()

    def _init_crs(self):
        # separated from the func above to make it easy to profile time use.
        self.c_r_score = CandidateRestrictedScore(
            score_array=self.score_array,
            C=self.C_array,
            K=self.p["constraints"]["K"],
            cc_tolerance=self.p["catastrophic_cancellation"]["tolerance"],
            cc_cache_size=self.p["catastrophic_cancellation"]["cache_size"],
            pruning_eps=self.p["pruning_tolerance"],
            score_sum_eps=self.p["scoresum_tolerance"],
            silent=self.log.silent(),
            debug=DEBUG,
        )
        del self.score_array

    def _precompute_candidate_complement_scoring(self):
        self.c_c_score = None
        if (
            self.p["constraints"]["K"] < self.data.n - 1
            and self.p["constraints"]["d"] > 0
        ):
            # NOTE: CandidateComplementScore gives error if K = n-1,
            #       and is unnecessary.
            # NOTE: Does this really need to be reinitialized?
            self.l_score = LocalScore(
                data=self.data,
                score=self.p["score"],
                prior=self.p["structure_prior"],
                maxid=self.p["constraints"]["d"],
            )
            self.c_c_score = CandidateComplementScore(
                localscore=self.l_score,
                C=self.C,
                d=self.p["constraints"]["d"],
                eps=self.p["scoresum_tolerance"],
            )
            del self.l_score

        self.score = Score(
            C=self.C, c_r_score=self.c_r_score, c_c_score=self.c_c_score
        )

    def _mcmc_init(self):

        self.mcmc = list()

        for i in range(self.p["mcmc"]["n_independent"]):

            if self.p["metropolis_coupling"]["params"]["M"] == 1:
                self.mcmc.append(
                    PartitionMCMC(
                        self.C,
                        self.score,
                        self.p["constraints"]["d"],
                        move_weights=self.p["mcmc"]["move_weights"],
                        R=self.p["mcmc"]["initial_rootpartition"],
                    )
                )

            elif self.p["metropolis_coupling"]["params"]["M"] > 1:
                if self.p["metropolis_coupling"]["name"] == "adaptive":
                    inv_temps = MC3.get_inv_temperatures(
                        "inv_linear",
                        self.p["metropolis_coupling"]["params"]["M"],
                        self.p["metropolis_coupling"]["params"]["delta_init"],
                    )
                else:  # static
                    inv_temps = MC3.get_inv_temperatures(
                        self.p["metropolis_coupling"]["params"]["heating"],
                        self.p["metropolis_coupling"]["params"]["M"],
                    )
                self.mcmc.append(
                    MC3(
                        [
                            PartitionMCMC(
                                self.C,
                                self.score,
                                self.p["constraints"]["d"],
                                inv_temp=inv_temps[i],
                                move_weights=self.p["mcmc"]["move_weights"],
                                R=self.p["mcmc"]["initial_rootpartition"],
                            )
                            for i in range(
                                self.p["metropolis_coupling"]["params"]["M"]
                            )
                        ],
                        self.p["metropolis_coupling"]["name"],
                        **self.p["metropolis_coupling"]["params"],
                    )
                )

    def _mcmc_run_normal(self):
        def burnin_cond():
            return (
                self._stats["mcmc"]["target_chain_iter_count"]
                < self.p["run_mode"]["params"]["n_target_chain_iters"]
                * self.p["mcmc"]["burnin"]
            )

        def mcmc_cond():
            return (
                self._stats["mcmc"]["target_chain_iter_count"]
                < self.p["run_mode"]["params"]["n_target_chain_iters"]
            )

        def dag_sampling_cond():
            return (
                self._stats["after_burnin"]["target_chain_iter_count"]
            ) >= (
                self.p["run_mode"]["params"]["n_target_chain_iters"]
                * (1 - self.p["mcmc"]["burnin"])
            ) / self.p[
                "mcmc"
            ][
                "n_dags"
            ] * len(
                self.dags
            )

        self._mcmc_run_burnin(burnin_cond=burnin_cond)
        self._mcmc_run_dag_sampling(
            mcmc_cond=mcmc_cond,
            dag_sampling_cond=dag_sampling_cond,
        )

    def _mcmc_run_time_budget(self):
        self._stats["burnin"]["deadline"] = (
            time.time() + self.p.left() * self.p["mcmc"]["burnin"]
        )
        self._stats["mcmc"]["deadline"] = time.time() + self.p.left()
        self.p.budget["mcmc"] = self.p.left()

        def burnin_cond():
            return time.time() < self._stats["burnin"]["deadline"]

        def mcmc_cond():
            return (
                len(self.dags) < self.p["mcmc"]["n_dags"]
                and time.time() < self._stats["mcmc"]["deadline"]
            )

        def dag_sampling_cond():
            return (
                len(self.dags)
                < (time.time() - self._stats["burnin"]["deadline"])
                / self._stats["mcmc"]["time_per_dag"]
            )

        self._mcmc_run_burnin(burnin_cond=burnin_cond)
        self._stats["mcmc"]["time_per_dag"] = (
            self._stats["mcmc"]["deadline"] - time.time()
        ) / self.p["mcmc"]["n_dags"]
        self._mcmc_run_dag_sampling(
            mcmc_cond=mcmc_cond,
            dag_sampling_cond=dag_sampling_cond,
        )

    def _mcmc_run_anytime(self):
        def burnin_cond():
            return True

        def mcmc_cond():
            return True

        def dag_sampling_cond():
            return (
                self._stats["after_burnin"]["target_chain_iter_count"] > 0
                and self._stats["after_burnin"]["target_chain_iter_count"]
                % self._stats["mcmc"]["thinning"]
                == 0
            )

        def periodic_msg():
            self.log(
                f"{len(self.dags)} DAGs "
                f"with thinning {self._stats['mcmc']['thinning']}."
            )

        def after_dag_sampling():
            if len(self.dags) == 2 * self.p["mcmc"]["n_dags"]:
                self.dags = self.dags[0::2]
                self._stats["mcmc"]["thinning"] *= 2

        try:
            self._mcmc_run_burnin(burnin_cond=burnin_cond)
        except KeyboardInterrupt:
            self._stats["burnin"]["iter_count"] = sum(
                mcmc.describe()["iter_count"] for mcmc in self.mcmc
            )
            self._stats["burnin"]["time_used"] = (
                time.time() - self._stats["burnin"]["time_start"]
            )

        try:
            self._mcmc_run_dag_sampling(
                mcmc_cond=mcmc_cond,
                dag_sampling_cond=dag_sampling_cond,
                periodic_msg=periodic_msg,
                after_dag_sampling=after_dag_sampling,
            )
        except KeyboardInterrupt:
            self._stats["mcmc"]["iter_count"] = sum(
                mcmc.describe()["iter_count"] for mcmc in self.mcmc
            )
            self._stats["after_burnin"]["iter_count"] = (
                self._stats["mcmc"]["iter_count"]
                - self._stats["burnin"]["iter_count"]
            )
            self._stats["after_burnin"]["time_used"] = (
                time.time() - self._stats["after_burnin"]["time_start"]
            )

    def _update_verbose_logs(self, indep_chain_idx, R_score):
        mcmc_stats = self.mcmc[indep_chain_idx].describe()
        cyclic_idx = (
            self._stats["mcmc"]["target_chain_iter_count"] % self._verbose_len
        )
        self._verbose["score"][indep_chain_idx][cyclic_idx] = R_score
        self._verbose["inv_temp"][indep_chain_idx][cyclic_idx] = mcmc_stats[
            "inv_temp"
        ]
        self._verbose["mc3_swap_prob"][indep_chain_idx][
            cyclic_idx
        ] = mcmc_stats["accept_prob"].get("mc3", np.array([]))
        self._verbose["mc3_local_swap_prob"][indep_chain_idx][
            cyclic_idx
        ] = mcmc_stats["accept_prob"].get("local_mc3", np.array([]))
        if (
            self._stats["mcmc"]["target_chain_iter_count"] > 0
            and self._stats["mcmc"]["target_chain_iter_count"]
            % (self._verbose_len - 1)
            == 0
        ):
            for verbose_output in self._verbose:
                for row in self._verbose[verbose_output][indep_chain_idx]:
                    self.log.verbose_logger[verbose_output][
                        indep_chain_idx
                    ].numpy(np.expand_dims(row, 0))

    def _mcmc_run_burnin(
        self,
        *,
        burnin_cond=None,
    ):

        self._stats["burnin"]["time_start"] = time.time()
        while burnin_cond():
            for i in range(self.p["mcmc"]["n_independent"]):
                R, R_score = self.mcmc[i].sample()
                if (
                    self._stats["highest_scoring_rootpartition"] is None
                    or R_score[0]
                    > self._stats["highest_scoring_rootpartition"][1]
                ):
                    self._stats["highest_scoring_rootpartition"] = (
                        R[0],
                        R_score[0],
                    )
                self._update_verbose_logs(i, R_score)

            self.log.periodic_stats()
            self._stats["mcmc"]["target_chain_iter_count"] += 1
            self._stats["burnin"]["target_chain_iter_count"] += 1

        self._stats["burnin"]["iter_count"] = sum(
            mcmc.describe()["iter_count"] for mcmc in self.mcmc
        )
        self._stats["burnin"]["time_used"] = (
            time.time() - self._stats["burnin"]["time_start"]
        )

    def _mcmc_run_dag_sampling(
        self,
        *,
        mcmc_cond=None,
        dag_sampling_cond=None,
        periodic_msg=None,
        after_dag_sampling=None,
    ):

        self._stats["after_burnin"]["time_start"] = time.time()
        self.log("Sampling DAGs...")
        self.log.br(2)

        while mcmc_cond():
            for i in range(self.p["mcmc"]["n_independent"]):
                R, R_score = self.mcmc[i].sample()
                if (
                    self._stats["highest_scoring_rootpartition"] is None
                    or R_score[0]
                    > self._stats["highest_scoring_rootpartition"][1]
                ):
                    self._stats["highest_scoring_rootpartition"] = (
                        R[0],
                        R_score[0],
                    )
                if dag_sampling_cond():
                    dag, score = self.score.sample_DAG(R[0])
                    self.dags.append(dag)
                    self.dag_scores.append(score)
                    if after_dag_sampling:
                        after_dag_sampling()
                self._update_verbose_logs(i, R_score)

            if self.log.periodic_stats() and periodic_msg:
                periodic_msg()
            self._stats["mcmc"]["target_chain_iter_count"] += 1
            self._stats["after_burnin"]["target_chain_iter_count"] += 1

        self._stats["mcmc"]["iter_count"] = sum(
            mcmc.describe()["iter_count"] for mcmc in self.mcmc
        )
        self._stats["after_burnin"]["iter_count"] = (
            self._stats["mcmc"]["iter_count"]
            - self._stats["burnin"]["iter_count"]
        )
        self._stats["after_burnin"]["time_used"] = (
            time.time() - self._stats["after_burnin"]["time_start"]
        )


class LocalScore:
    """Class for computing local scores given input data.

    Implemented scores are BDeu and BGe. The scores by default use the "fair"
    modular structure prior :footcite:`eggeling:2019`.

    """

    def __init__(
        self,
        *,
        data,
        score=None,
        prior={"name": "fair"},
        maxid=-1,
    ):
        self.data = Data(data)
        self.score = score
        if score is None:
            # TODO: decouple from Defaults
            self.score = Defaults()["score"](self.data)
        self.prior = prior
        self.priorf = {"fair": self._prior_fair, "unif": self._prior_unif}
        self.maxid = maxid
        self._precompute_prior()

        if self.data.N == 0:
            self.scorer = EmptyDataScore()

        elif self.score["name"] == "bdeu":
            self.scorer = BDeu(
                data=self.data.data,
                maxid=self.maxid,
                ess=self.score["params"]["ess"],
            )

        elif self.score["name"] == "bge":
            self.scorer = BGe(data=self.data, maxid=self.maxid)

        self.t_scorer = 0
        self.t_prior = 0

    def _prior_fair(self, indegree):
        return self._prior[indegree]

    def _prior_unif(self, indegree):
        return 0

    def _precompute_prior(self):
        if self.prior["name"] == "fair":
            self._prior = np.zeros(self.data.n)
            self._prior = -np.array(
                list(
                    map(
                        np.log,
                        [
                            float(comb(self.data.n - 1, k))
                            for k in range(self.data.n)
                        ],
                    )
                )
            )

    def local(self, v, pset):
        """Local score for input node v and pset, with score function
        self.scoref.

        This is the "safe" version, raising error if queried with invalid
        input.  The unsafe self._local will just segfault."""
        if v in pset:
            raise IndexError(
                "Attempting to query score for (v, pset) where v \in pset"
            )
        # Because min() will raise error with empty pset
        if v in range(self.data.n) and len(pset) == 0:
            return self._local(v, pset)
        if min(v, min(pset)) < 0 or max(v, max(pset)) >= self.data.n:
            raise IndexError(
                "Attempting to query score for (v, pset) "
                "where some variables don't exist in data"
            )
        return self._local(v, pset)

    def _local(self, v, pset):
        # NOTE: How expensive are nested function calls?
        return self.scorer.local(v, pset) + self.priorf[self.prior["name"]](
            len(pset)
        )

    def score_dag(self, dag):
        dag = validate.dag(dag)
        return sum([self.local(v, np.array(list(pset))) for v, pset in dag])

    def clear_cache(self):
        self.scorer.clear_cache()

    def candidate_scores(self, C=None):
        # There should be an option to return this for a given node
        if C is None:
            C = np.array(
                [
                    np.array([j for j in range(self.data.n) if j != i])
                    for i in range(self.data.n)
                ],
                dtype=np.int32,
            )
        prior = np.array([bin(i).count("1") for i in range(2 ** len(C[0]))])
        prior = np.array(
            list(map(lambda k: self.priorf[self.prior["name"]](k), prior))
        )
        return self.scorer.candidate_score_array(C) + prior

    def complement_psets_and_scores(self, v, C, d):
        psets, scores, pset_len = self.scorer.complement_psets_and_scores(
            v, C, d
        )
        prior = np.array(
            list(map(lambda k: self.priorf[self.prior["name"]](k), pset_len))
        )
        return psets, scores + prior

    def all_scores_dict(self, C=None):
        # NOTE: Not used in Gadget pipeline, but useful for example
        #       when computing input data for aps.
        scores = dict()
        if C is None:
            C = {
                v: tuple(sorted(set(range(self.data.n)).difference({v})))
                for v in range(self.data.n)
            }
        for v in C:
            tmp = dict()
            for pset in subsets(
                C[v], 0, [len(C[v]) if self.maxid == -1 else self.maxid][0]
            ):
                tmp[frozenset(pset)] = self._local(v, np.array(pset))
            scores[v] = tmp
        return scores


class EmptyDataScore:
    def __init__(self, **kwargs):
        pass

    def local(self, v, pset):
        return 0

    def candidate_score_array(self, C):
        return np.zeros((len(C), 2 ** len(C[0])))

    def clear_cache(self):
        pass

    def complement_psets_and_scores(self, v, C, d):
        n = len(C)
        k = (n - 1) // 64 + 1
        pset_tuple = list(
            filter(
                lambda ss: not set(ss).issubset(C[v]),
                subsets([u for u in C if u != v], 1, d),
            )
        )
        pset_len = np.array(list(map(len, pset_tuple)), dtype=np.int32)
        pset_bm = list(
            map(lambda pset: bm_to_np64(bm(set(pset)), k), pset_tuple)
        )
        scores = np.array([self.local(v, pset) for pset in pset_tuple])
        return np.array(pset_bm), scores, pset_len


class Score:  # should be renamed to e.g. ScoreHandler
    def __init__(self, *, C, c_r_score, c_c_score):

        self.C = C
        self.n = len(self.C)
        self.c_r_score = c_r_score
        self.c_c_score = c_c_score

    def score_rootpartition(self, R):
        # Utility for computing rootpartition score. The actual scoring for
        # simulation happens slightly differently in PartitionMCMC class.
        inpart = [0] * sum(len(R[i]) for i in range(len(R)))
        for i in range(len(R)):
            for v in R[i]:
                inpart[v] = i
        R_node_scores = [0] * len(inpart)
        for v in set().union(*R):
            if inpart[v] == 0:
                R_node_scores[v] = self.sum(v, set(), set())

            else:
                R_node_scores[v] = self.sum(
                    v, set().union(*R[: inpart[v]]), R[inpart[v] - 1]
                )
        return sum(R_node_scores)

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
           T (set): Parent sets must have at least one member in T
                    (if T is not empty).

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
            return self.c_c_score.sum(v, U, T, W_prime)  # [0]
        else:
            # empty pset handled in c_r_score
            return self.c_c_score.sum(v, U, U, W_prime)  # [0]

    def sample_pset(self, v, U, T=set()):

        U_bm = bm(U.intersection(self.C[v]), idx=self.C[v])
        T_bm = bm(T.intersection(self.C[v]), idx=self.C[v])

        if len(T) > 0 and T_bm == 0 and self.c_c_score is None:
            raise RuntimeError(
                "Cannot meet constraints if d=0 (c_c_score is None) "
                "and T does not intersect C[v]"
            )

        if len(T) > 0:
            if T_bm == 0:
                w_crs = -float("inf")
            else:
                w_crs = self.c_r_score.sum(v, U_bm, T_bm, isum=True)
        else:
            w_crs = self.c_r_score.sum(v, U_bm)

        w_ccs = -float("inf")
        if self.c_c_score is not None and not U.issubset(self.C[v]):
            if len(T) > 0:
                w_ccs = self.c_c_score.sum(v, U, T)
            else:
                # Empty pset is handled in c_r_score
                w_ccs = self.c_c_score.sum(v, U, U)

        if (
            self.c_c_score is None
            or -np.random.exponential() < w_crs - np.logaddexp(w_ccs, w_crs)
        ):
            # Sampling from candidate psets.
            pset, family_score = self.c_r_score.sample_pset(
                v, U_bm, T_bm, w_crs - np.random.exponential()
            )
            family = (v, set(self.C[v][i] for i in bm_to_ints(pset)))

        else:
            # Sampling from complement psets.
            if len(T) > 0:
                pset, family_score = self.c_c_score.sample_pset(
                    v, U, T, w_ccs - np.random.exponential()
                )
            else:
                pset, family_score = self.c_c_score.sample_pset(
                    v, U, U, w_ccs - np.random.exponential()
                )

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
                T = R[i - 1]
                family, family_score = self.sample_pset(v, U, T)
            DAG.append(family)
            DAG_score += family_score
        return DAG, DAG_score
