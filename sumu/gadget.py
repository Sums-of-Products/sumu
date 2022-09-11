"""The module implements the algorithm Gadget as first detailed in
:footcite:`viinikka:2020a`.
"""

import copy
import os
import pathlib
import sys
import time
from pathlib import Path

import numpy as np

try:
    import plotext as plt

    plot_trace = plt.__version__ == "4.1.3"
except ImportError:
    plot_trace = False

from . import validate
from .candidates import candidate_parent_algorithm as cpa
from .data import Data
from .mcmc import MC3, PartitionMCMC
from .scorer import BDeu, BGe
from .stats import Stats, stats
from .utils.bitmap import bm, bm_to_ints, bm_to_np64
from .utils.io import read_candidates
from .utils.math_utils import comb, subsets
from .weight_sum import CandidateComplementScore, CandidateRestrictedScore

# Debugging level. 0 = no debug prints.
DEBUG = 1


def DBUG(msg):
    if DEBUG:
        print("DEBUG: " + msg)


class Defaults:

    # default parameter values used by multiple classes

    def __init__(self):
        self.default = {
            "run_mode": {"name": "normal"},
            "mcmc": {
                "n_indep": 1,
                "n_target_chain_iters": 20000,
                "burn_in": 0.5,
                "n_dags": 10000,
                "move_weights": [1, 1, 2],
            },
            "metropolis_coupling_scheme": lambda name: {
                name
                == "adaptive": {
                    "name": name,
                    "params": {
                        "M": 2,
                        "p_target": 0.234,
                        "delta_t_init": 0.5,
                        "local_accept_history_size": 1000,
                        "update_freq": 100,
                        "smoothing": 2.0,
                        "slowdown": 1.0,
                    },
                },
                name
                not in ("adaptive",): {
                    "name": "linear" if name is None else name,
                    "params": {"M": 16, "local_accept_history_size": 100},
                },
            }.get(True),
            "score": lambda discrete: {"name": "bdeu", "params": {"ess": 10}}
            if discrete
            else {"name": "bge"},
            "structure_prior": {"name": "fair"},
            "constraints": {
                "max_id": -1,
                "K": lambda n: min(n - 1, 16),
                "d": lambda n: min(n - 1, 3),
                "pruning_eps": 0.001,
                "score_sum_eps": 0.1,
            },
            "candidate_parent_algorithm": {
                "name": "greedy",
                "params": {"k": 6, "criterion": "score"},
            },
            "catastrophic_cancellation": {
                "tolerance": 2 ** -32,
                "cache_size": 10 ** 7,
            },
            "logging": {
                "silent": False,
                "verbose_prefix": None,
                "stats_period": 15,
                "overwrite": False,
            },
        }

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
        initial_rootpartition=None,
        run_mode=dict(),
        mcmc=dict(),
        metropolis_coupling_scheme=dict(),
        score=dict(),
        structure_prior=dict(),
        constraints=dict(),
        candidate_parent_algorithm=dict(),
        candidate_parents_path=None,
        candidate_parents=None,
        catastrophic_cancellation=dict(),
        logging=dict(),
    ):
        # Save parameters initially given by user.
        # locals() has to be the first thing called in __init__.
        self.init = dict(**locals())
        del self.init["self"]
        del self.init["data"]

        # useful to set this to False when developing new features
        if validate.is_boolean(
            validate_params, msg="'validate_params' should be boolean"
        ):
            self._validate_parameters()
        del self.init["validate_params"]

        self.data = Data(data)
        self.default = Defaults()()
        self.p = copy.deepcopy(self.init)

        self._populate_default_parameters()
        self._complete_user_given_parameters()

        # if self.p["run_mode"]["name"] == "normal":
        #    self._adjust_inconsistent_parameters()
        if self.p["run_mode"]["name"] == "budget":
            if "t" in self.p["run_mode"]["params"]:
                self._adjust_to_time_budget()
            if "mem" in self.p["run_mode"]["params"]:
                self._adjust_to_mem_budget()

    def _validate_parameters(self):
        if self.init["initial_rootpartition"]:
            validate.rootpartition(self.init["initial_rootpartition"])
        validate.run_mode_args(self.init["run_mode"])
        validate.mcmc_args(self.init["mcmc"])
        validate.metropolis_coupling_scheme_args(
            self.init["metropolis_coupling_scheme"]
        )
        validate.score_args(self.init["score"])
        validate.structure_prior_args(self.init["structure_prior"])
        validate.constraints_args(self.init["constraints"])
        validate.catastrophic_cancellation_args(
            self.init["catastrophic_cancellation"]
        )
        validate.logging_args(self.init["logging"])
        # Ensure candidate parents are set only by one of the three ways and
        # remove all except the used param from self.init.
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
            [self.init[k] for k in alt_candp_params],
            msg=f"only one of {alt_candp_params} can be set",
        )
        removed = list()
        for k in alt_candp_params:
            if not bool(self.init[k]):
                del self.init[k]
                removed.append(k)
        if len(removed) == 3:
            self.init[alt_candp_params[0]] = dict()
        if alt_candp_params[0] in self.init:
            validate.candidate_parent_algorithm_args(
                self.init[alt_candp_params[0]]
            )
        if alt_candp_params[1] in self.init:
            validate.is_string(
                self.init[alt_candp_params[1]],
                msg=f"'{alt_candp_params[1]}' should be string",
            )
            self.init["constraints"]["K"] = len(
                read_candidates(self.init[alt_candp_params[1]])[0]
            )
        if alt_candp_params[2] in self.init:
            validate.candidates(self.init[alt_candp_params[2]])
            self.init["constraints"]["K"] = len(
                self.init[alt_candp_params[2]][0]
            )

    def _populate_default_parameters(self):
        # Some defaults are defined as functions of data.
        # Evaluate the functions here.
        self.default["constraints"]["K"] = self.default["constraints"]["K"](
            self.data.n
        )
        self.default["constraints"]["d"] = self.default["constraints"]["d"](
            self.data.n
        )
        self.default["score"] = self.default["score"](self.data.discrete)
        self.default["metropolis_coupling_scheme"] = self.default[
            "metropolis_coupling_scheme"
        ](self.p["metropolis_coupling_scheme"].get("name"))

    def _complete_user_given_parameters(self):
        for k in self.p:
            if k in {
                "initial_rootpartition",
                "candidate_parents_path",
                "candidate_parents",
            }:
                continue
            if (
                "name" in self.p[k]
                and self.p[k]["name"] != self.default[k]["name"]
            ):
                continue
            # if validate.candidates_is_valid(self.p[k]):
            #     continue
            self.p[k] = dict(self.default[k], **self.p[k])
            for k2 in self.p[k]:
                if type(self.p[k][k2]) == dict:
                    self.p[k][k2] = dict(self.default[k][k2], **self.p[k][k2])

    # def _adjust_inconsistent_parameters(self):
    #     iters = self.p["mcmc"]["iters"]
    #     M = self.p["mc3"].get("M", 1)
    #     burn_in = self.p["mcmc"]["burn_in"]
    #     n_dags = self.p["mcmc"]["n_dags"]
    #     self.p["mcmc"]["iters"] = iters // M * M
    #     self.p["mcmc"]["n_dags"] = min(
    #         (iters - int(iters * burn_in)) // M, n_dags
    #     )
    #     self.adjusted = (
    #         self.p["mcmc"]["iters"] != iters,
    #         self.p["mcmc"]["n_dags"] != n_dags,
    #     )

    def _adjust_to_time_budget(self):
        self.gb = GadgetTimeBudget(
            self.data, self.p["run_mode"]["params"]["t"]
        )
        params_to_predict = ["d", "K"]
        # dict of preset values for params if any
        preset_params = dict(
            i
            for i in (
                (k, self.init["constraints"].get(k)) for k in params_to_predict
            )
            if i[1] is not None
        )

        # params needs to be copied because of the remove below
        for k in list(params_to_predict):
            if k in preset_params:
                self.p["constraints"][k] = self.gb.get_and_pred(
                    k, preset_params[k]
                )
                params_to_predict.remove(k)
        for k in params_to_predict:
            self.p["constraints"][k] = self.gb.get_and_pred(k)

        if (
            "candidate_parents" in self.init
            or "candidate_parents_path" in self.init
        ):
            return

        candidate_parent_algorithm_is_given = (
            "candidate_parent_algorithm" in self.init
            and self.init["candidate_parent_algorithm"] != dict()
        )
        candidate_parent_algorithm_is_greedy = (
            candidate_parent_algorithm_is_given
            and self.init["candidate_parent_algorithm"]["name"]
            == Defaults()["candidate_parent_algorithm"]["name"]
        )
        candidate_parent_algorithm_params_is_given = (
            candidate_parent_algorithm_is_given
            and "params" in self.init["candidate_parent_algorithm"]
        )
        candidate_parent_algorithm_k_is_preset = (
            candidate_parent_algorithm_is_greedy
            and candidate_parent_algorithm_params_is_given
            and "k" in self.init["candidate_parent_algorithm"]["params"]
        )
        if candidate_parent_algorithm_k_is_preset:
            self.gb.preset.add("candidate_parent_algorithm")
        else:
            if not candidate_parent_algorithm_params_is_given:
                self.p["candidate_parent_algorithm"]["params"] = self.default[
                    "candidate_parent_algorithm"
                ]["params"]
            self.p["candidate_parent_algorithm"]["params"]["t_budget"] = int(
                self.gb.budget["candp"]
            )

    def _mem_estimate(self, n, K, d):
        def n_psets(n, K, d):
            return sum(comb(n - 1, i) - comb(K, i) for i in range(1, d + 1))

        return n * 6e-5 * (n_psets(n, K, d) + 2 ** K) + 71

    def _adjust_to_mem_budget(self):
        mem_budget = self.p["run_mode"]["params"]["mem"]
        n = self.data.n
        K = self.p["constraints"]["K"]
        d = self.p["constraints"]["d"]
        # Decrement d until we're in budget.
        d_changed = False
        while self._mem_estimate(n, K, d) > mem_budget and d > 0:
            d -= 1
            d_changed = True
        # We might be way below budget.
        # Return if incrementing K by one brings us over the budget.
        if (
            self._mem_estimate(n, K, d) < mem_budget
            and self._mem_estimate(n, K + 1, d) > mem_budget
        ):
            self.p["constraints"]["d"] = d
            return
        # If not, increment d by one, if it was changed,
        # and decrement K until budget constraint met.
        if d_changed:
            d += 1
        while self._mem_estimate(n, K, d) > mem_budget and K > 1:
            K -= 1
        self.p["constraints"]["K"] = K
        self.p["constraints"]["d"] = d

    def __getitem__(self, key):
        return self.p[key]

    def __contains__(self, key):
        return key in self.p


class GadgetTimeBudget:
    """Class for predicting run times."""

    def __init__(
        self,
        data,
        t_budget,
        share={"candp": 1 / 9, "crs": 1 / 9, "ccs": 1 / 9, "mcmc": 2 / 3},
    ):
        # The preferred order is: predict d, predict K,
        # remaining precomp budget to candp.
        self.t0 = time.time()
        self.n = data.n
        self.data = data
        self.share = share
        # NOTE: After get_d and get_K the sum of budgets might exceed
        #       total, since remaining budget is adjusted upwards if previous
        #       phase is not predicted to use all of its budget.
        self.budget = dict()
        self.budget["total"] = t_budget
        self.predicted = {phase: 0 for phase in share}
        self.used = {phase: 0 for phase in share}
        self._not_done = set(share)
        self._not_done.remove("mcmc")
        self._update_precomp_budgets()
        self.preset = set()

    def _update_precomp_budgets(self):
        precomp_budget_left = self.budget["total"]
        precomp_budget_left *= 1 - self.share["mcmc"]
        precomp_budget_left -= sum(self.predicted.values()) - sum(
            self.used.values()
        )
        normalizer = sum(self.share[phase] for phase in self._not_done)
        for phase in self._not_done:
            self.budget[phase] = (
                self.share[phase] / normalizer * precomp_budget_left
            )

    def get_and_pred(self, param, preset_value=None):
        # if preset_value given only the time use is predicted
        if param == "d":
            return self.get_and_pred_d(preset_value)
        elif param == "K":
            return self.get_and_pred_K(preset_value)

    def get_and_pred_d(self, d_preset=None):
        phase = "ccs"
        t0 = time.time()
        K_max = 25
        C = {
            v: tuple([u for u in range(self.n) if u != v])
            for v in range(self.n)
        }
        C = {v: C[v][:K_max] for v in C}
        ls = LocalScore(data=self.data)
        d = 0
        t_d = 0
        t_budget = self.budget[phase]

        if d_preset is None:
            d_cond = (
                lambda: d < self.n - 1
                and t_d * comb(self.n, d + 1) / comb(self.n, d) * self.n
                < t_budget
            )
        else:
            self.preset.add("d")
            d_cond = lambda: d <= d_preset - 1

        while d_cond():
            d += 1
            t_d = time.time()
            # This does not take into acount time used by initializing
            # IntersectSums, as it seemed negligible.
            ls.complement_psets_and_scores(0, C, d)
            t_d = time.time() - t_d
            DBUG(f"d={d} t_d={t_d}")

        self.predicted["ccs"] = t_d * self.n
        self.used["ccs"] = time.time() - t0
        self._not_done.remove(phase)
        self._update_precomp_budgets()
        return d

    def get_and_pred_K(self, K_preset=None):
        phase = "crs"
        t0 = time.time()
        K_high = min(self.n - 1, 13)
        K_low = max(K_high - 8, 1)
        t_budget = self.budget[phase]
        X = np.zeros((K_high - K_low + 1, 3))
        i = 0
        for K in range(K_low, K_high + 1):
            params = {
                "constraints": {"K": K},
                "candidate_parent_algorithm": {"name": "rnd"},
                "logging": {"silent": True},
            }
            g = Gadget(data=self.data, **params)
            g._find_candidate_parents()
            t_score = time.time()
            g._precompute_scores_for_all_candidate_psets()
            t_score = time.time() - t_score
            t = time.time()
            g._precompute_candidate_restricted_scoring()
            t = time.time() - t
            X[i] = np.array([1, K ** 2 * 2 ** K, t])
            i += 1
            DBUG(f"K={K} t_score={t_score / 2**K} t={t}")
        t_score = t_score / 2 ** K_high
        DBUG(f"final t_score={t_score}")

        a, b = np.linalg.lstsq(X[:, :-1], X[:, -1], rcond=None)[0]
        t_pred = 0
        K = K_high

        if K_preset is None:
            K_cond = lambda: K < self.n and t_pred < t_budget
        else:
            K_cond = lambda: K < K_preset + 1
            K = K_preset
            self.preset.add("K")

        while K_cond():
            K += 1
            t_pred = a + b * K ** 2 * 2 ** K + 2 ** K * t_score
            DBUG(f"K={K} t_pred={t_pred}")
        K -= 1
        t_pred = a + b * K ** 2 * 2 ** K + 2 ** K * t_score

        self.predicted["crs"] = t_pred
        self.used["crs"] = time.time() - t0
        self._not_done.remove(phase)
        self._update_precomp_budgets()
        return K

    def left(self):
        return self.budget["total"] - (time.time() - self.t0)


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
            print(string, file=self._logfile)
        else:
            with open(self._logfile, self._mode) as f:
                print(string, file=f)

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
            for i in range(gadget.p["mcmc"]["n_indep"]):
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
        for i in range(self.g.p["mcmc"]["n_indep"]):
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
            < self.g.p["logging"]["stats_period"]
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
            self(" " * 15 + "inv_temp")
            msg_tmpl = "{:<12.12} |" + " {:<5.5}" * s["M"]
            temps = [1.0]
            temps_labels = [1.0]
            temps = sorted(s["inv_temp"], reverse=True)
            temps_labels = [round(t, 2) for t in temps]
            moves = s["accept_prob"].keys()
            msg = msg_tmpl.format("move", *temps_labels) + "\n"
            msg = msg.replace("|", " ")
            hr = ["-"] * self._linewidth
            hr[13] = "+"
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
        n_indep = self.g.p["mcmc"]["n_indep"]

        percentage = ""
        # stats = self.g._stats
        if self.g.p["run_mode"]["name"] == "normal" or (
            self.g.p["run_mode"]["name"] == "budget"
            and "t" not in self.g.p["run_mode"]["params"]
        ):
            percentage = round(
                100
                * target_chain_iter_count
                / (self.g.p["mcmc"]["n_target_chain_iters"] * n_indep)
            )
        elif self.g.p["run_mode"]["name"] == "budget":
            percentage = round(
                100
                * (time.time() - self.g._stats["mcmc"]["time_start"])
                / self.g.p.gb.budget["mcmc"]
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
        for i in range(self.g.p["mcmc"]["n_indep"]):
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
        for i in range(self.g.p["mcmc"]["n_indep"]):
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

    In this documentation nested parameters are referenced as
    **outer:inner**, e.g., the ``ess`` parameter can be referenced as
    **score:params:ess**.

    - **run_mode**: Which mode to run Gadget in.

      - **name**: Name of the mode: ``normal``, ``budget`` or ``anytime``.

        - **Default**: ``normal``.

        ``normal``: All parameters are set manually.

        ``budget``: Gadget is run until a given time budget is used
        up. **cons:K**, **cons:d**, **mcmc:iters** and **candp** are set
        automatically, so that approximately one third of the budget is
        used on precomputations and the rest on MCMC sampling. The
        precomputation budget is split between

        - (1) finding candidate parents;
        - (2) precomputing candidate restricted scoring structures;
        - (3) precomputing complementary scoring structures.

        The time required by the third phase is factorial in **cons:d**
        (there are approximately :math:`\\binom{n}{d}` scores complementary
        to those restricted to candidate parents), so the amount of
        additional time required going from :math:`d` to :math:`d+1` can be
        very large. Therefore, as a first step :math:`d` is set to a value
        with which phase (3) is predicted to use at most :math:`1/3` of the
        precomputation budget (i.e., :math:`1/9` of the total). Then the
        remaining precomputation budget is adjusted to be the original
        subtracted by the predicted time use for phase (3) and the (small)
        amount of time required for the prediction itself.

        As a second step **cons:K** is set to a value with which phase (2)
        is predicted to use at most :math:`1/2` of the remaining
        precomputation budget. Again, the predicted time use and the amount
        of time required for the prediction of this phase is subtracted
        from the remaining precomputation budget.

        Then, the candidate parent selection algorithm (**candp**) is set
        to ``greedy-lite``, and its parameter :math:`k` is dynamically set
        during the running of the algorithm to a value for which
        :math:`k-1` is predicted to overuse the remaining precomputation
        budget.

        Finally, the MCMC phase uses the amount of budget that remains. The
        **mcmc:burn_in** parameter in this mode sets fraction of *time* to
        be used on the burn-in phase, rather than the fraction of
        iterations.

        Overrides **mcmc:iters**, **cons:K**, **cons:d** and **candp**.

        - **params**:

          - **t**: The time budget in seconds.

        ``anytime``: If ran in this mode the first CTRL-C after calling
        sample() stops the burn-in phase and starts sampling DAGs, and the
        second CTRL-C stops the sampling. DAG sampling first accumulates up to
        2 * **mcmc**:**n_dags** - 1 DAGs with thinning 1 (i.e., a DAG is
        sampled for each sampled root-partition), then each time the number of
        DAGs reaches 2 x **mcmc**:**n_dags** the thinning is doubled and every
        2nd already sampled DAG is deleted. Overrides **mcmc**:**iters** and
        **mcmc**:**burn_in**.

    - **mcmc**: General Markov Chain Monte Carlo arguments.

      - **n_indep**: Number of independent chains to run (each multiplied by
        **mc3**).  DAGs are sampled evenly from each.

        **Default**: 4.

      - **iters**: The total number of iterations across all the Metropolis
        coupled chains, i.e., if the number of coupled chains is :math:`k`
        then each runs for **iters/k** iterations. If the given **iters**
        is not a multiple of the number of chains it is adjusted downwards.

        **Default**: 320000.

      - **mc3**: The number of of Metropolis coupled chains. The
        temperatures of the chains are spread evenly between uniform
        and the target distribution.

        **Default**: 16.

      - **burn_in**: Ratio of how much of the iterations to use for burn-in
        (0.5 is 50%).

        **Default**: 0.5.

      - **n_dags**: Number of DAGs to sample. The maximum number of
        DAGs that can be sampled is **iters/mc3*(1-burn_in)**; if the given
        **n_dags** is higher than the maximum, it is adjusted
        downwards.

        **Default**: 10000.

    - **score**: The score to use.

      - **name**: Name of the score.

        **Default**: ``bdeu`` (i.e., Bayesian Dirichlet equivalent
        uniform) for discrete data, and ``bge`` (i.e., Bayesian
        Gaussian equivalent) for continuous data.

      - **params**: A dict of parameters for the score.

        **Default**: ``{"ess": 10}`` for ``bdeu``.

    - **structure_prior**: Modular structure prior to use.

      - **name**: Structure prior: *fair* or *unif*
        :footcite:`eggeling:2019`.

        **Default**: fair.

    - **cons**: Constraints on the explored DAG space.

      - **K**: Number of candidate parents per node.

        **Default**: :math:`\min(n-1, 16)`, where :math:`n` is the number
        of nodes.

      - **d**: Maximum size of parent sets that are not subsets of the
        candidate parents.

        **Default**: :math:`\min(n-1, 3)`, where :math:`n` is the number of
        nodes.

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

    - **catastrophic_cancellation**: Parameters determining how catastrofic
      cancellations are handled. Catastrofic cancellation occurs when a score
      sum :math:`\\tau_i(U,T)` computed as :math:`\\tau_i(U) - \\tau_i(U
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

      - **silent**: Whether to print output to ``sys.stdout`` or not.

        **Default**: ``False``.

      - **stats_period**: Interval in seconds for printing more statistics.

        **Default**: 15.

      - **verbose_prefix**: If set, more verbose output is created in files at
        <working directory>/prefix. For example prefix ``x/y`` would create
        files whose name starts with ``y`` in directory ``x``.

        **Default**: ``None``.

      - **overwrite**: If ``True`` files in ``verbose_prefix`` are
        overwritten if they exist.

        **Default**: ``False``.

    """

    def __init__(
        self,
        *,
        data,
        initial_rootpartition=None,
        validate_params=True,
        run_mode=dict(),
        mcmc=dict(),
        metropolis_coupling_scheme=dict(),
        score=dict(),
        structure_prior=dict(),
        constraints=dict(),
        candidate_parent_algorithm=dict(),
        candidate_parents_path=None,
        candidate_parents=None,
        catastrophic_cancellation=dict(),
        logging=dict(),
    ):

        # locals() has to be the first thing called in __init__.
        user_given_parameters = locals()
        del user_given_parameters["self"]

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
                for c in range(self.p["mcmc"]["n_indep"])
            }

        self.dags = list()
        self.dag_scores = list()

        log.h("PROBLEM INSTANCE")
        log.dict(self.data.info)
        log.br()

        log.h("RUN PARAMETERS")
        log.dict(self.p.p)
        mem_use_estimate = round(
            self.p._mem_estimate(
                self.data.n,
                self.p["constraints"]["K"],
                self.p["constraints"]["d"],
            )
        )
        log(f"Estimated memory use: {mem_use_estimate}MB")
        log.br()

        self.precomputations_done = False

    def precompute(self):

        log = self.log

        log.h("FINDING CANDIDATE PARENTS")
        self._stats["candp"]["time_start"] = time.time()
        self._find_candidate_parents()
        self._stats["candp"]["time_used"] = (
            time.time() - self._stats["candp"]["time_start"]
        )
        log.numpy(self.C_array, "%i")
        if (
            self.p["run_mode"]["name"] == "budget"
            and "t" in self.p["run_mode"]["params"]
            and "candidate_parent_algorithm" not in self.p.gb.preset
            and "candidate_parent_algorithm" in self.p
            and self.p["candidate_parent_algorithm"]["name"]
            == Defaults()["candidate_parent_algorithm"]["name"]
        ):
            log.br()
            log(f"Adjusted for time budget: k = {stats['C']['k']}")
            k = "candidate_parent_algorithm"  # to shorten next row
            log(f"time budgeted: {round(self.p[k]['params']['t_budget'])}s")
        log.br()
        log(f"time used: {round(self._stats['candp']['time_used'])}s")
        log.br(2)

        log.h("PRECOMPUTING SCORING STRUCTURES FOR CANDIDATE PARENT SETS")
        self._stats["crscore"]["time_start"] = time.time()
        self._precompute_scores_for_all_candidate_psets()
        self._precompute_candidate_restricted_scoring()
        self._stats["crscore"]["time_used"] = (
            time.time() - self._stats["crscore"]["time_start"]
        )
        if (
            self.p["run_mode"]["name"] == "budget"
            and "t" in self.p["run_mode"]["params"]
        ):
            log(f"time predicted: {round(self.p.gb.predicted['crs'])}s")
        log(f"time used: {round(self._stats['crscore']['time_used'])}s")
        log.br(2)

        log.h("PRECOMPUTING SCORING STRUCTURES FOR COMPLEMENTARY PARENT SETS")
        self._stats["ccscore"]["time_start"] = time.time()
        self._precompute_candidate_complement_scoring()
        self._stats["ccscore"]["time_used"] = (
            time.time() - self._stats["ccscore"]["time_start"]
        )
        if (
            self.p["run_mode"]["name"] == "budget"
            and "t" in self.p["run_mode"]["params"]
        ):
            log(f"time predicted: {round(self.p.gb.predicted['ccs'])}s")
        log(f"time used: {round(self._stats['crscore']['time_used'])}s")
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
        self.c_r_score = CandidateRestrictedScore(
            score_array=self.score_array,
            C=self.C_array,
            K=self.p["constraints"]["K"],
            cc_tolerance=self.p["catastrophic_cancellation"]["tolerance"],
            cc_cache_size=self.p["catastrophic_cancellation"]["cache_size"],
            pruning_eps=self.p["constraints"]["pruning_eps"],
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
                eps=self.p["constraints"]["score_sum_eps"],
            )
            del self.l_score

        self.score = Score(
            C=self.C, c_r_score=self.c_r_score, c_c_score=self.c_c_score
        )

    def _mcmc_init(self):

        self.mcmc = list()

        for i in range(self.p["mcmc"]["n_indep"]):

            if self.p["metropolis_coupling_scheme"]["params"]["M"] == 1:
                self.mcmc.append(
                    PartitionMCMC(
                        self.C,
                        self.score,
                        self.p["constraints"]["d"],
                        move_weights=self.p["mcmc"]["move_weights"],
                        R=self.p["initial_rootpartition"],
                    )
                )

            elif self.p["metropolis_coupling_scheme"]["params"]["M"] > 1:
                scheme = self.p["metropolis_coupling_scheme"]["name"]
                if scheme == "adaptive":
                    inv_temps = MC3.get_inv_temperatures(
                        "inv_linear",
                        self.p["metropolis_coupling_scheme"]["params"]["M"],
                        self.p["metropolis_coupling_scheme"]["params"][
                            "delta_t_init"
                        ],
                    )
                else:
                    inv_temps = MC3.get_inv_temperatures(
                        scheme,
                        self.p["metropolis_coupling_scheme"]["params"]["M"],
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
                                R=self.p["initial_rootpartition"],
                            )
                            for i in range(
                                self.p["metropolis_coupling_scheme"]["params"][
                                    "M"
                                ]
                            )
                        ],
                        scheme,
                        **self.p["metropolis_coupling_scheme"]["params"],
                    )
                )

    def _mcmc_run_normal(self):
        def burn_in_cond():
            return (
                self._stats["mcmc"]["target_chain_iter_count"]
                < self.p["mcmc"]["n_target_chain_iters"]
                * self.p["mcmc"]["burn_in"]
            )

        def mcmc_cond():
            return (
                self._stats["mcmc"]["target_chain_iter_count"]
                < self.p["mcmc"]["n_target_chain_iters"]
            )

        def dag_sampling_cond():
            return (
                self._stats["after_burnin"]["target_chain_iter_count"]
            ) >= (
                self.p["mcmc"]["n_target_chain_iters"]
                * (1 - self.p["mcmc"]["burn_in"])
            ) / self.p[
                "mcmc"
            ][
                "n_dags"
            ] * len(
                self.dags
            )

        self._mcmc_run_burnin(burn_in_cond=burn_in_cond)
        self._mcmc_run_dag_sampling(
            mcmc_cond=mcmc_cond,
            dag_sampling_cond=dag_sampling_cond,
        )

    def _mcmc_run_time_budget(self):
        self._stats["burnin"]["deadline"] = (
            time.time() + self.p.gb.left() * self.p["mcmc"]["burn_in"]
        )
        self._stats["mcmc"]["deadline"] = time.time() + self.p.gb.left()
        self.p.gb.budget["mcmc"] = self.p.gb.left()

        def burn_in_cond():
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

        self._mcmc_run_burnin(burn_in_cond=burn_in_cond)
        self._stats["mcmc"]["time_per_dag"] = (
            self._stats["mcmc"]["deadline"] - time.time()
        ) / self.p["mcmc"]["n_dags"]
        self._mcmc_run_dag_sampling(
            mcmc_cond=mcmc_cond,
            dag_sampling_cond=dag_sampling_cond,
        )

    def _mcmc_run_anytime(self):
        def burn_in_cond():
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
            self._mcmc_run_burnin(burn_in_cond=burn_in_cond)
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
        burn_in_cond=None,
    ):

        self._stats["burnin"]["time_start"] = time.time()
        while burn_in_cond():
            for i in range(self.p["mcmc"]["n_indep"]):
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
            for i in range(self.p["mcmc"]["n_indep"]):
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
        prior=Defaults()["structure_prior"],
        maxid=Defaults()["constraints"]["max_id"],
    ):
        self.data = Data(data)
        self.score = score
        if score is None:
            self.score = Defaults()["score"](self.data.discrete)
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
