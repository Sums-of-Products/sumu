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


class Defaults:

    # default parameter values used by multiple classes

    def __init__(self):
        self.default = {
            "run_mode": {"name": "normal"},
            "mcmc": {
                "n_indep": 1,
                "iters": 320000,
                "burn_in": 0.5,
                "n_dags": 10000,
                "move_weights": [1, 1, 2],
            },
            "mc3": lambda name: {
                name
                == "adaptive": {
                    "name": name,
                    "params": {
                        "M": 16,
                        "p_target": 0.234,
                        "delta_t_init": 1000,
                        "delta_t_max_delta": 1,
                        "local_accept_history_size": 100,
                        "update_freq": 100,
                    },
                },
                name
                == "adaptive-incremental": {
                    "name": name,
                    "params": {},
                },
                name
                not in ("adaptive", "adaptive-incremental"): {
                    "name": "linear" if name is None else name,
                    "params": {"M": 16, "local_accept_history_size": 100},
                },
            }.get(True),
            "score": lambda discrete: {"name": "bdeu", "params": {"ess": 10}}
            if discrete
            else {"name": "bge"},
            "prior": {"name": "fair"},
            "cons": {
                "max_id": -1,
                "K": lambda n: min(n - 1, 16),
                "d": lambda n: min(n - 1, 3),
                "pruning_eps": 0.001,
                "score_sum_eps": 0.1,
            },
            "candp": {
                "name": "greedy",
                "params": {"k": 6, "criterion": "score"},
            },
            "catc": {"tolerance": 2 ** -32, "cache_size": 10 ** 7},
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
        run_mode=dict(),
        mcmc=dict(),
        mc3=dict(),
        score=dict(),
        prior=dict(),
        cons=dict(),
        candp=dict(),
        catc=dict(),
        logging=dict(),
    ):
        # Save parameters initially given by user.
        # locals() has to be the first thing called in __init__.
        self.init = dict(**locals())
        del self.init["self"]
        del self.init["data"]

        self.data = Data(data)
        self.default = Defaults()()
        self.p = copy.deepcopy(self.init)

        self._populate_default_parameters()
        self._complete_user_given_parameters()
        self._validate_parameters()
        if self.p["run_mode"]["name"] == "normal":
            self._adjust_inconsistent_parameters()
        if self.p["run_mode"]["name"] == "budget":
            if "t" in self.p["run_mode"]["params"]:
                self._adjust_to_time_budget()
            if "mem" in self.p["run_mode"]["params"]:
                self._adjust_to_mem_budget()

    def _validate_parameters(self):
        # only validating possible user given candidate parents for now
        if "name" not in self.p["candp"] and "path" not in self.p["candp"]:
            validate.candidates(self.p["candp"])

    def _populate_default_parameters(self):
        # Some defaults are defined as functions of data.
        # Evaluate the functions here.
        self.default["cons"]["K"] = self.default["cons"]["K"](self.data.n)
        self.default["cons"]["d"] = self.default["cons"]["d"](self.data.n)
        self.default["score"] = self.default["score"](self.data.discrete)
        self.default["mc3"] = self.default["mc3"](self.p["mc3"].get("name"))

    def _complete_user_given_parameters(self):
        for k in self.p:
            if (
                "name" in self.p[k]
                and self.p[k]["name"] != self.default[k]["name"]
            ):
                continue
            if validate.candidates_is_valid(self.p[k]):
                continue
            self.p[k] = dict(self.default[k], **self.p[k])
            for k2 in self.p[k]:
                if type(self.p[k][k2]) == dict:
                    self.p[k][k2] = dict(self.default[k][k2], **self.p[k][k2])

    def _adjust_inconsistent_parameters(self):
        iters = self.p["mcmc"]["iters"]
        M = self.p["mc3"].get("M", 1)
        burn_in = self.p["mcmc"]["burn_in"]
        n_dags = self.p["mcmc"]["n_dags"]
        self.p["mcmc"]["iters"] = iters // M * M
        self.p["mcmc"]["n_dags"] = min(
            (iters - int(iters * burn_in)) // M, n_dags
        )
        self.adjusted = (
            self.p["mcmc"]["iters"] != iters,
            self.p["mcmc"]["n_dags"] != n_dags,
        )

    def _adjust_to_time_budget(self):
        self.gb = GadgetTimeBudget(
            self.data, self.p["run_mode"]["params"]["t"]
        )
        params_to_predict = ["d", "K"]
        # dict of preset values for params if any
        preset_params = dict(
            i
            for i in ((k, self.init["cons"].get(k)) for k in params_to_predict)
            if i[1] is not None
        )

        # params needs to be copied because of the remove below
        for k in list(params_to_predict):
            if k in preset_params:
                self.p["cons"][k] = self.gb.get_and_pred(k, preset_params[k])
                params_to_predict.remove(k)
        for k in params_to_predict:
            self.p["cons"][k] = self.gb.get_and_pred(k)

        candp_is_given = self.init["candp"] != dict()
        candp_is_greedy = (
            candp_is_given
            and self.init["candp"]["name"] == Defaults()["candp"]["name"]
        )
        candp_params_is_given = (
            candp_is_given and "params" in self.init["candp"]
        )
        candp_k_is_preset = (
            candp_is_greedy
            and candp_params_is_given
            and "k" in self.init["candp"]["params"]
        )
        if candp_k_is_preset:
            self.gb.preset.add("candp")
        else:
            if not candp_params_is_given:
                self.p["candp"]["params"] = self.default["candp"]["params"]
            self.p["candp"]["params"]["t_budget"] = int(
                self.gb.budget["candp"]
            )

    def _mem_estimate(self, n, K, d):
        def n_psets(n, K, d):
            return sum(comb(n - 1, i) - comb(K, i) for i in range(1, d + 1))

        return n * 6e-5 * (n_psets(n, K, d) + 2 ** K) + 71

    def _adjust_to_mem_budget(self):
        mem_budget = self.p["run_mode"]["params"]["mem"]
        n = self.data.n
        K = self.p["cons"]["K"]
        d = self.p["cons"]["d"]
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
            self.p["cons"]["d"] = d
            return
        # If not, increment d by one, if it was changed,
        # and decrement K until budget constraint met.
        if d_changed:
            d += 1
        while self._mem_estimate(n, K, d) > mem_budget and K > 1:
            K -= 1
        self.p["cons"]["K"] = K
        self.p["cons"]["d"] = d

    def __getitem__(self, key):
        return self.p[key]


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
                "cons": {"K": K},
                "candp": {"name": "rnd"},
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

        t_score = t_score / 2 ** K_high
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
        K -= 1

        self.predicted["crs"] = a + b * K ** 2 * 2 ** K + 2 ** K * t_score
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


class GadgetLogger(Logger):
    """Stuff for printing stuff."""

    def __init__(self, gadget):

        super().__init__(
            logfile=None if gadget.p["logging"]["silent"] else sys.stdout,
        )

        log_params = gadget.p["logging"]

        self.score = dict()
        self.inv_temp = dict()

        for i in range(gadget.p["mcmc"]["n_indep"]):
            score_log_path = None
            inv_temp_log_path = None
            if log_params["verbose_prefix"] is not None:
                score_log_path = (
                    Path(log_params["verbose_prefix"])
                    .absolute()
                    .with_suffix(f".{i}.score.tmp")
                )
                inv_temp_log_path = (
                    Path(log_params["verbose_prefix"])
                    .absolute()
                    .with_suffix(f".{i}.inv_temp.tmp")
                )

                # Just to raise error if the final log files
                # without .tmp suffix exist when overwrite=False
                Logger(
                    logfile=score_log_path.with_suffix(""),
                    overwrite=log_params["overwrite"],
                )
                Logger(
                    logfile=inv_temp_log_path.with_suffix(""),
                    overwrite=log_params["overwrite"],
                )

            self.score[i] = Logger(
                logfile=score_log_path,
                overwrite=log_params["overwrite"],
            )
            self.inv_temp[i] = Logger(
                logfile=inv_temp_log_path,
                overwrite=log_params["overwrite"],
            )

        self._running_sec_num = 0
        self._linewidth = 80
        self.g = gadget

    def finalize(self):
        if self.g.p["logging"]["verbose_prefix"] is None:
            return
        for i in range(self.g.p["mcmc"]["n_indep"]):
            M_max = 0
            with open(self.score[i]._logfile, "r") as f:
                for line in f:
                    M_max = max(M_max, line.count(" ") + 1)
            with open(self.score[i]._logfile.with_suffix(""), "w") as f_output:
                f_output.write(" ".join(map(str, range(M_max))) + "\n")
                with open(self.score[i]._logfile, "r") as f_input:
                    for line in f_input:
                        f_output.write(line)
                self.score[i].unlink()
            with open(
                self.inv_temp[i]._logfile.with_suffix(""), "w"
            ) as f_output:
                f_output.write(" ".join(map(str, range(M_max))) + "\n")
                with open(self.inv_temp[i]._logfile, "r") as f_input:
                    for line in f_input:
                        f_output.write(line)
                self.inv_temp[i].unlink()

    def h(self, title):
        self._running_sec_num += 1
        end = "." * (self._linewidth - len(title) - 4)
        title = f"{self._running_sec_num}. {title} {end}"
        self(title)
        self.br()

    def periodic_stats(self, stats, header=False):
        M = len(self.g.mcmc[0].chains)
        msg_tmpl = "{:<12.12}" + " {:<5.5}" * M
        temps = [1.0]
        temps_labels = [1.0]
        temps = sorted(stats["inv_temp"], reverse=True)
        temps_labels = [round(t, 2) for t in temps]
        moves = stats["accept_prob"].keys()

        def print_stats_title():
            msg = "Periodic statistics on:\n"
            msg += (
                "1. Cumulative acceptance probability by move "
                "and inverse temperature.\n"
            )
            if plot_trace:
                msg += (
                    "2. Root-partition score traces "
                    "for each independent chain.\n"
                )
            else:
                msg += (
                    "2. Last root-partition score "
                    "for each independent chain.\n"
                )
            self(msg)

        if header:
            print_stats_title()

        msg = msg_tmpl.format("move", *temps_labels)
        msg += "\n" + "-" * self._linewidth
        self(msg)

        for m in moves:
            ar = [
                round(r, 2) if not np.isnan(r) else ""
                for r in stats["accept_prob"][m]
            ]
            msg = msg_tmpl.format(m, *ar)
            self(msg)

        self.br()

    def run_stats(self):
        w_iters = str(max(len("iters"), len(str(stats["iters"]["total"]))) + 2)
        w_seconds = str(len(str(int(stats["t"]["mcmc"]))) + 2)
        msg_title_tmpl = (
            "{:<20}{:<" + w_iters + "}{:<13}{:<" + w_seconds + "}{:<9}{:<13}"
        )
        msg_tmpl = (
            "{:<20}{:<"
            + w_iters
            + "}{:<13.3}{:<"
            + w_seconds
            + "}{:<9.3}{:<13.3}"
        )
        msg = (
            msg_title_tmpl.format(
                "phase", "iters", "iters/total", "s", "s/total", "s/iter"
            )
            + "\n"
        )
        msg += "-" * self._linewidth + "\n"

        phases = ["burn-in", "after burn-in"]
        if self.g.p["mc3"]["name"] == "adaptive-incremental":
            phases = ["adaptive tempering"] + phases

        for phase in phases:
            msg += (
                msg_tmpl.format(
                    phase,
                    stats["iters"][phase],
                    stats["iters"][phase] / stats["iters"]["total"],
                    round(stats["t"][phase]),
                    stats["t"][phase] / stats["t"]["mcmc"],
                    (
                        stats["t"][phase] / stats["iters"][phase]
                        if stats["iters"][phase] > 0
                        else "-"
                    ),
                )
                + "\n"
            )
        msg += msg_tmpl.format(
            "mcmc total",
            stats["iters"]["total"],
            1.0,
            round(stats["t"]["mcmc"]),
            1.0,
            stats["t"]["mcmc"] / stats["iters"]["total"],
        )
        self(msg)
        self.br()

    def progress(self, t, t_elapsed):
        percentage = None
        # BUG: With variable M this is not correct!
        iterations = t * self.g.p["mc3"]["params"]["M"]
        if self.g.p["run_mode"]["name"] == "normal" or (
            self.g.p["run_mode"]["name"] == "budget"
            and "t" not in self.g.p["run_mode"]["params"]
        ):
            percentage = round(
                100
                * t
                / (self.g.p["mcmc"]["iters"] // self.g.p["mc3"]["params"]["M"])
            )
        elif self.g.p["run_mode"]["name"] == "budget":
            percentage = round(100 * t_elapsed / self.g.p.gb.budget["mcmc"])
        if percentage is not None:
            msg = f"Progress: {percentage}% ({iterations} iterations)"
        else:  # run_mode == anytime
            msg = f"Progress: {iterations} iterations"
        self(msg)

    def r_scores(self, t, M, R_scores):
        msg = "Last root-partition scores: " + " ".join(
            str(int(R_scores[i][t % 1000][0]))
            for i in range(self.g.p["mcmc"]["n_indep"])
        )
        self(msg)
        self.br()

    def plot_score_trace(self, t, M, R_scores):
        r = len(R_scores[0])
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
        if t < 1000:
            xticks = [int(w * t) for w in np.arange(0, 1 + 1 / 3, 1 / 3)]
            xlabels = [str(round(x * M / 1000, 1)) + "k" for x in xticks]
        else:
            xticks = np.array([0, 333, 666, 999])
            xlabels = [
                str(round((x + t) * M / 1000, 1)) + "k"
                for x in -1 * xticks[::-1]
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

    - **prior**: Modular structure prior to use.

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
        run_mode=dict(),
        mcmc=dict(),
        mc3=dict(),
        score=dict(),
        prior=dict(),
        cons=dict(),
        candp=dict(),
        catc=dict(),
        logging=dict(),
    ):

        # locals() has to be the first thing called in __init__.
        user_given_parameters = locals()
        del user_given_parameters["self"]
        self.p = GadgetParameters(**user_given_parameters)
        self.data = self.p.data

        self.log = GadgetLogger(self)
        log = self.log

        log.h("PROBLEM INSTANCE")
        log.dict(self.data.info)
        log.br()

        log.h("RUN PARAMETERS")
        log.dict(self.p.p)
        if self.p["run_mode"]["name"] == "normal":
            if any(self.p.adjusted):
                log("WARNING")
            if self.p.adjusted[0]:
                log("iters adjusted downwards: needs to be multiple of mc3.")
            if self.p.adjusted[1]:
                log(
                    "n_dags adjusted downwards: "
                    "max is (iters * (1 - burn_in)) / mc3."
                )
            if any(self.p.adjusted):
                log.br()
        mem_use_estimate = round(
            self.p._mem_estimate(
                self.data.n, self.p["cons"]["K"], self.p["cons"]["d"]
            )
        )
        log(f"Estimated memory use: {mem_use_estimate}MB")
        log.br()

    def sample(self):

        log = self.log

        log.h("FINDING CANDIDATE PARENTS")
        stats["t"]["C"] = time.time()
        self._find_candidate_parents()
        stats["t"]["C"] = time.time() - stats["t"]["C"]
        log.numpy(self.C_array, "%i")
        if (
            self.p["run_mode"]["name"] == "budget"
            and "t" in self.p["run_mode"]["params"]
            and "candp" not in self.p.gb.preset
            and self.p["candp"]["name"] == Defaults()["candp"]["name"]
        ):
            log.br()
            log(f"Adjusted for time budget: k = {stats['C']['k']}")
            log(
                "time budgeted: "
                f"{round(self.p['candp']['params']['t_budget'])}s"
            )
        log.br()
        log(f"time used: {round(stats['t']['C'])}s")
        log.br(2)

        log.h("PRECOMPUTING SCORING STRUCTURES FOR CANDIDATE PARENT SETS")
        stats["t"]["crscore"] = time.time()
        self._precompute_scores_for_all_candidate_psets()
        self._precompute_candidate_restricted_scoring()
        stats["t"]["crscore"] = time.time() - stats["t"]["crscore"]
        if (
            self.p["run_mode"]["name"] == "budget"
            and "t" in self.p["run_mode"]["params"]
        ):
            log(f"time predicted: {round(self.p.gb.predicted['crs'])}s")
        log(f"time used: {round(stats['t']['crscore'])}s")
        log.br(2)

        log.h("PRECOMPUTING SCORING STRUCTURES FOR COMPLEMENTARY PARENT SETS")
        stats["t"]["ccscore"] = time.time()
        self._precompute_candidate_complement_scoring()
        stats["t"]["ccscore"] = time.time() - stats["t"]["ccscore"]
        if (
            self.p["run_mode"]["name"] == "budget"
            and "t" in self.p["run_mode"]["params"]
        ):
            log(f"time predicted: {round(self.p.gb.predicted['ccs'])}s")
        log(f"time used: {round(stats['t']['ccscore'])}s")
        log.br(2)

        log.h("RUNNING MCMC")
        stats["t"]["mcmc"] = time.time()
        self._mcmc_init()
        stats["t"]["adaptive tempering"] = time.time() - stats["t"]["mcmc"]
        if self.p["run_mode"]["name"] == "anytime":
            self._mcmc_run_anytime()
        else:
            if self.p["mc3"]["name"] == "adaptive-incremental":
                self._mcmc_run(t_elapsed_init=stats["t"]["adaptive tempering"])
            else:
                self._mcmc_run()
        stats["t"]["mcmc"] = time.time() - stats["t"]["mcmc"]
        log(f"time used: {round(stats['t']['mcmc'])}s")
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
            stats=stats,
        )

    def _find_candidate_parents(self):
        self.l_score = LocalScore(
            data=self.data,
            score=self.p["score"],
            prior=self.p["prior"],
            maxid=self.p["cons"]["max_id"],
        )

        if "path" in self.p["candp"]:
            self.C = read_candidates(self.p["candp"]["path"])

        elif "name" in self.p["candp"]:
            self.C, stats["C"] = cpa[self.p["candp"]["name"]](
                self.p["cons"]["K"],
                scores=self.l_score,
                data=self.data,
                params=self.p["candp"].get("params"),
            )
        else:
            self.C = self.p["candp"]

        self.C_array = np.empty(
            (self.data.n, self.p["cons"]["K"]), dtype=np.int32
        )
        for v in self.C:
            self.C_array[v] = np.array(self.C[v])

    def _precompute_scores_for_all_candidate_psets(self):
        self.score_array = self.l_score.candidate_scores(self.C_array)

    def _precompute_candidate_restricted_scoring(self):
        self.c_r_score = CandidateRestrictedScore(
            score_array=self.score_array,
            C=self.C_array,
            K=self.p["cons"]["K"],
            cc_tolerance=self.p["catc"]["tolerance"],
            cc_cache_size=self.p["catc"]["cache_size"],
            pruning_eps=self.p["cons"]["pruning_eps"],
            silent=self.log.silent(),
        )
        del self.score_array

    def _precompute_candidate_complement_scoring(self):
        self.c_c_score = None
        if self.p["cons"]["K"] < self.data.n - 1 and self.p["cons"]["d"] > 0:
            # NOTE: CandidateComplementScore gives error if K = n-1,
            #       and is unnecessary.
            # NOTE: Does this really need to be reinitialized?
            self.l_score = LocalScore(
                data=self.data,
                score=self.p["score"],
                prior=self.p["prior"],
                maxid=self.p["cons"]["d"],
            )
            self.c_c_score = CandidateComplementScore(
                localscore=self.l_score,
                C=self.C,
                d=self.p["cons"]["d"],
                eps=self.p["cons"]["score_sum_eps"],
            )
            del self.l_score

    def _mcmc_init(self):

        self.score = Score(
            C=self.C, c_r_score=self.c_r_score, c_c_score=self.c_c_score
        )

        self.mcmc = list()

        for i in range(self.p["mcmc"]["n_indep"]):

            if self.p["mc3"]["name"] == "adaptive-incremental":
                self.log("Adaptive incremental tempering")
                self.log.br()
                self.mcmc.append(
                    MC3.adaptive_incremental(
                        PartitionMCMC(
                            self.C,
                            self.score,
                            self.p["cons"]["d"],
                            move_weights=self.p["mcmc"]["move_weights"],
                        ),
                        t_budget=self.p.gb.left()
                        if self.p["run_mode"]["name"] == "budget"
                        else None,
                        stats=stats,
                        log=self.log,
                        **dict(
                            dict(),
                            **(
                                self.p["mc3"]["params"]
                                if "params" in self.p["mc3"]
                                else dict()
                            ),
                        ),
                    )
                )
                self.p["mc3"]["params"]["M"] = len(self.mcmc[0].chains)
                self.log.br()

            elif self.p["mc3"]["params"]["M"] == 1:
                self.mcmc.append(
                    PartitionMCMC(
                        self.C,
                        self.score,
                        self.p["cons"]["d"],
                        move_weights=self.p["mcmc"]["move_weights"],
                        stats=stats,
                    )
                )

            elif self.p["mc3"]["params"]["M"] > 1:
                scheme = self.p["mc3"]["name"]
                if scheme == "adaptive":
                    inv_temps = MC3.get_inv_temperatures(
                        "inv_linear",
                        self.p["mc3"]["params"]["M"],
                        self.p["mc3"]["params"]["delta_t_init"],
                    )
                else:
                    inv_temps = MC3.get_inv_temperatures(
                        scheme, self.p["mc3"]["params"]["M"]
                    )
                self.mcmc.append(
                    MC3(
                        [
                            PartitionMCMC(
                                self.C,
                                self.score,
                                self.p["cons"]["d"],
                                inv_temp=inv_temps[i],
                                move_weights=self.p["mcmc"]["move_weights"],
                            )
                            for i in range(self.p["mc3"]["params"]["M"])
                        ],
                        scheme,
                        **self.p["mc3"]["params"],
                    )
                )

    def _mcmc_run(self, t_elapsed_init=0):

        r = 1000  # max number of iterations to plot in score trace

        self.dags = list()
        self.dag_scores = list()

        R_scores = {
            c: [
                None for i in range(r)
            ]  # will contain up to r arrays of scores
            # representing each mc3 chain
            for c in range(self.p["mcmc"]["n_indep"])
        }

        inv_temps = {
            c: [
                None for i in range(r)
            ]  # will contain up to r arrays of inv_temps
            # representing each mc3 chain
            for c in range(self.p["mcmc"]["n_indep"])
        }

        timer = time.time()
        first = True

        if self.p["run_mode"]["name"] == "normal" or (
            self.p["run_mode"]["name"] == "budget"
            and "t" not in self.p["run_mode"]["params"]
        ):
            iters_burn_in = int(
                self.p["mcmc"]["iters"]
                / self.p["mc3"]["params"]["M"]
                * self.p["mcmc"]["burn_in"]
            )
            iters_burn_in = int(iters_burn_in)
            iters_dag_sampling = (
                self.p["mcmc"]["iters"] // self.p["mc3"]["params"]["M"]
                - iters_burn_in
            )
            if self.p["mc3"]["name"] == "adaptive-incremental":
                iters_burn_in -= int(
                    stats["iters"]["adaptive tempering"]
                    / self.p["mc3"]["params"]["M"]
                )
            burn_in_cond = lambda: t < iters_burn_in
            mcmc_cond = lambda: t < iters_dag_sampling
            dag_sample_cond = (
                lambda: t
                >= iters_dag_sampling / self.p["mcmc"]["n_dags"] * dag_count
            )

        elif (
            self.p["run_mode"]["name"] == "budget"
            and "t" in self.p["run_mode"]["params"]
        ):
            self.p.gb.budget["mcmc"] = self.p.gb.left()
            t_b_burnin = self.p.gb.budget["mcmc"] * self.p["mcmc"]["burn_in"]
            burn_in_cond = lambda: t_elapsed < t_b_burnin
            mcmc_cond = (
                lambda: dag_count < self.p["mcmc"]["n_dags"]
                and t_elapsed < t_b_mcmc
            )
            dag_sample_cond = lambda: dag_count < t_elapsed / t_per_dag

        t = 0
        t_elapsed = t_elapsed_init
        t0 = time.time()
        while burn_in_cond():
            for i in range(self.p["mcmc"]["n_indep"]):
                R, R_score = self.mcmc[i].sample()
                R_scores[i][t % r] = R_score
                inv_temps[i][t % r] = self.mcmc[i].inverse_temperatures()
                R, R_score = R[0], R_score[0]
                if t > 0 and t % (r - 1) == 0:
                    for row in R_scores[i]:
                        self.log.score[i].numpy(np.expand_dims(row, 0))
                    for row in inv_temps[i]:
                        self.log.inv_temp[i].numpy(np.expand_dims(row, 0))
            if time.time() - timer > self.p["logging"]["stats_period"]:
                timer = time.time()
                self.log.periodic_stats(self.mcmc[0].describe(), first)
                self.log.progress(t, time.time() - t0)
                if plot_trace:
                    self.log.br()
                    self.log.plot_score_trace(
                        t, self.p["mc3"]["params"]["M"], R_scores
                    )
                else:
                    self.log.r_scores(
                        t, self.p["mc3"]["params"]["M"], R_scores
                    )
                first = False

            t += 1
            t_elapsed = t_elapsed_init + time.time() - t0

        stats["t"]["burn-in"] = time.time() - t0
        if (
            self.p["run_mode"]["name"] == "budget"
            and "t" in self.p["run_mode"]["params"]
        ):
            t_b_mcmc = self.p["run_mode"]["params"]["t"] - (
                time.time() - self.p.gb.t0
            )
            t_per_dag = t_b_mcmc / self.p["mcmc"]["n_dags"]

        self.log("Sampling DAGs...")
        self.log.br(2)

        dag_count = 0
        iters_burn_in = t
        t = 0
        t_elapsed = 0
        t0 = time.time()
        while mcmc_cond():
            if time.time() - timer > self.p["logging"]["stats_period"]:
                timer = time.time()
                self.log.periodic_stats(self.mcmc[0].describe(), first)
                self.log.progress(
                    t + iters_burn_in,
                    time.time() - t0 + stats["t"]["burn-in"],
                )
                if plot_trace:
                    self.log.br()
                    self.log.plot_score_trace(
                        t + iters_burn_in,
                        self.p["mc3"]["params"]["M"],
                        R_scores,
                    )
                else:
                    self.log.r_scores(
                        t + iters_burn_in,
                        self.p["mc3"]["params"]["M"],
                        R_scores,
                    )
                first = False
            if dag_sample_cond():
                # TODO: Fix stupidities in loop structure
                for i in range(self.p["mcmc"]["n_indep"]):
                    dag_count += 1
                    R, R_score = self.mcmc[i].sample()
                    R_scores[i][(t + iters_burn_in) % r] = R_score
                    inv_temps[i][(t + iters_burn_in) % r] = self.mcmc[
                        i
                    ].inverse_temperatures()
                    R, R_score = R[0], R_score[0]
                    dag, score = self.score.sample_DAG(R)
                    self.dags.append(dag)
                    self.dag_scores.append(score)
            else:
                for i in range(self.p["mcmc"]["n_indep"]):
                    R, R_score = self.mcmc[i].sample()
                    R_scores[i][(t + iters_burn_in) % r] = R_score
                    inv_temps[i][(t + iters_burn_in) % r] = self.mcmc[
                        i
                    ].inverse_temperatures()
                    R, R_score = R[0], R_score[0]
            if t > 0 and t % (r - 1) == 0:
                for i in range(self.p["mcmc"]["n_indep"]):
                    for row in R_scores[i]:
                        self.log.score[i].numpy(np.expand_dims(row, 0))
                    for row in inv_temps[i]:
                        self.log.inv_temp[i].numpy(np.expand_dims(row, 0))

            t += 1
            t_elapsed = time.time() - t0
        stats["t"]["after burn-in"] = t_elapsed

        self.log.periodic_stats(self.mcmc[0].describe(), first)

        stats["iters"]["burn-in"] = (
            iters_burn_in * self.p["mc3"]["params"]["M"]
        )
        stats["iters"]["after burn-in"] = t * self.p["mc3"]["params"]["M"]
        stats["iters"]["total"] = (iters_burn_in + t) * self.p["mc3"][
            "params"
        ]["M"]
        if "adaptive tempering" in stats["iters"]:
            stats["iters"]["total"] += stats["iters"]["adaptive tempering"]

    def _mcmc_run_anytime(self):

        r = 1000  # max number of iterations to plot in score trace

        self.dags = list()
        self.dag_scores = list()

        R_scores = {
            c: [
                None for i in range(r)
            ]  # will contain up to r arrays of scores
            # representing each mc3 chain
            for c in range(self.p["mcmc"]["n_indep"])
        }

        inv_temps = {
            c: [
                None for i in range(r)
            ]  # will contain up to r arrays of inv_temps
            # representing each mc3 chain
            for c in range(self.p["mcmc"]["n_indep"])
        }

        timer = time.time()
        t0 = timer
        first = True

        try:
            t_b = -1
            while True:
                t_b += 1
                for i in range(self.p["mcmc"]["n_indep"]):
                    R, R_score = self.mcmc[i].sample()
                    R_scores[i][t_b % r] = R_score
                    inv_temps[i][t_b % r] = self.mcmc[i].inverse_temperatures()
                    R, R_score = R[0], R_score[0]
                if t_b > 0 and t_b % (r - 1) == 0:
                    for row in R_scores[i]:
                        self.log.score[i].numpy(np.expand_dims(row, 0))
                    for row in inv_temps[i]:
                        self.log.inv_temp[i].numpy(np.expand_dims(row, 0))
                if time.time() - timer > self.p["logging"]["stats_period"]:
                    timer = time.time()
                    self.log.periodic_stats(self.mcmc[0].describe(), first)
                    self.log.progress(t_b, 0)
                    if plot_trace:
                        self.log.br()
                        self.log.plot_score_trace(t_b, R_scores)
                    else:
                        self.log.r_scores(
                            t_b, self.p["mc3"]["params"]["M"], R_scores
                        )
                    first = False
        except KeyboardInterrupt:
            stats["t"]["burn-in"] = time.time() - t0
            stats["iters"]["burn-in"] = t_b

        self.log("Sampling DAGs...")
        self.log.br(2)

        try:
            t0 = time.time()
            thinning = 1
            dag_count = 0
            t = -1
            while True:
                t += 1
                if time.time() - timer > self.p["logging"]["stats_period"]:
                    timer = time.time()
                    self.log.periodic_stats(self.mcmc[0].describe(), first)
                    self.log.progress(t_b + t, 0)
                    if plot_trace:
                        self.log.br()
                        self.log.plot_score_trace(t + t_b, R_scores)
                    else:
                        self.log.r_scores(
                            t + t_b, self.p["mc3"]["params"]["M"], R_scores
                        )
                    first = False
                    msg = "{} DAGs with thinning {}."
                    self.log(msg.format(len(self.dags), thinning))
                    self.log.br()
                if t > 0 and t % thinning == 0:
                    for i in range(self.p["mcmc"]["n_indep"]):
                        dag_count += 1
                        R, R_score = self.mcmc[i].sample()
                        R_scores[i][(t + t_b) % r] = R_score
                        inv_temps[i][(t + t_b) % r] = self.mcmc[
                            i
                        ].inverse_temperatures()
                        R, R_score = R[0], R_score[0]
                        dag, score = self.score.sample_DAG(R)
                        self.dags.append(dag)
                        self.dag_scores.append(score)
                        if dag_count == 2 * self.p["mcmc"]["n_dags"]:
                            self.dags = self.dags[0::2]
                            dag_count = len(self.dags)
                            thinning *= 2
                else:
                    for i in range(self.p["mcmc"]["n_indep"]):
                        R, R_score = self.mcmc[i].sample()
                        R_scores[i][(t + t_b) % r] = R_score
                        inv_temps[i][(t + t_b) % r] = self.mcmc[
                            i
                        ].inverse_temperatures()
                        R, R_score = R[0], R_score[0]
                if t % (r - 1) == 0:
                    for row in R_scores[i]:
                        self.log.score[i].numpy(np.expand_dims(row, 0))
                    for row in inv_temps[i]:
                        self.log.inv_temp[i].numpy(np.expand_dims(row, 0))

        except KeyboardInterrupt:
            stats["t"]["after burn-in"] = time.time() - t0
            stats["iters"]["after burn-in"] = t
            stats["iters"]["total"] = (
                stats["iters"]["burn-in"] + stats["iters"]["after burn-in"]
            )

        if first:
            self.log.periodic_stats(self.mcmc[0].describe(), first)


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
        prior=Defaults()["prior"],
        maxid=Defaults()["cons"]["max_id"],
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
