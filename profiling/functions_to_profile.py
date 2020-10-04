""" Collection of functions to profile.

If it weren't for memory_profiler, this module could just import
sumu and use its functions like in time_gadget(). However, memory
profiling things accurately doesn't possible seem possible without
expanding the functions inner workings like in mem_gadget(). That then
results in the needs to

1. copy the contents of the memory profiled functions into a function
   defined in this module,
2. and explicitly import everything the function contents need.

"""

from config import params

import sumu
from sumu.gadget import *
from sumu.CandidateRestrictedScore import CandidateRestrictedScore

import numpy as np
from sumu.weight_sum import weight_sum, weight_sum_contribs
# from sumu.zeta_transform import solve as zeta_transform

from sumu.mcmc import PartitionMCMC, MC3

from sumu.utils.bitmap import bm, bm_to_ints, msb, bm_to_pyint_chunks, bm_to_np64, bms_to_np64, np64_to_bm, fbit, kzon, dkbit, ikbit, subsets_size_k, ssets
from sumu.utils.core_utils import arg
from sumu.utils.io import read_candidates, get_n
from sumu.utils.math_utils import log_minus_exp, close, comb, subsets

from sumu.scoring import DiscreteData, ContinuousData, BDeu, BGe

import sumu.candidates_no_r as cnd


def time_gadget():
    g = sumu.Gadget(**params)
    g.sample()


def mem_gadget():

    self = sumu.Gadget(**params)

    # def _find_candidate_parents(self):
    self.l_score = LocalScore(self.datapath, scoref=self.scoref,
                              maxid=self.maxid, ess=self.ess,
                              stats=self.stats)

    if self.cp_path is None:
        # NOTE: datapath is only used by ges, pc and hc which are
        #       not in the imported candidates_no_r
        self.C = cnd.algo[self.cp_algo](self.K, n=self.n,
                                        scores=self.l_score,
                                        datapath=self.datapath)
    else:
        self.C = read_candidates(self.cp_path)

    # def _precompute_scores_for_all_candidate_psets(self):
    self.score_array = self.l_score.as_array(self.C)

    # def _precompute_candidate_restricted_scoring(self):
    C = np.empty((self.n, self.K), dtype=np.int32)
    for v in self.C:
        C[v] = np.array(self.C[v])

    self.c_r_score = CandidateRestrictedScore(self.score_array, C, self.K,
                                              tolerance=self.tolerance)

    # def _precompute_candidate_complement_scoring(self):
    self.c_c_score = None
    if self.K < self.n - 1:
        # NOTE: CandidateComplementScore gives error if K >= n-1.
        self.l_score = LocalScore(self.datapath, scoref=self.scoref,
                                  maxid=self.d, ess=self.ess)
        self.c_c_score = CandidateComplementScore(self.C, self.l_score, self.d)
        del self.l_score

    # def _init_mcmc(self):

    self.score = Score(C=self.C,
                       score_array=self.score_array,
                       c_r_score=self.c_r_score,
                       c_c_score=self.c_c_score)

    if self.mc3_chains > 1:
        self.mcmc = MC3([PartitionMCMC(self.C, self.score, self.d,
                                       temperature=i/(self.mc3_chains-1),
                                       stats=self.stats)
                         for i in range(self.mc3_chains)],
                        stats=self.stats)
    else:
        self.mcmc = PartitionMCMC(self.C, self.score, self.d,
                                  stats=self.stats)

    # def _run_mcmc(self):
    for i in range(self.burn_in):
        self.mcmc.sample()
    self.Rs = list()
    for i in range(self.iterations):
        if i % self.thinning == 0:
            self.Rs.append(self.mcmc.sample()[0])
        else:
            self.mcmc.sample()

    # def _sample_dags(self):
    ds = DAGR(self.score, self.C, tolerance=self.tolerance,
              stats=self.stats)
    self.dags = [[] for i in range(len(self.Rs))]
    self.dag_scores = [0]*len(self.Rs)
    for v in self.C:
        ds.precompute_pset_sampling(v)
        for i in range(len(self.Rs)):
            family, family_score = ds.sample_pset(v, self.Rs[i],
                                                  score=True)
            self.dags[i].append(family)
            self.dag_scores[i] += family_score
    return self.dags, self.dag_scores
