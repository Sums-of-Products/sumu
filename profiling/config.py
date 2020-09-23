import os

test_path = os.path.dirname(os.path.realpath(__file__))
bn_path = "/".join(test_path.split("/")[:-1]) + "/sumppy/tests/insurance.dsc"
data_path = test_path + "/test_data.csv"

params = {"bn_path": bn_path,
          "datapath": data_path,
          "scoref": "bdeu",
          "ess": 10,
          "max_id": -1,
          "K": 14,
          "d": 3,
          "cp_algo": "greedy-lite",
          "mc3_chains": 16,
          "burn_in": 10000,
          "iterations": 10000,
          "thinning": 10,
          "tolerance": 2**(-32)}

keys = ["_find_candidate_parents",
        "_precompute_scores_for_all_candidate_psets",
        "_precompute_candidate_restricted_scoring",
        "_precompute_candidate_complement_scoring",
        "_run_mcmc",
        "precompute_pset_sampling",
        "sample_pset"]
