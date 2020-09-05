import argparse
import time

import numpy as np

import sumppy.candidates_no_R as cnd
from sumppy.samplers import MCMC
from sumppy.utils import pset_posteriors, write_jkl, DAG_to_str, read_candidates, pretty_dict


def main():

    parser = argparse.ArgumentParser(description="MCMC")
    parser.add_argument("datapath", help="path to data file")
    parser.add_argument("K", help="how many candidates to include", type=int)
    parser.add_argument("-f", "--candidates-from-file", help="load candidates from file, row number equals node label, space separated")
    parser.add_argument("-s", "--score", help="score function to use", choices=["bdeu", "bge"], default="bdeu")
    parser.add_argument("-e", "--ess", help="equivalent sample size for BDeu", type=int, default=10)
    parser.add_argument("-m", "--max-id", help="maximum indegree for scores (default no max-indegree)", type=int, default=-1)
    parser.add_argument("-d", help="maximum indegree for psets which are not subsets of candidates (default 2)", type=int, default=2)
    parser.add_argument("-t", "--tolerance", help="max relative difference 1 - a/b for a and b to be considered too close (default 2.33e-10 = 2^-32)", type=float, default=2**(-32))

    parser.add_argument("-c", "--candidate-algorithm", help="candidate algorithm to use", choices=cnd.algo.keys(), default="greedy-lite")

    parser.add_argument("-b", "--burn-in", help="number of burn-in samples", type=int, default=1000)
    parser.add_argument("-i", "--iterations", help="number of iterations after burn-in", type=int, default=1000)
    parser.add_argument("-n", "--nth", help="sample dag every nth iteration", type=int, default=1)
    parser.add_argument("-nc", "--n-chains", help="number of Metropolis coupled MCMC chains (default 16)", type=int, default=16)

    parser.add_argument("-o", "--output-path-prefix", help="prefix for outputs of DAGs (.dag), DAG score (.trace), and pset frequencies (.psetfreq)")
    # parser.add_argument("-w", "--overwrite", help="if set, overwrite the output files if they already exist", action="store_true")

    parser.add_argument("-r", "--randomseed", help="random seed", type=int)
    parser.add_argument("-v", "--verbose", help="print info during run", action="store_true")

    args = parser.parse_args()

    stats = dict()

    if args.randomseed is not None:
        np.random.seed(args.randomseed)

    # scores : function to allow evaluation of any local score
    scores = MCMC.Score(args.datapath, scoref=args.score, maxid=args.max_id, ess=args.ess, stats=stats)

    t0 = time.process_time()
    if args.candidates_from_file is None:
        C = cnd.algo[args.candidate_algorithm](args.K, n=scores.n, scores=scores, datapath=args.datapath)
    else:
        C = read_candidates(args.candidates_from_file)

    t_C = time.process_time() - t0
    if args.verbose:
        print("Candidates")
        for v in C:
            print("{}: {}".format(v, " ".join([str(u) for u in C[v]])))
        print("")

    if args.verbose:
        print("Time usage")
        print("1. candidates:\t\t{}".format(round(t_C, 3)))

    t0 = time.process_time()
    # scores : scores for all possible candidate parents precomputed
    scores = scores.all_scores_list(C)
    t_scores = time.process_time() - t0
    if args.verbose:
        print("2. precompute all local scores for candidates:\t\t{}".format(round(t_scores, 3)))

    complementary_scores = MCMC.Score(args.datapath, scoref=args.score, maxid=args.d, ess=args.ess)
    complementary_scores = MCMC.CScoreR(C, complementary_scores, args.d)

    t0 = time.process_time()
    # scores : special scoring structure for root-partition space
    scores = MCMC.ScoreR(scores, C, tolerance=args.tolerance, cscores=complementary_scores, stats=stats)
    t_scorer = time.process_time() - t0
    if args.verbose:
        print("3. precompute data structure for scoring root-partitions:\t\t{}".format(round(t_scorer, 3)))

    t0 = time.process_time()
    if args.n_chains > 1:
        mcmc = MCMC.MC3([MCMC.PartitionMCMC(C, scores, complementary_scores, temperature=i/(args.n_chains-1), stats=stats) for i in range(args.n_chains)], stats=stats)
    else:
        mcmc = MCMC.PartitionMCMC(C, scores, stats=stats)
    t_mcmc_init = time.process_time() - t0
    if args.verbose:
        print("4. initialize mcmc chains:\t\t{}".format(round(t_mcmc_init, 3)))

    t0 = time.process_time()
    for i in range(args.burn_in):
        mcmc.sample()
    t_mcmc_burnin = time.process_time() - t0
    if args.verbose:
        print("5. {} burn-in mcmc iterations:\t\t{}".format(args.burn_in, round(t_mcmc_burnin, 3)))

    t0 = time.process_time()
    Rs = list()
    for i in range(args.iterations):
        if i % args.nth == 0:
            Rs.append(mcmc.sample()[0])
        else:
            mcmc.sample()
    t_mcmc_iterations = time.process_time() - t0
    if args.verbose:
        print("6. {} mcmc iterations, {} root-partitions stored:\t\t{}".format(args.iterations, len(Rs), round(t_mcmc_iterations, 3)))

    if MCMC.PartitionMCMC.__name__ in stats:
        for temp in [i/(args.n_chains-1) for i in range(args.n_chains)]:

            k_mcmc = MCMC.PartitionMCMC.__name__
            k_move_basic = MCMC.PartitionMCMC.R_basic_move.__name__
            k_move_swap = MCMC.PartitionMCMC.R_swap_any.__name__

            n_valid = stats[k_mcmc][temp][k_move_basic]["candidate-valid"]["n"]
            n_invalid = stats[k_mcmc][temp][k_move_basic]["candidate-invalid"]["n"]
            acc_valid = stats[k_mcmc][temp][k_move_basic]["candidate-valid"]["accepted"]
            acc_invalid = stats[k_mcmc][temp][k_move_basic]["candidate-invalid"]["accepted"]
            stats[k_mcmc][temp][k_move_basic]["candidate-valid"]["ratio"] = round(n_valid / (n_valid + n_invalid), 3)
            stats[k_mcmc][temp][k_move_basic]["candidate-invalid"]["ratio"] = round(n_invalid / (n_valid + n_invalid), 3)
            if n_valid > 0:
                stats[k_mcmc][temp][k_move_basic]["candidate-valid"]["accepted"] = round(acc_valid / n_valid, 3)
            if n_invalid > 0:
                stats[k_mcmc][temp][k_move_basic]["candidate-invalid"]["accepted"] = round(acc_invalid / n_invalid, 3)
            del stats[k_mcmc][temp][k_move_basic]["candidate-valid"]["n"]
            del stats[k_mcmc][temp][k_move_basic]["candidate-invalid"]["n"]

            n_valid = stats[k_mcmc][temp][k_move_swap]["candidate-valid"]["n"]
            n_invalid = stats[k_mcmc][temp][k_move_swap]["candidate-invalid"]["n"]
            acc_valid = stats[k_mcmc][temp][k_move_swap]["candidate-valid"]["accepted"]
            acc_invalid = stats[k_mcmc][temp][k_move_swap]["candidate-invalid"]["accepted"]
            stats[k_mcmc][temp][k_move_swap]["candidate-valid"]["ratio"] = round(n_valid / (n_valid + n_invalid), 3)
            stats[k_mcmc][temp][k_move_swap]["candidate-invalid"]["ratio"] = round(n_invalid / (n_valid + n_invalid), 3)
            if n_valid > 0:
                stats[k_mcmc][temp][k_move_swap]["candidate-valid"]["accepted"] = round(acc_valid / n_valid, 3)
            if n_invalid > 0:
                stats[k_mcmc][temp][k_move_swap]["candidate-invalid"]["accepted"] = round(acc_invalid / n_invalid, 3)
            del stats[k_mcmc][temp][k_move_swap]["candidate-valid"]["n"]
            del stats[k_mcmc][temp][k_move_swap]["candidate-invalid"]["n"]

    if MCMC.MC3.__name__ in stats:
        stats[MCMC.MC3.__name__]["swaps accepted %"] = [round(stats[MCMC.MC3.__name__]["swaps accepted"][i]/stats[MCMC.MC3.__name__]["swaps proposed"][i], 3)
                                                        if stats[MCMC.MC3.__name__]["swaps proposed"][i] != 0 else 0 for i in range(args.n_chains)]
        stats[MCMC.MC3.__name__]["swaps proposed"] = [int(x) for x in list(stats[MCMC.MC3.__name__]["swaps proposed"])]
        stats[MCMC.MC3.__name__]["swaps accepted"] = [int(x) for x in list(stats[MCMC.MC3.__name__]["swaps accepted"])]

    t_dagr = 0
    t0 = time.process_time()
    # DAGR : special structure for sampling psets given root-partition
    ds = MCMC.DAGR(scores, C, complementary_scores, tolerance=args.tolerance, stats=stats)
    t_dagr += time.process_time() - t0

    t_dags = 0
    DAGs = [[] for i in range(len(Rs))]
    DAG_scores = [0]*len(Rs)
    for v in C:
        t0 = time.process_time()
        ds.precompute(v)
        t_dagr += time.process_time() - t0
        t0 = time.process_time()
        for i in range(len(Rs)):
            family, family_score = ds.sample_pset(v, Rs[i], score=True)
            DAGs[i].append(family)
            DAG_scores[i] += family_score
        t_dags += time.process_time() - t0

    if args.verbose:
        print("7. precompute data structure for DAG sampling:\t\t{}".format(round(t_dagr, 3)))

    if args.verbose:
        print("8. {} DAGs sampled:\t\t{}".format(len(DAGs), round(t_dags, 3)))

    t0 = time.process_time()
    ppost = pset_posteriors(DAGs)
    t_ppost = time.process_time() - t0
    if args.verbose:
        print("9. parent set frequencies:\t\t{}".format(round(t_ppost, 3)))

    if args.output_path_prefix is not None:
        with open(args.output_path_prefix + ".dag", "w") as f:
            for DAG in DAGs:
                f.write(DAG_to_str(DAG) + "\n")

        with open(args.output_path_prefix + ".trace", "w") as f:
            f.write(",".join(str(score) for score in DAG_scores))

        write_jkl(ppost, args.output_path_prefix + ".psetfreq")

    if args.verbose:
        print("")
        print("Statistics")
        pretty_dict(stats)


if __name__ == '__main__':
    main()
