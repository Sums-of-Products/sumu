import argparse

import numpy as np

from utils import read_jkl
from MCMC import Score
import candidates as cnd


def main():

    algo = {
        "rnd": cnd.rnd,
        "top": cnd.top,
        "pc": cnd.pc,
        "mb": cnd.mb,
        "hc": cnd.hc,
        "greedy": cnd.greedy,
        "pessy": cnd.pessy,
        "greedy-1": cnd.greedy_1,
        "greedy-2": cnd.greedy_2,
        "greedy-3": cnd.greedy_3,
        "greedy-1-double": cnd.greedy_1_double,
        "greedy-1-sum": cnd.greedy_1_sum,
        "greedy-1-double-sum": cnd.greedy_1_double_sum,
        "greedy-2-inverse": cnd.greedy_2_inverse,
        "greedy-2-s": cnd.greedy_2_s,
        "greedy-2-s-inverse": cnd.greedy_2_s_inverse,
        "greedy-backward-forward": cnd.greedy_backward_forward,
        "hybrid": cnd.hybrid,
        "opt": cnd.opt
    }

    parser = argparse.ArgumentParser(description="Evaluate candidate parentsets")
    parser.add_argument("algorithm", help="algorithm to use", choices=algo.keys())
    parser.add_argument("K", help="how many candidates to include", type=int)
    parser.add_argument("-noeval", help="return candidate sets without evaluation", action="store_true")
    parser.add_argument("-p", "--posteriorspath", help="path to pset posteriors")
    parser.add_argument("-rs", "--randomseed", help="random seed", type=int)
    parser.add_argument("-n", "--nvars", help="number of variables", type=int)
    parser.add_argument("-b", "--bsamples", help="how many bootstrap samples to use for pcb and mbt", type=int)
    parser.add_argument("-gs", help="s param for greedy and pessy", type=int)
    parser.add_argument("-d", "--datapath", help="path to data file")
    parser.add_argument("-s", "--score", help="score function to use", choices=["bdeu", "bge"])
    # parser.add_argument("-s", "--scorepath", help="path to score file")
    parser.add_argument("-a", "--alpha", help="alpha value to use in CI-test in PC and IAMB", type=float)
    parser.add_argument("-m", "--maxsx", help="max conditioning set size to use in CI-test in PC and IAMB", type=int)
    parser.add_argument("-f", "--fill", help="how to ensure K candidates for algos for which it's not guaranteed", choices=["none", "random", "top"])
    parser.add_argument("-ha", "--halgos", help="algorithms to use for hybrid", nargs='+', choices=algo.keys())
    parser.add_argument("-hf", "--hfill", help="how to break ties in hybrid", choices=["random", "top"])
    args = parser.parse_args()

    kwargs = {
        "pset_posteriors": None,
        "scores": None,
        "n": args.nvars,
        "B": args.bsamples,
        "s": args.gs,
        "datapath": args.datapath,
        "alpha": args.alpha,
        "max_sx": args.maxsx,
        "fill": args.fill,
        "halgos": args.halgos,
        "hfill": args.hfill
    }

    if args.randomseed is not None:
        np.random.seed(args.randomseed)

    if not args.noeval and not args.posteriorspath:
        parser.error("evaluating candidates requires pset posteriors path (-p)")

    if args.posteriorspath is not None:
        pset_posteriors = read_jkl(args.posteriorspath)
        for node in pset_posteriors:
            normalizer = np.logaddexp.reduce(list(pset_posteriors[node].values()))
            for pset in pset_posteriors[node]:
                pset_posteriors[node][pset] -= normalizer
        kwargs["pset_posteriors"] = pset_posteriors

    if args.score:
        # kwargs["scores"] = read_jkl(args.scorepath)
        kwargs["scores"] = Score(args.datapath, scoref=args.score)

    C = algo[args.algorithm](args.K, **kwargs)

    if args.noeval:
        for node in sorted(C.keys()):
            print(C[node])
    else:
        arg_list = [args.datapath, args.score, args.posteriorspath,
                    args.K, args.algorithm,
                    [args.halgos if args.halgos is None else '|'.join(args.halgos)][0], args.hfill,
                    args.fill,
                    args.gs, args.alpha, args.maxsx, args.bsamples,
                    args.randomseed]
        posterior_covers = cnd.eval_candidates(C, pset_posteriors)
        eval_results = ['|'.join([str(x) for x in posterior_covers.values()]),
                        cnd.eval_candidates_amean(posterior_covers)]
        print(','.join([str(item) for item in arg_list + eval_results + [cnd.candidates_to_str(C)]]))


if __name__ == '__main__':
    main()
