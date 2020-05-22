import argparse

from MCMC import Score
from utils import write_jkl


def main():

    parser = argparse.ArgumentParser(description="MCMC")
    parser.add_argument("datapath", help="path to data file")
    parser.add_argument("-s", "--score", help="score function to use", choices=["bdeu", "bge"], default="bdeu")
    parser.add_argument("-e", "--ess", help="equivalent sample size for BDeu", type=int, default=10)
    parser.add_argument("-m", "--max-id", help="maximum indegree for scores (default no max-indegree)", type=int, default=-1)

    # Score-class uses the "fair" prior
    # parser.add_argument("-p", "--prior", help="structure prior (default uniform)", default="unif")

    parser.add_argument("-o", "--output-path", help="output path for .jkl file")
    # parser.add_argument("-w", "--overwrite", help="if set, overwrite the output files if they already exist", action="store_true")

    args = parser.parse_args()

    # scores : function to allow evaluation of any local score
    scores = Score(args.datapath, scoref=args.score, maxid=args.max_id, ess=args.ess)
    scores = scores.all_scores_dict()
    write_jkl(scores, args.output_path)


if __name__ == '__main__':
    main()
