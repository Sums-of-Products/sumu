import sys
import numpy as np

from sumppy.samplers.MCMC import LayeringMCMC
from sumppy.utils import read_jkl

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 5:
        print("Usage: python layeringMCMC.py scorepath M max_indegree iterations seed")
        exit()

    scores = read_jkl(args[0])
    M = int(args[1])
    max_indegree = int(args[2])
    iterations = int(args[3])
    seed = int(args[4])
    np.random.seed(seed)

    mcmc = LayeringMCMC(M, max_indegree, scores)

    # MCMC(M, iterations, max_indegree, scores, seed=seed, print_steps=True)

    for i in range(iterations):
        print(mcmc.sample())
