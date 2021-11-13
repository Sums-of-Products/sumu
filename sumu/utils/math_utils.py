import math
from itertools import chain, combinations
import numpy as np


def close(a, b, tolerance):
    return max(a, b) - min(a, b) < tolerance


def subsets(iterable, fromsize, tosize):
    s = list(iterable)
    step = 1 + (fromsize > tosize) * -2
    return chain.from_iterable(
        combinations(s, i) for i in range(fromsize, tosize + step, step)
    )


def comb(n, r):
    if r > n:
        return 0
    f = math.factorial
    return f(n) // f(r) // f(n - r)


def log_minus_exp(p1, p2):
    if np.exp(min(p1, p2) - max(p1, p2)) == 1:
        return -float("inf")
    return max(p1, p2) + np.log1p(-np.exp(min(p1, p2) - max(p1, p2)))
