
# Sumu

<a href="https://badge.fury.io/py/sumu" class="reference external image-reference"><img src="https://badge.fury.io/py/sumu.svg" alt="https://badge.fury.io/py/sumu.svg" /></a> ![](https://github.com/jussiviinikka/sumu/workflows/build/badge.svg)<a href="https://codecov.io/gh/jussiviinikka/sumu" class="reference external image-reference"><img src="https://codecov.io/gh/jussiviinikka/sumu/branch/master/graph/badge.svg?token=2QFOVD3BBD" alt="https://codecov.io/gh/jussiviinikka/sumu/branch/master/graph/badge.svg?token=2QFOVD3BBD" /></a> ![](https://img.shields.io/pypi/dm/sumu.svg)

Python library for working with probabilistic and causal graphical models.

Developed at the <a href="https://www.cs.helsinki.fi/u/mkhkoivi/sopu.html#sopu" class="reference external">Sums of Products research group</a> at the University of Helsinki.

> :warning: **The library in its current state is very much a work-in-progress.**

The library aims to facilitate:

-   Academic workflows, by giving modular access to highly optimized low level algorithms.
-   Industry workflows, by providing easy to use wrappers using the low level components for well-defined practical use cases.

**Documentation can be found at** <http://sumu.readthedocs.io>.

## Requirements

Sumu has been tested to work with Python versions 3.6, 3.7 and 3.8. If you would like to use the library in R see <a href="https://sumu.readthedocs.io/en/latest/use-in-R.html" class="reference internal"><span class="doc">use in R</span></a>, after installing as per the <a href="https://sumu.readthedocs.io/en/latest/installation.html" class="reference internal"><span class="doc">installation instructions</span></a>.

## Installation

    $ pip install sumu

For more details see <a href="https://sumu.readthedocs.io/en/latest/installation.html" class="reference internal"><span class="doc">installation instructions</span></a>.

## Features

Currently the following core algorithms are implemented in Sumu:
-   **Gadget** (Generating Acyclic DiGraphs from Target) for MCMC sampling of DAGs <a href="#viinikka-2020a" id="id1" class="footnote-reference brackets">1</a>.
-   **Beeps** (Bayesian Estimation of Effect Posterior by Sampling) for sampling from the posterior of linear causal effects <a href="#viinikka-2020a" id="id2" class="footnote-reference brackets">1</a>.
-   **APS** (All Parent Sets) for computing for each variable and parent set the total weight of DAGs where that variable has those parents <a href="#pensar-2020" id="id3" class="footnote-reference brackets">2</a>.

Additionally, the library includes supporting functionality, for example functions for local score computations.

## Getting started
```python

 import sumu

 # A path to a space separated file of either discrete or continuous
 # data. No header rows for variable names or arities (in the discrete
 # case) are assumed. Discrete data is assumed to be integer encoded;
 # continuous data uses "." as decimal separator.
 data_path = "path_to_continuous_data_with_n_variables.csv"

 data = sumu.Data(data_path, discrete=False)

 params = {"data": data,
           "scoref": "bge",           # Or "bdeu" for discrete data.
           "ess": 10,                 # If using BDeu.
           "max_id": -1,              # Max indegree, -1 for none.
           "K": 15,                   # Number of candidate parents per variable (< n).
           "d": 3,                    # Max size for parent sets not constrained to candidates.
           "cp_algo": "greedy-lite",  # Algorithm for finding the candidate parents.
           "mc3_chains": 16,          # Number of parallel Metropolis coupled Markov chains.
           "burn_in": 10000,          # Number of burn-in iterations in the chain.
           "iterations": 10000,       # Number of iterations after burn-in.
           "thinning": 10}            # Sample a DAG at every nth iteration.

 g = sumu.Gadget(**params)

 # dags is a list of tuples, where the first element is an int encoding a node
 # and the second element is a (possibly empty) tuple of its parents.
 dags, scores = g.sample()

 # Causal effect computations only for continuous data.
 # dags are first converted to adjacency matrices.
 dags = [sumu.bnet.family_sequence_to_adj_mat(dag) for dag in dags]

 # All pairwise causal effects for each sampled DAG.
 # causal_effects[i] : effects for ith DAG,
 # where the the first n-1 values represent the effects from variable 1 to 2, ..., n,
 # the following n-1 values represent the effects from variable 2 to 1, 3, ..., n, etc.
 causal_effects = sumu.beeps(dags, data)

```
## Citing

If you use the library in your research work please cite the appropriate sources, e.g., <a href="#viinikka-2020a" id="id4" class="footnote-reference brackets">1</a> if you use **Gadget** or **Beeps**, or <a href="#pensar-2020" id="id5" class="footnote-reference brackets">2</a> if you use **APS**.

## References

<span class="brackets">1</span><span class="fn-backref">([1](#id1),[2](#id2),[3](#id4))</span>  
Jussi Viinikka, Antti Hyttinen, Johan Pensar, and Mikko Koivisto. Towards scalable Bayesian learning of causal DAGs. In *NeurIPS 2020*. in press.

<span class="brackets">2</span><span class="fn-backref">([1](#id3),[2](#id5))</span>  
Johan Pensar, Topi Talvitie, Antti Hyttinen, and Mikko Koivisto. A Bayesian approach for estimating causal effects from observational data. In *The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020*. AAAI Press, 2020.
