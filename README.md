# Sumu

> *sumu*
> 
> 1.  [Finnish](https://en.wiktionary.org/wiki/sumu#Finnish): Very thick vapor obscuring the visibility near the ground, translated usually as "mist" or "fog".
> 2.  [Japanese (澄む)](https://en.wiktionary.org/wiki/%E6%BE%84%E3%82%80#Japanese): For the weather to clear up; become clear, become transparent.
> 3.  Bunch of other meanings not worthy of mention.

**Sumu** is a Python library for working with probabilistic and causal graphical models, developed at the [Sums of Products research group](https://www.cs.helsinki.fi/u/mkhkoivi/sopu.html#sopu) at the University of Helsinki.

The library aims to facilitate:

-   Academic workflows, by giving modular access to highly optimized low level algorithms.
-   Industry workflows, by providing easy to use wrappers using the low level components for well-defined practical use cases.

The library in its current state is very much a work-in-progress.


## Installation

You will probably want to install this in a conda environment, so you should first [install conda](https://docs.conda.io/en/latest/miniconda.html), if you haven't already, and create an environment. Sumu has been tested to work with at least Python 3.7.6.

~~The easiest way to install the package is with pip: `pip install sumu`.~~ (Not available yet.)

To alternatively install from sources you should clone the repository and run `pip install .` in the repository root. Installing from sources requires having `Cython` and `Numpy` installed first.

Finally, if you wish to work on the source code it is preferable to install with `pip install --verbose --no-build-isolation --editable .` as it allows you to modify the Python code without reinstallation. Any changes to compiled code (i.e., C++ or Cython) will still need to be recompiled by running the same `pip` command.


## Features

At the moment the library implements the algorithms

-   **Gadget** (Generating Acyclic DiGraphs from Target) for MCMC sampling of dags,
-   **Beeps** (Bayesian Estimation of Effect Posterior by Sampling) for sampling from the posterior of linear causal effects,

as presented in the paper [Towards Scalable Bayesian Learning of Causal DAGs](https://arxiv.org/abs/2010.00684) (NeurIPS 2020). 


## Getting started

    import sumu
    
    data = sumu.Data(path_to_continuous_data.csv, discrete=False)
    
    params = {"data": data,
    	  "scoref": "bge",  # or "bdeu" for discrete data
    	  "ess": 10,
    	  "max_id": -1,
    	  "K": 15,
    	  "d": 3,
    	  "cp_algo": "greedy-lite",
    	  "mc3_chains": 16,
    	  "burn_in": 10000,
    	  "iterations": 10000,
    	  "thinning": 10,
    	  "tolerance": 2**(-32)}
    
    g = sumu.Gadget(**params)
    dags, scores = g.sample()
    
    # The following only for continuous data 
    dags = [sumu.bnet.family_sequence_to_adj_mat(dag) for dag in dags]
    causal_effects = sumu.beeps(dags, data)


## Citing

If you use the library in your research work please cite the paper *Towards Scalable Bayesian Learning of Causal DAGs (NeurIPS 2020)*, e.g., using the bibtex entry

    @inproceedings{viinikka:2020a,
      author    = {Jussi Viinikka and
    	       Antti Hyttinen and
    	       Johan Pensar and
    	       Mikko Koivisto},
      title     = {Towards Scalable Bayesian Learning of Causal DAGs},
      booktitle = {NeurIPS 2020},
      year      = {in press}
    }

