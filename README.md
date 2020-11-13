
# Sumu

**Sumu** is a Python library for working with probabilistic and causal graphical models, developed at the [Sums of Products research group](https://www.cs.helsinki.fi/u/mkhkoivi/sopu.html#sopu) at the University of Helsinki.

The library aims to facilitate:

-   Academic workflows, by giving modular access to highly optimized low level algorithms.
-   Industry workflows, by providing easy to use wrappers using the low level components for well-defined practical use cases.

The library in its current state is very much a work-in-progress.

> *sumu*
> 
> 1.  [Finnish](https://en.wiktionary.org/wiki/sumu#Finnish): Very thick vapor obscuring the visibility near the ground, translated usually as "mist" or "fog".
> 2.  [Japanese (澄む)](https://en.wiktionary.org/wiki/%E6%BE%84%E3%82%80#Japanese): For the weather to clear up; become clear, become transparent.
> 3.  Bunch of other meanings not worthy of mention.


## Installation

You will probably want to install this in a conda environment, so you should first [install conda](https://docs.conda.io/en/latest/miniconda.html), if you haven't already, and create an environment. Sumu has been tested to work with Python versions 3.6, 3.7 and 3.8.

The easiest way to install the package is by running `pip install sumu`.

To alternatively install from sources you should clone the repository and run `pip install .` in the repository root.

Finally, if you wish to work on the source code it is preferable to install with `pip install --verbose --no-build-isolation --editable .` as it allows you to modify the Python code without reinstallation. Any changes to compiled code (i.e., C++ or Cython) will still need to be recompiled by running the same `pip` command. For the `editable` install to work you first need to install the build dependencies, that is `Cython>=0.29.17` and `Numpy`.


## Features

Sumu, at the moment, implements the following core algorithms:

-   **Gadget** (Generating Acyclic DiGraphs from Target) for MCMC sampling of DAGs [[1]​](#org5756234).
-   **Beeps** (Bayesian Estimation of Effect Posterior by Sampling) for sampling from the posterior of linear causal effects [[1]​](#org5756234).
-   **APS** (All Parent Sets) for computing for each variable and parent set the total weight of DAGs where that variable has those parents [[2]​](#org19467ac).

Additionally, the library includes supporting functionality, for example functions for local score computations.

**TODO**: Proper documentation of the library and its features.


## Getting started

    import sumu
    
    data = sumu.Data(path_to_continuous_data.csv, discrete=False)
    
    params = {"data": data,
    	  "scoref": "bge",  # Or "bdeu" for discrete data.
    	  "ess": 10,        # If using BDeu.
    	  "max_id": -1,
    	  "K": 15,
    	  "d": 3,
    	  "cp_algo": "greedy-lite",
    	  "mc3_chains": 16,
    	  "burn_in": 10000,
    	  "iterations": 10000,
    	  "thinning": 10}
    
    g = sumu.Gadget(**params)
    dags, scores = g.sample()
    
    # The following only for continuous data 
    dags = [sumu.bnet.family_sequence_to_adj_mat(dag) for dag in dags]
    causal_effects = sumu.beeps(dags, data)


## Citing

If you use the library in your research work please cite the appropriate sources. For APS you should cite [[2]​](#org19467ac) and for everything else [[1]​](#org5756234).


## References

[<a id="org5756234"></a>1] [Jussi Viinikka, Antti Hyttinen, Johan Pensar, and Mikko Koivisto. Towards Scalable Bayesian Learning of Causal DAGs. In *NeurIPS 2020*, in press.](https://arxiv.org/abs/2010.00684)

[<a id="org19467ac"></a>2] [Johan Pensar, Topi Talvitie, Antti Hyttinen, and Mikko Koivisto. A Bayesian approach for estimating causal effects from observational data. In The Thirty-Fourth AAAI Conference on Artiﬁcial Intelligence, AAAI 2020. AAAI Press, 2020.](https://ojs.aaai.org//index.php/AAAI/article/view/5988)

