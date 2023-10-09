Sumu
====

.. image:: https://badge.fury.io/py/sumu.svg
    :target: https://badge.fury.io/py/sumu

.. image:: https://github.com/jussiviinikka/sumu/actions/workflows/github-deploy.yml/badge.svg

.. image:: https://codecov.io/gh/jussiviinikka/sumu/branch/master/graph/badge.svg?token=2QFOVD3BBD
	   :target: https://codecov.io/gh/jussiviinikka/sumu

.. image:: https://img.shields.io/pypi/dm/sumu.svg

Python library for working with probabilistic and causal
graphical models.
	   
Developed at the `Sums of Products research
group <https://www.cs.helsinki.fi/u/mkhkoivi/sopu.html#sopu>`__ at the
University of Helsinki.

.. note:: The library in its current state is very much a work-in-progress.
	  
The library aims to facilitate:

-  Academic workflows, by giving modular access to highly optimized low
   level algorithms.
-  Industry workflows, by providing easy to use wrappers using the low
   level components for well-defined practical use cases.

**Documentation can be found at** http://sumu.readthedocs.io.

Requirements
------------

Sumu has been tested to work with Python versions 3.7, 3.8 and 3.9, but might work with earlier versions too.
If you would like to use the library in R see :doc:`use in R <use-in-R>`,
after installing as per the :doc:`installation instructions <installation>`.

Installation
------------

::

    $ pip install sumu

For more details see :doc:`installation instructions <installation>`.

.. note:: The documentation aims to reflect the use of the latest development version. If you do install the PyPI version, you might find these instructions useful: https://www.cs.helsinki.fi/group/sop/gadget-beeps/. **TODO**: versioned documentation.

Features
--------

Currently the following core algorithms are implemented in Sumu:

-  **Gadget** (Generating Acyclic DiGraphs from Target) for MCMC
   sampling of DAGs :footcite:`viinikka:2020a`.
-  **Beeps** (Bayesian Estimation of Effect Posterior by Sampling) for
   sampling from the posterior of linear causal effects :footcite:`viinikka:2020a`.
-  **APS** (All Parent Sets) for computing for each variable and parent
   set the total weight of DAGs where that variable has those parents
   :footcite:`pensar:2020`.

Additionally, the library includes supporting functionality, for example
functions for local score computations.

Getting started
---------------

.. code:: python

   import sumu

   # A path to a space separated file of either discrete or continuous
   # data. No header rows for variable names or arities (in the discrete
   # case) are assumed. Discrete data is assumed to be integer encoded;
   # continuous data uses "." as decimal separator.
   data = sumu.Data("data.csv")
   
   dags, scores = sumu.Gadget(data=data).sample()

   # Causal effect computations only for continuous data.
   # dags are first converted to adjacency matrices.
   dags = [sumu.bnet.family_sequence_to_adj_mat(dag) for dag in dags]

   # All pairwise causal effects for each sampled DAG.
   # causal_effects[i] : effects for ith DAG,
   # where the the first n-1 values represent the effects from variable 1 to 2, ..., n,
   # the following n-1 values represent the effects from variable 2 to 1, 3, ..., n, etc.
   causal_effects = sumu.beeps(dags, data)

See :py:class:`~sumu.gadget.Gadget` for help on how to adjust all the sampling parameters. 
   
Citing
------

If you use the library in your research work please cite the appropriate
sources, e.g., :footcite:`viinikka:2020a` if you use **Gadget** or **Beeps**, or :footcite:`pensar:2020` if you use **APS**.

References
----------

.. footbibliography::


