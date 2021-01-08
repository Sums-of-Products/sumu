Use in R
--------

Sumu seems somewhat usable in R by employing the `reticulate
<https://github.com/rstudio/reticulate>`_ interoperability package. 

The R equivalent to :ref:`Getting started` is as follows:

.. code:: R

   library("reticulate")
   use_condaenv("name_of_your_conda_environment")
   sumu <- import("sumu")

   data_path <- "path_continuous_data_with_n_variables.csv"

   data <- sumu$Data(data_path, discrete=FALSE)

   g <- sumu$Gadget(
          "data" = data,
          "scoref" = "bge",                 # Or "bdeu" for discrete data.
          "ess" = as.integer(10),           # If using BDeu.
          "max_id" = as.integer(-1),        # Max indegree, -1 for none.
          "K" = as.integer(15),             # Number of candidate parents per variable (< n).
          "d" = as.integer(3),              # Max size for parent sets not constrained to candidates.
          "cp_algo" = "greedy-lite",        # Algorithm for finding the candidate parents.
          "mc3_chains" = as.integer(16),    # Number of parallel Metropolis coupled Markov chains.
          "burn_in" = as.integer(10000),    # Number of burn-in iterations in the chain.
          "iterations" = as.integer(10000), # Number of iterations after burn-in.
          "thinning" = as.integer(10)       # Sample a DAG at every nth iteration.
   )

   results <- g$sample()

   dags <- results[1]
   scores <- results[2]


The output format for DAGs does not seem very pretty, but might be
parseable to something more convenient. Computing the causal effects
might also be possible but would require some refactoring in the
Python code to be convenient.
