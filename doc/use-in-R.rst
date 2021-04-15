Use in R
--------

Sumu seems somewhat usable in R by employing the `reticulate
<https://github.com/rstudio/reticulate>`_ interoperability package. 

The R equivalent to :ref:`Getting started` is as follows:

.. code:: R

   library("reticulate")
   use_condaenv("name_of_your_conda_environment")
   sumu <- import("sumu")

   data <- sumu$Data("data.csv")

   results <- sumu$Gadget("data"=data)$sample()

   dags <- results[1]
   scores <- results[2]


The output format for DAGs does not seem very pretty, but might be
parseable to something more convenient. Computing the causal effects
might also be possible but would require some refactoring in the
Python code to be convenient.
