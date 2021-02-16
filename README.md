# Tasks
* Full extensive analysis about what works and what not in the model: 500 pounds
* Substantial improvement in code speed: 500 pounds

# Goals
* Get a method that produces at least as accurate/precise results than the current method, but way more efficiently computationally
* Get a really efficient method which might be a bit less accurate (for the web application)


# Some facts about the model

* Time series of development indicators
* Parameters
  ** M indicators
  ** N periods

# Potential calibration/estimation approaches

* Bayesian methods / Gaussian processes / Hyperopt
* Evolutionary approach / Genetic algorithms
* Greedy heuristic search / Ceteris paribus partial optimization approach
  ** This works best currently
* **Approximate Markov-chain methods / Linear dynamical systemts approximation **
  ** This seems to give a really poor approximation and not too fast either
* Scipy multivariate nonlinear optimization
  ** Seems to be fast (3x) but less accurate
* Particle method (approximation via convex optimization)
  ** A good reference for this method is available here: https://web.stanford.edu/class/ee364b/lectures/seq_notes.pdf
