# Energy-based models with non-convergent MCMC

A JAX implementation of generative CNN energy-based models using non-convergent MCMC demonstrated on MNIST.

This implementation draws heavily on the suggestions of [Nijkamp et al. 2019a](https://arxiv.org/abs/1903.12370) and [Nijkamp et al. 2019b](https://arxiv.org/abs/1904.09770). It is based on the [Flax](https://github.com/google/flax) library for deep learning and uses [tinymcmc](https://github.com/MatthewQuenneville/tinymcmc) for sampling.

![example samples]( example_samples.png )
