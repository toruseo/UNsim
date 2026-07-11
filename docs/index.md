# UNsim: Traffic simulation with Autodiff

UNsim is a differentiable macroscopic network traffic simulator in Python.
It also provides a JAX-based differentiable simulation engine that enables lightning fast simulation using GPU and gradient-based applications such as OD demand calibration.

This documentation site is still under development.

<img width="400" alt="Simulation animation of a grid network" src="https://github.com/user-attachments/assets/c0508731-a352-4ca6-aa9c-17e7e838f4f1" />

60000 vehicles travel through a 10 km grid network over 3 hours.
Dark colors indicate congestion (slow speeds).
The simulation wall-clock time was 5 seconds on a 2.0 GHz CPU in pure Python mode.

## Main Features

- Simple, lightweight, and easy-to-use Python implementation of modern standard models of dynamic network traffic flow
- An end-to-end differentiable simulation using JAX
- Lightning-fast JAX mode with a GPU: 0.3 sec for forward simulation on the Chicago-Sketch dataset (2500 links, 1 million vehicles, 3 hours), and 0.5 sec for backward differentiation
- The basic features and syntax are almost identical to those of the [UXsim](https://github.com/toruseo/UXsim) traffic flow simulator

## Contents

```{toctree}
:maxdepth: 2

getting_started
tutorial
reference
```

## Links

- [GitHub repository](https://github.com/toruseo/UNsim)
- [arXiv preprint](https://doi.org/10.48550/arXiv.2604.11380)
- [Toru Seo (Author)](https://toruseo.jp)
