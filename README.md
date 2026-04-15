# UNsim: Differentiable network traffic simulation in Python

[![PyPi](https://img.shields.io/pypi/v/unsim.svg)](https://pypi.python.org/pypi/unsim)
[![arXiv](https://img.shields.io/badge/arXiv-2604.11380-b31b1b.svg)](https://doi.org/10.48550/arXiv.2604.11380)

> [!IMPORTANT]
>  **Still an early development stage.** There may be still bugs and inconsistency. The performance need to be optimized (especially the memory consumption). Documents will be added. The code and API will be significantly changed in the future.

## Main Features

- Simple, lightweight, and easy-to-use Python implementation of modern standard models of dynamic network traffic flow for Python
- Based on Link Transmission Model and Dynamic User Optimum route choice model
- The features and syntax are almost identical to [UXsim](https://github.com/toruseo/UXsim) traffic flow simulator
- An end-to-end differentiable simulation using JAX
- Lightning-fast JAX mode on a good GPU server: 0.3 sec forward simulation for Chicago-Sketch dataset (2500 links, 1 million vehicles, 3 hours), and 0.5 sec for backward differentiation

## Simulation Examples

Simple demonstration will be added.
See `examples/` for now.

## Usage

```python
from unsim import World

W = World(name="", deltat=5, tmax=1200, print_mode=1, save_mode=1)

W.addNode("orig1", 0, 0)
W.addNode("orig2", 0, 2)
W.addNode("merge", 1, 1)
W.addNode("dest", 2, 1)
link1 = W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
link2 = W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
link3 = W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
W.adddemand("orig1", "dest", 0, 1000, 0.45)
W.adddemand("orig2", "dest", 400, 1000, 0.6)

W.exec_simulation()

W.analyzer.print_simple_stats()

W.analyzer.time_space_diagram(mode="k_norm", links="link1", vmax=1)
W.analyzer.time_space_diagram(mode="k_norm", links="link2", vmax=1)
W.analyzer.time_space_diagram(mode="k_norm", links="link3", vmax=1)
```

To be extended.

## Install

```bash
pip install unsim
```

If you want to use acceleration using JAX, install your preferred JAX by doing something like
```bash
pip install jax[cpu]
pip install jax[cuda13]
```
The optimal installation depends on your hardware and software configuration, so please check the details by yourself.

## Terms of Use & License

UNsim is released under the MIT License. You are free to use it as long as the source is acknowledged.

If you use the code, please cite the arXiv article:
- Toru Seo. [End-to-end differentiable network traffic simulation with dynamic route choice](https://doi.org/10.48550/arXiv.2604.11380), arXiv preprint arXiv: 2604.11380, 2026

```bibtex
@Article{seo2026unsim_arxiv,
  author  = {Toru Seo},
  journal = {arXiv preprint arXiv: 2604.11380},
  title   = {End-to-end differentiable network traffic simulation with dynamic route choice},
  year    = {2026},
  doi     = {10.48550/arXiv.2604.11380},
}
```

## Related Links

- [Toru Seo (Author)](https://toruseo.jp/index_en.html)
- [Seo Laboratory, Institute of Science Tokyo](http://seo.cv.ens.titech.ac.jp/en/)
