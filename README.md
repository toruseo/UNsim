# UNsim: Differentiable network traffic simulation in Python

[![PyPi](https://img.shields.io/pypi/v/unsim.svg)](https://pypi.python.org/pypi/unsim)
[![arXiv](https://img.shields.io/badge/arXiv-2604.11380-b31b1b.svg)](https://doi.org/10.48550/arXiv.2604.11380)

> [!IMPORTANT]
>  **Early development stage.** There may be bugs and inconsistencies. The performance needs to be optimized (especially the memory consumption). Documentation should be added. This is more like a research prototype than a library. The code and API will change significantly in the future.

## Main Features

- Simple, lightweight, and easy-to-use Python implementation of modern standard models of dynamic network traffic flow
- An end-to-end differentiable simulation using JAX
- Lightning-fast JAX mode on a good GPU server: 0.3 sec for forward simulation on the Chicago-Sketch dataset (2500 links, 1 million vehicles, 3 hours), and 0.5 sec for backward differentiation
- The features and syntax are almost identical to those of the [UXsim](https://github.com/toruseo/UXsim) traffic flow simulator

## Simulation Examples


<img width="500" alt="grid11x11_anim_linkbased_3hours_11km_60000vehs_5sec_by_2GHz_cpu" src="https://github.com/user-attachments/assets/c0508731-a352-4ca6-aa9c-17e7e838f4f1" />

60000 vehicles travel through a 10 km grid network over 3 hours. 
Dark colors indicate congestion (slow speeds). 
The simulation wall-clock time was 5 seconds on a 2.0 GHz CPU in pure Python mode.

## Usage

Simple scenario in a Y-shaped merge network:
```python
from unsim import World
import matplotlib.pyplot as plt

# Define the main simulation
# Units are standardized to seconds (s) and meters (m)
W = World(name="merge", deltat=5, tmax=1200,    
          print_mode=1, save_mode=1, show_mode=1)

# Define the network
W.addNode("orig1", x=0, y=0)
W.addNode("orig2", x=0, y=2)
W.addNode("merge", x=1, y=1)
W.addNode("dest", x=2, y=1)
W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20, capacity=0.8, merge_priority=1)
W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20, capacity=0.8, merge_priority=1)
W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20)

# Define the vehicle demand
W.adddemand("orig1", "dest", t_start=0, t_end=1000, flow=0.45)
W.adddemand("orig2", "dest", t_start=400, t_end=1000, flow=0.6)

# Run the simulation
W.exec_simulation()

# Analysis
W.analyzer.print_simple_stats()

W.analyzer.network(t=200)
W.analyzer.network(t=800)
plt.show()
```

Results:
```text
Simulation completed. merge
  Simulation Results:
    Total trips:     810.0
    Completed trips: 740.0
    Total travel time: 136825.0 s
    Avg travel time: 184.9 s
    Avg delay:       84.9 s
```

<p float="left">
  <img width="400" alt="network_t200" src="https://github.com/user-attachments/assets/cac09936-3672-4bd4-9922-df8d4c7aeacb" />
  <img width="400" alt="network_t800" src="https://github.com/user-attachments/assets/cd702ac6-dfc3-4f34-b807-7b8148b7a9c9" />
</p>

## Install

```bash
pip install unsim
```
If you want to use JAX acceleration, install your preferred JAX build such as `jax[cpu]` and `jax[cuda13]`.
The optimal installation depends on your hardware and software configuration, so please check the [JAX official document](https://docs.jax.dev/en/latest/installation.html).

## Technical Note

This simulator implements the following transportation scientific models:
- [Link Transmission Model](https://www.mech.kuleuven.be/cib/verkeer/dwn/pub/P2007A.pdf): very efficient and accurate macroscopic traffic flow model
- [Incremental Node Model](https://doi.org/10.1016/j.trb.2011.04.001): general intersection or junction model
- [Dynamic User Optimum](https://doi.org/10.1016/S0191-2615(00)00005-9): route choice model in which each traveler chooses the real-time minimum travel time path

For the details, please see [arXiv preprint](https://doi.org/10.48550/arXiv.2604.11380).

## Terms of Use & License

UNsim is released under the MIT License. You are free to use it as long as the source is acknowledged.

If you use the code, please cite the arXiv article:
- Toru Seo. [End-to-end differentiable network traffic simulation with dynamic route choice](https://doi.org/10.48550/arXiv.2604.11380). arXiv preprint arXiv:2604.11380, 2026.

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
