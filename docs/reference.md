# Technical Reference

API reference of UNsim. This page is under development; the contents will be expanded.

## Simulation core

### World

```{eval-rst}
.. autoclass:: unsim.World
   :members:
   :undoc-members:
```

### Node

```{eval-rst}
.. autoclass:: unsim.Node
   :members:
```

### Link

```{eval-rst}
.. autoclass:: unsim.Link
   :members:
```

## Result analysis

### Analyzer

```{eval-rst}
.. autoclass:: unsim.Analyzer
   :members:
```

## Differentiable simulation (JAX)

```{eval-rst}
.. automodule:: unsim.unsim_diff
   :members: world_to_jax, simulate, simulate_duo, total_travel_time
```
