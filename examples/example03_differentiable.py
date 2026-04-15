"""
Example: JAX differentiable simulation and gradient computation.

Computes gradients of total travel time with respect to:
- demand flow rates
- free-flow speed
- merge priority

Requires JAX: pip install jax[cpu]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unsim import *
from unsim.unsim_diff import world_to_jax, simulate, total_travel_time, travel_time
import jax
import jax.numpy as jnp

# --- Build scenario (same API as non-differentiable version) ---

W = World(name="", deltat=5, tmax=2000, print_mode=0)
W.addNode("orig1", 0, 0)
W.addNode("orig2", 0, 2)
W.addNode("merge", 1, 1)
W.addNode("dest", 2, 1)
W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
W.adddemand("orig1", "dest", 0, 1000, 0.45)
W.adddemand("orig2", "dest", 400, 1000, 0.6)

# --- Convert to JAX and run ---

params, config = world_to_jax(W)
state = simulate(params, config)
ttt = total_travel_time(state, config)
print(f"Total travel time: {ttt:.1f} s")

# --- Gradient of TTT w.r.t. demand ---

def loss_demand(demand_rate):
    p = params._replace(demand_rate=demand_rate)
    s = simulate(p, config)
    return total_travel_time(s, config)

grad_demand = jax.grad(loss_demand)(params.demand_rate)
print(f"\nGradient of TTT w.r.t. demand_rate (sum per node):")
for node in W.NODES:
    g = float(jnp.sum(grad_demand[node.id]))
    if abs(g) > 1e-6:
        print(f"  {node.name}: {g:.2f}")

# --- Gradient of TTT w.r.t. free-flow speed ---

def loss_speed(u):
    p = params._replace(u=u)
    s = simulate(p, config)
    return total_travel_time(s, config)

grad_u = jax.grad(loss_speed)(params.u)
print(f"\nGradient of TTT w.r.t. free-flow speed:")
for i, link in enumerate(W.LINKS):
    print(f"  {link.name}: {float(grad_u[i]):.2f}")

# --- Gradient of TTT w.r.t. merge priority ---

def loss_priority(merge_priority):
    p = params._replace(merge_priority=merge_priority)
    s = simulate(p, config)
    return total_travel_time(s, config)

grad_mp = jax.grad(loss_priority)(params.merge_priority)
print(f"\nGradient of TTT w.r.t. merge_priority:")
for i, link in enumerate(W.LINKS):
    print(f"  {link.name}: {float(grad_mp[i]):.2f}")

# --- Partial derivative of each link's travel time w.r.t. merge_priority of link1 ---

def per_link_ttt(mp_link1):
    """Per-link total travel time as a function of link1's merge_priority."""
    mp = params.merge_priority.at[0].set(mp_link1)
    p = params._replace(merge_priority=mp)
    s = simulate(p, config)
    dt = config.deltat
    n_on_link = s.cum_arrival[:, :config.tsize] - s.cum_departure[:, :config.tsize]
    return jnp.sum(jnp.maximum(n_on_link, 0.0), axis=1) * dt

grad_per_link = jax.jacfwd(per_link_ttt)(params.merge_priority[0])
print(f"\nPartial derivative of each link's TTT w.r.t. merge_priority of link1:")
for i, link in enumerate(W.LINKS):
    print(f"  {link.name}: {float(grad_per_link[i]):.2f}")

# --- Partial derivative of OD travel time w.r.t. merge_priority of link1 ---

od_paths = {
    "orig1->dest": [0, 2],  # link1 -> link3
    "orig2->dest": [1, 2],  # link2 -> link3
}
t_departs = jnp.arange(0, 1000, 50, dtype=jnp.float32)

print(f"\nPartial derivative of OD travel time w.r.t. merge_priority of link1:")
for od_name, path in od_paths.items():
    print(f"  {od_name}:")

    def od_travel_time(mp_link1, t_dep):
        mp = params.merge_priority.at[0].set(mp_link1)
        p = params._replace(merge_priority=mp)
        s = simulate(p, config)
        return travel_time(path, t_dep, s, p, config)

    grad_fn = jax.grad(od_travel_time, argnums=0)
    for t in t_departs:
        tt = travel_time(path, t, state, params, config)
        g = grad_fn(params.merge_priority[0], t)
        print(f"    t_depart={float(t):6.0f}s: TT={float(tt):7.1f}s, dTT/dmp1={float(g):8.2f}")
