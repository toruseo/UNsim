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
from unsim.unsim_diff import world_to_jax, simulate, total_travel_time, travel_time, travel_time_auto
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# --- Build scenario (same API as non-differentiable version) ---

def create_World():
    W = World(name="", deltat=5, tmax=2000, print_mode=1, show_mode=1, save_mode=1)
    W.addNode("orig1", 0, 0)
    W.addNode("orig2", 0, 2)
    W.addNode("merge", 1, 1)
    W.addNode("dest", 2, 1)
    W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
    W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
    W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig1", "dest", 0, 1000, 0.45)
    W.adddemand("orig2", "dest", 400, 1000, 0.6)

    return W

# --- Python version ---

"""
W = create_World()

W.exec_simulation()

W.analyzer.print_simple_stats()

W.analyzer.network(t=0)
W.analyzer.network(t=300)
W.analyzer.network(t=800)
W.analyzer.network(t=2000)
W.analyzer.time_space_diagram(mode="k_norm", links="link1")
W.analyzer.time_space_diagram(mode="k_norm", links="link2")
W.analyzer.time_space_diagram(mode="k_norm", links="link3")
plt.show()
"""

# --- Convert to JAX and run ---

W = create_World()

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

# --- Partial derivative of each link's travel time w.r.t. merge_priority of link1 ---

def per_link_ttt(mp_link1):
    """Per-link total travel time as a function of link1's merge_priority."""
    mp = params.merge_priority.at[0].set(mp_link1)
    p = params._replace(merge_priority=mp)
    s = simulate(p, config)
    dt = config.deltat
    n_on_link = s.cum_arrival[:, :config.tsize] - s.cum_departure[:, :config.tsize]
    return jnp.sum(n_on_link, axis=1) * dt

grad_per_link = jax.jacfwd(per_link_ttt)(params.merge_priority[0])
print(f"\nPartial derivative of each link's TTT w.r.t. merge_priority of link1:")
for i, link in enumerate(W.LINKS):
    print(f"  {link.name}: {float(grad_per_link[i]):.2f}")

# --- Partial derivative of vehicle trajectory travel time w.r.t. merge_priority of link1 ---

od_pairs = {
    "orig1->dest": (0, 3),  # (origin_node_id, dest_node_id)
    "orig2->dest": (1, 3),
}
t_departs = [100, 500]

print(f"\nPartial derivative of OD travel time w.r.t. merge_priority of link1:")
for od_name, (orig, dest) in od_pairs.items():
    print(f"  {od_name}:")

    def od_travel_time(mp_link1, t_dep, orig=orig, dest=dest):
        mp = params.merge_priority.at[0].set(mp_link1)
        p = params._replace(merge_priority=mp)
        s = simulate(p, config)
        return travel_time_auto(orig, dest, t_dep, s, p, config)

    grad_fn = jax.grad(od_travel_time, argnums=0)
    for t in t_departs:
        tt = travel_time_auto(orig, dest, t, state, params, config)
        g = grad_fn(params.merge_priority[0], t)
        print(f"    t_depart={float(t):6.0f}s: TT={float(tt):7.1f}s, dTT/dmp1={float(g):8.2f}")

# --- Finite difference validation of all AD gradients ---
# Forward: (f(x+d) - f(x)) / d
# Backward: (f(x) - f(x-d)) / d
# Central: (f(x+d) - f(x-d)) / 2d

import numpy as np

deltas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

def print_finited_comparison(label, ad_val, eval_fn):
    """Print AD vs FiniteD (forward/backward/central) table.

    Parameters
    ----------
    label : str
    ad_val : float
        AD gradient value.
    eval_fn : callable
        eval_fn(delta) returns f(x0 + delta).  delta=0 gives the base value.
    """
    f0 = eval_fn(0.0)
    print(f"\n  {label}:  AD = {ad_val:.6f}")
    print(f"    {'delta':>8}  {'forward':>12}  {'backward':>12}  {'central':>12}"
          f"  {'err(fwd)':>10}  {'err(bwd)':>10}  {'err(cent)':>10}")
    print(f"    {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}"
          f"  {'-'*10}  {'-'*10}  {'-'*10}")
    for delta in deltas:
        fp = eval_fn(delta)
        fm = eval_fn(-delta)
        fwd = (fp - f0) / delta
        bwd = (f0 - fm) / delta
        ctr = (fp - fm) / (2 * delta)
        def rel(v):
            if abs(ad_val) > 1e-10:
                return f"{abs(v - ad_val) / abs(ad_val):.2e}"
            return f"{'N/A':>10}"
        print(f"    {delta:>8.0e}  {fwd:>12.6f}  {bwd:>12.6f}  {ctr:>12.6f}"
              f"  {rel(fwd):>10}  {rel(bwd):>10}  {rel(ctr):>10}")

print(f"\n{'='*70}")
print("  AD vs FiniteD: validation of all gradients")
print(f"{'='*70}")

# (1) TTT w.r.t. demand_rate (sum per node = directional derivative)
print(f"\n--- (1) dTTT/d(demand_rate) [sum per node] ---")
for node in W.NODES:
    ad_val = float(jnp.sum(grad_demand[node.id]))
    if abs(ad_val) < 1e-6:
        continue
    def eval_fn(delta, nid=node.id):
        dr = params.demand_rate.at[nid, :].add(delta)
        return float(total_travel_time(
            simulate(params._replace(demand_rate=dr), config), config))
    print_finited_comparison(f"node {node.name}", ad_val, eval_fn)

# (2) TTT w.r.t. free-flow speed
print(f"\n--- (2) dTTT/d(u) ---")
for i, link in enumerate(W.LINKS):
    ad_val = float(grad_u[i])
    def eval_fn(delta, i=i):
        return float(total_travel_time(
            simulate(params._replace(u=params.u.at[i].add(delta)), config), config))
    print_finited_comparison(link.name, ad_val, eval_fn)

# (4) Per-link TTT w.r.t. merge_priority of link1
print(f"\n--- (4) d(link_TTT)/d(merge_priority of link1) ---")
mp1_base = float(params.merge_priority[0])
for i, link in enumerate(W.LINKS):
    ad_val = float(grad_per_link[i])
    def eval_fn(delta, i=i):
        mp = params.merge_priority.at[0].set(mp1_base + delta)
        s = simulate(params._replace(merge_priority=mp), config)
        dt = config.deltat
        n_on = s.cum_arrival[i, :config.tsize] - s.cum_departure[i, :config.tsize]
        return float(jnp.sum(n_on) * dt)
    print_finited_comparison(link.name, ad_val, eval_fn)

# (5) vehicle trajectory travel time w.r.t. merge_priority of link1
print(f"\n--- (5) d(OD_TT)/d(merge_priority of link1) ---")
for od_name, (orig, dest) in od_pairs.items():
    for t in t_departs:
        def od_tt_fn(mp1, orig=orig, dest=dest, t=t):
            mp = params.merge_priority.at[0].set(mp1)
            p = params._replace(merge_priority=mp)
            s = simulate(p, config)
            return travel_time_auto(orig, dest, t, s, p, config)
        ad_val = float(jax.grad(od_tt_fn)(params.merge_priority[0]))
        def eval_fn(delta, orig=orig, dest=dest, t=t):
            mp = params.merge_priority.at[0].set(mp1_base + delta)
            p = params._replace(merge_priority=mp)
            s = simulate(p, config)
            return float(travel_time_auto(orig, dest, t, s, p, config))
        print_finited_comparison(f"{od_name}, t={float(t):.0f}s", ad_val, eval_fn)
