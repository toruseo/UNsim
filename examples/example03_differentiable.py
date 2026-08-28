"""
Example: differentiable simulation and gradient computation with the object-oriented API.

Computes gradients of total travel time with respect to:
- demand flow rates
- free-flow speed
- merge priority

The same analyses with the low-level JAX API are in example03d_differentiable_old.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unsim import World
import jax.numpy as jnp

# --- Build scenario (same API as non-differentiable version) ---

W = World(name="", deltat=5, tmax=2000, print_mode=0)
W.addNode("orig1", 0, 0)
W.addNode("orig2", 0, 2)
W.addNode("merge", 1, 1)
W.addNode("dest", 2, 1)
link1 = W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
link2 = W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
link3 = W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
demand1 = W.adddemand("orig1", "dest", 0, 1000, 0.45)
demand2 = W.adddemand("orig2", "dest", 400, 1000, 0.6)

# --- Compile and run ---

M = W.compile(backend="jax")
R = M.run()
print(f"Total travel time: {R.metrics.total_travel_time():.1f} s")

# --- Gradients of TTT w.r.t. demand flows, free-flow speeds, and merge priorities ---

q_refs = [M.parameter(d, "flow") for d in (demand1, demand2)]
u_refs = [M.parameter(l, "free_flow_speed") for l in (link1, link2, link3)]
mp_refs = [M.parameter(l, "merge_priority") for l in (link1, link2, link3)]

ttt = M.objective(lambda R: R.metrics.total_travel_time())
value, grad = ttt.value_and_gradient(wrt=q_refs + u_refs + mp_refs)

print(f"\nGradient of TTT w.r.t. demand flow:")
for ref in q_refs:
    print(f"  {ref.name}: {float(grad[ref]):.2f}")

print(f"\nGradient of TTT w.r.t. free-flow speed:")
for ref in u_refs:
    print(f"  {ref.name}: {float(grad[ref]):.2f}")

print(f"\nGradient of TTT w.r.t. merge_priority:")
for ref in mp_refs:
    print(f"  {ref.name}: {float(grad[ref]):.2f}")

# --- Partial derivative of each link's travel time w.r.t. merge_priority of link1 ---

mp1 = M.parameter(link1, "merge_priority")


def link_ttt(link):
    """Objective: total travel time spent on one link."""
    def fn(R):
        tsize = R.config.tsize
        n_on = R.link(link).cum_arrival()[:tsize] - R.link(link).cum_departure()[:tsize]
        return jnp.sum(jnp.maximum(n_on, 0.0)) * R.config.deltat
    return fn


print(f"\nPartial derivative of each link's TTT w.r.t. merge_priority of link1:")
for link in (link1, link2, link3):
    _, g = M.objective(link_ttt(link)).value_and_gradient(wrt=[mp1])
    print(f"  {link.name}: {float(g[mp1]):.2f}")

# --- Partial derivative of OD travel time w.r.t. merge_priority of link1 ---

od_paths = {
    "orig1->dest": M.path([link1, link3]),
    "orig2->dest": M.path([link2, link3]),
}

print(f"\nPartial derivative of OD travel time w.r.t. merge_priority of link1:")
for od_name, path in od_paths.items():
    print(f"  {od_name}:")
    for t in range(0, 1000, 50):
        obj = M.objective(lambda R, path=path, t=t: R.path(path).travel_time(float(t)))
        tt, g = obj.value_and_gradient(wrt=[mp1])
        print(f"    t_depart={t:6.0f}s: TT={float(tt):7.1f}s, dTT/dmp1={float(g[mp1]):8.2f}")
