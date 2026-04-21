"""
Example: JAX differentiable simulation with DUO logit route choice.

Computes gradients of several travel-time metrics with respect to the
capacity (q*) of link "fast2":
- TTT_all   : total travel time over all links
- TTT_fast1 : total travel time on link "fast1"
- TTT_fast2 : total travel time on link "fast2"
- OD travel times for several departure times (orig -> dest)

Requires JAX: pip install jax[cpu]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unsim import *
from unsim.unsim_diff import (
    world_to_jax, simulate_duo, total_travel_time,
    travel_time, travel_time_auto,
)
import jax
import jax.numpy as jnp


def create_world_2routes():
    W = World(name="", deltat=5, tmax=4000, print_mode=0,
              route_choice="duo_logit")
    W.LOGIT_TEMPERATURE = 60.0
    W.addNode("orig", 0, 0)
    W.addNode("mid1", 1, 1)
    W.addNode("mid2", 1, -1)
    W.addNode("dest", 2, 0)
    W.addLink("fast1", "orig", "mid1", length=1000, free_flow_speed=20, capacity=0.8)
    W.addLink("fast2", "mid1", "dest", length= 500, free_flow_speed=20, capacity=0.6)
    W.addLink("slow1", "orig", "mid2", length=1000, free_flow_speed=10, capacity=0.8)
    W.addLink("slow2", "mid2", "dest", length= 500, free_flow_speed=10, capacity=0.8)
    W.adddemand("orig", "dest", 0, 3000, 0.6)
    W.adddemand("orig", "dest", 500, 2000, 0.2)
    return W


# --- Python version (reference) ---

"""
W = create_world_2routes()
W.exec_simulation()
W.analyzer.print_simple_stats()
W.analyzer.time_space_diagram(mode="k_norm", links=["fast1","fast2"])
W.analyzer.time_space_diagram(mode="k_norm", links=["slow1","slow2"])
show()
W.analyzer.network(t=300)
W.analyzer.network(t=800)
show()
"""

# --- Convert to JAX and run DUO logit simulation ---

W = create_world_2routes()
params, config = world_to_jax(W)

# Name -> index lookup
link_name_to_id = {l.name: i for i, l in enumerate(W.LINKS)}
node_name_to_id = {n.name: i for i, n in enumerate(W.NODES)}
fast1_id = link_name_to_id["fast1"]
fast2_id = link_name_to_id["fast2"]
slow1_id = link_name_to_id["slow1"]
slow2_id = link_name_to_id["slow2"]
orig_id = node_name_to_id["orig"]
dest_id = node_name_to_id["dest"]

fast_path = (fast1_id, fast2_id)
slow_path = (slow1_id, slow2_id)

state = simulate_duo(params, config)

def avg_tt_per_link(state, config):
    """Per-link average travel time (s): TTT_link / vehicles_used_link.

    vehicles_used_link = cum_departure at the end of the simulation
    (equals cum_arrival once all vehicles have flushed out).
    """
    dt = config.deltat
    n_on = state.cum_arrival[:, :config.tsize] - state.cum_departure[:, :config.tsize]
    ttt = jnp.sum(n_on, axis=1) * dt
    n_used = state.cum_departure[:, config.tsize]
    return ttt / jnp.maximum(n_used, 1e-10)


ttt_all = float(total_travel_time(state, config))
avg_tt_link = avg_tt_per_link(state, config)

avg_tt_targets = [
    ("fast1", fast1_id),
    ("fast2", fast2_id),
    ("slow1", slow1_id),
    ("slow2", slow2_id),
]

print(f"Base simulation (DUO logit):")
print(f"  TTT_all       = {ttt_all:10.1f} s")
for lname, lid in avg_tt_targets:
    print(f"  avg_TT_{lname:<5} = {float(avg_tt_link[lid]):10.2f} s/veh")

# --- AD: gradient w.r.t. fast2.capacity (== params.q_star[fast2_id]) ---

q_star_base = params.q_star


def _replace_fast2_cap(cap):
    """Return a new Params with fast2's q* set to cap."""
    new_q = q_star_base.at[fast2_id].set(cap)
    return params._replace(q_star=new_q)


def ttt_all_of_cap(cap):
    p = _replace_fast2_cap(cap)
    s = simulate_duo(p, config)
    return total_travel_time(s, config)


def avg_tt_of_cap(cap, link_id):
    p = _replace_fast2_cap(cap)
    s = simulate_duo(p, config)
    return avg_tt_per_link(s, config)[link_id]


cap0 = float(q_star_base[fast2_id])

grad_ttt_all = float(jax.grad(ttt_all_of_cap)(cap0))
grad_avg_tt = {
    lname: float(jax.grad(avg_tt_of_cap)(cap0, lid))
    for lname, lid in avg_tt_targets
}

print(f"\nAD: d(.)/d(fast2.capacity) at capacity = {cap0:.3f} veh/s")
print(f"  dTTT_all        / dcap = {grad_ttt_all:12.2f}")
for lname, _ in avg_tt_targets:
    print(f"  d(avg_TT_{lname:<5})/ dcap = {grad_avg_tt[lname]:12.4f}")

# --- AD: OD travel time (shortest-path) for several departure times ---

t_departs = [100.0, 1500.0]


def od_tt_of_cap(cap, t_dep):
    p = _replace_fast2_cap(cap)
    s = simulate_duo(p, config)
    return travel_time_auto(orig_id, dest_id, t_dep, s, p, config)


grad_od_tt_fn = jax.grad(od_tt_of_cap, argnums=0)

print(f"\nAD: d(OD_TT)/d(fast2.capacity)  (orig -> dest, shortest path)")
print(f"  {'t_depart':>10}  {'TT (s)':>10}  {'dTT/dcap':>12}")
print(f"  {'-'*10}  {'-'*10}  {'-'*12}")
for t in t_departs:
    tt = float(travel_time_auto(orig_id, dest_id, t, state, params, config))
    g = float(grad_od_tt_fn(cap0, t))
    print(f"  {t:>10.0f}  {tt:>10.1f}  {g:>12.4f}")

# --- AD: travel time along explicit fixed paths (fast vs slow) ---


def path_tt_of_cap(cap, t_dep, path_link_ids):
    p = _replace_fast2_cap(cap)
    s = simulate_duo(p, config)
    return travel_time(path_link_ids, t_dep, s, p, config)


grad_path_tt_fn = jax.grad(path_tt_of_cap, argnums=0)

for path_name, path_ids in (("fast (fast1->fast2)", fast_path),
                            ("slow (slow1->slow2)", slow_path)):
    print(f"\nAD: d(path_TT)/d(fast2.capacity)  path = {path_name}")
    print(f"  {'t_depart':>10}  {'TT (s)':>10}  {'dTT/dcap':>12}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*12}")
    for t in t_departs:
        tt = float(travel_time(path_ids, t, state, params, config))
        g = float(grad_path_tt_fn(cap0, t, path_ids))
        print(f"  {t:>10.0f}  {tt:>10.1f}  {g:>12.4f}")


# --- Finite difference validation of all AD gradients ---
# Forward: (f(x+d) - f(x)) / d
# Backward: (f(x) - f(x-d)) / d
# Central: (f(x+d) - f(x-d)) / 2d

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
print("  AD vs FiniteD: validation of all gradients (cap perturbation)")
print(f"{'='*70}")


def _sim_with_cap(cap):
    return simulate_duo(_replace_fast2_cap(cap), config)


# (1) TTT_all w.r.t. fast2.capacity
print(f"\n--- (1) dTTT_all/d(fast2.capacity) ---")
def eval_ttt_all(delta):
    return float(total_travel_time(_sim_with_cap(cap0 + delta), config))
print_finited_comparison("TTT_all", grad_ttt_all, eval_ttt_all)

# (2) avg_TT_link w.r.t. fast2.capacity (fast1 / fast2 / slow1 / slow2)
print(f"\n--- (2) d(avg_TT_link)/d(fast2.capacity) ---")
for lname, link_id in avg_tt_targets:
    grad_val = grad_avg_tt[lname]
    def eval_avg(delta, lid=link_id):
        return float(avg_tt_per_link(_sim_with_cap(cap0 + delta), config)[lid])
    print_finited_comparison(f"avg_TT_{lname}", grad_val, eval_avg)

# (3) OD travel time (shortest-path, orig->dest) w.r.t. fast2.capacity
print(f"\n--- (3) d(OD_TT, shortest-path)/d(fast2.capacity) ---")
for t in t_departs:
    ad_val = float(grad_od_tt_fn(cap0, t))
    def eval_od(delta, t=t):
        s = _sim_with_cap(cap0 + delta)
        p = _replace_fast2_cap(cap0 + delta)
        return float(travel_time_auto(orig_id, dest_id, t, s, p, config))
    print_finited_comparison(f"OD_TT, t={t:.0f}s", ad_val, eval_od)

# (4) Path travel times (fast/slow, fixed route) w.r.t. fast2.capacity
print(f"\n--- (4) d(path_TT)/d(fast2.capacity) ---")
for path_name, path_ids in (("fast", fast_path), ("slow", slow_path)):
    for t in t_departs:
        ad_val = float(grad_path_tt_fn(cap0, t, path_ids))
        def eval_path(delta, t=t, path_ids=path_ids):
            s = _sim_with_cap(cap0 + delta)
            p = _replace_fast2_cap(cap0 + delta)
            return float(travel_time(path_ids, t, s, p, config))
        print_finited_comparison(f"{path_name}, t={t:.0f}s", ad_val, eval_path)
