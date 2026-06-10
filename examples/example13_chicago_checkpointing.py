"""
Chicago-Sketch gradient computation with checkpointing to reduce GPU memory.

Demonstrates the ``checkpoint_every`` option of ``simulate_duo``. Reverse-mode automatic differentiation through the simulation loop normally stores intermediate values for every timestep, so peak GPU memory grows linearly with the number of timesteps. With ``checkpoint_every=k``, only the loop state at every k-th timestep is stored and the rest is recomputed during the backward pass: AD-tape memory scales as O(tsize/k + k) instead of O(tsize), at the cost of roughly one extra forward pass (~25-35 percent wall time). k ~ sqrt(tsize) minimizes the AD-tape memory; k=None gives the default (fastest, highest memory). Results are identical.

This script computes the gradient of total travel time with respect to the OD demand rates once and reports wall time and peak GPU memory. On this network (tsize=360), CHECKPOINT_EVERY=19 reduces peak memory from ~17 GiB (default) to ~2 GiB.

Requires JAX: pip install jax[cuda12] (or jax[cpu])

Sample result with RTX 5060 Ti in WSL:
Network: 927 nodes, 2557 links, 200 destinations, tsize=360
CHECKPOINT_EVERY: 19
Device: cuda:0
TTT: 2.387023e+09 s
forward compile: 1.76 s
forward run:     0.33 s
grad compile:    7.36 s
grad run:        1.47 s
peak memory (forward): 1.02 GiB
peak memory (grad):    1.99 GiB
"""

import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Disable XLA preallocation so memory statistics reflect actual use.
# Must be set before importing jax.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
from unsim.unsim_diff import world_to_jax, simulate_duo, total_travel_time

CHECKPOINT_EVERY = 19  # ~sqrt(tsize); None disables checkpointing

# --- Build the calibrated Chicago-Sketch network (same as example12a) ---

code = open(os.path.join(os.path.dirname(__file__),
                         "example06b_chicago_calibrated.py")).read()
g = {"__file__": os.path.join(os.path.dirname(__file__),
                              "example06b_chicago_calibrated.py")}
exec(code.split("W.exec_simulation()")[0], g)
W = g["W"]

params, config = world_to_jax(W)
print(f"Network: {config.n_nodes} nodes, {config.n_links} links, "
      f"{config.n_dests} destinations, tsize={config.tsize}")
print(f"CHECKPOINT_EVERY: {CHECKPOINT_EVERY}")
print(f"Device: {jax.devices()[0]}")

# --- Forward simulation (no grad) ---

def forward_fn(od_demand_rate):
    p = params._replace(od_demand_rate=od_demand_rate)
    state = simulate_duo(p, config, checkpoint_every=CHECKPOINT_EVERY)
    return total_travel_time(state, config)

t0 = time.time()
fwd_compiled = jax.jit(forward_fn).lower(params.od_demand_rate).compile()
t_compile_fwd = time.time() - t0

dev = jax.local_devices()[0]
def _peak_bytes():
    stats = dev.memory_stats()
    if stats and "peak_bytes_in_use" in stats:
        return stats["peak_bytes_in_use"]
    return None

peak_before_fwd = _peak_bytes()
t0 = time.time()
ttt = fwd_compiled(params.od_demand_rate)
jax.block_until_ready(ttt)
t_fwd = time.time() - t0
peak_after_fwd = _peak_bytes()

# --- Gradient of total travel time w.r.t. OD demand rates ---

t0 = time.time()
vg = jax.jit(jax.value_and_grad(forward_fn)).lower(params.od_demand_rate).compile()
t_compile_grad = time.time() - t0

t0 = time.time()
ttt_g, grad = vg(params.od_demand_rate)
jax.block_until_ready(grad)
t_grad = time.time() - t0
peak_after_grad = _peak_bytes()

print(f"TTT: {float(ttt):.6e} s")
print(f"forward compile: {t_compile_fwd:.2f} s")
print(f"forward run:     {t_fwd:.2f} s")
print(f"grad compile:    {t_compile_grad:.2f} s")
print(f"grad run:        {t_grad:.2f} s")
if peak_after_grad is not None:
    print(f"peak memory (forward): {peak_after_fwd / 2**30:.2f} GiB")
    print(f"peak memory (grad):    {peak_after_grad / 2**30:.2f} GiB")
else:
    print("Something is wrong. It looks like GPU was not used")
