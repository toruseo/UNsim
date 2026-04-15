"""
Dynamic congestion pricing optimization for Chicago-Sketch network.

Uses duo_logit (soft logit routing) for differentiable toll optimization.
Optimizes per-link, per-time-period toll values to minimize:
  L(toll) = TTT(toll) + lambda * ||toll||_2^2

Usage:
  python examples/example12b_toll_optimization.py

Environment variables:
  PEAK_FACTOR : peak demand multiplier (default 1.5)
  STEPS       : optimizer steps (default 200)
  LR          : learning rate (default 1.0)
  REG_LAMBDA  : L2 regularization coefficient (default auto)
  TEMPERATURE : logit temperature (default 60)
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from unsim.unsim_diff import simulate_duo, total_travel_time, trip_completed

# ================================================================
# 1. Build scenario (duo_logit version of example12a)
# ================================================================

print("=" * 60)
print("  Step 1: Build scenario (duo_logit)")
print("=" * 60)

TEMPERATURE = float(os.environ.get("TEMPERATURE", "60"))

from unsim import World
from unsim.unsim_diff import world_to_jax

N_FACTOR = float(os.environ.get("PEAK_FACTOR", "1.5"))

# Load base Chicago network
code = open(os.path.join(os.path.dirname(__file__),
                         "example06b_chicago_calibrated.py")).read()
parts = code.split("W.exec_simulation()")
g = {"__file__": os.path.join(os.path.dirname(__file__),
                               "example06b_chicago_calibrated.py")}
exec(parts[0], g)
W = g["W"]

# Override route choice to duo_logit
W.ROUTE_CHOICE = "duo_logit"
W.LOGIT_TEMPERATURE = TEMPERATURE

# 3-period demand
original_demands = list(W.demand_info)
W.demand_info = []
for orig, dest, ts, te, flow in original_demands:
    dur = te - ts
    t1 = ts + dur / 3
    t2 = ts + 2 * dur / 3
    W.demand_info.append((orig, dest, ts, t1, flow))
    W.demand_info.append((orig, dest, t1, t2, flow * N_FACTOR))
    W.demand_info.append((orig, dest, t2, te, flow))

W.finalize_scenario()
params, config = world_to_jax(W)

n_links = int(config.n_links)
n_toll_steps = int(config.n_toll_steps)

# Identify congested links via baseline simulation
state0 = simulate_duo(params, config)
jax.block_until_ready(state0.cum_departure)

cum_arr = np.asarray(state0.cum_arrival[:, :config.tsize])
cum_dep = np.asarray(state0.cum_departure[:, :config.tsize])
max_count = (cum_arr - cum_dep).max(axis=1)
u_np = np.asarray(params.u)
lengths_np = np.asarray(config.link_lengths)
q_star_np = np.asarray(params.q_star)
ff_vehicles = q_star_np * lengths_np / u_np
congested_idx = np.where(max_count > ff_vehicles * 1.1)[0]

n_congested = len(congested_idx)
n_params = n_congested * n_toll_steps
congested_idx_jnp = jnp.array(congested_idx, dtype=jnp.int32)

ttt0 = float(total_travel_time(state0, config))
trips0 = float(trip_completed(state0, config))
total_vol = np.asarray(state0.cum_departure[:, -1])
ff_ttt = float(np.sum(total_vol * lengths_np / u_np))

print(f"  Network: {config.n_nodes} nodes, {n_links} links")
print(f"  Congested links: {n_congested}/{n_links}")
print(f"  Toll time steps: {n_toll_steps}")
print(f"  Decision variables: {n_params}")
print(f"  Logit temperature: {TEMPERATURE}s")
print(f"  TTT baseline: {ttt0/3600:.0f} veh-hr")
print(f"  Free-flow TTT: {ff_ttt/3600:.0f} veh-hr")
print(f"  Delay ratio: {(ttt0-ff_ttt)/ttt0*100:.1f}%")
print(f"  Trips completed: {trips0:.0f}")


# ================================================================
# 2. Loss function
# ================================================================

REG_LAMBDA = float(os.environ.get("REG_LAMBDA", "0"))
if REG_LAMBDA <= 0:
    toll_scale = 60.0
    REG_LAMBDA = 0.01 * ttt0 / max(n_params * toll_scale ** 2, 1.0)
print(f"\n  REG_LAMBDA: {REG_LAMBDA:.6f}")

base_toll = params.toll


def build_toll_array(theta_flat):
    """Build full toll array from congested-link parameters."""
    toll_cong = theta_flat.reshape(n_congested, n_toll_steps)
    return base_toll.at[congested_idx_jnp].set(toll_cong)


def loss_fn(theta_flat):
    """TTT + L2 regularization."""
    toll_arr = build_toll_array(theta_flat)
    p = params._replace(toll=toll_arr)
    state = simulate_duo(p, config)
    ttt = total_travel_time(state, config)
    l2 = jnp.sum(theta_flat ** 2)
    return ttt + REG_LAMBDA * l2


# ================================================================
# 3. JIT compile
# ================================================================

print(f"\n{'='*60}")
print("  Step 2: JIT compile")
print("=" * 60)

theta0 = jnp.zeros(n_params)

t0 = time.time()
loss_jit = jax.jit(loss_fn)
l0 = float(loss_jit(theta0))
t_fwd = time.time() - t0
print(f"  Forward compile+run: {t_fwd:.1f}s, loss={l0:.0f}")

t0 = time.time()
grad_jit = jax.jit(jax.grad(loss_fn))
g0 = grad_jit(theta0)
jax.block_until_ready(g0)
t_grad = time.time() - t0
g0_norm = float(jnp.linalg.norm(g0))
g0_nnz = int(jnp.sum(jnp.abs(g0) > 1e-10))
g0_nan = int(jnp.sum(jnp.isnan(g0)))
print(f"  Gradient compile+run: {t_grad:.1f}s")
print(f"  |grad|={g0_norm:.1f}, nonzero={g0_nnz}/{n_params}, nan={g0_nan}")

# Cached speed
t0 = time.time()
_ = float(loss_jit(theta0))
t_fwd_cached = time.time() - t0
t0 = time.time()
_ = grad_jit(theta0)
jax.block_until_ready(_)
t_grad_cached = time.time() - t0
print(f"  Cached: forward {t_fwd_cached:.2f}s, gradient {t_grad_cached:.2f}s")


# ================================================================
# 4. L-BFGS-B optimization
# ================================================================

STEPS = int(os.environ.get("STEPS", "200"))
LR = float(os.environ.get("LR", "1.0"))

print(f"\n{'='*60}")
print(f"  Step 3: Adam optimization (steps={STEPS}, lr={LR})")
print("=" * 60)

theta = jnp.zeros(n_params)
m_adam = jnp.zeros(n_params)
v_adam = jnp.zeros(n_params)
beta1, beta2, eps = 0.9, 0.999, 1e-8

GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "1e8"))

opt_losses = []
opt_ttts = []
opt_times = []
best_ttt = ttt0
best_theta = theta
t_start = time.time()

for step in range(1, STEPS + 1):
    g = grad_jit(theta)
    g_norm = float(jnp.linalg.norm(g))
    if g_norm > GRAD_CLIP:
        g = g * (GRAD_CLIP / g_norm)

    m_adam = beta1 * m_adam + (1 - beta1) * g
    v_adam = beta2 * v_adam + (1 - beta2) * g ** 2
    m_hat = m_adam / (1 - beta1 ** step)
    v_hat = v_adam / (1 - beta2 ** step)
    theta = theta - LR * m_hat / (jnp.sqrt(v_hat) + eps)
    theta = jnp.maximum(theta, 0.0)

    if step == 1 or step % 10 == 0 or step == STEPS:
        l = float(loss_jit(theta))
        toll_arr = build_toll_array(theta)
        p = params._replace(toll=toll_arr)
        state = simulate_duo(p, config)
        ttt_val = float(total_travel_time(state, config))
        elapsed = time.time() - t_start
        toll_norm = float(jnp.linalg.norm(theta))

        opt_losses.append(l)
        opt_ttts.append(ttt_val)
        opt_times.append(elapsed)

        if ttt_val < best_ttt:
            best_ttt = ttt_val
            best_theta = theta

        if step == 1 or step % 50 == 0 or step == STEPS:
            print(f"    step {step:4d}: loss={l:.0f}, TTT={ttt_val/3600:.0f} veh-hr "
                  f"({(ttt_val-ttt0)/ttt0*100:+.2f}%), "
                  f"|toll|={toll_norm:.1f}, |g|={g_norm:.0f}, t={elapsed:.0f}s",
                  flush=True)

theta_opt = best_theta
toll_opt = build_toll_array(theta_opt)
state_opt = simulate_duo(params._replace(toll=toll_opt), config)
jax.block_until_ready(state_opt.cum_departure)
ttt_opt = float(total_travel_time(state_opt, config))
trips_opt = float(trip_completed(state_opt, config))
t_total = time.time() - t_start

reduction = (ttt0 - ttt_opt) / ttt0 * 100
toll_vals = np.asarray(theta_opt)
toll_mean = float(np.mean(toll_vals[toll_vals > 0])) if np.any(toll_vals > 0) else 0
toll_max = float(np.max(toll_vals))
n_tolled = int(np.sum(toll_vals > 0.1))


# ================================================================
# 5. Summary
# ================================================================

print(f"\n{'='*60}")
print("  RESULT")
print("=" * 60)
print(f"  Baseline TTT:   {ttt0/3600:.0f} veh-hr")
print(f"  Optimized TTT:  {ttt_opt/3600:.0f} veh-hr ({reduction:+.2f}%)")
print(f"  Free-flow TTT:  {ff_ttt/3600:.0f} veh-hr")
print(f"  Trips baseline: {trips0:.0f}")
print(f"  Trips optimized:{trips_opt:.0f}")
print(f"  Toll: mean={toll_mean:.1f}s, max={toll_max:.1f}s, active={n_tolled}/{n_params}")
print(f"  Adam: {STEPS} steps, {t_total:.0f}s")
print("=" * 60)


# ================================================================
# 6. Save results
# ================================================================

outdir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(outdir, exist_ok=True)

# Save full toll array (n_links, n_toll_steps) and metadata
save_path = os.path.join(outdir, "toll_optimized.npz")
np.savez(save_path,
         toll_full=np.asarray(toll_opt),          # (n_links, n_toll_steps)
         toll_congested=np.asarray(theta_opt),     # (n_congested * n_toll_steps,)
         congested_idx=congested_idx,              # (n_congested,)
         n_toll_steps=n_toll_steps,
         temperature=TEMPERATURE,
         reg_lambda=REG_LAMBDA,
         ttt_baseline=ttt0,
         ttt_optimized=ttt_opt,
         ff_ttt=ff_ttt,
         losses=np.array(opt_losses),
         ttts=np.array(opt_ttts),
         times=np.array(opt_times))
print(f"\n  Saved: {save_path}")
