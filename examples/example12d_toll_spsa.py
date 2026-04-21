"""
Dynamic congestion pricing optimization via SPSA (no autodiff).

Same problem setting as example12b (Adam + autodiff) but uses
Simultaneous Perturbation Stochastic Approximation -- a gradient-free
method that estimates the gradient with only 2 function evaluations
per iteration, regardless of dimensionality.

Usage:
  python examples/example12d_toll_spsa.py

Environment variables:
  PEAK_FACTOR : peak demand multiplier (default 1.5)
  STEPS       : optimizer steps (default 5000)
  TEMPERATURE : logit temperature (default 10)
  REG_LAMBDA  : L2 regularization coefficient (default 0.001)
  SPSA_A      : stability constant A (default 100)
  SPSA_C      : perturbation magnitude c (default 10.0)
  SPSA_a      : step size a (default auto)
  SPSA_ALPHA  : step size decay exponent (default 0.602)
  SPSA_GAMMA  : perturbation decay exponent (default 0.101)
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from unsim.unsim_diff import simulate_duo, total_travel_time, trip_completed

# ================================================================
# 1. Build scenario (identical to example12b)
# ================================================================

print("=" * 60)
print("  Step 1: Build scenario (duo_logit)")
print("=" * 60)

TEMPERATURE = float(os.environ.get("TEMPERATURE", "10"))

from unsim import World
from unsim.unsim_diff import world_to_jax

N_FACTOR = float(os.environ.get("PEAK_FACTOR", "1.5"))

code = open(os.path.join(os.path.dirname(__file__),
                         "example06b_chicago_calibrated.py")).read()
parts = code.split("W.exec_simulation()")
g = {"__file__": os.path.join(os.path.dirname(__file__),
                               "example06b_chicago_calibrated.py")}
exec(parts[0], g)
W = g["W"]

W.ROUTE_CHOICE = "duo_logit"
W.LOGIT_TEMPERATURE = TEMPERATURE

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
# 2. Loss function (forward-only, no grad needed)
# ================================================================

REG_LAMBDA = float(os.environ.get("REG_LAMBDA", "0.001"))
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
# 3. JIT compile (forward only)
# ================================================================

print(f"\n{'='*60}")
print("  Step 2: JIT compile (forward only)")
print("=" * 60)

theta0 = jnp.zeros(n_params)

t0 = time.time()
loss_jit = jax.jit(loss_fn)
l0 = float(loss_jit(theta0))
t_fwd = time.time() - t0
print(f"  Forward compile+run: {t_fwd:.1f}s, loss={l0:.0f}")

t0 = time.time()
_ = float(loss_jit(theta0))
t_fwd_cached = time.time() - t0
print(f"  Cached forward: {t_fwd_cached:.2f}s")
print(f"  (No gradient compilation -- SPSA is gradient-free)")


# ================================================================
# 4. SPSA optimization
# ================================================================

STEPS = int(os.environ.get("STEPS", "5000"))

# SPSA hyperparameters (Spall 1998 guidelines)
SPSA_A = float(os.environ.get("SPSA_A", "100"))
SPSA_c = float(os.environ.get("SPSA_C", "10.0"))
SPSA_alpha = float(os.environ.get("SPSA_ALPHA", "0.602"))
SPSA_gamma = float(os.environ.get("SPSA_GAMMA", "0.101"))

# Auto-calibrate step size 'a' from initial gradient estimate
# Do a few pilot perturbations to estimate gradient magnitude
print(f"\n  Calibrating SPSA step size...")
rng = np.random.RandomState(42)
pilot_gnorms = []
theta_np = np.zeros(n_params, dtype=np.float32)
for _ in range(3):
    delta = rng.choice([-1.0, 1.0], size=n_params).astype(np.float32)
    theta_plus = jnp.array(np.maximum(theta_np + SPSA_c * delta, 0.0))
    theta_minus = jnp.array(np.maximum(theta_np - SPSA_c * delta, 0.0))
    lp = float(loss_jit(theta_plus))
    lm = float(loss_jit(theta_minus))
    g_hat = (lp - lm) / (2.0 * SPSA_c * delta)
    pilot_gnorms.append(np.linalg.norm(g_hat))

avg_gnorm = np.mean(pilot_gnorms)
SPSA_a = float(os.environ.get("SPSA_a", "0"))
if SPSA_a <= 0:
    # Target initial step ~ 5.0 (matching Adam effective step)
    SPSA_a = 5.0 * (SPSA_A + 1) ** SPSA_alpha / max(avg_gnorm, 1e-10)
print(f"  Pilot |g_hat| avg: {avg_gnorm:.1f}")
print(f"  SPSA params: a={SPSA_a:.4f}, c={SPSA_c}, A={SPSA_A}, "
      f"alpha={SPSA_alpha}, gamma={SPSA_gamma}")

print(f"\n{'='*60}")
print(f"  Step 3: SPSA optimization (steps={STEPS})")
print("=" * 60)

theta_np = np.zeros(n_params, dtype=np.float32)
best_ttt = ttt0
best_theta_np = theta_np.copy()

opt_losses = []
opt_ttts = []
opt_times = []
t_start = time.time()

for step in range(1, STEPS + 1):
    # Decaying step size and perturbation
    a_k = SPSA_a / (SPSA_A + step) ** SPSA_alpha
    c_k = SPSA_c / step ** SPSA_gamma

    # Random perturbation: Bernoulli +/-1
    delta = rng.choice([-1.0, 1.0], size=n_params).astype(np.float32)

    # Evaluate loss at theta +/- c_k * delta (with projection to >=0)
    theta_plus = jnp.array(np.maximum(theta_np + c_k * delta, 0.0))
    theta_minus = jnp.array(np.maximum(theta_np - c_k * delta, 0.0))
    lp = float(loss_jit(theta_plus))
    lm = float(loss_jit(theta_minus))

    # SPSA gradient estimate
    g_hat = (lp - lm) / (2.0 * c_k * delta)

    # Update with projection to non-negative
    theta_np = np.maximum(theta_np - a_k * g_hat, 0.0)

    # Logging
    if step == 1 or step % 10 == 0 or step == STEPS:
        theta_jnp = jnp.array(theta_np)
        l = float(loss_jit(theta_jnp))
        toll_arr = build_toll_array(theta_jnp)
        p = params._replace(toll=toll_arr)
        state = simulate_duo(p, config)
        ttt_val = float(total_travel_time(state, config))
        elapsed = time.time() - t_start
        toll_norm = float(np.linalg.norm(theta_np))
        g_norm = float(np.linalg.norm(g_hat))

        opt_losses.append(l)
        opt_ttts.append(ttt_val)
        opt_times.append(elapsed)

        if ttt_val < best_ttt:
            best_ttt = ttt_val
            best_theta_np = theta_np.copy()

        if step == 1 or step % 50 == 0 or step == STEPS:
            print(f"    step {step:5d}: loss={l:.0f}, TTT={ttt_val/3600:.0f} veh-hr "
                  f"({(ttt_val-ttt0)/ttt0*100:+.2f}%), "
                  f"|toll|={toll_norm:.1f}, |g|={g_norm:.0f}, "
                  f"a_k={a_k:.3f}, c_k={c_k:.2f}, t={elapsed:.0f}s",
                  flush=True)

theta_opt = jnp.array(best_theta_np)
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
print(f"  SPSA: {STEPS} steps ({STEPS*2} fwd evals), {t_total:.0f}s")
print("=" * 60)


# ================================================================
# 6. Save results
# ================================================================

outdir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(outdir, exist_ok=True)

save_path = os.path.join(outdir, "toll_spsa.npz")
np.savez(save_path,
         toll_full=np.asarray(toll_opt),
         toll_congested=best_theta_np,
         congested_idx=congested_idx,
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
