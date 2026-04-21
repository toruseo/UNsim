"""
Analyze optimized congestion pricing results from example12d.

Loads toll_spsa.npz, runs baseline and tolled simulations,
and generates comparison plots.

Usage:
  python examples/example12c_toll_analysis.py
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": plt.rcParams["font.size"] * 1.5})

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from unsim import World
from unsim.unsim_diff import world_to_jax, simulate_duo, total_travel_time, trip_completed

# ================================================================
# 1. Load toll data and build scenario
# ================================================================

print("=" * 60)
print("  Loading toll data and building scenario")
print("=" * 60)

result_dir = os.path.join(os.path.dirname(__file__), "..", "results")
data = np.load(os.path.join(result_dir, "toll_spsa.npz"))
toll_full = data["toll_full"]
TEMPERATURE = float(data["temperature"])
ttt_opt_saved = float(data["ttt_optimized"])
ttt_base_saved = float(data["ttt_baseline"])
print(f"  Saved result: TTT {ttt_base_saved/3600:.0f} -> {ttt_opt_saved/3600:.0f} veh-hr "
      f"({(ttt_base_saved-ttt_opt_saved)/ttt_base_saved*100:.1f}% reduction)")
print(f"  Temperature: {TEMPERATURE}")

# Build scenario (same as example12d)
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
dt = float(config.deltat)
tsize = int(config.tsize)
n_links = int(config.n_links)


# ================================================================
# 2. Run both simulations
# ================================================================

print("\n  Running baseline simulation...")
state_base = simulate_duo(params, config)
jax.block_until_ready(state_base.cum_departure)
ttt_base = float(total_travel_time(state_base, config))
print(f"  Baseline TTT: {ttt_base/3600:.0f} veh-hr")

print("  Running tolled simulation...")
toll_jnp = jnp.array(toll_full, dtype=jnp.float32)
state_toll = simulate_duo(params._replace(toll=toll_jnp), config)
jax.block_until_ready(state_toll.cum_departure)
ttt_toll = float(total_travel_time(state_toll, config))
print(f"  Tolled TTT: {ttt_toll/3600:.0f} veh-hr "
      f"({(ttt_base-ttt_toll)/ttt_base*100:.1f}% reduction)")

# Convert to numpy
ca_base = np.asarray(state_base.cum_arrival)   # (n_links, tsize+1)
cd_base = np.asarray(state_base.cum_departure)
ca_toll = np.asarray(state_toll.cum_arrival)
cd_toll = np.asarray(state_toll.cum_departure)
lengths = np.asarray(config.link_lengths)
u_vals = np.asarray(params.u)

# Time axis (hours)
t_hours = np.arange(tsize) * dt / 3600

imgdir = os.path.join(os.path.dirname(__file__), "..", "docs", "img")
os.makedirs(imgdir, exist_ok=True)


# ================================================================
# 3. Total toll amount time series
# ================================================================

print("\n  Generating plots...")

toll_np = np.asarray(toll_jnp)  # (n_links, n_toll_steps)
toll_step_size = int(config.toll_step_size)
n_toll_steps = int(config.n_toll_steps)

# Per-timestep toll revenue: sum over links of (toll * throughput)
# Throughput at each step: outflow_rate = (cum_dep[t+1] - cum_dep[t]) / dt
outflow_base = np.diff(cd_base, axis=1) / dt  # (n_links, tsize)
outflow_toll = np.diff(cd_toll, axis=1) / dt

# Toll at each simulation timestep
toll_per_step = np.zeros((n_links, tsize))
for t in range(tsize):
    k = min(t // toll_step_size, n_toll_steps - 1)
    toll_per_step[:, t] = toll_np[:, k]

# Total toll collected per timestep (veh*s / s = veh-seconds per second)
toll_revenue = np.sum(toll_per_step * outflow_toll, axis=0)
# Sum toll across all links per timestep (the "total toll level")
toll_total = np.sum(toll_per_step, axis=0)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_hours, toll_total / 3600, "b-", linewidth=1)
ax.set_xlabel("Time (h)")
ax.set_ylabel("Total toll across all links (h)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(imgdir, "spsa_total_timeseries.png"), dpi=150)
plt.close(fig)
print("    spsa_total_timeseries.png")


# ================================================================
# 4. Network vehicle count time series (with/without toll)
# ================================================================

n_veh_base = np.sum(np.maximum(ca_base[:, :tsize] - cd_base[:, :tsize], 0), axis=0)
n_veh_toll = np.sum(np.maximum(ca_toll[:, :tsize] - cd_toll[:, :tsize], 0), axis=0)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_hours, n_veh_base, "r-", label="No toll", linewidth=1)
ax.plot(t_hours, n_veh_toll, "b-", label="With toll", linewidth=1)
ax.set_xlabel("Time (h)")
ax.set_ylabel("Vehicles in network")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(imgdir, "spsa_vehicle_count.png"), dpi=150)
plt.close(fig)
print("    spsa_vehicle_count.png")


# ================================================================
# 5. Network average speed time series (with/without toll)
# ================================================================

# Average speed = total throughput / total vehicles
# Throughput: sum of (outflow * length) across all links
throughput_base = np.sum(outflow_base * lengths[:, None], axis=0)  # (tsize,) m/s total
throughput_toll = np.sum(outflow_toll * lengths[:, None], axis=0)

speed_base = np.where(n_veh_base > 1, throughput_base / n_veh_base, 0)
speed_toll = np.where(n_veh_toll > 1, throughput_toll / n_veh_toll, 0)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_hours, speed_base * 3.6, "r-", label="No toll", linewidth=1)
ax.plot(t_hours, speed_toll * 3.6, "b-", label="With toll", linewidth=1)
ax.set_xlabel("Time (h)")
ax.set_ylabel("Network average speed (km/h)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(imgdir, "spsa_avg_speed.png"), dpi=150)
plt.close(fig)
print("    spsa_avg_speed.png")


# ================================================================
# 6. MFD: vehicles vs throughput (with/without toll)
# ================================================================

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(n_veh_base, throughput_base, c="r", s=3, alpha=0.5, label="No toll")
ax.scatter(n_veh_toll, throughput_toll, c="b", s=3, alpha=0.5, label="With toll")
ax.set_xlabel("Network vehicle count")
ax.set_ylabel("Network throughput (veh-m/s)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(imgdir, "spsa_mfd.png"), dpi=150)
plt.close(fig)
print("    spsa_mfd.png")


# ================================================================
# 7. Network average plots (with/without toll) via Python simulation
# ================================================================

# Re-run Python simulation for analyzer access
def run_python_sim(use_toll):
    """Run Python simulation and return World with analyzer."""
    code2 = open(os.path.join(os.path.dirname(__file__),
                               "example06b_chicago_calibrated.py")).read()
    parts2 = code2.split("W.exec_simulation()")
    g2 = {"__file__": os.path.join(os.path.dirname(__file__),
                                    "example06b_chicago_calibrated.py")}
    exec(parts2[0], g2)
    Wp = g2["W"]
    Wp.ROUTE_CHOICE = "duo_logit"
    Wp.LOGIT_TEMPERATURE = TEMPERATURE

    orig_d = list(Wp.demand_info)
    Wp.demand_info = []
    for orig, dest, ts, te, flow in orig_d:
        dur = te - ts
        t1p = ts + dur / 3
        t2p = ts + 2 * dur / 3
        Wp.demand_info.append((orig, dest, ts, t1p, flow))
        Wp.demand_info.append((orig, dest, t1p, t2p, flow * N_FACTOR))
        Wp.demand_info.append((orig, dest, t2p, te, flow))

    if use_toll:
        # Apply toll via congestion_pricing on each link
        toll_step_dt = toll_step_size * dt
        for i, link in enumerate(Wp.LINKS):
            toll_vals_link = toll_full[i]
            if np.any(toll_vals_link > 0.01):
                def make_pricing(vals, step_dt, n_steps):
                    def pricing(t):
                        k = min(int(t / step_dt), n_steps - 1)
                        return float(vals[k])
                    return pricing
                link.congestion_pricing = make_pricing(toll_vals_link, toll_step_dt, n_toll_steps)

    Wp.exec_simulation()
    Wp.analyzer.basic_analysis()
    return Wp


print("\n  Running Python baseline for network_average plot...")
W_base = run_python_sim(use_toll=False)
fig_base = W_base.analyzer.network_average(show_labels=False, node_size=0)
fig_base.savefig(os.path.join(imgdir, "spsa_network_avg_baseline.png"),
                 dpi=150, bbox_inches="tight")
plt.close(fig_base)
print("    spsa_network_avg_baseline.png")

print("  Running Python tolled for network_average plot...")
W_toll_py = run_python_sim(use_toll=True)
fig_toll = W_toll_py.analyzer.network_average(show_labels=False, node_size=0)
fig_toll.savefig(os.path.join(imgdir, "spsa_network_avg_tolled.png"),
                 dpi=150, bbox_inches="tight")
plt.close(fig_toll)
print("    spsa_network_avg_tolled.png")


# ================================================================
# 8. Link-level time-averaged toll (network_average style)
# ================================================================

# Time-averaged toll per link (seconds)
link_avg_toll = np.mean(toll_np, axis=1)  # (n_links,)

fig, ax = plt.subplots(figsize=(6, 6))
cmap = plt.colormaps["YlOrRd"]
max_toll_val = np.max(link_avg_toll) if np.max(link_avg_toll) > 0 else 1

for i, link in enumerate(W_toll_py.LINKS):
    ox, oy = W_toll_py.analyzer._link_offset(link, left_handed=True)
    x1, y1 = link.start_node.x + ox, link.start_node.y + oy
    x2, y2 = link.end_node.x + ox, link.end_node.y + oy

    toll_val = link_avg_toll[i]
    color_val = np.clip(toll_val / max_toll_val, 0, 1)
    color = cmap(color_val)

    ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.5,
            solid_capstyle="butt", zorder=6)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_toll_val / 60))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
cbar.set_label("Time-averaged toll (min)")

ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(os.path.join(imgdir, "spsa_link_avg.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)
print("    spsa_link_avg.png")


# ================================================================
# 9. Convergence curves (loss, TTT, |g| vs iteration)
# ================================================================

# Parse optimization log if available
logpath = os.path.join(result_dir, "spsa_final.log")
if os.path.exists(logpath):
    import re
    steps_log, losses_log, ttts_log, gnorms_log = [], [], [], []
    with open(logpath) as f:
        for line in f:
            m = re.search(
                r"step\s+(\d+):\s+loss=(\d+),\s+TTT=(\d+)\s.*\|g\|=([0-9.e+]+)",
                line)
            if m:
                steps_log.append(int(m.group(1)))
                losses_log.append(float(m.group(2)))
                ttts_log.append(float(m.group(3)))
                gnorms_log.append(float(m.group(4)))

    if steps_log:
        steps_arr = np.array(steps_log)
        losses_arr = np.array(losses_log)
        ttts_arr = np.array(ttts_log)
        gnorms_arr = np.array(gnorms_log)

        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

        axes[0].plot(steps_arr, losses_arr / 1e9, "b-", linewidth=0.8)
        axes[0].set_ylabel("Loss (x1e9)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps_arr, ttts_arr / 1000, "r-", linewidth=0.8)
        axes[1].set_ylabel("TTT (x1000 veh-hr)")
        axes[1].grid(True, alpha=0.3)

        axes[2].semilogy(steps_arr, gnorms_arr, "g-", linewidth=0.5,
                          alpha=0.7)
        axes[2].set_ylabel("|grad|")
        axes[2].set_xlabel("Iteration")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(imgdir, "spsa_convergence.png"), dpi=150)
        plt.close(fig)
        print("    spsa_convergence.png")
else:
    print("    (no spsa_final.log found, skipping convergence plot)")


# ================================================================
# Summary
# ================================================================

print(f"\n{'='*60}")
print("  Analysis complete")
print("=" * 60)
print(f"  Baseline TTT: {ttt_base/3600:.0f} veh-hr")
print(f"  Tolled TTT:   {ttt_toll/3600:.0f} veh-hr ({(ttt_base-ttt_toll)/ttt_base*100:.1f}%)")
print(f"  Plots saved to: {imgdir}/")
print("=" * 60)
