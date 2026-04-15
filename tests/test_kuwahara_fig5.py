"""
Quantitative comparison against Kuwahara & Akamatsu (2001) Fig. 5-7.

Checks:
- Initial demand slopes match OD demand rates
- Queue forms at link (6,3) bottleneck around t=1.35h
- Route switching occurs for dest 3 (arterial used after congestion)
- Off-ramp (5,2) is not used significantly by dest 3
- Departure rate from (5,6) bottleneck ~ 3000 veh/h after congestion
- Dest 2 route switch timing around t=2.0h
"""
import sys, os, pytest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unsim import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_kuwahara():
    W = World(tmax=5*3600, print_mode=0, route_choice="duo_multipoint")
    W.addNode("1", -2, 0); W.addNode("2", 0, -1); W.addNode("3", 2, -1)
    W.addNode("4", -1, 1); W.addNode("5", 0.5, 1); W.addNode("6", 1.5, 1)
    for s, e, L, lw, lwp, km, fm in [
        ("1","4",2,0.05,0.1,450,6000), ("4","5",16,0.2,0.8,375,6000),
        ("5","6",8,0.1,0.4,250,4000), ("6","3",2,0.05,0.1,225,3000),
        ("1","2",16,0.4,0.8,450,6000), ("2","3",12,0.3,0.6,450,6000),
        ("5","2",2,0.05,0.1,450,6000)]:
        W.addLink(f"{s}_{e}", s, e, length=L*1000,
                  free_flow_speed=(L/lw)*1000/3600,
                  backward_wave_speed=(L/lwp)*1000/3600,
                  capacity=fm/3600)
    W.adddemand("1", "2", 0, 3600, 1000/3600)
    W.adddemand("1", "2", 3600, 10800, 2000/3600)
    W.adddemand("1", "3", 0, 3600, 2000/3600)
    W.adddemand("1", "3", 3600, 10800, 4000/3600)
    return W


W = build_kuwahara()
W.exec_simulation()
dt = W.DELTAT

print(f"dt = {dt:.1f}s = {dt/3600:.4f}h")
print(f"TSIZE = {W.TSIZE}, tmax = {W.TSIZE * dt:.0f}s = {W.TSIZE * dt / 3600:.2f}h")

# Extract per-destination cumulative arrivals
dests = W._destinations  # destination names
print(f"Destinations: {dests}")

# Time array in hours
t_hours = np.array([t * dt / 3600 for t in range(W.TSIZE + 1)])

# Links of interest
links = {name: W.get_link(name) for name in ["1_4", "4_5", "5_6", "6_3", "1_2", "2_3", "5_2"]}

# Per-destination cumulative arrivals
cum_arr_d = {}
for lname, link in links.items():
    cum_arr_d[lname] = {}
    for d in dests:
        cum_arr_d[lname][d] = np.array(link.cum_arrival_d[d])

# Aggregate cumulative arrivals/departures
cum_arr = {lname: np.array(link.cum_arrival) for lname, link in links.items()}
cum_dep = {lname: np.array(link.cum_departure) for lname, link in links.items()}

# ===== Quantitative checks =====
print("\n===== Quantitative Checks =====")

# 1. Initial slopes: dest-3 demand = 2000 veh/h for t<1h
# All dest-3 initially on freeway, so A_14^3 slope should be ~2000 veh/h
t_check = int(0.5 * 3600 / dt)  # t=0.5h
if t_check > 0:
    slope_14_d3 = cum_arr_d["1_4"]["3"][t_check] / (t_check * dt) * 3600  # veh/h
    print(f"1. Initial slope A_14^3 at t=0.5h: {slope_14_d3:.0f} veh/h (expected ~2000)")
    # Dest-2 initial demand = 1000 veh/h, also on freeway initially
    slope_14_d2 = cum_arr_d["1_4"]["2"][t_check] / (t_check * dt) * 3600
    print(f"   Initial slope A_14^2 at t=0.5h: {slope_14_d2:.0f} veh/h (expected ~1000)")
    # Total on freeway = 3000 veh/h
    slope_14_total = cum_arr["1_4"][t_check] / (t_check * dt) * 3600
    print(f"   Initial slope A_14 total at t=0.5h: {slope_14_total:.0f} veh/h (expected ~3000)")

# 2. After t=1h, dest-3 demand increases to 4000 veh/h
# Check slope around t=1.2h (before queue formation at 1.35h)
t1 = int(1.05 * 3600 / dt)
t2 = int(1.25 * 3600 / dt)
if t2 > t1:
    slope_14_d3_peak = (cum_arr_d["1_4"]["3"][t2] - cum_arr_d["1_4"]["3"][t1]) / ((t2 - t1) * dt) * 3600
    print(f"\n2. Slope A_14^3 at t=1.05-1.25h: {slope_14_d3_peak:.0f} veh/h (expected ~4000)")
    slope_14_d2_peak = (cum_arr_d["1_4"]["2"][t2] - cum_arr_d["1_4"]["2"][t1]) / ((t2 - t1) * dt) * 3600
    print(f"   Slope A_14^2 at t=1.05-1.25h: {slope_14_d2_peak:.0f} veh/h (expected ~2000)")

# 3. Bottleneck at link (6,3): capacity = 3000 veh/h
# Queue should form when dest-3 demand (4000) > capacity (3000)
# Check departure rate from (5,6) after congestion
t_cong = int(2.0 * 3600 / dt)  # well after congestion
t_cong_prev = int(1.8 * 3600 / dt)
if t_cong > t_cong_prev:
    dep_rate_56 = (cum_dep["5_6"][t_cong] - cum_dep["5_6"][t_cong_prev]) / ((t_cong - t_cong_prev) * dt) * 3600
    print(f"\n3. Departure rate from (5,6) at t=1.8-2.0h: {dep_rate_56:.0f} veh/h (expected ~3000)")

# 4. Off-ramp (5,2) should carry minimal dest-3 traffic
# Paper says off-ramp never used by dest 3
final_52_d3 = cum_arr_d["5_2"]["3"][-1]
final_52_d2 = cum_arr_d["5_2"]["2"][-1]
print(f"\n4. Off-ramp (5,2) total arrivals:")
print(f"   dest 3: {final_52_d3:.0f} veh (expected ~0)")
print(f"   dest 2: {final_52_d2:.0f} veh")

# 5. Route switching: arterial link (1,2) should eventually carry dest-3 traffic
# Paper shows A_12^3 appears around t~2h
final_12_d3 = cum_arr_d["1_2"]["3"][-1]
print(f"\n5. Arterial (1,2) total dest-3 arrivals: {final_12_d3:.0f} veh (expected > 0, route switch)")

# Check when arterial first gets significant dest-3 traffic
for t in range(W.TSIZE + 1):
    if cum_arr_d["1_2"]["3"][t] > 50:
        print(f"   First significant A_12^3 > 50 at t={t_hours[t]:.2f}h (paper: ~2.01h)")
        break

# 6. Route switching for dest 2: should start around t~2.01h
for t in range(W.TSIZE + 1):
    if cum_arr_d["1_2"]["2"][t] > 50:
        print(f"\n6. First significant A_12^2 > 50 at t={t_hours[t]:.2f}h (paper: ~2.01h)")
        break

# 7. Queue formation timing: N_U > N_D on link (5,6) indicates queue
for t in range(W.TSIZE + 1):
    n_on = cum_arr["5_6"][t] - cum_dep["5_6"][t]
    if n_on > 100:  # significant queue
        print(f"\n7. First significant queue on (5,6) (>100 veh) at t={t_hours[t]:.2f}h (paper: ~1.35h)")
        break

# ===== Generate Fig. 6-like plot =====
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Fig 6: Dest 3 cumulative arrivals
ax = axes[0]
for lname, label in [("1_4", "$A_{14}^3$"), ("4_5", "$A_{45}^3$"),
                       ("5_6", "$A_{56}^3$"), ("6_3", "$A_{63}^3$"),
                       ("1_2", "$A_{12}^3$"), ("2_3", "$A_{23}^3$")]:
    arr = cum_arr_d[lname]["3"]
    if arr[-1] > 10:
        ax.plot(t_hours, arr, label=label)
ax.set_xlabel("Time (h)"); ax.set_ylabel("Cumulative Trips (veh)")
ax.set_title("Dest 3: Cumulative Arrivals (cf. Fig. 6)")
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 5); ax.set_ylim(0, max(cum_arr_d["1_4"]["3"][-1], 1) * 1.1)

# Fig 7: Dest 2 cumulative arrivals
ax = axes[1]
for lname, label in [("1_4", "$A_{14}^2$"), ("4_5", "$A_{45}^2$"),
                       ("5_2", "$A_{52}^2$"), ("1_2", "$A_{12}^2$"),
                       ("2_3", "$A_{23}^2$")]:
    if lname in cum_arr_d:
        arr = cum_arr_d[lname]["2"]
        if arr[-1] > 10:
            ax.plot(t_hours, arr, label=label)
ax.set_xlabel("Time (h)"); ax.set_ylabel("Cumulative Trips (veh)")
ax.set_title("Dest 2: Cumulative Arrivals (cf. Fig. 7)")
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 5); ax.set_ylim(0, max(cum_arr_d["1_4"]["2"][-1], 1) * 1.1)

plt.tight_layout()
outdir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, "kuwahara_fig6_7_comparison.png")
fig.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved to {outpath}")
