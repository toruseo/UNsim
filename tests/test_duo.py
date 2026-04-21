"""
Tests for DUO route choice models (duo, duo_logit, JAX duo_logit)
and dynamic congestion pricing (toll).
Based on Kuwahara & Akamatsu (2001) Fig. 5 scenario.

Also includes standalone DUO comparison report (run as script):
  python tests/test_duo.py

Network: Freeway + Arterial with off-ramp
  1(Work) -> 4 -> 5 -> 6 -> 3(Residential)   [Freeway]
  1(Work) -> 2 -> 3(Residential)             [Arterial]
  5 -> 2                                     [Off-ramp]

OD: 1->2, 1->3 (time-varying)
"""

import time
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unsim import *


# ================================================================
# Scenario builders
# ================================================================

def _build_kuwahara(route_choice="duo", logit_temperature=60.0):
    """Build the Kuwahara & Akamatsu (2001) Fig. 5 network.

    Parameters
    ----------
    route_choice : str
        "duo" or "duo_logit".
    logit_temperature : float
        Temperature for duo_logit (ignored for duo).
    """
    kwargs = {"tmax": 5*3600, "print_mode": 0, "route_choice": route_choice}
    if route_choice == "duo_logit":
        kwargs["logit_temperature"] = logit_temperature
    W = World(**kwargs)

    W.addNode("1", -2, 0)
    W.addNode("2", 0, -1)
    W.addNode("3", 2, -1)
    W.addNode("4", -1, 1)
    W.addNode("5", 0.5, 1)
    W.addNode("6", 1.5, 1)

    link_data = [
        ("1", "4", 2,  0.05, 0.1, 450, 6000),
        ("4", "5", 16, 0.2,  0.8, 375, 6000),
        ("5", "6", 8,  0.1,  0.4, 250, 4000),
        ("6", "3", 2,  0.05, 0.1, 225, 3000),
        ("1", "2", 16, 0.4,  0.8, 450, 6000),
        ("2", "3", 12, 0.3,  0.6, 450, 6000),
        ("5", "2", 2,  0.05, 0.1, 450, 6000),
    ]
    for s, e, L, lw, lwp, km, fm in link_data:
        W.addLink(f"{s}_{e}", s, e, length=L*1000,
                  free_flow_speed=(L/lw)*1000/3600,
                  backward_wave_speed=(L/lwp)*1000/3600,
                  capacity=fm/3600)

    W.adddemand("1", "2", 0, 3600, 1000/3600)
    W.adddemand("1", "2", 3600, 10800, 2000/3600)
    W.adddemand("1", "3", 0, 3600, 2000/3600)
    W.adddemand("1", "3", 3600, 10800, 4000/3600)
    return W


# ================================================================
# Python DUO tests
# ================================================================

class TestDUO:
    """Python DUO (hard route choice) tests."""

    def test_runs(self):
        W = _build_kuwahara("duo")
        W.exec_simulation()
        W.analyzer.basic_analysis()
        assert W.analyzer.trip_all > 0

    def test_initial_freeflow_route(self):
        """Initially, all dest-3 traffic uses freeway (faster in free flow)."""
        W = _build_kuwahara("duo")
        W.exec_simulation()
        dt = W.DELTAT
        t_idx = int(1000 / dt)
        assert W.get_link("1_4").cum_arrival[min(t_idx, W.TSIZE)] > \
               W.get_link("1_2").cum_arrival[min(t_idx, W.TSIZE)]

    def test_route_switching(self):
        """After congestion, some traffic switches to arterial."""
        W = _build_kuwahara("duo")
        W.exec_simulation()
        assert W.get_link("1_2").cum_arrival[-1] > 100

    def test_flow_conservation(self):
        W = _build_kuwahara("duo")
        W.exec_simulation()
        W.analyzer.basic_analysis()
        total_demand = W.analyzer.trip_all
        total_absorbed = sum(n.absorbed_count for n in W.NODES)
        remaining = W.get_node("1").demand_queue + \
                    sum(l.cum_arrival[-1] - l.cum_departure[-1] for l in W.LINKS)
        assert equal_tolerance(total_absorbed + remaining, total_demand, rel_tol=0.1)

    def test_per_destination_consistency(self):
        """Sum of per-destination counts equals aggregate."""
        W = _build_kuwahara("duo")
        W.exec_simulation()
        for link in W.LINKS:
            for t in range(0, W.TSIZE + 1, max(1, W.TSIZE // 10)):
                sum_arr = sum(link.cum_arrival_d[d][t] for d in W._destinations)
                assert abs(link.cum_arrival[t] - sum_arr) < 1


# ================================================================
# Python DUO logit tests
# ================================================================

class TestDUOLogit:
    """Python DUO logit (soft route choice) tests."""

    def test_runs(self):
        W = _build_kuwahara("duo_logit")
        W.exec_simulation()
        W.analyzer.basic_analysis()
        assert W.analyzer.trip_all > 0

    def test_initial_freeflow_route(self):
        """Most dest-3 traffic uses freeway initially (with logit leakage)."""
        W = _build_kuwahara("duo_logit")
        W.exec_simulation()
        dt = W.DELTAT
        t_idx = int(1000 / dt)
        assert W.get_link("1_4").cum_arrival[min(t_idx, W.TSIZE)] > \
               W.get_link("1_2").cum_arrival[min(t_idx, W.TSIZE)]

    def test_route_switching(self):
        W = _build_kuwahara("duo_logit")
        W.exec_simulation()
        assert W.get_link("1_2").cum_arrival[-1] > 100

    def test_flow_conservation(self):
        W = _build_kuwahara("duo_logit")
        W.exec_simulation()
        W.analyzer.basic_analysis()
        total_demand = W.analyzer.trip_all
        total_absorbed = sum(n.absorbed_count for n in W.NODES)
        remaining = W.get_node("1").demand_queue + \
                    sum(l.cum_arrival[-1] - l.cum_departure[-1] for l in W.LINKS)
        assert equal_tolerance(total_absorbed + remaining, total_demand, rel_tol=0.1)

    def test_per_destination_consistency(self):
        W = _build_kuwahara("duo_logit")
        W.exec_simulation()
        for link in W.LINKS:
            for t in range(0, W.TSIZE + 1, max(1, W.TSIZE // 10)):
                sum_arr = sum(link.cum_arrival_d[d][t] for d in W._destinations)
                assert abs(link.cum_arrival[t] - sum_arr) < 1

    def test_low_temperature_approaches_duo(self):
        """Very low temperature logit should approximate hard DUO."""
        W_duo = _build_kuwahara("duo")
        W_duo.exec_simulation()
        W_duo.analyzer.basic_analysis()

        W_logit = _build_kuwahara("duo_logit", logit_temperature=1.0)
        W_logit.exec_simulation()
        W_logit.analyzer.basic_analysis()

        assert equal_tolerance(W_logit.analyzer.total_travel_time,
                               W_duo.analyzer.total_travel_time, rel_tol=0.3)


# ================================================================
# JAX DUO logit tests
# ================================================================

try:
    import jax
    import jax.numpy as jnp
    from unsim.unsim_diff import world_to_jax, simulate_duo, total_travel_time
    HAS_JAX = True
except (ImportError, RuntimeError):
    HAS_JAX = False


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestDUOLogitJAX:
    """JAX duo_logit: numerical agreement and gradient support."""

    def test_jax_runs(self):
        W = _build_kuwahara("duo_logit")
        params, config = world_to_jax(W)
        state = simulate_duo(params, config)
        ttt = total_travel_time(state, config)
        assert jnp.isfinite(ttt) and float(ttt) > 0

    def test_jax_matches_python(self):
        W = _build_kuwahara("duo_logit")
        W.exec_simulation()
        W.analyzer.basic_analysis()
        ttt_py = W.analyzer.total_travel_time

        W2 = _build_kuwahara("duo_logit")
        params, config = world_to_jax(W2)
        ttt_jax = float(total_travel_time(simulate_duo(params, config), config))
        assert equal_tolerance(ttt_jax, ttt_py, rel_tol=0.3)

    def test_grad_od_demand_finite(self):
        W = _build_kuwahara("duo_logit")
        params, config = world_to_jax(W)
        grads = jax.grad(lambda p: total_travel_time(
            simulate_duo(p, config), config))(params)
        assert jnp.all(jnp.isfinite(grads.od_demand_rate))

    def test_grad_toll_nonzero(self):
        """Toll gradient is non-zero for duo_logit."""
        W = World(tmax=3600, print_mode=0, route_choice="duo_logit",
                  logit_temperature=60.0)
        W.addNode("O", 0, 0)
        W.addNode("D", 1, 0)
        W.addLink("a", "O", "D", length=1000, free_flow_speed=25,
                  congestion_pricing=lambda t: 50)
        W.addLink("b", "O", "D", length=1000, free_flow_speed=20)
        W.adddemand("O", "D", 0, 3600, 0.2)
        params, config = world_to_jax(W)
        grads = jax.grad(lambda p: total_travel_time(
            simulate_duo(p, config), config))(params)
        assert jnp.all(jnp.isfinite(grads.toll))
        assert float(jnp.linalg.norm(grads.toll)) > 1e-6


# ================================================================
# Dynamic congestion pricing (toll) tests
# ================================================================

def _build_two_route(toll_a=None, toll_b=None,
                     u_a=20, u_b=20, tmax=3600, demand_flow=0.2):
    """Build 2-route parallel network with optional toll."""
    W = World(tmax=tmax, print_mode=0, route_choice="duo")
    W.addNode("O", 0, 0)
    W.addNode("D", 1, 0)
    W.addLink("link_a", "O", "D", length=1000, free_flow_speed=u_a,
              congestion_pricing=toll_a)
    W.addLink("link_b", "O", "D", length=1000, free_flow_speed=u_b,
              congestion_pricing=toll_b)
    W.adddemand("O", "D", 0, tmax, demand_flow)
    return W


class TestTollPython:
    """Congestion pricing controls DUO route choice (Python)."""

    def test_toll_shifts_traffic(self):
        W = _build_two_route(toll_a=lambda t: 100, u_a=25, u_b=20)
        W.exec_simulation()
        arr_a = W.LINKS_NAME_DICT["link_a"].cum_arrival[W.TSIZE]
        arr_b = W.LINKS_NAME_DICT["link_b"].cum_arrival[W.TSIZE]
        assert arr_b > 0
        assert arr_a < arr_b * 0.05

    def test_no_toll_faster_preferred(self):
        W = _build_two_route(u_a=25, u_b=20)
        W.exec_simulation()
        arr_a = W.LINKS_NAME_DICT["link_a"].cum_arrival[W.TSIZE]
        arr_b = W.LINKS_NAME_DICT["link_b"].cum_arrival[W.TSIZE]
        assert arr_a > arr_b

    def test_toll_reverses_preference(self):
        W_free = _build_two_route(u_a=25, u_b=20)
        W_free.exec_simulation()
        W_toll = _build_two_route(toll_a=lambda t: 100, u_a=25, u_b=20)
        W_toll.exec_simulation()
        assert W_free.LINKS_NAME_DICT["link_a"].cum_arrival[W_free.TSIZE] > \
               W_free.LINKS_NAME_DICT["link_b"].cum_arrival[W_free.TSIZE]
        assert W_toll.LINKS_NAME_DICT["link_b"].cum_arrival[W_toll.TSIZE] > \
               W_toll.LINKS_NAME_DICT["link_a"].cum_arrival[W_toll.TSIZE]

    def test_time_varying_toll(self):
        mid = 1800
        W = World(tmax=3600, print_mode=0, route_choice="duo")
        W.addNode("O", 0, 0); W.addNode("D", 1, 0)
        W.addLink("link_a", "O", "D", length=1000, free_flow_speed=20,
                  congestion_pricing=lambda t: 200 if t < mid else 0)
        W.addLink("link_b", "O", "D", length=1000, free_flow_speed=20,
                  congestion_pricing=lambda t: 0 if t < mid else 200)
        W.adddemand("O", "D", 0, 3600, 0.2)
        W.exec_simulation()
        t_mid = int(mid / W.DELTAT)
        la = W.LINKS_NAME_DICT["link_a"]
        lb = W.LINKS_NAME_DICT["link_b"]
        assert lb.cum_arrival[t_mid] > la.cum_arrival[t_mid]
        assert (la.cum_arrival[W.TSIZE] - la.cum_arrival[t_mid]) > \
               (lb.cum_arrival[W.TSIZE] - lb.cum_arrival[t_mid])

    def test_zero_toll_no_effect(self):
        W_none = _build_two_route(u_a=25, u_b=20)
        W_none.exec_simulation()
        W_zero = _build_two_route(toll_a=lambda t: 0, toll_b=lambda t: 0,
                                  u_a=25, u_b=20)
        W_zero.exec_simulation()
        for ln in ["link_a", "link_b"]:
            a1 = W_none.LINKS_NAME_DICT[ln].cum_arrival[W_none.TSIZE]
            a2 = W_zero.LINKS_NAME_DICT[ln].cum_arrival[W_zero.TSIZE]
            assert abs(a1 - a2) < 0.1


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestTollJAX:
    """Congestion pricing in JAX DUO."""

    def test_jax_runs(self):
        W = _build_two_route(toll_a=lambda t: 100, u_a=25, u_b=20)
        params, config = world_to_jax(W)
        state = simulate_duo(params, config)
        assert jnp.isfinite(total_travel_time(state, config))

    def test_jax_shifts_traffic(self):
        W = _build_two_route(toll_a=lambda t: 100, u_a=25, u_b=20)
        params, config = world_to_jax(W)
        state = simulate_duo(params, config)
        assert float(state.cum_arrival[0, -1]) < float(state.cum_arrival[1, -1]) * 0.05

    def test_python_jax_agreement(self):
        W_py = _build_two_route(toll_a=lambda t: 100, u_a=25, u_b=20)
        W_py.exec_simulation(); W_py.analyzer.basic_analysis()
        W_jax = _build_two_route(toll_a=lambda t: 100, u_a=25, u_b=20)
        params, config = world_to_jax(W_jax)
        state = simulate_duo(params, config)
        assert equal_tolerance(float(total_travel_time(state, config)),
                               W_py.analyzer.total_travel_time, rel_tol=0.3)

    def test_grad_finite(self):
        W = _build_two_route(toll_a=lambda t: 50, u_a=25, u_b=20)
        params, config = world_to_jax(W)
        grads = jax.grad(lambda p: total_travel_time(
            simulate_duo(p, config), config))(params)
        assert jnp.all(jnp.isfinite(grads.toll))

    def test_toll_array_populated(self):
        W = _build_two_route(toll_a=lambda t: 42, toll_b=lambda t: 0)
        params, config = world_to_jax(W)
        assert jnp.all(params.toll[0] == 42.0)
        assert jnp.all(params.toll[1] == 0.0)


# ================================================================
# Standalone DUO comparison report (run as: python tests/test_duo.py)
# ================================================================

def _build_grid():
    """Build 6x6 grid network with 6 OD pairs."""
    GRID = 6
    W = World(tmax=21600, print_mode=0, route_choice="duo")
    for i in range(GRID):
        for j in range(GRID):
            W.addNode(f"n{i}{j}", j*2, -i*2)
    for i in range(GRID):
        for j in range(GRID-1):
            u_r = 18 + 0.5*j + 0.3*i
            u_l = 17 + 0.4*(GRID-j) + 0.2*i
            W.addLink(f"h{i}{j}r", f"n{i}{j}", f"n{i}{j+1}",
                      length=2000, free_flow_speed=u_r, backward_wave_speed=5, capacity=0.6)
            W.addLink(f"h{i}{j}l", f"n{i}{j+1}", f"n{i}{j}",
                      length=2000, free_flow_speed=u_l, backward_wave_speed=5, capacity=0.6)
    for i in range(GRID-1):
        for j in range(GRID):
            u_d = 19 + 0.3*i + 0.4*j
            u_u = 18 + 0.2*(GRID-i) + 0.5*j
            W.addLink(f"v{i}{j}d", f"n{i}{j}", f"n{i+1}{j}",
                      length=2000, free_flow_speed=u_d, backward_wave_speed=5, capacity=0.6)
            W.addLink(f"v{i}{j}u", f"n{i+1}{j}", f"n{i}{j}",
                      length=2000, free_flow_speed=u_u, backward_wave_speed=5, capacity=0.6)
    od_data = [
        ("o1","n00","d1","n55",0,5400,0.55), ("o2","n05","d2","n50",0,5400,0.55),
        ("o3","n02","d3","n52",600,5400,0.45), ("o4","n20","d4","n25",600,5400,0.45),
        ("o5","n50","d5","n05",0,5400,0.45), ("o6","n55","d6","n00",0,5400,0.45),
    ]
    for on, og, dn, dg, ts, te, fl in od_data:
        oi, oj = int(og[1]), int(og[2])
        di, dj = int(dg[1]), int(dg[2])
        if on not in [n.name for n in W.NODES]:
            W.addNode(on, oj*2-1, -oi*2+1)
            W.addLink(f"l_{on}", on, og, length=2000, free_flow_speed=20,
                      backward_wave_speed=5, capacity=1.0)
        if dn not in [n.name for n in W.NODES]:
            W.addNode(dn, dj*2+1, -di*2-1)
            W.addLink(f"l_{dg}_{dn}", dg, dn, length=2000, free_flow_speed=20,
                      backward_wave_speed=5, capacity=1.0)
        W.adddemand(on, dn, ts, te, fl)
    return W


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / max(ss_tot, 1e-10)


def _compare_py_jax(name, build_fn):
    """Compare Python DUO vs JAX DUO for a scenario."""
    print(f"\n{'='*60}\n  {name}\n{'='*60}")

    W = build_fn()
    t0 = time.time(); W.exec_simulation(); t_py = time.time() - t0
    W.analyzer.basic_analysis()
    dt = W.DELTAT

    py_vols = np.array([l.cum_departure[-1] for l in W.LINKS])
    py_tts = np.array([
        sum(max(l.cum_arrival[t]-l.cum_departure[t], 0)*dt for t in range(W.TSIZE))
        / max(l.cum_departure[-1], 1e-10) for l in W.LINKS])

    W2 = build_fn(); W2.exec_simulation()
    params, config = world_to_jax(W2)
    _ = simulate_duo(params, config)
    t0 = time.time()
    state = simulate_duo(params, config)
    jax.block_until_ready(state.cum_arrival)
    t_jax = time.time() - t0

    jax_vols = np.array([float(state.cum_departure[i, -1]) for i in range(len(W.LINKS))])
    jax_tts = np.array([
        float(jnp.sum(jnp.maximum(
            state.cum_arrival[i, :config.tsize] - state.cum_departure[i, :config.tsize], 0))) * dt
        / max(float(state.cum_departure[i, -1]), 1e-10) for i in range(len(W.LINKS))])

    active = py_vols > 1
    print(f"  Python: {t_py:.3f}s | JAX: {t_jax:.3f}s")
    print(f"  Volume R2={_r2(py_vols, jax_vols):.4f}, "
          f"TT R2={_r2(py_tts[active], jax_tts[active]):.4f}")

    return {"py_vols": py_vols, "jax_vols": jax_vols,
            "py_tts": py_tts[active], "jax_tts": jax_tts[active]}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imgdir = os.path.join(os.path.dirname(__file__), "..", "docs", "img")
    os.makedirs(imgdir, exist_ok=True)

    # Part 1: Kuwahara quantitative checks
    print(f"\n{'='*60}\n  Kuwahara & Akamatsu (2001) Fig. 5-7 Checks\n{'='*60}")
    W = _build_kuwahara("duo_multipoint")
    W.exec_simulation()
    dt = W.DELTAT
    t_hours = np.array([t * dt / 3600 for t in range(W.TSIZE + 1)])
    links = {n: W.get_link(n) for n in ["1_4","4_5","5_6","6_3","1_2","2_3","5_2"]}
    dests = W._destinations
    cum_arr_d = {ln: {d: np.array(l.cum_arrival_d[d]) for d in dests} for ln, l in links.items()}
    cum_dep = {ln: np.array(l.cum_departure) for ln, l in links.items()}

    t_check = int(0.5 * 3600 / dt)
    if t_check > 0:
        s3 = cum_arr_d["1_4"]["3"][t_check] / (t_check * dt) * 3600
        s2 = cum_arr_d["1_4"]["2"][t_check] / (t_check * dt) * 3600
        print(f"  Initial slopes at t=0.5h: d3={s3:.0f} (exp~2000), d2={s2:.0f} (exp~1000)")
    t_c, t_cp = int(2.0*3600/dt), int(1.8*3600/dt)
    if t_c > t_cp:
        dr = (cum_dep["5_6"][t_c] - cum_dep["5_6"][t_cp]) / ((t_c-t_cp)*dt) * 3600
        print(f"  Departure rate (5,6) at t=1.8-2.0h: {dr:.0f} veh/h (exp~3000)")
    print(f"  Off-ramp (5,2) dest-3: {cum_arr_d['5_2']['3'][-1]:.0f} veh (exp~0)")
    print(f"  Arterial (1,2) dest-3: {cum_arr_d['1_2']['3'][-1]:.0f} veh (exp>0)")
    for t in range(W.TSIZE + 1):
        if cum_arr_d["1_2"]["3"][t] > 50:
            print(f"  Route switch dest-3 at t={t_hours[t]:.2f}h (paper~2.01h)")
            break

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, dest, pairs in [
        (axes[0], "3", [("1_4","$A_{14}^3$"),("4_5","$A_{45}^3$"),("5_6","$A_{56}^3$"),
                         ("6_3","$A_{63}^3$"),("1_2","$A_{12}^3$"),("2_3","$A_{23}^3$")]),
        (axes[1], "2", [("1_4","$A_{14}^2$"),("4_5","$A_{45}^2$"),("5_2","$A_{52}^2$"),
                         ("1_2","$A_{12}^2$"),("2_3","$A_{23}^2$")])]:
        for ln, label in pairs:
            if ln in cum_arr_d:
                arr = cum_arr_d[ln][dest]
                if arr[-1] > 10:
                    ax.plot(t_hours, arr, label=label)
        ax.set_xlabel("Time (h)"); ax.set_ylabel("Cumulative Trips (veh)")
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(0, 5)
    plt.tight_layout()
    fig.savefig(os.path.join(imgdir, "kuwahara_fig6_7_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: kuwahara_fig6_7_comparison.png")

    # Part 2: Python vs JAX comparison
    results = [
        _compare_py_jax("Kuwahara (6n/7l/2d)", _build_kuwahara),
        _compare_py_jax("Grid 6x6 (48n/132l/6d)", _build_grid),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for idx, res in enumerate(results):
        ax = axes[idx, 0]
        ax.scatter(res["py_vols"], res["jax_vols"], s=20, alpha=0.7)
        vmax = max(res["py_vols"].max(), res["jax_vols"].max()) * 1.1
        ax.plot([0, vmax], [0, vmax], "k--", lw=0.5)
        ax.set_xlabel("Python Volume (veh)"); ax.set_ylabel("JAX Volume (veh)")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax = axes[idx, 1]
        ax.scatter(res["py_tts"], res["jax_tts"], s=20, alpha=0.7, color="C1")
        tmax_val = max(res["py_tts"].max(), res["jax_tts"].max()) * 1.1
        ax.plot([0, tmax_val], [0, tmax_val], "k--", lw=0.5)
        ax.set_xlabel("Python Travel Time (s)"); ax.set_ylabel("JAX Travel Time (s)")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(imgdir, "duo_py_vs_jax.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: duo_py_vs_jax.png")
