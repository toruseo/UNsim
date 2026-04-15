"""
Tests for DUO logit (soft routing) route choice model.
Based on the same Kuwahara & Akamatsu (2001) Fig. 5 scenario as test_duo.py,
but using route_choice="duo_logit" with softmax-based route probabilities.

Network: Freeway + Arterial with off-ramp
  1(Work) -> 4 -> 5 -> 6 -> 3(Residential)   [Freeway]
  1(Work) -> 2 -> 3(Residential)             [Arterial]
  5 -> 2                                     [Off-ramp]

OD: 1->2, 1->3 (time-varying)
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unsim import *


def build_kuwahara_logit(logit_temperature=60.0):
    """Build the Kuwahara scenario with duo_logit route choice."""
    W = World(tmax=5*3600, print_mode=0, route_choice="duo_logit",
              logit_temperature=logit_temperature)

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

    for start, end, L_km, l_w, l_wp, kmax, fmax in link_data:
        L = L_km * 1000
        u = (L_km / l_w) * 1000 / 3600
        w = (L_km / l_wp) * 1000 / 3600
        kap = kmax / 1000
        cap = fmax / 3600
        W.addLink(f"{start}_{end}", start, end,
                  length=L, free_flow_speed=u, backward_wave_speed=w,
                  capacity=cap)

    W.adddemand("1", "2", 0, 3600, 1000/3600)
    W.adddemand("1", "2", 3600, 10800, 2000/3600)
    W.adddemand("1", "3", 0, 3600, 2000/3600)
    W.adddemand("1", "3", 3600, 10800, 4000/3600)

    return W


# ================================================================
# Python duo_logit tests
# ================================================================

def test_duo_logit_runs():
    """DUO logit simulation runs without error."""
    W = build_kuwahara_logit()
    W.exec_simulation()
    W.analyzer.basic_analysis()
    assert W.analyzer.trip_all > 0
    assert W.analyzer.trip_completed > 0


def test_duo_logit_initial_freeflow_route():
    """Initially, most traffic to dest 3 uses freeway (faster in free flow).

    With logit, some traffic leaks to arterial, but freeway should dominate.
    """
    W = build_kuwahara_logit()
    W.exec_simulation()

    link_14 = W.get_link("1_4")
    link_12 = W.get_link("1_2")

    dt = W.DELTAT
    t_idx = int(1000 / dt)
    arr_14 = link_14.cum_arrival[min(t_idx, W.TSIZE)]
    arr_12 = link_12.cum_arrival[min(t_idx, W.TSIZE)]
    assert arr_14 > arr_12, \
        f"Expected freeway to dominate initially: arr_14={arr_14:.0f} vs arr_12={arr_12:.0f}"


def test_duo_logit_route_switching():
    """After congestion on freeway, traffic switches to arterial."""
    W = build_kuwahara_logit()
    W.exec_simulation()

    link_12 = W.get_link("1_2")
    final_arr = link_12.cum_arrival[-1]
    assert final_arr > 100, \
        f"Expected route switch to arterial, but 1_2 only got {final_arr:.0f} veh"


def test_duo_logit_flow_conservation():
    """Total absorbed should equal total demand (minus queue and on-links)."""
    W = build_kuwahara_logit()
    W.exec_simulation()
    W.analyzer.basic_analysis()

    total_demand = W.analyzer.trip_all
    total_absorbed = sum(n.absorbed_count for n in W.NODES)
    remaining_queue = W.get_node("1").demand_queue
    remaining_on_links = sum(
        link.cum_arrival[-1] - link.cum_departure[-1] for link in W.LINKS)

    assert equal_tolerance(total_absorbed + remaining_queue + remaining_on_links,
                           total_demand, rel_tol=0.1), \
        f"demand={total_demand:.0f}, absorbed={total_absorbed:.0f}, " \
        f"queue={remaining_queue:.0f}, on_links={remaining_on_links:.0f}"


def test_duo_logit_per_destination_consistency():
    """Sum of per-destination cumulative counts should equal aggregate."""
    W = build_kuwahara_logit()
    W.exec_simulation()

    for link in W.LINKS:
        for t in range(0, W.TSIZE + 1, max(1, W.TSIZE // 10)):
            sum_arr_d = sum(link.cum_arrival_d[d][t] for d in W._destinations)
            sum_dep_d = sum(link.cum_departure_d[d][t] for d in W._destinations)
            assert abs(link.cum_arrival[t] - sum_arr_d) < 1, \
                f"{link.name} t={t}: arr={link.cum_arrival[t]:.1f} vs sum_d={sum_arr_d:.1f}"
            assert abs(link.cum_departure[t] - sum_dep_d) < 1, \
                f"{link.name} t={t}: dep={link.cum_departure[t]:.1f} vs sum_d={sum_dep_d:.1f}"


def test_duo_logit_low_temperature_approaches_duo():
    """With very low temperature, logit approaches hard DUO behavior."""
    W_duo = World(tmax=5*3600, print_mode=0, route_choice="duo")
    W_logit = World(tmax=5*3600, print_mode=0, route_choice="duo_logit",
                    logit_temperature=1.0)

    # Build identical networks
    for W in [W_duo, W_logit]:
        W.addNode("1", -2, 0); W.addNode("2", 0, -1); W.addNode("3", 2, -1)
        W.addNode("4", -1, 1); W.addNode("5", 0.5, 1); W.addNode("6", 1.5, 1)
        link_data = [
            ("1","4",2,0.05,0.1,450,6000), ("4","5",16,0.2,0.8,375,6000),
            ("5","6",8,0.1,0.4,250,4000), ("6","3",2,0.05,0.1,225,3000),
            ("1","2",16,0.4,0.8,450,6000), ("2","3",12,0.3,0.6,450,6000),
            ("5","2",2,0.05,0.1,450,6000)]
        for s, e, L, lw, lwp, km, fm in link_data:
            W.addLink(f"{s}_{e}", s, e, length=L*1000,
                      free_flow_speed=(L/lw)*1000/3600,
                      backward_wave_speed=(L/lwp)*1000/3600,
                      capacity=fm/3600)
        W.adddemand("1", "2", 0, 3600, 1000/3600)
        W.adddemand("1", "2", 3600, 10800, 2000/3600)
        W.adddemand("1", "3", 0, 3600, 2000/3600)
        W.adddemand("1", "3", 3600, 10800, 4000/3600)

    W_duo.exec_simulation()
    W_duo.analyzer.basic_analysis()
    W_logit.exec_simulation()
    W_logit.analyzer.basic_analysis()

    # TTT should be similar (within 30%)
    assert equal_tolerance(W_logit.analyzer.total_travel_time,
                           W_duo.analyzer.total_travel_time, rel_tol=0.3), \
        f"Logit T=1 TTT={W_logit.analyzer.total_travel_time:.0f} " \
        f"vs DUO TTT={W_duo.analyzer.total_travel_time:.0f}"


# ================================================================
# JAX duo_logit tests
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
    """Verify JAX duo_logit matches Python and supports gradients."""

    def _build(self):
        return build_kuwahara_logit()

    def test_jax_runs(self):
        """JAX duo_logit simulation runs without error."""
        W = self._build()
        params, config = world_to_jax(W)
        assert config.use_logit is True
        state = simulate_duo(params, config)
        ttt = total_travel_time(state, config)
        assert jnp.isfinite(ttt)
        assert float(ttt) > 0

    def test_jax_matches_python(self):
        """JAX duo_logit TTT matches Python duo_logit TTT."""
        W = self._build()
        W.exec_simulation()
        W.analyzer.basic_analysis()
        ttt_py = W.analyzer.total_travel_time

        W2 = self._build()
        params, config = world_to_jax(W2)
        state = simulate_duo(params, config)
        ttt_jax = float(total_travel_time(state, config))
        assert equal_tolerance(ttt_jax, ttt_py, rel_tol=0.3), \
            f"JAX TTT={ttt_jax:.0f} vs Python TTT={ttt_py:.0f}"

    def test_grad_od_demand_finite(self):
        """Gradient of TTT w.r.t. OD demand is finite."""
        W = self._build()
        params, config = world_to_jax(W)

        def loss(p):
            return total_travel_time(simulate_duo(p, config), config)

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.od_demand_rate))

    def test_grad_toll_nonzero(self):
        """Gradient of TTT w.r.t. toll is non-zero (differentiable routing)."""
        W = World(tmax=3600, print_mode=0, route_choice="duo_logit",
                  logit_temperature=60.0)
        W.addNode("O", 0, 0)
        W.addNode("D", 1, 0)
        W.addLink("a", "O", "D", length=1000, free_flow_speed=25,
                  congestion_pricing=lambda t: 50)
        W.addLink("b", "O", "D", length=1000, free_flow_speed=20)
        W.adddemand("O", "D", 0, 3600, 0.2)

        params, config = world_to_jax(W)

        def loss(p):
            return total_travel_time(simulate_duo(p, config), config)

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.toll)), "Toll gradient should be finite"
        assert float(jnp.linalg.norm(grads.toll)) > 1e-6, \
            "Toll gradient should be non-zero for duo_logit"
