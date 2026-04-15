"""
Tests for dynamic congestion pricing (toll) in DUO route choice.

Verifies that congestion_pricing on links correctly influences
DUO route selection via generalized cost = travel time + toll.

Network topology: 2-route parallel network
  Origin O ---(link_a)---> Destination D
  Origin O ---(link_b)---> Destination D
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unsim import World, equal_tolerance


# ================================================================
# Helpers
# ================================================================

def build_two_route(toll_a=None, toll_b=None,
                    u_a=20, u_b=20, length=1000,
                    tmax=3600, demand_flow=0.2):
    """Build a 2-route parallel network with optional congestion pricing.

    Parameters
    ----------
    toll_a : callable or None
        congestion_pricing function for link_a.
    toll_b : callable or None
        congestion_pricing function for link_b.
    u_a : float
        Free flow speed for link_a (m/s).
    u_b : float
        Free flow speed for link_b (m/s).
    length : float
        Link length for both links (m).
    tmax : float
        Simulation duration (s).
    demand_flow : float
        OD demand flow rate (veh/s).

    Returns
    -------
    World
    """
    W = World(tmax=tmax, print_mode=0, route_choice="duo")
    W.addNode("O", 0, 0)
    W.addNode("D", 1, 0)
    W.addLink("link_a", "O", "D", length=length, free_flow_speed=u_a,
              congestion_pricing=toll_a)
    W.addLink("link_b", "O", "D", length=length, free_flow_speed=u_b,
              congestion_pricing=toll_b)
    W.adddemand("O", "D", 0, tmax, demand_flow)
    return W


def get_link_arrivals(W, link_name):
    """Get total cumulative arrivals for a link."""
    link = W.LINKS_NAME_DICT[link_name]
    return link.cum_arrival[W.TSIZE]


# ================================================================
# Python DUO toll tests
# ================================================================

class TestTollPython:
    """Verify congestion pricing controls DUO route choice (Python)."""

    def test_toll_shifts_traffic(self):
        """High toll on faster route shifts all traffic to slower route."""
        # link_a: faster (u=25, TT=40s) but tolled (100s)
        # link_b: slower (u=20, TT=50s) but free
        # Generalized cost: link_a = 40+100 = 140 > 50 = link_b
        W = build_two_route(
            toll_a=lambda t: 100,
            u_a=25, u_b=20, tmax=3600, demand_flow=0.2)
        W.exec_simulation()

        arr_a = get_link_arrivals(W, "link_a")
        arr_b = get_link_arrivals(W, "link_b")
        # Nearly all traffic should use link_b
        assert arr_b > 0, "link_b should carry traffic"
        assert arr_a < arr_b * 0.05, \
            f"link_a arrivals ({arr_a:.1f}) should be negligible vs link_b ({arr_b:.1f})"

    def test_no_toll_faster_route_preferred(self):
        """Without toll, faster route carries all traffic."""
        W = build_two_route(
            toll_a=None, toll_b=None,
            u_a=25, u_b=20, tmax=3600, demand_flow=0.2)
        W.exec_simulation()

        arr_a = get_link_arrivals(W, "link_a")
        arr_b = get_link_arrivals(W, "link_b")
        # link_a is faster (TT=40s vs 50s), should carry most traffic
        assert arr_a > arr_b, \
            f"Faster link_a ({arr_a:.1f}) should carry more than link_b ({arr_b:.1f})"

    def test_toll_reverses_preference(self):
        """Toll on faster route reverses route preference vs no-toll case."""
        # No toll: link_a preferred (faster)
        W_no_toll = build_two_route(
            toll_a=None, u_a=25, u_b=20, tmax=3600, demand_flow=0.2)
        W_no_toll.exec_simulation()
        arr_a_free = get_link_arrivals(W_no_toll, "link_a")
        arr_b_free = get_link_arrivals(W_no_toll, "link_b")

        # With toll: link_b preferred
        W_toll = build_two_route(
            toll_a=lambda t: 100, u_a=25, u_b=20, tmax=3600, demand_flow=0.2)
        W_toll.exec_simulation()
        arr_a_toll = get_link_arrivals(W_toll, "link_a")
        arr_b_toll = get_link_arrivals(W_toll, "link_b")

        # Preference reversal
        assert arr_a_free > arr_b_free, "No-toll: link_a should be preferred"
        assert arr_b_toll > arr_a_toll, "With toll: link_b should be preferred"

    def test_time_varying_toll(self):
        """Time-varying toll switches route preference mid-simulation."""
        tmax = 3600
        mid = tmax / 2

        # Phase 1 (0-1800s): toll on link_a -> traffic uses link_b
        # Phase 2 (1800-3600s): toll on link_b -> traffic uses link_a
        def toll_a(t):
            return 200 if t < mid else 0

        def toll_b(t):
            return 0 if t < mid else 200

        W = World(tmax=tmax, print_mode=0, route_choice="duo")
        W.addNode("O", 0, 0)
        W.addNode("D", 1, 0)
        W.addLink("link_a", "O", "D", length=1000, free_flow_speed=20,
                  congestion_pricing=toll_a)
        W.addLink("link_b", "O", "D", length=1000, free_flow_speed=20,
                  congestion_pricing=toll_b)
        W.adddemand("O", "D", 0, tmax, 0.2)
        W.exec_simulation()

        # Check mid-point cumulative arrivals vs final
        t_mid_idx = int(mid / W.DELTAT)
        link_a = W.LINKS_NAME_DICT["link_a"]
        link_b = W.LINKS_NAME_DICT["link_b"]

        # Phase 1 arrivals (0 to mid)
        arr_a_phase1 = link_a.cum_arrival[t_mid_idx]
        arr_b_phase1 = link_b.cum_arrival[t_mid_idx]

        # Phase 2 arrivals (mid to end)
        arr_a_phase2 = link_a.cum_arrival[W.TSIZE] - link_a.cum_arrival[t_mid_idx]
        arr_b_phase2 = link_b.cum_arrival[W.TSIZE] - link_b.cum_arrival[t_mid_idx]

        # Phase 1: link_b should dominate (link_a has toll)
        assert arr_b_phase1 > arr_a_phase1, \
            f"Phase 1: link_b ({arr_b_phase1:.1f}) should dominate over link_a ({arr_a_phase1:.1f})"
        # Phase 2: link_a should dominate (link_b has toll)
        assert arr_a_phase2 > arr_b_phase2, \
            f"Phase 2: link_a ({arr_a_phase2:.1f}) should dominate over link_b ({arr_b_phase2:.1f})"

    def test_zero_toll_no_effect(self):
        """Zero toll produces same result as no congestion_pricing."""
        W_none = build_two_route(
            toll_a=None, toll_b=None,
            u_a=25, u_b=20, tmax=3600, demand_flow=0.2)
        W_none.exec_simulation()

        W_zero = build_two_route(
            toll_a=lambda t: 0, toll_b=lambda t: 0,
            u_a=25, u_b=20, tmax=3600, demand_flow=0.2)
        W_zero.exec_simulation()

        for ln in ["link_a", "link_b"]:
            arr_none = get_link_arrivals(W_none, ln)
            arr_zero = get_link_arrivals(W_zero, ln)
            assert abs(arr_none - arr_zero) < 0.1, \
                f"{ln}: zero toll ({arr_zero:.1f}) should match no toll ({arr_none:.1f})"


# ================================================================
# JAX toll tests
# ================================================================

try:
    import jax
    import jax.numpy as jnp
    from unsim.unsim_diff import world_to_jax, simulate_duo, total_travel_time
    HAS_JAX = True
except (ImportError, RuntimeError):
    HAS_JAX = False


@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestTollJAX:
    """Verify congestion pricing in JAX DUO matches Python."""

    def test_toll_jax_runs(self):
        """JAX DUO with toll runs without error."""
        W = build_two_route(
            toll_a=lambda t: 100, u_a=25, u_b=20,
            tmax=3600, demand_flow=0.2)
        params, config = world_to_jax(W)
        state = simulate_duo(params, config)
        ttt = total_travel_time(state, config)
        assert jnp.isfinite(ttt)
        assert float(ttt) > 0

    def test_toll_jax_shifts_traffic(self):
        """JAX DUO: high toll shifts traffic to untolled route."""
        W = build_two_route(
            toll_a=lambda t: 100, u_a=25, u_b=20,
            tmax=3600, demand_flow=0.2)
        params, config = world_to_jax(W)
        state = simulate_duo(params, config)

        # link_a (id=0) should have negligible arrivals
        # link_b (id=1) should carry most traffic
        arr_a = float(state.cum_arrival[0, -1])
        arr_b = float(state.cum_arrival[1, -1])
        assert arr_b > 0, "link_b should carry traffic"
        assert arr_a < arr_b * 0.05, \
            f"link_a arrivals ({arr_a:.1f}) should be negligible vs link_b ({arr_b:.1f})"

    def test_toll_python_jax_agreement(self):
        """JAX and Python DUO produce similar results with toll."""
        W_py = build_two_route(
            toll_a=lambda t: 100, u_a=25, u_b=20,
            tmax=3600, demand_flow=0.2)
        W_py.exec_simulation()
        W_py.analyzer.basic_analysis()
        ttt_py = W_py.analyzer.total_travel_time

        W_jax = build_two_route(
            toll_a=lambda t: 100, u_a=25, u_b=20,
            tmax=3600, demand_flow=0.2)
        params, config = world_to_jax(W_jax)
        state = simulate_duo(params, config)
        ttt_jax = float(total_travel_time(state, config))

        assert equal_tolerance(ttt_jax, ttt_py, rel_tol=0.3), \
            f"JAX TTT={ttt_jax:.0f} vs Python TTT={ttt_py:.0f}"

        # Per-link cumulative arrivals agreement
        for i, link in enumerate(W_py.LINKS):
            py_ca = np.array(link.cum_arrival)
            jax_ca = np.array(state.cum_arrival[i])
            assert np.allclose(py_ca, jax_ca, atol=1.0), \
                f"Link {link.name}: cum_arrival mismatch"

    def test_toll_grad_finite(self):
        """Gradient of TTT w.r.t. toll values is finite."""
        W = build_two_route(
            toll_a=lambda t: 50, u_a=25, u_b=20,
            tmax=3600, demand_flow=0.2)
        params, config = world_to_jax(W)

        def loss(p):
            s = simulate_duo(p, config)
            return total_travel_time(s, config)

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.toll)), \
            "Gradient of TTT w.r.t. toll should be finite"

    def test_toll_array_populated(self):
        """world_to_jax correctly populates toll array."""
        W = build_two_route(
            toll_a=lambda t: 42, toll_b=lambda t: 0,
            tmax=3600, demand_flow=0.2)
        params, config = world_to_jax(W)

        # link_a (id=0) should have toll=42 at all steps
        assert jnp.all(params.toll[0] == 42.0), \
            f"link_a toll should be 42, got {params.toll[0]}"
        # link_b (id=1) should have toll=0 at all steps
        assert jnp.all(params.toll[1] == 0.0), \
            f"link_b toll should be 0, got {params.toll[1]}"
