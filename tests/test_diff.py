"""
Tests for the JAX differentiable LTM simulator (unsim_diff.py).
Verifies numerical agreement with unsim.py and gradient computation.

Requires JAX to be installed. Skipped if JAX is unavailable.
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import jax
    import jax.numpy as jnp
except (ImportError, RuntimeError):
    pytest.skip("JAX not available on this platform", allow_module_level=True)

from unsim import World, equal_tolerance
from unsim.unsim_diff import (
    world_to_jax, simulate, total_travel_time, trip_completed,
    average_travel_time, compute_N, invert_interp_1d, link_exit_time,
    travel_time, NetworkConfig, Params, SimState,
)


# ================================================================
# Helper
# ================================================================

def run_both(world_factory):
    """Run simulation with both unsim.py and unsim_diff.py, return results."""
    # Original
    W = world_factory()
    W.exec_simulation()
    W.analyzer.basic_analysis()

    # JAX
    W2 = world_factory()
    params, config = world_to_jax(W2)
    state = simulate(params, config)

    return W, params, config, state


# ================================================================
# Numerical agreement tests
# ================================================================

class TestNumericalAgreement:
    """Verify JAX simulation matches original unsim.py results."""

    def test_1link_freeflow(self):
        """Single link, free flow."""
        def factory():
            W = World(name="", deltat=5, tmax=2000, print_mode=0)
            W.addNode("orig", 0, 0)
            W.addNode("dest", 1, 1)
            W.addLink("link", "orig", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("orig", "dest", 0, 500, 0.5)
            return W

        W, params, config, state = run_both(factory)

        # Cumulative counts match
        orig_ca = np.array(W.LINKS[0].cum_arrival)
        jax_ca = np.array(state.cum_arrival[0])
        assert np.allclose(orig_ca, jax_ca, atol=0.1), \
            f"cum_arrival mismatch: max diff={np.max(np.abs(orig_ca - jax_ca))}"

        orig_cd = np.array(W.LINKS[0].cum_departure)
        jax_cd = np.array(state.cum_departure[0])
        assert np.allclose(orig_cd, jax_cd, atol=0.1), \
            f"cum_departure mismatch: max diff={np.max(np.abs(orig_cd - jax_cd))}"

        # Total travel time
        ttt_orig = W.analyzer.total_travel_time
        ttt_jax = float(total_travel_time(state, config))
        assert equal_tolerance(ttt_jax, ttt_orig), \
            f"TTT mismatch: orig={ttt_orig}, jax={ttt_jax}"

    def test_1link_maxflow(self):
        """Single link, overcapacity demand."""
        def factory():
            W = World(name="", deltat=5, tmax=2000, print_mode=0)
            W.addNode("orig", 0, 0)
            W.addNode("dest", 1, 1)
            W.addLink("link", "orig", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("orig", "dest", 0, 2000, 2)
            return W

        W, params, config, state = run_both(factory)
        orig_cd = np.array(W.LINKS[0].cum_departure)
        jax_cd = np.array(state.cum_departure[0])
        assert np.allclose(orig_cd, jax_cd, atol=0.5)

    def test_2link_bottleneck(self):
        """2-link bottleneck due to speed difference."""
        def factory():
            W = World(name="", deltat=5, tmax=2000, print_mode=0)
            W.addNode("orig", 0, 0)
            W.addNode("mid", 1, 1)
            W.addNode("dest", 2, 2)
            W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
            W.addLink("link2", "mid", "dest", length=1000, free_flow_speed=10, jam_density=0.2)
            W.adddemand("orig", "dest", 0, 500, 0.8)
            W.adddemand("orig", "dest", 500, 1500, 0.4)
            return W

        W, params, config, state = run_both(factory)
        ttt_orig = W.analyzer.total_travel_time
        ttt_jax = float(total_travel_time(state, config))
        assert equal_tolerance(ttt_jax, ttt_orig)

    def test_merge_fair(self):
        """2-to-1 merge, equal priority, congestion."""
        def factory():
            W = World(name="", deltat=5, tmax=1200, print_mode=0)
            W.addNode("orig1", 0, 0)
            W.addNode("orig2", 0, 2)
            W.addNode("merge", 1, 1)
            W.addNode("dest", 2, 1)
            W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20,
                       jam_density=0.2, merge_priority=1)
            W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20,
                       jam_density=0.2, merge_priority=1)
            W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("orig1", "dest", 0, 1000, 0.5)
            W.adddemand("orig2", "dest", 0, 1000, 0.5)
            return W

        W, params, config, state = run_both(factory)
        ttt_orig = W.analyzer.total_travel_time
        ttt_jax = float(total_travel_time(state, config))
        assert equal_tolerance(ttt_jax, ttt_orig)

    def test_merge_unfair(self):
        """2-to-1 merge, priority 1:2."""
        def factory():
            W = World(name="", deltat=5, tmax=1200, print_mode=0)
            W.addNode("orig1", 0, 0)
            W.addNode("orig2", 0, 2)
            W.addNode("merge", 1, 1)
            W.addNode("dest", 2, 1)
            W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20,
                       jam_density=0.2, merge_priority=1)
            W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20,
                       jam_density=0.2, merge_priority=2)
            W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("orig1", "dest", 0, 1000, 0.8)
            W.adddemand("orig2", "dest", 0, 1000, 0.8)
            return W

        W, params, config, state = run_both(factory)
        ttt_orig = W.analyzer.total_travel_time
        ttt_jax = float(total_travel_time(state, config))
        assert equal_tolerance(ttt_jax, ttt_orig)

    def test_diverge(self):
        """1-to-2 diverge."""
        def factory():
            W = World(name="", deltat=5, tmax=1200, print_mode=0)
            W.addNode("orig", 0, 0)
            W.addNode("mid", 1, 1)
            W.addNode("dest1", 2, 0)
            W.addNode("dest2", 2, 2)
            W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
            W.addLink("link2", "mid", "dest1", length=1000, free_flow_speed=20, jam_density=0.2)
            W.addLink("link3", "mid", "dest2", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("orig", "dest1", 0, 1000, 0.3)
            W.adddemand("orig", "dest2", 0, 1000, 0.3)
            return W

        W, params, config, state = run_both(factory)
        ttt_orig = W.analyzer.total_travel_time
        ttt_jax = float(total_travel_time(state, config))
        assert equal_tolerance(ttt_jax, ttt_orig)


# ================================================================
# Gradient tests
# ================================================================

class TestGradient:
    """Verify jax.grad works and produces finite gradients."""

    def test_grad_demand(self):
        """Gradient of TTT w.r.t. demand rate."""
        W = World(name="", deltat=5, tmax=1000, print_mode=0)
        W.addNode("orig", 0, 0)
        W.addNode("dest", 1, 1)
        W.addLink("link", "orig", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
        W.adddemand("orig", "dest", 0, 500, 0.5)

        params, config = world_to_jax(W)

        def loss(p):
            s = simulate(p, config)
            return total_travel_time(s, config)

        grad_fn = jax.grad(loss)
        grads = grad_fn(params)

        # demand_rate gradients should be finite
        assert jnp.all(jnp.isfinite(grads.demand_rate))
        # Positive demand should increase TTT
        assert jnp.sum(grads.demand_rate) > 0

    def test_grad_freeflow_speed(self):
        """Gradient of TTT w.r.t. free flow speed."""
        W = World(name="", deltat=5, tmax=1000, print_mode=0)
        W.addNode("orig", 0, 0)
        W.addNode("dest", 1, 1)
        W.addLink("link", "orig", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
        W.adddemand("orig", "dest", 0, 500, 0.5)

        params, config = world_to_jax(W)

        def loss(p):
            s = simulate(p, config)
            return total_travel_time(s, config)

        grad_fn = jax.grad(loss)
        grads = grad_fn(params)

        # Speed gradient should be finite
        assert jnp.all(jnp.isfinite(grads.u))
        # Increasing speed should decrease TTT (negative gradient)
        assert grads.u[0] < 0

    def test_grad_merge_priority(self):
        """Gradient of TTT w.r.t. merge priority."""
        W = World(name="", deltat=5, tmax=1000, print_mode=0)
        W.addNode("orig1", 0, 0)
        W.addNode("orig2", 0, 2)
        W.addNode("merge", 1, 1)
        W.addNode("dest", 2, 1)
        W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20,
                   jam_density=0.2, merge_priority=1)
        W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20,
                   jam_density=0.2, merge_priority=2)
        W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
        W.adddemand("orig1", "dest", 0, 800, 0.5)
        W.adddemand("orig2", "dest", 0, 800, 0.5)

        params, config = world_to_jax(W)

        def loss(p):
            s = simulate(p, config)
            return total_travel_time(s, config)

        grad_fn = jax.grad(loss)
        grads = grad_fn(params)

        assert jnp.all(jnp.isfinite(grads.merge_priority))

    def test_jit_consistency(self):
        """jax.jit produces same results as non-jit."""
        W = World(name="", deltat=5, tmax=1000, print_mode=0)
        W.addNode("orig", 0, 0)
        W.addNode("dest", 1, 1)
        W.addLink("link", "orig", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
        W.adddemand("orig", "dest", 0, 500, 0.5)

        params, config = world_to_jax(W)

        state_nojit = simulate(params, config)

        @jax.jit
        def simulate_jit(p):
            return simulate(p, config)

        state_jit = simulate_jit(params)

        assert jnp.allclose(state_nojit.cum_arrival, state_jit.cum_arrival, atol=1e-5)
        assert jnp.allclose(state_nojit.cum_departure, state_jit.cum_departure, atol=1e-5)


# ================================================================
# INM (general node) tests
# ================================================================

class TestINM:
    """Verify JAX INM matches Python INM and supports gradients."""

    def _build_flotterod(self, sw_capacity=None):
        """Build Floetteroed 3x3 intersection scenario."""
        W = World(name="", deltat=5, tmax=3000, print_mode=0)
        W.addNode("O_S", 0, 0)
        W.addNode("O_E", 2, 0)
        W.addNode("O_N", 1, 2)
        W.addNode("intersection", 1, 1,
                  turning_fractions={
                      "P_S": {"S_N": 0.5, "S_W": 0.5},
                      "P_E": {"S_W": 1.0},
                      "P_N": {"S_W": 0.5, "S_S": 0.5},
                  })
        W.addNode("D_N", 1, 3)
        W.addNode("D_W", -1, 1)
        W.addNode("D_S", 1, -1)
        W.addLink("P_S", "O_S", "intersection", length=1000,
                  free_flow_speed=20, backward_wave_speed=5, jam_density=0.2, merge_priority=1.0)
        W.addLink("P_E", "O_E", "intersection", length=1000,
                  free_flow_speed=20, backward_wave_speed=5, jam_density=0.2, merge_priority=0.1)
        W.addLink("P_N", "O_N", "intersection", length=1000,
                  free_flow_speed=20, backward_wave_speed=5, jam_density=0.2, merge_priority=10.0)
        W.addLink("S_N", "intersection", "D_N", length=1000,
                  free_flow_speed=20, backward_wave_speed=5, jam_density=0.2)
        if sw_capacity is not None:
            W.addLink("S_W", "intersection", "D_W", length=1000,
                      free_flow_speed=20, backward_wave_speed=5, capacity=sw_capacity)
        else:
            W.addLink("S_W", "intersection", "D_W", length=1000,
                      free_flow_speed=20, backward_wave_speed=5, jam_density=0.2)
        W.addLink("S_S", "intersection", "D_S", length=1000,
                  free_flow_speed=20, backward_wave_speed=5, jam_density=0.2)
        W.adddemand("O_S", "D_N", 0, 2000, 600/3600)
        W.adddemand("O_E", "D_W", 0, 2000, 100/3600)
        W.adddemand("O_N", "D_S", 0, 2000, 600/3600)
        return W

    def test_inm_uncongested(self):
        """JAX INM matches Python for Floetteroed Table 1 (uncongested)."""
        W = self._build_flotterod()
        W.exec_simulation()
        W.analyzer.basic_analysis()

        params, config = world_to_jax(W)
        state = simulate(params, config)

        ttt_orig = W.analyzer.total_travel_time
        ttt_jax = float(total_travel_time(state, config))
        assert equal_tolerance(ttt_jax, ttt_orig)

    def test_inm_congested(self):
        """JAX INM matches Python for Floetteroed Table 2 (congested)."""
        W = self._build_flotterod(sw_capacity=400/3600)
        W.exec_simulation()
        W.analyzer.basic_analysis()

        params, config = world_to_jax(W)
        state = simulate(params, config)

        ttt_orig = W.analyzer.total_travel_time
        ttt_jax = float(total_travel_time(state, config))
        assert equal_tolerance(ttt_jax, ttt_orig)

    def test_inm_grad_turning_fractions(self):
        """Gradient of TTT w.r.t. turning fractions is finite."""
        W = self._build_flotterod()
        params, config = world_to_jax(W)

        def loss(p):
            s = simulate(p, config)
            return total_travel_time(s, config)

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.turning_fractions))


# ================================================================
# DUO tests
# ================================================================

class TestDUO:
    """Verify JAX DUO matches Python DUO and supports gradients."""

    def _build_kuwahara(self):
        """Build Kuwahara & Akamatsu (2001) Fig.5 scenario."""
        W = World(tmax=5*3600, print_mode=0, route_choice="duo")
        W.addNode("1", -2, 0); W.addNode("2", 0, -1); W.addNode("3", 2, -1)
        W.addNode("4", -1, 1); W.addNode("5", 0.5, 1); W.addNode("6", 1.5, 1)
        link_data = [
            ("1","4",2,0.05,0.1,450,6000), ("4","5",16,0.2,0.8,375,6000),
            ("5","6",8,0.1,0.4,250,4000), ("6","3",2,0.05,0.1,225,3000),
            ("1","2",16,0.4,0.8,450,6000), ("2","3",12,0.3,0.6,450,6000),
            ("5","2",2,0.05,0.1,450,6000)]
        for s, e, L, lw, lwp, km, fm in link_data:
            W.addLink(f"{s}_{e}", s, e, length=L*1000,
                      free_flow_speed=(L/lw)*1000/3600, backward_wave_speed=(L/lwp)*1000/3600,
                      capacity=fm/3600)
        W.adddemand("1", "2", 0, 3600, 1000/3600)
        W.adddemand("1", "2", 3600, 10800, 2000/3600)
        W.adddemand("1", "3", 0, 3600, 2000/3600)
        W.adddemand("1", "3", 3600, 10800, 4000/3600)
        return W

    def test_duo_jax_runs(self):
        """JAX DUO simulation runs without error."""
        from unsim.unsim_diff import world_to_jax, simulate_duo, total_travel_time
        W = self._build_kuwahara()
        W.exec_simulation()  # Python DUO
        params, config = world_to_jax(W)
        state = simulate_duo(params, config)
        ttt = total_travel_time(state, config)
        assert jnp.isfinite(ttt)
        assert float(ttt) > 0

    def test_duo_jax_matches_python(self):
        """JAX DUO TTT matches Python DUO TTT."""
        from unsim.unsim_diff import world_to_jax, simulate_duo, total_travel_time
        W = self._build_kuwahara()
        W.exec_simulation()
        W.analyzer.basic_analysis()
        ttt_py = W.analyzer.total_travel_time

        params, config = world_to_jax(W)
        state = simulate_duo(params, config)
        ttt_jax = float(total_travel_time(state, config))
        assert equal_tolerance(ttt_jax, ttt_py, rel_tol=0.3), \
            f"JAX TTT={ttt_jax:.0f} vs Python TTT={ttt_py:.0f}"

    def test_duo_grad_od_demand(self):
        """Gradient of TTT w.r.t. OD demand is finite."""
        from unsim.unsim_diff import world_to_jax, simulate_duo, total_travel_time
        W = self._build_kuwahara()
        params, config = world_to_jax(W)

        def loss(p):
            s = simulate_duo(p, config)
            return total_travel_time(s, config)

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.od_demand_rate))


# ================================================================
# Virtual vehicle travel time tests
# ================================================================

class TestTravelTime:
    """Verify differentiable travel_time matches Analyzer.travel_time."""

    def test_1link_freeflow(self):
        """Single link, free flow: travel time = d/u."""
        def factory():
            W = World(name="", deltat=5, tmax=2000, print_mode=0)
            W.addNode("orig", 0, 0)
            W.addNode("dest", 1, 1)
            W.addLink("link", "orig", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("orig", "dest", 0, 500, 0.5)
            return W

        W, params, config, state = run_both(factory)
        link_id = 0
        t_dep = 100.0

        tt_jax = float(travel_time([link_id], t_dep, state, params, config))
        tt_py = W.analyzer.travel_time("orig", "dest", t_dep, path=["link"])

        assert abs(tt_jax - 1000/20) < 1.0, f"Free flow: expected 50, got {tt_jax}"
        assert abs(tt_jax - tt_py) < 1.0, f"Mismatch: jax={tt_jax}, py={tt_py}"

    def test_2link_freeflow(self):
        """Two links, free flow: travel time = d1/u1 + d2/u2."""
        def factory():
            W = World(name="", deltat=5, tmax=2000, print_mode=0)
            W.addNode("orig", 0, 0)
            W.addNode("mid", 1, 1)
            W.addNode("dest", 2, 2)
            W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
            W.addLink("link2", "mid", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("orig", "dest", 0, 500, 0.3)
            return W

        W, params, config, state = run_both(factory)
        tt_jax = float(travel_time([0, 1], 100.0, state, params, config))
        tt_py = W.analyzer.travel_time("orig", "dest", 100.0, path=["link1", "link2"])

        expected = 1000/20 + 1000/20  # 100s
        assert abs(tt_jax - expected) < 2.0, f"Free flow 2-link: expected {expected}, got {tt_jax}"
        assert abs(tt_jax - tt_py) < 2.0, f"Mismatch: jax={tt_jax}, py={tt_py}"

    def test_bottleneck_congestion(self):
        """Bottleneck causes travel time > free flow."""
        def factory():
            W = World(name="", deltat=5, tmax=2000, print_mode=0)
            W.addNode("orig", 0, 0)
            W.addNode("mid", 1, 1)
            W.addNode("dest", 2, 2)
            W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
            W.addLink("link2", "mid", "dest", length=1000, free_flow_speed=10, jam_density=0.2)
            W.adddemand("orig", "dest", 0, 1000, 0.8)
            return W

        W, params, config, state = run_both(factory)

        # Late departure should experience congestion
        tt_jax = float(travel_time([0, 1], 500.0, state, params, config))
        tt_py = W.analyzer.travel_time("orig", "dest", 500.0, path=["link1", "link2"])

        free_flow_tt = 1000/20 + 1000/10  # 150s
        assert tt_jax > free_flow_tt, f"Should exceed free flow: {tt_jax} <= {free_flow_tt}"
        assert abs(tt_jax - tt_py) / max(tt_py, 1.0) < 0.05, \
            f"Mismatch: jax={tt_jax:.1f}, py={tt_py:.1f}"

    def test_grad_travel_time_demand(self):
        """AD gradient of travel_time w.r.t. demand matches Newell theory.

        Scenario: 2-link bottleneck, demand = q1* = 0.8 veh/s.
          q1* = u1*w1*kappa/(u1+w1) = 20*5*0.2/25 = 0.8
          q2* = u2*w2*kappa/(u2+w2) = 10*5*0.2/15 = 2/3

        Theory: demand = q1* places the origin node exactly at the
        min(demand, supply) boundary. The supply branch is selected,
        so d(origin_flow)/d(demand_rate) = 0. Consequently the entire
        gradient d(travel_time)/d(demand_rate) = 0.
        """
        W = World(name="", deltat=5, tmax=2000, print_mode=0)
        W.addNode("orig", 0, 0)
        W.addNode("mid", 1, 1)
        W.addNode("dest", 2, 2)
        W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
        W.addLink("link2", "mid", "dest", length=1000, free_flow_speed=10, jam_density=0.2)
        W.adddemand("orig", "dest", 0, 1000, 0.8)

        params, config = world_to_jax(W)
        path = [0, 1]

        def loss(p):
            s = simulate(p, config)
            return travel_time(path, 500.0, s, p, config)

        grads = jax.grad(loss)(params)

        assert jnp.all(jnp.isfinite(grads.demand_rate))
        # Theory: gradient = 0 (demand = q1*, supply branch selected at origin)
        assert jnp.allclose(grads.demand_rate, 0.0, atol=1e-6), \
            f"Expected 0, got sum={float(jnp.sum(grads.demand_rate))}"

    def test_grad_invert_interp_1d(self):
        """Unit test: invert_interp_1d gradient matches theory.

        array = [0, 2.5, 5, 7.5, 10], value = 3.75
        result = 1.5 (between indices 1 and 2)
        slope = 2.5
        Theory: d(result)/d(value) = 1/slope = 0.4
                d(result)/d(array[1]) = -(1-0.5)/2.5 = -0.2
                d(result)/d(array[2]) = -0.5/2.5 = -0.2
        """
        arr = jnp.array([0.0, 2.5, 5.0, 7.5, 10.0])
        val = jnp.float32(3.75)

        # Forward check
        result = float(invert_interp_1d(arr, val))
        assert abs(result - 1.5) < 1e-10, f"Forward: {result} != 1.5"

        # Gradient w.r.t. value
        grad_val = float(jax.grad(lambda v: invert_interp_1d(arr, v))(val))
        assert abs(grad_val - 0.4) < 0.01, \
            f"d/d(value): AD={grad_val:.4f}, theory=0.4"

        # Gradient w.r.t. array
        grad_arr = jax.grad(lambda a: invert_interp_1d(a, val))(arr)
        assert abs(float(grad_arr[1]) - (-0.2)) < 0.01, \
            f"d/d(arr[1]): AD={float(grad_arr[1]):.4f}, theory=-0.2"
        assert abs(float(grad_arr[2]) - (-0.2)) < 0.01, \
            f"d/d(arr[2]): AD={float(grad_arr[2]):.4f}, theory=-0.2"

    def test_grad_link_exit_time_direct(self):
        """link_exit_time gradient with frozen state (no simulate gradient).

        1 link free flow, t_enter=10. t_freeflow = t_queue = 60 (tie).
        jnp.maximum(a,b) = 0.5*(a+b+|a-b|), so at tie d/da = d/db = 0.5.
        With frozen state: d(t_queue)/d(u) = 0 (stop_gradient).

        Theory: 0.5 * d(t_freeflow)/d(u) + 0.5 * 0
              = 0.5 * (-d/u^2) = 0.5 * (-2.5) = -1.25
        """
        W = World(name="", deltat=5, tmax=2000, print_mode=0)
        W.addNode("orig", 0, 0)
        W.addNode("dest", 1, 1)
        W.addLink("link", "orig", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
        W.adddemand("orig", "dest", 0, 500, 0.5)

        params, config = world_to_jax(W)

        state = simulate(params, config)
        frozen_ca = jax.lax.stop_gradient(state.cum_arrival)
        frozen_cd = jax.lax.stop_gradient(state.cum_departure)
        frozen_state = state._replace(cum_arrival=frozen_ca, cum_departure=frozen_cd)

        def loss_direct(p):
            return link_exit_time(0, 10.0, frozen_state, p, config) - 10.0

        grads = jax.grad(loss_direct)(params)
        ad_grad = float(grads.u[0])
        expected = 0.5 * (-1000.0 / 20.0 ** 2)  # -1.25
        assert abs(ad_grad - expected) < 0.1, \
            f"AD={ad_grad:.4f}, theory={expected:.4f}"

    def test_grad_travel_time_speed(self):
        """Full pipeline: simulate + travel_time, gradient w.r.t. speed.

        1 link free flow, t_enter=10. t_freeflow = t_queue = 60 (tie).
        jnp.maximum at tie gives 50/50 gradient split.
        Additionally, in simulate, jnp.maximum(D, 0) at D=0 (step k=9)
        also gives 50/50, halving d(cum_departure)/d(u).

        Theory:
          d(t_freeflow)/d(u) = -d/u^2 = -2.5  (direct)
          d(t_queue)/d(u) = -1.25              (halved by D=0 tie in simulate)
          total = 0.5*(-2.5) + 0.5*(-1.25) = -1.875
        """
        W = World(name="", deltat=5, tmax=2000, print_mode=0)
        W.addNode("orig", 0, 0)
        W.addNode("dest", 1, 1)
        W.addLink("link", "orig", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
        W.adddemand("orig", "dest", 0, 500, 0.5)

        params, config = world_to_jax(W)

        def loss_full(p):
            s = simulate(p, config)
            return travel_time([0], 10.0, s, p, config)

        grads = jax.grad(loss_full)(params)
        ad_grad = float(grads.u[0])
        expected = 0.5 * (-2.5) + 0.5 * (-1.25)  # -1.875
        assert abs(ad_grad - expected) < 0.1, \
            f"AD={ad_grad:.4f}, theory={expected:.4f}"
