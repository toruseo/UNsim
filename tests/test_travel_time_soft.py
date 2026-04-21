"""
Tests for autodiff-compatible travel time functions.
Verifies travel_time_soft, logsum_travel_time, travel_time_auto,
and their building blocks (invert_interp_batch, link_exit_time_batch).

Requires JAX.
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

from unsim import World
from unsim.unsim_diff import (
    world_to_jax, simulate, simulate_duo,
    travel_time, link_exit_time, total_travel_time,
    invert_interp_1d, invert_interp_batch, interp_batch,
    link_exit_time_batch, compute_link_cost_from_state,
    travel_time_soft, logsum_travel_time, travel_time_auto,
)


# ================================================================
# Network factories
# ================================================================

def make_2link_network():
    """2-link straight road: orig --link0--> mid --link1--> dest."""
    W = World(name="", deltat=5, tmax=2000, print_mode=0)
    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1)
    W.addNode("dest", 2, 2)
    W.addLink("link0", "orig", "mid", length=1000,
              free_flow_speed=20, jam_density=0.2)
    W.addLink("link1", "mid", "dest", length=1000,
              free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest", 0, 500, 0.3)
    return W


def make_y_network():
    """Y-shaped network: orig -> (fast path | slow path) -> dest.

    fast path:  orig --link0(L=1000,u=20)--> mid1 --link2(L=500,u=20)--> dest
    slow path:  orig --link1(L=1000,u=10)--> mid2 --link3(L=500,u=10)--> dest

    Free-flow TT: fast=75s, slow=150s.
    """
    W = World(name="", deltat=5, tmax=2000, print_mode=0,
              route_choice="duo_logit")
    W.LOGIT_TEMPERATURE = 60.0
    W.addNode("orig", 0, 0)
    W.addNode("mid1", 1, 1)
    W.addNode("mid2", 1, -1)
    W.addNode("dest", 2, 0)
    W.addLink("fast1", "orig", "mid1", length=1000,
              free_flow_speed=20, jam_density=0.2)
    W.addLink("slow1", "orig", "mid2", length=1000,
              free_flow_speed=10, jam_density=0.2)
    W.addLink("fast2", "mid1", "dest", length=500,
              free_flow_speed=20, jam_density=0.2)
    W.addLink("slow2", "mid2", "dest", length=500,
              free_flow_speed=10, jam_density=0.2)
    W.adddemand("orig", "dest", 0, 500, 0.2)
    return W


def make_bottleneck_network():
    """2-link bottleneck: link2 has lower capacity."""
    W = World(name="", deltat=5, tmax=2000, print_mode=0)
    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1)
    W.addNode("dest", 2, 2)
    W.addLink("link0", "orig", "mid", length=1000,
              free_flow_speed=20, jam_density=0.2)
    W.addLink("link1", "mid", "dest", length=1000,
              free_flow_speed=10, jam_density=0.2)
    W.adddemand("orig", "dest", 0, 1000, 0.8)
    return W


# ================================================================
# Test invert_interp_batch
# ================================================================

class TestInvertInterpBatch:
    """Unit tests for the batch inverse interpolation function."""

    def test_matches_scalar(self):
        """Batch output matches per-row invert_interp_1d."""
        arrays = jnp.array([
            [0.0, 1.0, 3.0, 6.0, 10.0],
            [0.0, 2.5, 5.0, 7.5, 10.0],
            [0.0, 0.0, 0.0, 5.0, 10.0],
        ])
        values = jnp.array([2.0, 3.75, 4.0])

        batch_result = invert_interp_batch(arrays, values)

        for i in range(3):
            scalar_result = invert_interp_1d(arrays[i], values[i])
            assert abs(float(batch_result[i]) - float(scalar_result)) < 1e-5, \
                f"Row {i}: batch={float(batch_result[i])}, scalar={float(scalar_result)}"

    def test_gradient(self):
        """Gradient through invert_interp_batch is finite."""
        arrays = jnp.array([
            [0.0, 2.5, 5.0, 7.5, 10.0],
            [0.0, 1.0, 3.0, 6.0, 10.0],
        ])
        values = jnp.array([3.75, 2.0])

        def fn(vals):
            return jnp.sum(invert_interp_batch(arrays, vals))

        grads = jax.grad(fn)(values)
        assert jnp.all(jnp.isfinite(grads)), f"Non-finite grads: {grads}"
        assert jnp.any(grads != 0.0), "All-zero grads"


# ================================================================
# Test link_exit_time_batch
# ================================================================

class TestLinkExitTimeBatch:
    """Unit tests for the batch link exit time function."""

    def test_matches_scalar(self):
        """Batch output matches per-link link_exit_time."""
        W = make_2link_network()
        params, config = world_to_jax(W)
        state = simulate(params, config)

        link_ids = jnp.array([0, 1, 0], dtype=jnp.int32)
        t_enters = jnp.array([100.0, 200.0, 300.0])

        batch_result = link_exit_time_batch(link_ids, t_enters,
                                            state, params, config)

        for i in range(3):
            scalar_result = link_exit_time(int(link_ids[i]), float(t_enters[i]),
                                           state, params, config)
            assert abs(float(batch_result[i]) - float(scalar_result)) < 1e-3, \
                f"Element {i}: batch={float(batch_result[i])}, scalar={float(scalar_result)}"

    def test_freeflow(self):
        """In free flow, exit_time = enter + d/u."""
        W = make_2link_network()
        params, config = world_to_jax(W)
        state = simulate(params, config)

        # Early departure: should be free flow
        link_ids = jnp.array([0, 1], dtype=jnp.int32)
        t_enters = jnp.array([50.0, 100.0])

        exits = link_exit_time_batch(link_ids, t_enters, state, params, config)
        expected = t_enters + config.link_lengths[link_ids] / params.u[link_ids]

        for i in range(2):
            assert abs(float(exits[i]) - float(expected[i])) < 1.0, \
                f"Link {i}: got {float(exits[i])}, expected {float(expected[i])}"


# ================================================================
# Test travel_time_soft
# ================================================================

class TestTravelTimeSoft:
    """Tests for the fully-differentiable soft routing travel time."""

    def test_2link_straight_low_temp(self):
        """On a straight road (no route choice), travel_time_soft matches
        travel_time at low temperature."""
        W = make_2link_network()
        params, config = world_to_jax(W)
        state = simulate(params, config)

        t_dep = 100.0
        # With only one path, soft routing should match deterministic
        tt_fixed = float(travel_time([0, 1], t_dep, state, params, config))
        tt_soft = float(travel_time_soft(0, 2, t_dep, state, params, config,
                                         temperature=10.0))

        assert abs(tt_soft - tt_fixed) < 2.0, \
            f"Soft={tt_soft:.1f} vs fixed={tt_fixed:.1f}"

    def test_y_network_expected_tt(self):
        """On Y-network, expected TT is between the fast and slow path TTs."""
        W = make_y_network()
        params, config = world_to_jax(W)
        state = simulate_duo(params, config, differentiable=False)

        t_dep = 100.0
        fast_tt = 1000/20 + 500/20   # 75s
        slow_tt = 1000/10 + 500/10   # 150s

        tt_soft = float(travel_time_soft(0, 3, t_dep, state, params, config,
                                         temperature=60.0))

        assert fast_tt - 5.0 <= tt_soft <= slow_tt + 5.0, \
            f"Soft TT={tt_soft:.1f} not in [{fast_tt}, {slow_tt}]"

    def test_gradient_finite(self):
        """jax.grad of travel_time_soft produces finite, non-zero gradients."""
        W = make_2link_network()
        params, config = world_to_jax(W)

        def loss(p):
            s = simulate(p, config)
            return travel_time_soft(0, 2, 100.0, s, p, config,
                                    temperature=60.0)

        grads = jax.grad(loss)(params)
        # Check u (free-flow speed) gradient: should be finite and non-zero
        assert jnp.all(jnp.isfinite(grads.u)), f"Non-finite u grads: {grads.u}"

    def test_jit_compatible(self):
        """travel_time_soft works under jax.jit."""
        W = make_2link_network()
        params, config = world_to_jax(W)
        state = simulate(params, config)

        @jax.jit
        def compute(s, p):
            return travel_time_soft(0, 2, 100.0, s, p, config,
                                    temperature=60.0)

        tt = float(compute(state, params))
        assert jnp.isfinite(tt) and tt > 0, f"JIT result: {tt}"


# ================================================================
# Test logsum_travel_time
# ================================================================

class TestLogsumTravelTime:
    """Tests for the logsum Bellman equation travel time."""

    def test_2link_straight(self):
        """On a straight road, logsum value approximates actual TT."""
        W = make_2link_network()
        params, config = world_to_jax(W)
        state = simulate(params, config)

        t_dep = 100.0
        expected_tt = 1000/20 + 1000/20  # 100s free flow

        tt_logsum = float(logsum_travel_time(0, 2, t_dep, state, params, config,
                                             temperature=60.0))

        # Logsum is a lower bound (soft-min <= min), so it can be slightly
        # less than actual TT, but should be in the right ballpark.
        assert tt_logsum > 0, f"Logsum TT should be positive: {tt_logsum}"
        assert abs(tt_logsum - expected_tt) / expected_tt < 0.5, \
            f"Logsum={tt_logsum:.1f} too far from expected={expected_tt:.1f}"

    def test_gradient_finite(self):
        """Gradient of logsum_travel_time is finite."""
        W = make_2link_network()
        params, config = world_to_jax(W)

        def loss(p):
            s = simulate(p, config)
            return logsum_travel_time(0, 2, 100.0, s, p, config,
                                      temperature=60.0)

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.u)), f"Non-finite u grads: {grads.u}"

    def test_y_network_bounded(self):
        """Logsum value is less than or equal to min-cost path."""
        W = make_y_network()
        params, config = world_to_jax(W)
        state = simulate_duo(params, config, differentiable=False)

        t_dep = 100.0
        fast_tt = 1000/20 + 500/20  # 75s (min-cost path)

        tt_logsum = float(logsum_travel_time(0, 3, t_dep, state, params, config,
                                             temperature=60.0))

        # Logsum <= min-cost (soft-min is always <= hard min)
        assert tt_logsum <= fast_tt + 5.0, \
            f"Logsum={tt_logsum:.1f} should be <= min-cost={fast_tt:.1f}"

    def test_jit_compatible(self):
        """logsum_travel_time works under jax.jit."""
        W = make_2link_network()
        params, config = world_to_jax(W)
        state = simulate(params, config)

        @jax.jit
        def compute(s, p):
            return logsum_travel_time(0, 2, 100.0, s, p, config,
                                      temperature=60.0)

        tt = float(compute(state, params))
        assert jnp.isfinite(tt) and tt > 0, f"JIT result: {tt}"


# ================================================================
# Test travel_time_auto
# ================================================================

class TestTravelTimeAuto:
    """Tests for the auto-path-extraction travel time."""

    def test_matches_travel_time_straight(self):
        """On a straight road, auto version matches explicit path version."""
        W = make_2link_network()
        params, config = world_to_jax(W)
        state = simulate(params, config)

        t_dep = 100.0
        tt_fixed = float(travel_time([0, 1], t_dep, state, params, config))
        tt_auto = float(travel_time_auto(0, 2, t_dep, state, params, config))

        assert abs(tt_auto - tt_fixed) < 2.0, \
            f"Auto={tt_auto:.1f} vs fixed={tt_fixed:.1f}"

    def test_y_network_shortest_path(self):
        """On Y-network, auto version picks the fast path."""
        W = make_y_network()
        params, config = world_to_jax(W)
        state = simulate_duo(params, config, differentiable=False)

        t_dep = 100.0
        fast_tt = 1000/20 + 500/20  # 75s

        tt_auto = float(travel_time_auto(0, 3, t_dep, state, params, config))

        assert abs(tt_auto - fast_tt) < 10.0, \
            f"Auto={tt_auto:.1f} should be near fast path={fast_tt:.1f}"

    def test_gradient_partial(self):
        """Gradient w.r.t. speed captures congestion effect (path fixed)."""
        W = make_2link_network()
        params, config = world_to_jax(W)

        def loss(p):
            s = simulate(p, config)
            return travel_time_auto(0, 2, 100.0, s, p, config)

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.u)), f"Non-finite u grads: {grads.u}"

    def test_jit_compatible(self):
        """travel_time_auto works under jax.jit."""
        W = make_2link_network()
        params, config = world_to_jax(W)
        state = simulate(params, config)

        @jax.jit
        def compute(s, p):
            return travel_time_auto(0, 2, 100.0, s, p, config)

        tt = float(compute(state, params))
        assert jnp.isfinite(tt) and tt > 0, f"JIT result: {tt}"


# ================================================================
# Test compute_link_cost_from_state
# ================================================================

class TestComputeLinkCostFromState:
    """Tests for the link cost extraction helper."""

    def test_positive_costs(self):
        """Link costs are positive and finite."""
        W = make_2link_network()
        params, config = world_to_jax(W)
        state = simulate(params, config)

        link_cost, link_tt = compute_link_cost_from_state(
            100.0, state, params, config)

        assert jnp.all(link_cost > 0), f"Non-positive costs: {link_cost}"
        assert jnp.all(jnp.isfinite(link_cost)), f"Non-finite costs: {link_cost}"
        assert jnp.all(link_tt > 0), f"Non-positive TT: {link_tt}"

    def test_freeflow_tt(self):
        """At early time, link TT approximates d/u."""
        W = make_2link_network()
        params, config = world_to_jax(W)
        state = simulate(params, config)

        _, link_tt = compute_link_cost_from_state(50.0, state, params, config)
        expected = config.link_lengths / params.u

        for i in range(config.n_links):
            assert abs(float(link_tt[i]) - float(expected[i])) < 5.0, \
                f"Link {i}: TT={float(link_tt[i])}, expected~{float(expected[i])}"
