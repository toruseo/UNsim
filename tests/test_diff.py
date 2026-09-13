"""
Tests for the JAX differentiable LTM simulator (unsim_diff.py).
Verifies numerical agreement with unsim.py, gradient computation,
and AD vs finite-difference regression.

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
    world_to_jax, simulate, simulate_duo, total_travel_time, trip_completed,
    average_travel_time, compute_N, invert_interp_1d, link_exit_time,
    travel_time, travel_time_auto, NetworkConfig, Params, SimState,
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

    def test_merge_3inlinks(self):
        """3-to-1 merge, multiple priorities and congestion (Issue #18)."""

        def factory():
            W = World(name="", deltat=5, tmax=1200, print_mode=0)
            W.addNode("orig1", 0, 0)
            W.addNode("orig2", 0, 2)
            W.addNode("orig3", 0, 4)
            W.addNode("merge", 1, 2)
            W.addNode("dest", 2, 2)
            W.addLink(
                "link1",
                "orig1",
                "merge",
                length=1000,
                free_flow_speed=20,
                jam_density=0.2,
                merge_priority=1,
            )
            W.addLink(
                "link2",
                "orig2",
                "merge",
                length=1000,
                free_flow_speed=20,
                jam_density=0.2,
                merge_priority=2,
            )
            W.addLink(
                "link3",
                "orig3",
                "merge",
                length=1000,
                free_flow_speed=20,
                jam_density=0.2,
                merge_priority=1,
            )
            W.addLink(
                "link4",
                "merge",
                "dest",
                length=1000,
                free_flow_speed=20,
                jam_density=0.2,
            )
            W.adddemand("orig1", "dest", 0, 1000, 0.4)
            W.adddemand("orig2", "dest", 0, 1000, 0.4)
            W.adddemand("orig3", "dest", 0, 1000, 0.4)
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


    @staticmethod
    def _assert_all_equal_tolerance(val_arr, check_arr, rel_tol=0.1, abs_tol=0.1, err_msg=""):
        """Check that all elements in val_arr match check_arr using equal_tolerance."""
        val_flat = jnp.ravel(val_arr)
        check_flat = jnp.ravel(check_arr)
        assert len(val_flat) == len(check_flat), f"Length mismatch: {len(val_flat)} vs {len(check_flat)}"
        mismatches = []
        for idx, (v, c) in enumerate(zip(val_flat, check_flat)):
            if not equal_tolerance(float(v), float(c), rel_tol=rel_tol, abs_tol=abs_tol):
                mismatches.append((idx, float(v), float(c)))
        if mismatches:
            details = ", ".join([f"idx {i}: val={v:.4f} vs check={c:.4f}" for i, v, c in mismatches])
            raise AssertionError(f"{err_msg} Mismatches found (showing first {len(mismatches)}): {details}")

    def _check_linkwise(self, W, state, config, rel_tol=0.1, abs_tol=0.5):
        for link_id, link in enumerate(W.LINKS):
            orig_ca = jnp.array(link.cum_arrival)
            jax_ca = jnp.array(state.cum_arrival[link_id])
            self._assert_all_equal_tolerance(
                jax_ca, orig_ca, rel_tol=rel_tol, abs_tol=abs_tol,
                err_msg=f"Link {link.name} (id={link_id}) cum_arrival mismatch:"
            )

            orig_cd = jnp.array(link.cum_departure)
            jax_cd = jnp.array(state.cum_departure[link_id])
            self._assert_all_equal_tolerance(
                jax_cd, orig_cd, rel_tol=rel_tol, abs_tol=abs_tol,
                err_msg=f"Link {link.name} (id={link_id}) cum_departure mismatch:"
            )

    @staticmethod
    def _merge_node_jax_formula(D, p, S):
        """Merge node calculation matching unsim_diff.py."""
        total_D = jnp.sum(D)
        alphas = p / jnp.maximum(jnp.sum(p), 1e-10)
        base_q = jnp.minimum(D, alphas * S)
        rem_S = jnp.maximum(S - jnp.sum(base_q), 0.0)
        surplus_cap = D - base_q
        cum_surplus = jnp.cumsum(surplus_cap)
        extra_q = jnp.minimum(surplus_cap, jnp.maximum(rem_S - (cum_surplus - surplus_cap), 0.0))
        merge_q = jnp.where(total_D <= S, D, base_q + extra_q)
        return merge_q

    def test_merge_2inlinks_fair_linkwise(self):
        """2-to-1 merge, equal priority: check every link's cumulative arrival and departure."""
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
        self._check_linkwise(W, state, config)

    def test_merge_2inlinks_unfair_linkwise(self):
        """2-to-1 merge, priority 1:2: check every link's cumulative arrival and departure."""
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
        self._check_linkwise(W, state, config)

    def test_merge_3inlinks_linkwise(self):
        """3-to-1 merge (Issue #18): check every link's cumulative arrival and departure."""
        def factory():
            W = World(name="", deltat=5, tmax=1200, print_mode=0)
            W.addNode("orig1", 0, 0)
            W.addNode("orig2", 0, 2)
            W.addNode("orig3", 0, 4)
            W.addNode("merge", 1, 2)
            W.addNode("dest", 2, 2)
            W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20,
                       jam_density=0.2, merge_priority=1)
            W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20,
                       jam_density=0.2, merge_priority=2)
            W.addLink("link3", "orig3", "merge", length=1000, free_flow_speed=20,
                       jam_density=0.2, merge_priority=1)
            W.addLink("link4", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("orig1", "dest", 0, 1000, 0.4)
            W.adddemand("orig2", "dest", 0, 1000, 0.4)
            W.adddemand("orig3", "dest", 0, 1000, 0.4)
            return W

        W, params, config, state = run_both(factory)
        self._check_linkwise(W, state, config)

    def test_merge_surplus_reallocation_linkwise(self):
        """3-to-1 merge with surplus supply reallocation: link1 demand < alpha1 * S."""
        def factory():
            W = World(name="", deltat=5, tmax=1200, print_mode=0)
            W.addNode("orig1", 0, 0)
            W.addNode("orig2", 0, 2)
            W.addNode("orig3", 0, 4)
            W.addNode("merge", 1, 2)
            W.addNode("dest", 2, 2)
            W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20,
                       jam_density=0.2, merge_priority=1)
            W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20,
                       jam_density=0.2, merge_priority=1)
            W.addLink("link3", "orig3", "merge", length=1000, free_flow_speed=20,
                       jam_density=0.2, merge_priority=1)
            W.addLink("link4", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("orig1", "dest", 0, 1000, 0.1)
            W.adddemand("orig2", "dest", 0, 1000, 0.5)
            W.adddemand("orig3", "dest", 0, 1000, 0.5)
            return W

        W, params, config, state = run_both(factory)
        self._check_linkwise(W, state, config)

    def test_merge_4inlinks_linkwise(self):
        """4-to-1 merge: check link-wise cumulative counts for 4 inlinks."""
        def factory():
            W = World(name="", deltat=5, tmax=1200, print_mode=0)
            for i in range(1, 5):
                W.addNode(f"orig{i}", 0, i * 2)
            W.addNode("merge", 1, 4)
            W.addNode("dest", 2, 4)
            priorities = [1, 2, 1, 3]
            for i in range(1, 5):
                W.addLink(f"link{i}", f"orig{i}", "merge", length=1000, free_flow_speed=20,
                           jam_density=0.2, merge_priority=priorities[i-1])
            W.addLink("link_out", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
            for i in range(1, 5):
                W.adddemand(f"orig{i}", "dest", 0, 1000, 0.3)
            return W

        W, params, config, state = run_both(factory)
        self._check_linkwise(W, state, config)

    @staticmethod
    def _simulate_minimal_merge(factory, D, p, S, return_linkwise=False):
        """Run minimal merge World through simulate and return merge node flow(s)."""
        W = factory()
        params, config = world_to_jax(W)
        n_in = len(D)
        outlink_id = n_in
        params = params._replace(
            demand_rate=params.demand_rate.at[:n_in, :].set(D[:, None]),
            merge_priority=params.merge_priority.at[:n_in].set(p),
            q_star=params.q_star.at[outlink_id].set(S),
        )
        state = simulate(params, config)
        if return_linkwise:
            return jnp.array(
                [state.cum_departure[i][2] - state.cum_departure[i][1] for i in range(n_in)]
            )
        else:
            return state.cum_arrival[outlink_id][2] - state.cum_arrival[outlink_id][1]

    @pytest.mark.parametrize("scale", [1e-12, 1e-6, 1e-3, 0.1, 0.5, 2.0, 10.0, 1e3, 1e6])
    def test_priority_scale_invariance_formula(self, scale):
        """Minimal merge simulation check: flows must be invariant when scaling p by any positive factor."""
        def factory():
            W = World(name="", deltat=1, tmax=3, print_mode=0)
            for i in range(3):
                W.addNode(f"orig{i}", 0, i)
            W.addNode("merge", 1, 1)
            W.addNode("dest", 2, 1)
            for i in range(3):
                W.addLink(
                    f"in{i}", f"orig{i}", "merge", length=1, free_flow_speed=1,
                    jam_density=10.0, capacity=10.0
                )
            W.addLink(
                "out", "merge", "dest", length=1, free_flow_speed=1,
                jam_density=10.0, capacity=1.0
            )
            for i in range(3):
                W.adddemand(f"orig{i}", "dest", 0, 3, 1.0)
            return W

        D = jnp.array([0.6, 0.8, 0.7])
        p_base = jnp.array([1.0, 2.0, 3.0])
        S = jnp.float32(1.0)

        p_scaled = p_base * scale

        flow_base = self._simulate_minimal_merge(factory, D, p_base, S, return_linkwise=True)
        flow_scaled = self._simulate_minimal_merge(factory, D, p_scaled, S, return_linkwise=True)

        self._assert_all_equal_tolerance(flow_scaled, flow_base, rel_tol=1e-4, abs_tol=1e-5)

    def test_tiny_priority_allocation(self):
        """Review comment P2: tiny priority must preserve true allocation ratio 1:2:1 -> [0.25, 0.5, 0.25]."""
        def factory():
            W = World(name="", deltat=1, tmax=3, print_mode=0)
            for i in range(3):
                W.addNode(f"orig{i}", 0, i)
            W.addNode("merge", 1, 1)
            W.addNode("dest", 2, 1)
            for i in range(3):
                W.addLink(
                    f"in{i}", f"orig{i}", "merge", length=1, free_flow_speed=1,
                    jam_density=10.0, capacity=10.0
                )
            W.addLink(
                "out", "merge", "dest", length=1, free_flow_speed=1,
                jam_density=10.0, capacity=1.0
            )
            for i in range(3):
                W.adddemand(f"orig{i}", "dest", 0, 3, 1.0)
            return W

        D = jnp.array([1.0, 1.0, 1.0])
        p_tiny = jnp.array([1e-12, 2e-12, 1e-12])
        S = jnp.float32(1.0)

        flows = self._simulate_minimal_merge(factory, D, p_tiny, S, return_linkwise=True)
        expected = jnp.array([0.25, 0.50, 0.25])
        assert jnp.allclose(flows, expected, atol=1e-4), f"Expected {expected}, got {flows}"

    @pytest.mark.parametrize("scale", [1e-12, 1e-3, 0.1, 2.0, 10.0, 1e3])
    def test_simulation_scale_invariance(self, scale):
        """Simulation check: scaling merge priority by constant k should not change simulation results."""
        def factory(priority_multiplier=1.0):
            W = World(name="", deltat=5, tmax=1200, print_mode=0)
            W.addNode("orig1", 0, 0)
            W.addNode("orig2", 0, 2)
            W.addNode("orig3", 0, 4)
            W.addNode("merge", 1, 2)
            W.addNode("dest", 2, 2)
            W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20,
                       jam_density=0.2, merge_priority=1.0 * priority_multiplier)
            W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20,
                       jam_density=0.2, merge_priority=2.0 * priority_multiplier)
            W.addLink("link3", "orig3", "merge", length=1000, free_flow_speed=20,
                       jam_density=0.2, merge_priority=1.0 * priority_multiplier)
            W.addLink("link4", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("orig1", "dest", 0, 1000, 0.4)
            W.adddemand("orig2", "dest", 0, 1000, 0.4)
            W.adddemand("orig3", "dest", 0, 1000, 0.4)
            return W

        W_base = factory(1.0)
        params_base, config_base = world_to_jax(W_base)
        state_base = simulate(params_base, config_base)

        W_scaled = factory(scale)
        params_scaled, config_scaled = world_to_jax(W_scaled)
        state_scaled = simulate(params_scaled, config_scaled)

        ttt_base = float(total_travel_time(state_base, config_base))
        ttt_scaled = float(total_travel_time(state_scaled, config_scaled))
        assert equal_tolerance(ttt_scaled, ttt_base, rel_tol=1e-4, abs_tol=1e-3)

        for link_id in range(config_base.n_links):
            self._assert_all_equal_tolerance(
                jnp.array(state_scaled.cum_arrival[link_id]),
                jnp.array(state_base.cum_arrival[link_id]),
                rel_tol=1e-4, abs_tol=1e-3,
                err_msg=f"scale {scale} cum_arrival link {link_id} mismatch:"
            )
            self._assert_all_equal_tolerance(
                jnp.array(state_scaled.cum_departure[link_id]),
                jnp.array(state_base.cum_departure[link_id]),
                rel_tol=1e-4, abs_tol=1e-3,
                err_msg=f"scale {scale} cum_departure link {link_id} mismatch:"
            )


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


    @staticmethod
    def _prior_merge_2inlinks(D, p, S):
        """Prior implementation from e66a512~1 using differentiable_mid."""
        D1, D2 = D[0], D[1]
        p1, p2 = p[0], p[1]
        total_p = p1 + p2
        a1 = p1 / jnp.sum(p)
        a2 = p2 / jnp.sum(p)
        total_D = D1 + D2
        def mid(a, b, c):
            return a + b + c - jnp.minimum(a, jnp.minimum(b, c)) - jnp.maximum(a, jnp.maximum(b, c))
        q1_cong = jnp.maximum(mid(D1, S - D2, a1 * S), 0.0)
        q2_cong = jnp.maximum(mid(D2, S - D1, a2 * S), 0.0)
        q1 = jnp.where(total_D <= S, D1, q1_cong)
        q2 = jnp.where(total_D <= S, D2, q2_cong)
        return jnp.array([q1, q2])

    @staticmethod
    def _simulate_minimal_merge(factory, D, p, S, return_linkwise=False):
        """Run minimal merge World through simulate and return merge node flow(s)."""
        W = factory()
        params, config = world_to_jax(W)
        n_in = len(D)
        outlink_id = n_in
        params = params._replace(
            demand_rate=params.demand_rate.at[:n_in, :].set(D[:, None]),
            merge_priority=params.merge_priority.at[:n_in].set(p),
            q_star=params.q_star.at[outlink_id].set(S),
        )
        state = simulate(params, config)
        if return_linkwise:
            return jnp.array(
                [state.cum_departure[i][2] - state.cum_departure[i][1] for i in range(n_in)]
            )
        else:
            return state.cum_arrival[outlink_id][2] - state.cum_arrival[outlink_id][1]

    def test_grad_merge_2inlinks_vs_prior_smooth(self):
        """Compare gradients with prior implementation at a smooth congested point.

        D = [0.8, 0.8], p = [1.0, 2.0], S = 1.0.
        Both inlinks are bottlenecked, alphas = [1/3, 2/3].
        Check dQ/dD, dQ/dp, dQ/dS.
        """
        def factory():
            W = World(name="", deltat=1, tmax=3, print_mode=0)
            for i in range(2):
                W.addNode(f"orig{i}", 0, i)
            W.addNode("merge", 1, 1)
            W.addNode("dest", 2, 1)
            for i in range(2):
                W.addLink(
                    f"in{i}", f"orig{i}", "merge", length=1, free_flow_speed=1,
                    jam_density=10.0, capacity=10.0
                )
            W.addLink(
                "out", "merge", "dest", length=1, free_flow_speed=1,
                jam_density=10.0, capacity=1.0
            )
            for i in range(2):
                W.adddemand(f"orig{i}", "dest", 0, 3, 1.0)
            return W

        D = jnp.array([0.8, 0.8])
        p = jnp.array([1.0, 2.0])
        S = jnp.float32(1.0)

        prior_dQ_dD = jax.grad(lambda d: jnp.sum(self._prior_merge_2inlinks(d, p, S)))(D)
        prior_dQ_dp = jax.grad(lambda pr: jnp.sum(self._prior_merge_2inlinks(D, pr, S)))(p)
        prior_dQ_dS = jax.grad(lambda s: jnp.sum(self._prior_merge_2inlinks(D, p, s)))(S)

        curr_dQ_dD = jax.grad(lambda d: self._simulate_minimal_merge(factory, d, p, S))(D)
        curr_dQ_dp = jax.grad(lambda pr: self._simulate_minimal_merge(factory, D, pr, S))(p)
        curr_dQ_dS = jax.grad(lambda s: self._simulate_minimal_merge(factory, D, p, s))(S)

        assert jnp.allclose(prior_dQ_dD, 0.0, atol=1e-5)
        assert jnp.allclose(prior_dQ_dp, 0.0, atol=1e-5)
        assert jnp.allclose(prior_dQ_dS, 1.0, atol=1e-5)

        assert jnp.allclose(curr_dQ_dD, prior_dQ_dD, atol=1e-5)
        assert jnp.allclose(curr_dQ_dp, prior_dQ_dp, atol=1e-5)
        assert jnp.allclose(curr_dQ_dS, prior_dQ_dS, atol=1e-5)

    def test_grad_merge_2inlinks_prior_at_boundary(self):
        """Verify prior implementation has correct gradients at boundary point D[0] == alpha[0]*S."""
        D = jnp.array([0.5, 1.0])
        p = jnp.array([1.0, 1.0])
        S = jnp.float32(1.0)

        prior_dQ_dD = jax.grad(lambda d: jnp.sum(self._prior_merge_2inlinks(d, p, S)))(D)
        prior_dQ_dp = jax.grad(lambda pr: jnp.sum(self._prior_merge_2inlinks(D, pr, S)))(p)
        prior_dQ_dS = jax.grad(lambda s: jnp.sum(self._prior_merge_2inlinks(D, p, s)))(S)

        assert jnp.allclose(prior_dQ_dD, jnp.array([0.0, 0.0]), atol=1e-5)
        assert jnp.allclose(prior_dQ_dp, jnp.array([0.0, 0.0]), atol=1e-5)
        assert jnp.allclose(prior_dQ_dS, 1.0, atol=1e-5)

    def test_grad_merge_2inlinks_vs_prior_uncongested(self):
        """Compare gradients with prior implementation in uncongested regime."""
        def factory():
            W = World(name="", deltat=1, tmax=3, print_mode=0)
            for i in range(2):
                W.addNode(f"orig{i}", 0, i)
            W.addNode("merge", 1, 1)
            W.addNode("dest", 2, 1)
            for i in range(2):
                W.addLink(
                    f"in{i}", f"orig{i}", "merge", length=1, free_flow_speed=1,
                    jam_density=10.0, capacity=10.0
                )
            W.addLink(
                "out", "merge", "dest", length=1, free_flow_speed=1,
                jam_density=10.0, capacity=1.0
            )
            for i in range(2):
                W.adddemand(f"orig{i}", "dest", 0, 3, 1.0)
            return W

        D = jnp.array([0.3, 0.4])
        p = jnp.array([1.0, 1.0])
        S = jnp.float32(1.0)

        prior_dQ_dD = jax.grad(lambda d: jnp.sum(self._prior_merge_2inlinks(d, p, S)))(D)
        prior_dQ_dp = jax.grad(lambda pr: jnp.sum(self._prior_merge_2inlinks(D, pr, S)))(p)
        prior_dQ_dS = jax.grad(lambda s: jnp.sum(self._prior_merge_2inlinks(D, p, s)))(S)

        curr_dQ_dD = jax.grad(lambda d: self._simulate_minimal_merge(factory, d, p, S))(D)
        curr_dQ_dp = jax.grad(lambda pr: self._simulate_minimal_merge(factory, D, pr, S))(p)
        curr_dQ_dS = jax.grad(lambda s: self._simulate_minimal_merge(factory, D, p, s))(S)

        assert jnp.allclose(prior_dQ_dD, jnp.array([1.0, 1.0]), atol=1e-5)
        assert jnp.allclose(prior_dQ_dp, jnp.array([0.0, 0.0]), atol=1e-5)
        assert jnp.allclose(prior_dQ_dS, 0.0, atol=1e-5)

        assert jnp.allclose(curr_dQ_dD, prior_dQ_dD, atol=1e-5)
        assert jnp.allclose(curr_dQ_dp, prior_dQ_dp, atol=1e-5)
        assert jnp.allclose(curr_dQ_dS, prior_dQ_dS, atol=1e-5)

    def test_grad_merge_2inlinks_vs_prior_one_under_capacity(self):
        """Compare gradients when one link is under priority share, taking residual supply."""
        def factory():
            W = World(name="", deltat=1, tmax=3, print_mode=0)
            for i in range(2):
                W.addNode(f"orig{i}", 0, i)
            W.addNode("merge", 1, 1)
            W.addNode("dest", 2, 1)
            for i in range(2):
                W.addLink(
                    f"in{i}", f"orig{i}", "merge", length=1, free_flow_speed=1,
                    jam_density=10.0, capacity=10.0
                )
            W.addLink(
                "out", "merge", "dest", length=1, free_flow_speed=1,
                jam_density=10.0, capacity=1.0
            )
            for i in range(2):
                W.adddemand(f"orig{i}", "dest", 0, 3, 1.0)
            return W

        D = jnp.array([0.2, 0.9])
        p = jnp.array([1.0, 1.0])
        S = jnp.float32(1.0)

        prior_dQ_dD = jax.grad(lambda d: jnp.sum(self._prior_merge_2inlinks(d, p, S)))(D)
        prior_dQ_dp = jax.grad(lambda pr: jnp.sum(self._prior_merge_2inlinks(D, pr, S)))(p)
        prior_dQ_dS = jax.grad(lambda s: jnp.sum(self._prior_merge_2inlinks(D, p, s)))(S)

        curr_dQ_dD = jax.grad(lambda d: self._simulate_minimal_merge(factory, d, p, S))(D)
        curr_dQ_dp = jax.grad(lambda pr: self._simulate_minimal_merge(factory, D, pr, S))(p)
        curr_dQ_dS = jax.grad(lambda s: self._simulate_minimal_merge(factory, D, p, s))(S)

        assert jnp.allclose(prior_dQ_dD, jnp.array([0.0, 0.0]), atol=1e-5)
        assert jnp.allclose(prior_dQ_dp, jnp.array([0.0, 0.0]), atol=1e-5)
        assert jnp.allclose(prior_dQ_dS, 1.0, atol=1e-5)

        assert jnp.allclose(curr_dQ_dD, prior_dQ_dD, atol=1e-5)
        assert jnp.allclose(curr_dQ_dp, prior_dQ_dp, atol=1e-5)
        assert jnp.allclose(curr_dQ_dS, prior_dQ_dS, atol=1e-5)

    def test_grad_merge_2inlinks_boundary(self):
        """Under supply constraint at boundary D[0] == alpha[0]*S, true sensitivities must be:
        dQ/dD = [0, 0], dQ/dp = [0, 0], dQ/dS = 1.0.
        """
        def factory():
            W = World(name="", deltat=1, tmax=3, print_mode=0)
            for i in range(2):
                W.addNode(f"orig{i}", 0, i)
            W.addNode("merge", 1, 1)
            W.addNode("dest", 2, 1)
            for i in range(2):
                W.addLink(
                    f"in{i}", f"orig{i}", "merge", length=1, free_flow_speed=1,
                    jam_density=10.0, capacity=10.0
                )
            W.addLink(
                "out", "merge", "dest", length=1, free_flow_speed=1,
                jam_density=10.0, capacity=1.0
            )
            for i in range(2):
                W.adddemand(f"orig{i}", "dest", 0, 3, 1.0)
            return W

        D = jnp.array([0.5, 1.0])
        p = jnp.array([1.0, 1.0])
        S = jnp.float32(1.0)

        curr_dQ_dD = jax.grad(lambda d: self._simulate_minimal_merge(factory, d, p, S))(D)
        curr_dQ_dp = jax.grad(lambda pr: self._simulate_minimal_merge(factory, D, pr, S))(p)
        curr_dQ_dS = jax.grad(lambda s: self._simulate_minimal_merge(factory, D, p, s))(S)

        assert jnp.allclose(curr_dQ_dD, jnp.array([0.0, 0.0]), atol=1e-4), f"Expected dQ/dD=[0, 0], got {curr_dQ_dD}"
        assert jnp.allclose(curr_dQ_dp, jnp.array([0.0, 0.0]), atol=1e-4), f"Expected dQ/dp=[0, 0], got {curr_dQ_dp}"
        assert jnp.isclose(float(curr_dQ_dS), 1.0, atol=1e-4), f"Expected dQ/dS=1.0, got {curr_dQ_dS}"

    def test_grad_merge_3inlinks_boundary(self):
        """3-inlink boundary test: D = [0.25, 0.75, 0.75], p = [1.0, 1.0, 2.0], S = 1.0.

        Here alpha = [0.25, 0.25, 0.50], so D[0] == alpha[0] * S = 0.25.
        Under supply constraint, true sensitivities must be: dQ/dD[0] = 0.0, dQ/dS = 1.0.
        """
        def factory():
            W = World(name="", deltat=1, tmax=3, print_mode=0)
            for i in range(3):
                W.addNode(f"orig{i}", 0, i)
            W.addNode("merge", 1, 1)
            W.addNode("dest", 2, 1)
            for i in range(3):
                W.addLink(
                    f"in{i}", f"orig{i}", "merge", length=1, free_flow_speed=1,
                    jam_density=10.0, capacity=10.0
                )
            W.addLink(
                "out", "merge", "dest", length=1, free_flow_speed=1,
                jam_density=10.0, capacity=1.0
            )
            for i in range(3):
                W.adddemand(f"orig{i}", "dest", 0, 3, 1.0)
            return W

        D = jnp.array([0.25, 0.75, 0.75])
        p = jnp.array([1.0, 1.0, 2.0])
        S = jnp.float32(1.0)

        curr_dQ_dD = jax.grad(lambda d: self._simulate_minimal_merge(factory, d, p, S))(D)
        curr_dQ_dS = jax.grad(lambda s: self._simulate_minimal_merge(factory, D, p, s))(S)

        assert jnp.isclose(float(curr_dQ_dD[0]), 0.0, atol=1e-4), f"Expected dQ/dD[0]=0, got {curr_dQ_dD[0]}"
        assert jnp.isclose(float(curr_dQ_dS), 1.0, atol=1e-4), f"Expected dQ/dS=1.0, got {curr_dQ_dS}"


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
        In simulate, jnp.maximum(D, 0) at D=0 (step k=9) also gives
        50/50, halving d(cum_departure)/d(u).  But the straight-through
        clip in invert_interp_1d prevents further halving of d(t_queue).

        AD gradient verified against central finite differences.
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

        # Finite-difference reference
        eps = 1e-4
        u0 = float(params.u[0])
        pp = params._replace(u=params.u.at[0].set(u0 + eps))
        pm = params._replace(u=params.u.at[0].set(u0 - eps))
        fd_grad = (float(loss_full(pp)) - float(loss_full(pm))) / (2 * eps)

        assert abs(ad_grad - fd_grad) / max(abs(fd_grad), 1.0) < 0.15, \
            f"AD={ad_grad:.4f}, FD={fd_grad:.4f}"


# ================================================================
# AD vs Finite-Difference regression tests
# ================================================================

_REG_REL_TOL = 0.20
_REG_ABS_TOL = 0.1
_FD_DELTA = 1e-3


def _check_reg(actual, expected, label=""):
    """Assert actual matches expected within rel_tol=20% or abs_tol=0.1."""
    diff = abs(actual - expected)
    if abs(expected) > _REG_ABS_TOL:
        ok = diff / abs(expected) < _REG_REL_TOL or diff < _REG_ABS_TOL
    else:
        ok = diff < _REG_ABS_TOL
    assert ok, (
        f"{label}: actual={actual:.6f}, expected={expected:.6f}, "
        f"diff={diff:.6f}")


def _build_merge_regression():
    W = World(name="", deltat=5, tmax=2000, print_mode=0)
    W.addNode("orig1", 0, 0); W.addNode("orig2", 0, 2)
    W.addNode("merge", 1, 1); W.addNode("dest", 2, 1)
    W.addLink("link1", "orig1", "merge", length=1000,
              free_flow_speed=20, jam_density=0.2, merge_priority=1)
    W.addLink("link2", "orig2", "merge", length=1000,
              free_flow_speed=20, jam_density=0.2, merge_priority=1)
    W.addLink("link3", "merge", "dest", length=1000,
              free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig1", "dest", 0, 1000, 0.45)
    W.adddemand("orig2", "dest", 400, 1000, 0.6)
    return W


class TestMergeAD:
    """AD gradient regression for merge scenario."""

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, request):
        W = _build_merge_regression()
        params, config = world_to_jax(W)
        request.cls.params = params
        request.cls.config = config
        request.cls.mp1_base = params.merge_priority[0]

    def test_ad_demand_orig1(self):
        p, c = self.params, self.config
        grad = jax.grad(lambda dr: total_travel_time(
            simulate(p._replace(demand_rate=dr), c), c))(p.demand_rate)
        _check_reg(float(jnp.sum(grad[0])), 437453.62, "demand_orig1")

    def test_ad_demand_orig2(self):
        p, c = self.params, self.config
        grad = jax.grad(lambda dr: total_travel_time(
            simulate(p._replace(demand_rate=dr), c), c))(p.demand_rate)
        _check_reg(float(jnp.sum(grad[1])), 421502.56, "demand_orig2")

    def test_ad_speed_link1(self):
        p, c = self.params, self.config
        grad = jax.grad(lambda u: total_travel_time(
            simulate(p._replace(u=u), c), c))(p.u)
        _check_reg(float(grad[0]), -1278.28, "speed_link1")

    def test_ad_speed_link2(self):
        p, c = self.params, self.config
        grad = jax.grad(lambda u: total_travel_time(
            simulate(p._replace(u=u), c), c))(p.u)
        _check_reg(float(grad[1]), -616.88, "speed_link2")

    def test_ad_speed_link3(self):
        p, c = self.params, self.config
        grad = jax.grad(lambda u: total_travel_time(
            simulate(p._replace(u=u), c), c))(p.u)
        _check_reg(float(grad[2]), -2024.69, "speed_link3")

    def test_ad_linkTTT_link1(self):
        p, c, mp1 = self.params, self.config, self.mp1_base
        def fn(m):
            s = simulate(p._replace(
                merge_priority=p.merge_priority.at[0].set(m)), c)
            n = s.cum_arrival[:, :c.tsize] - s.cum_departure[:, :c.tsize]
            return jnp.sum(n, axis=1) * c.deltat
        _check_reg(float(jax.jacfwd(fn)(mp1)[0]), -45900.04, "linkTTT_link1")

    def test_ad_linkTTT_link2(self):
        p, c, mp1 = self.params, self.config, self.mp1_base
        def fn(m):
            s = simulate(p._replace(
                merge_priority=p.merge_priority.at[0].set(m)), c)
            n = s.cum_arrival[:, :c.tsize] - s.cum_departure[:, :c.tsize]
            return jnp.sum(n, axis=1) * c.deltat
        _check_reg(float(jax.jacfwd(fn)(mp1)[1]), 40725.04, "linkTTT_link2")

    def test_ad_odTT_orig1_t100(self):
        p, c, mp1 = self.params, self.config, self.mp1_base
        def fn(m):
            pp = p._replace(merge_priority=p.merge_priority.at[0].set(m))
            return travel_time_auto(0, 3, 100.0, simulate(pp, c), pp, c)
        _check_reg(float(jax.grad(fn)(mp1)), 0.0, "odTT_orig1_t100")

    def test_ad_odTT_orig2_t500(self):
        p, c, mp1 = self.params, self.config, self.mp1_base
        def fn(m):
            pp = p._replace(merge_priority=p.merge_priority.at[0].set(m))
            return travel_time_auto(1, 3, 500.0, simulate(pp, c), pp, c)
        _check_reg(float(jax.grad(fn)(mp1)), 75.0, "odTT_orig2_t500")


class TestMergeFD:
    """Finite-difference gradient regression for merge scenario."""

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, request):
        W = _build_merge_regression()
        params, config = world_to_jax(W)
        request.cls.params = params
        request.cls.config = config
        request.cls.mp1_base = float(params.merge_priority[0])

    def _fd(self, fn):
        return (fn(_FD_DELTA) - fn(-_FD_DELTA)) / (2 * _FD_DELTA)

    def test_fd_demand_orig1(self):
        p, c = self.params, self.config
        def fn(d):
            return float(total_travel_time(
                simulate(p._replace(demand_rate=p.demand_rate.at[0,:].add(d)), c), c))
        _check_reg(self._fd(fn), 448125.00, "fd_demand_orig1")

    def test_fd_speed_link1(self):
        p, c = self.params, self.config
        def fn(d):
            return float(total_travel_time(
                simulate(p._replace(u=p.u.at[0].add(d)), c), c))
        _check_reg(self._fd(fn), -1351.56, "fd_speed_link1")

    def test_fd_odTT_orig2_t500(self):
        p, c, mp1 = self.params, self.config, self.mp1_base
        def fn(d):
            pp = p._replace(merge_priority=p.merge_priority.at[0].set(mp1+d))
            return float(travel_time_auto(1, 3, 500.0, simulate(pp, c), pp, c))
        _check_reg(self._fd(fn), 75.04, "fd_odTT_orig2_t500")


def _build_duo_regression():
    W = World(name="", deltat=5, tmax=4000, print_mode=0,
              route_choice="duo_logit")
    W.LOGIT_TEMPERATURE = 60.0
    W.addNode("orig", 0, 0); W.addNode("mid1", 1, 1)
    W.addNode("mid2", 1, -1); W.addNode("dest", 2, 0)
    W.addLink("fast1", "orig", "mid1", length=1000, free_flow_speed=20, capacity=0.8)
    W.addLink("fast2", "mid1", "dest", length=500, free_flow_speed=20, capacity=0.6)
    W.addLink("slow1", "orig", "mid2", length=1000, free_flow_speed=10, capacity=0.8)
    W.addLink("slow2", "mid2", "dest", length=500, free_flow_speed=10, capacity=0.8)
    W.adddemand("orig", "dest", 0, 3000, 0.6)
    W.adddemand("orig", "dest", 500, 2000, 0.2)
    return W


class TestDuoAD:
    """AD gradient regression for DUO logit scenario."""

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, request):
        W = _build_duo_regression()
        params, config = world_to_jax(W)
        li = {l.name: i for i, l in enumerate(W.LINKS)}
        ni = {n.name: i for i, n in enumerate(W.NODES)}
        request.cls.params = params
        request.cls.config = config
        request.cls.fast2_id = li["fast2"]
        request.cls.orig_id = ni["orig"]
        request.cls.dest_id = ni["dest"]
        request.cls.fast_path = (li["fast1"], li["fast2"])
        request.cls.cap0 = float(params.q_star[li["fast2"]])

    def _replace_cap(self, cap):
        return self.params._replace(
            q_star=self.params.q_star.at[self.fast2_id].set(cap))

    def test_ad_ttt_all(self):
        g = float(jax.grad(lambda c: total_travel_time(
            simulate_duo(self._replace_cap(c), self.config),
            self.config))(self.cap0))
        _check_reg(g, -586318.62, "ttt_all")

    def test_ad_odTT_t1500(self):
        def fn(c):
            p = self._replace_cap(c)
            return travel_time_auto(self.orig_id, self.dest_id,
                                    1500.0, simulate_duo(p, self.config), p, self.config)
        _check_reg(float(jax.grad(fn)(self.cap0)), -570.35, "odTT_t1500")

    def test_ad_pathTT_fast_t1500(self):
        def fn(c):
            p = self._replace_cap(c)
            return travel_time(self.fast_path, 1500.0,
                               simulate_duo(p, self.config), p, self.config)
        _check_reg(float(jax.grad(fn)(self.cap0)), -570.35, "pathTT_fast_t1500")


class TestDuoFD:
    """Finite-difference gradient regression for DUO logit scenario."""

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, request):
        W = _build_duo_regression()
        params, config = world_to_jax(W)
        li = {l.name: i for i, l in enumerate(W.LINKS)}
        ni = {n.name: i for i, n in enumerate(W.NODES)}
        request.cls.params = params
        request.cls.config = config
        request.cls.fast2_id = li["fast2"]
        request.cls.orig_id = ni["orig"]
        request.cls.dest_id = ni["dest"]
        request.cls.fast_path = (li["fast1"], li["fast2"])
        request.cls.cap0 = float(params.q_star[li["fast2"]])

    def _replace_cap(self, cap):
        return self.params._replace(
            q_star=self.params.q_star.at[self.fast2_id].set(cap))

    def _fd(self, fn):
        return (fn(self.cap0+_FD_DELTA) - fn(self.cap0-_FD_DELTA)) / (2*_FD_DELTA)

    def test_fd_ttt_all(self):
        def fn(c):
            return float(total_travel_time(
                simulate_duo(self._replace_cap(c), self.config), self.config))
        _check_reg(self._fd(fn), -585710.94, "fd_ttt_all")

    def test_fd_odTT_t1500(self):
        def fn(c):
            p = self._replace_cap(c)
            return float(travel_time_auto(
                self.orig_id, self.dest_id, 1500.0,
                simulate_duo(p, self.config), p, self.config))
        _check_reg(self._fd(fn), -570.13, "fd_odTT_t1500")
