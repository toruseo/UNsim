"""
Tests for gradient checkpointing (checkpoint_every option) in unsim_diff.

Verifies that simulate / simulate_duo / simulate_aon with
checkpoint_every produce the same states and gradients as the default
(non-checkpointed) path, including segment lengths that do not divide
tsize evenly.

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

from unsim import World
from unsim.unsim_diff import (
    world_to_jax, simulate, simulate_duo, simulate_aon, total_travel_time,
)


def make_ltm_world():
    """2-link bottleneck network for plain LTM simulation."""
    W = World(name="", deltat=5, tmax=2000, print_mode=0)
    W.addNode("orig", 0, 0)
    W.addNode("mid", 0.5, 0)
    W.addNode("dest", 1, 1)
    W.addLink("l1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    W.addLink("l2", "mid", "dest", length=1000, free_flow_speed=10, jam_density=0.2)
    W.adddemand("orig", "dest", 0, 500, 0.4)
    return W


def make_duo_world():
    """2-route network with congestion for DUO route choice."""
    W = World(name="", deltat=5, tmax=2000, print_mode=0, route_choice="duo")
    W.addNode("orig", 0, 0)
    W.addNode("mid1", 0.5, 0)
    W.addNode("mid2", 0.5, 1)
    W.addNode("dest", 1, 1)
    W.addLink("l1", "orig", "mid1", length=1000, free_flow_speed=20, jam_density=0.2)
    W.addLink("l2", "orig", "mid2", length=1500, free_flow_speed=20, jam_density=0.2)
    W.addLink("l3", "mid1", "dest", length=1000, free_flow_speed=10, jam_density=0.2)
    W.addLink("l4", "mid2", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest", 0, 500, 0.6)
    return W


# checkpoint_every values: divisor of tsize=400, non-divisor (remainder),
# 1 (per-step), equal to tsize, and larger than tsize (clamped)
CE_VALUES = [1, 7, 100, 400, 1000]


class TestSimulateCheckpoint:
    """simulate() with checkpoint_every matches the default path."""

    def setup_method(self):
        W = make_ltm_world()
        self.params, self.config = world_to_jax(W)
        self.state_ref = simulate(self.params, self.config)
        self.grad_ref = jax.grad(
            lambda p: total_travel_time(simulate(p, self.config), self.config)
        )(self.params)

    @pytest.mark.parametrize("ce", CE_VALUES)
    def test_state_matches(self, ce):
        state = simulate(self.params, self.config, checkpoint_every=ce)
        assert np.allclose(np.array(self.state_ref.cum_arrival),
                           np.array(state.cum_arrival), atol=1e-5)
        assert np.allclose(np.array(self.state_ref.cum_departure),
                           np.array(state.cum_departure), atol=1e-5)

    @pytest.mark.parametrize("ce", CE_VALUES)
    def test_grad_matches(self, ce):
        grad = jax.grad(
            lambda p: total_travel_time(
                simulate(p, self.config, checkpoint_every=ce), self.config)
        )(self.params)
        assert np.allclose(np.array(self.grad_ref.q_star),
                           np.array(grad.q_star), atol=1e-4)
        assert np.allclose(np.array(self.grad_ref.u),
                           np.array(grad.u), atol=1e-4)
        assert np.allclose(np.array(self.grad_ref.demand_rate),
                           np.array(grad.demand_rate), atol=1e-4)

    def test_invalid_checkpoint_every(self):
        with pytest.raises(ValueError):
            simulate(self.params, self.config, checkpoint_every=0)
        with pytest.raises(ValueError):
            simulate(self.params, self.config, checkpoint_every=-5)


class TestSimulateDuoCheckpoint:
    """simulate_duo() with checkpoint_every matches the default path."""

    def setup_method(self):
        W = make_duo_world()
        self.params, self.config = world_to_jax(W)
        self.state_ref = simulate_duo(self.params, self.config)
        self.grad_ref = jax.grad(
            lambda p: total_travel_time(simulate_duo(p, self.config), self.config)
        )(self.params)

    @pytest.mark.parametrize("ce", CE_VALUES)
    def test_state_matches(self, ce):
        state = simulate_duo(self.params, self.config, checkpoint_every=ce)
        assert np.allclose(np.array(self.state_ref.cum_arrival),
                           np.array(state.cum_arrival), atol=1e-5)
        assert np.allclose(np.array(self.state_ref.cum_departure_d),
                           np.array(state.cum_departure_d), atol=1e-5)

    @pytest.mark.parametrize("ce", CE_VALUES)
    def test_grad_matches(self, ce):
        grad = jax.grad(
            lambda p: total_travel_time(
                simulate_duo(p, self.config, checkpoint_every=ce), self.config)
        )(self.params)
        assert np.allclose(np.array(self.grad_ref.q_star),
                           np.array(grad.q_star), atol=1e-4)
        assert np.allclose(np.array(self.grad_ref.od_demand_rate),
                           np.array(grad.od_demand_rate), atol=1e-4)


class TestSimulateAonCheckpoint:
    """simulate_aon() passes checkpoint_every through to simulate_duo."""

    def test_state_and_grad_match(self):
        W = make_duo_world()
        params, config = world_to_jax(W)
        state_ref = simulate_aon(params, config)
        state = simulate_aon(params, config, checkpoint_every=60)
        assert np.allclose(np.array(state_ref.cum_arrival),
                           np.array(state.cum_arrival), atol=1e-5)

        grad_ref = jax.grad(
            lambda p: total_travel_time(simulate_aon(p, config), config)
        )(params)
        grad = jax.grad(
            lambda p: total_travel_time(
                simulate_aon(p, config, checkpoint_every=60), config)
        )(params)
        assert np.allclose(np.array(grad_ref.od_demand_rate),
                           np.array(grad.od_demand_rate), atol=1e-4)


class TestCheckpointUnderJit:
    """checkpoint_every works under jax.jit with static argument."""

    def test_jit_value_and_grad(self):
        W = make_ltm_world()
        params, config = world_to_jax(W)

        def loss(p, ce):
            return total_travel_time(
                simulate(p, config, checkpoint_every=ce), config)

        loss_ref = float(total_travel_time(simulate(params, config), config))
        jitted = jax.jit(jax.value_and_grad(loss), static_argnums=(1,))
        val, grad = jitted(params, 50)
        assert np.isclose(float(val), loss_ref, rtol=1e-5)
        grad_ref = jax.grad(
            lambda p: total_travel_time(simulate(p, config), config))(params)
        assert np.allclose(np.array(grad_ref.q_star),
                           np.array(grad.q_star), atol=1e-4)
