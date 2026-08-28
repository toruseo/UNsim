"""Tests for Params.route_bias (additive logit bias on duo_logit
route choice in unsim_diff)."""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsim import World
from unsim.unsim_diff import world_to_jax, simulate_duo


def _two_route_world(route_choice="duo_logit"):
    """Origin -> two parallel links -> destination."""
    W = World(name="bias-test", tmax=3600, route_choice=route_choice,
              logit_temperature=120.0)
    W.addNode("O", 0, 0)
    W.addNode("A", 0, 1)
    W.addNode("D", 1, 0)
    W.addLink("conn", "O", "A", length=2000, free_flow_speed=15,
              jam_density=0.2)
    W.addLink("fast", "A", "D", length=2000, free_flow_speed=20,
              jam_density=0.2)
    W.addLink("slow", "A", "D", length=2000, free_flow_speed=10,
              jam_density=0.2)
    W.adddemand("O", "D", 0, 1800, 0.3)
    W.finalize_scenario()
    return W


def _link_id(W, name):
    return W.LINKS_NAME_DICT[name].id


def test_default_bias_is_zero():
    W = _two_route_world()
    params, config = world_to_jax(W)
    assert params.route_bias.shape[1] == config.n_nodes
    assert float(jnp.abs(params.route_bias).max()) == 0.0


def test_bias_shifts_split():
    """Positive bias on the slow link's slot increases its share."""
    W = _two_route_world()
    params, config = world_to_jax(W)
    state0 = simulate_duo(params, config, differentiable=False)
    slow = _link_id(W, "slow")
    fast = _link_id(W, "fast")
    base_slow = float(state0.cum_arrival[slow, -1])
    base_fast = float(state0.cum_arrival[fast, -1])
    assert base_fast > base_slow  # logit prefers the faster link

    # find the (node A, outlink slot) of the slow link
    node_a = int(np.asarray(config.link_start_node)[slow])
    outlinks = np.asarray(config.node_outlinks)[node_a]
    slot = int(np.where(outlinks == slow)[0][0])
    rb = params.route_bias.at[:, node_a, slot].set(4.0)
    state1 = simulate_duo(params._replace(route_bias=rb), config,
                          differentiable=False)
    assert float(state1.cum_arrival[slow, -1]) > base_slow + 1.0
    assert float(state1.cum_arrival[fast, -1]) < base_fast - 1.0


def test_bias_grad_nonzero():
    W = _two_route_world()
    params, config = world_to_jax(W)

    def ttt(rb):
        s = simulate_duo(params._replace(route_bias=rb), config)
        return jnp.sum(s.cum_arrival[:, -1])

    g = jax.grad(ttt)(params.route_bias)
    assert np.isfinite(np.asarray(g)).all()
    assert float(jnp.abs(g).max()) > 0.0


def test_hard_duo_ignores_bias():
    """Non-logit (hard) DUO routing must be unaffected by route_bias."""
    W = _two_route_world(route_choice="duo")
    params, config = world_to_jax(W)
    s0 = simulate_duo(params, config, differentiable=False)
    rb = params.route_bias + 5.0
    s1 = simulate_duo(params._replace(route_bias=rb), config,
                      differentiable=False)
    np.testing.assert_allclose(np.asarray(s0.cum_arrival),
                               np.asarray(s1.cum_arrival), atol=1e-4)
