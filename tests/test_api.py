"""
Tests for the object-oriented facade (unsim.api) over the JAX core.

Verifies that the facade reproduces raw-core results exactly, that FD parameterization provenance gives the intended gradient semantics, and that snapshot immutability and analyzer write-back work.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from unsim import World
from unsim.unsim_diff import (
    world_to_jax, simulate, total_travel_time, trip_completed,
    average_travel_time, travel_time, travel_time_soft,
)
from unsim.api import ParameterRef, VariableSet, PiecewiseConstant, CustomVariable
from scenario_parallel_routes import create_world

REFERENCE_NPZ = os.path.join(os.path.dirname(__file__), "data", "parallel_routes_reference.npz")


def build_merge_world():
    """Two origins merging into one bottleneck link (route_choice=fix)."""
    W = World(name="", deltat=5, tmax=2000, print_mode=0)
    W.addNode("orig1", 0, 0)
    W.addNode("orig2", 0, 2)
    W.addNode("merge", 1, 1)
    W.addNode("dest", 2, 1)
    W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
    W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
    W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig1", "dest", 0, 1000, 0.45)
    W.adddemand("orig2", "dest", 400, 1000, 0.6)
    return W


def build_capacity_world():
    """Same merge network but link3 uses the (u, w, capacity) parameterization."""
    W = World(name="", deltat=5, tmax=2000, print_mode=0)
    W.addNode("orig1", 0, 0)
    W.addNode("orig2", 0, 2)
    W.addNode("merge", 1, 1)
    W.addNode("dest", 2, 1)
    W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20, jam_density=0.2)
    W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20, jam_density=0.2)
    W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, capacity=0.7)
    W.adddemand("orig1", "dest", 0, 1000, 0.45)
    W.adddemand("orig2", "dest", 400, 1000, 0.6)
    return W


# ================================================================
# Forward equivalence
# ================================================================

class TestForwardEquivalence:
    def test_run_matches_raw(self):
        W = build_merge_world()
        params, config = world_to_jax(W)
        state_raw = simulate(params, config)

        M = build_merge_world().compile(backend="jax")
        R = M.run()

        np.testing.assert_allclose(np.asarray(R.state.cum_arrival), np.asarray(state_raw.cum_arrival), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(R.state.cum_departure), np.asarray(state_raw.cum_departure), rtol=1e-6)

    def test_forward_only_matches(self):
        M = build_merge_world().compile(backend="jax")
        R1 = M.run(differentiable=True)
        R2 = M.run(differentiable=False)
        np.testing.assert_allclose(
            float(R1.metrics.total_travel_time()),
            float(R2.metrics.total_travel_time()), rtol=1e-4)

    def test_metrics_match_core(self):
        W = build_merge_world()
        params, config = world_to_jax(W)
        state = simulate(params, config)

        M = build_merge_world().compile(backend="jax")
        R = M.run()
        assert float(R.metrics.total_travel_time()) == pytest.approx(float(total_travel_time(state, config)), rel=1e-6)
        assert float(R.metrics.completed_trips()) == pytest.approx(float(trip_completed(state, config)), rel=1e-6)
        assert float(R.metrics.average_travel_time()) == pytest.approx(float(average_travel_time(state, config)), rel=1e-6)


# ================================================================
# Gradients
# ================================================================

class TestGradient:
    def test_merge_priority_matches_raw(self):
        W = build_merge_world()
        params, config = world_to_jax(W)

        def raw_loss(mp1):
            p = params._replace(merge_priority=params.merge_priority.at[0].set(mp1))
            return total_travel_time(simulate(p, config), config)

        g_raw = float(jax.grad(raw_loss)(params.merge_priority[0]))

        M = build_merge_world().compile(backend="jax")
        link1 = M._snapshot_link(0)
        mp1 = M.parameter(link1, "merge_priority")
        objective = M.objective(lambda R: R.metrics.total_travel_time())
        value, grad = objective.value_and_gradient(wrt=[mp1])

        assert float(grad[mp1]) == pytest.approx(g_raw, rel=1e-5)
        assert float(value) == pytest.approx(float(raw_loss(params.merge_priority[0])), rel=1e-6)

    def test_fd_u_kappa_tau_consistent_gradient(self):
        # Under (u, kappa, tau), varying u must also vary q* = u*w*kappa/(u+w) with w = 1/(tau*kappa).
        W = build_merge_world()
        params, config = world_to_jax(W)
        link3 = W.LINKS[2]
        tau = link3.tau
        kappa3 = float(params.kappa[2])
        w3 = 1.0 / (tau * kappa3)

        def raw_loss_consistent(u3):
            q3 = u3 * w3 * kappa3 / (u3 + w3)
            p = params._replace(u=params.u.at[2].set(u3), q_star=params.q_star.at[2].set(q3))
            return total_travel_time(simulate(p, config), config)

        def raw_loss_naive(u3):
            p = params._replace(u=params.u.at[2].set(u3))
            return total_travel_time(simulate(p, config), config)

        g_consistent = float(jax.grad(raw_loss_consistent)(params.u[2]))
        g_naive = float(jax.grad(raw_loss_naive)(params.u[2]))

        M = build_merge_world().compile(backend="jax")
        u3_ref = M.parameter("link3", "free_flow_speed")
        objective = M.objective(lambda R: R.metrics.total_travel_time())
        _, grad = objective.value_and_gradient(wrt=[u3_ref])

        assert float(grad[u3_ref]) == pytest.approx(g_consistent, rel=1e-5)
        # link3 is the active bottleneck, so the naive gradient (q* fixed) must differ.
        assert abs(g_consistent - g_naive) > 1e-3 * max(abs(g_consistent), 1.0)

    def test_multiple_refs(self):
        M = build_merge_world().compile(backend="jax")
        u3 = M.parameter("link3", "free_flow_speed")
        mp1 = M.parameter("link1", "merge_priority")
        objective = M.objective(lambda R: R.metrics.total_travel_time())
        _, grad = objective.value_and_gradient(wrt=[u3, mp1])
        _, grad_u3 = objective.value_and_gradient(wrt=[u3])
        _, grad_mp1 = objective.value_and_gradient(wrt=[mp1])
        assert float(grad[u3]) == pytest.approx(float(grad_u3[u3]), rel=1e-5)
        assert float(grad[mp1]) == pytest.approx(float(grad_mp1[mp1]), rel=1e-5)


# ================================================================
# FD parameterization provenance
# ================================================================

class TestFDProvenance:
    def test_default_parameterization_recorded(self):
        W = build_merge_world()
        assert W.LINKS[0].fd_parameterization == "u_kappa_tau"
        W2 = build_capacity_world()
        assert W2.LINKS[2].fd_parameterization == "u_w_capacity"

    def test_derived_quantity_error(self):
        M = build_merge_world().compile(backend="jax")
        with pytest.raises(ValueError, match="derived"):
            M.parameter("link1", "capacity")
        M2 = build_capacity_world().compile(backend="jax")
        with pytest.raises(ValueError, match="derived"):
            M2.parameter("link3", "jam_density")

    def test_inject_roundtrip(self):
        # Injecting the initial theta must reproduce the base params for all parameterizations.
        for builder, link_name, fld in [
            (build_merge_world, "link3", "free_flow_speed"),
            (build_merge_world, "link3", "jam_density"),
            (build_capacity_world, "link3", "free_flow_speed"),
            (build_capacity_world, "link3", "capacity"),
        ]:
            M = builder().compile(backend="jax")
            ref = M.parameter(link_name, fld)
            vs = VariableSet(M, [ref])
            p2 = vs.inject(vs.initial_theta(), M.raw.params)
            np.testing.assert_allclose(np.asarray(p2.u), np.asarray(M.raw.params.u), rtol=1e-6)
            np.testing.assert_allclose(np.asarray(p2.kappa), np.asarray(M.raw.params.kappa), rtol=1e-6)
            np.testing.assert_allclose(np.asarray(p2.q_star), np.asarray(M.raw.params.q_star), rtol=1e-6)

    def test_capacity_gradient_u_w_capacity(self):
        # Under (u, w, capacity), varying capacity must also vary kappa = q*(u+w)/(u*w).
        W = build_capacity_world()
        params, config = world_to_jax(W)
        link3 = W.LINKS[2]
        u3 = float(params.u[2])
        w3 = link3.w

        def raw_loss(q3):
            kappa3 = q3 * (u3 + w3) / (u3 * w3)
            p = params._replace(q_star=params.q_star.at[2].set(q3), kappa=params.kappa.at[2].set(kappa3))
            return total_travel_time(simulate(p, config), config)

        g_raw = float(jax.grad(raw_loss)(params.q_star[2]))

        M = build_capacity_world().compile(backend="jax")
        q3_ref = M.parameter("link3", "capacity")
        objective = M.objective(lambda R: R.metrics.total_travel_time())
        _, grad = objective.value_and_gradient(wrt=[q3_ref])
        assert float(grad[q3_ref]) == pytest.approx(g_raw, rel=1e-5)


# ================================================================
# Result views
# ================================================================

class TestResultViews:
    def test_path_travel_time(self):
        W = build_merge_world()
        params, config = world_to_jax(W)
        state = simulate(params, config)
        tt_raw = float(travel_time([0, 2], 300.0, state, params, config))

        M = build_merge_world().compile(backend="jax")
        R = M.run()
        path = M.path(["link1", "link3"])
        assert float(R.path(path).travel_time(300.0)) == pytest.approx(tt_raw, rel=1e-6)

    def test_od_soft_travel_time(self):
        W = build_merge_world()
        params, config = world_to_jax(W)
        state = simulate(params, config)
        o = W.NODES_NAME_DICT["orig1"].id
        d = W.NODES_NAME_DICT["dest"].id
        tt_raw = float(travel_time_soft(o, d, 300.0, state, params, config))

        M = build_merge_world().compile(backend="jax")
        R = M.run()
        assert float(R.od("orig1", "dest").travel_time(300.0, method="soft")) == pytest.approx(tt_raw, rel=1e-6)

    def test_link_and_node_views(self):
        M = build_merge_world().compile(backend="jax")
        R = M.run()
        n_veh = float(R.link("link3").vehicle_count(600.0))
        assert n_veh > 0.0
        dens = float(R.link("link3").density(600.0))
        assert dens == pytest.approx(n_veh / 1000.0, rel=1e-6)
        q = float(R.node("orig2").queue(600.0))
        assert q >= 0.0

    def test_analyzer_writeback(self):
        M = build_merge_world().compile(backend="jax")
        R = M.run()
        analyzer = R.analyzer
        analyzer.basic_analysis()
        ttt_facade = float(R.metrics.total_travel_time())
        assert analyzer.total_travel_time == pytest.approx(ttt_facade, rel=1e-3)
        assert analyzer.trip_completed == pytest.approx(float(R.metrics.completed_trips()), rel=1e-3)


# ================================================================
# Compilation semantics
# ================================================================

class TestCompilation:
    def test_snapshot_immutability(self):
        W = build_merge_world()
        M1 = W.compile(backend="jax")
        ttt1 = float(M1.run().metrics.total_travel_time())

        W.addNode("extra_a", 5, 5)
        W.addNode("extra_b", 6, 5)
        W.addLink("extra_link", "extra_a", "extra_b", length=1000, free_flow_speed=20)
        M2 = W.compile(backend="jax")

        assert M1.raw.config.n_links == 3
        assert M2.raw.config.n_links == 4
        assert float(M1.run().metrics.total_travel_time()) == pytest.approx(ttt1, rel=1e-8)

    def test_route_update_interval_override(self):
        W = build_merge_world()
        W.ROUTE_CHOICE = "duo_logit"
        M = W.compile(backend="jax", route_update_interval=100)
        assert M.raw.config.route_update_interval == 20
        assert M.raw.config.toll_step_size == 20
        toll_ref = M.parameter("link1", "toll")
        assert toll_ref.shape == (int(M.raw.config.n_toll_steps),)

    def test_parameter_ref_equality(self):
        M = build_merge_world().compile(backend="jax")
        r1 = M.parameter("link1", "free_flow_speed")
        r2 = M.parameter("link1", "free_flow_speed")
        assert r1 == r2
        assert hash(r1) == hash(r2)

    def test_parameter_fields(self):
        W = build_capacity_world()
        M = W.compile(backend="jax")
        # link3 has capacity given -> (u, w, capacity) parameterization.
        assert M.parameter_fields("link3") == [
            "free_flow_speed", "backward_wave_speed", "capacity",
            "merge_priority", "capacity_out", "capacity_in", "toll"]
        # link1 uses the default (u, kappa, tau) parameterization.
        assert M.parameter_fields("link1") == [
            "free_flow_speed", "jam_density",
            "merge_priority", "capacity_out", "capacity_in", "toll"]
        assert M.parameter_fields("orig1") == ["flow_capacity", "absorption_ratio"]
        assert M.parameter_fields(W.DEMANDS[0]) == ["flow"]
        # Every listed field must be accepted by parameter().
        for entity in ["link1", "link3", "orig1", W.DEMANDS[0]]:
            for f in M.parameter_fields(entity):
                M.parameter(entity, f)

    def test_unknown_entities(self):
        M = build_merge_world().compile(backend="jax")
        with pytest.raises(KeyError):
            M.parameter("no_such_link", "free_flow_speed")
        with pytest.raises(ValueError):
            M.parameter("link1", "no_such_field")

    def test_raw_access(self):
        M = build_merge_world().compile(backend="jax")
        state = M.raw.simulate(M.raw.params, M.raw.config)
        R = M.run()
        np.testing.assert_allclose(np.asarray(state.cum_arrival), np.asarray(R.state.cum_arrival), rtol=1e-6)


# ================================================================
# Demand as a first-class object
# ================================================================

class TestDemand:
    def test_adddemand_returns_demand(self):
        W = World(name="", deltat=5, tmax=100, print_mode=0)
        W.addNode("a", 0, 0)
        W.addNode("b", 1, 0)
        W.addLink("ab", "a", "b", length=100, free_flow_speed=20)
        d = W.adddemand("a", "b", 0, 50, flow=0.1)
        assert d.orig == "a" and d.dest == "b"
        assert d.flow == pytest.approx(0.1)
        assert W.DEMANDS[d.id] is d

    def test_demand_gradient_matches_raw(self):
        W = build_merge_world()
        params, config = world_to_jax(W)
        d0 = W.DEMANDS[0]
        oid = W.NODES_NAME_DICT[d0.orig].id
        dt = config.deltat
        i0 = max(int(np.ceil(d0.t_start / dt - 1e-10)), 0)
        i1 = min(int(np.ceil(d0.t_end / dt - 1e-10)), config.tsize)

        def raw_loss(flow):
            delta = flow - d0.flow
            p = params._replace(demand_rate=params.demand_rate.at[oid, i0:i1].add(delta))
            return total_travel_time(simulate(p, config), config)

        g_raw = float(jax.grad(raw_loss)(jnp.float32(d0.flow)))

        W2 = build_merge_world()
        M = W2.compile(backend="jax")
        q0 = M.parameter(W2.DEMANDS[0], "flow")
        objective = M.objective(lambda R: R.metrics.total_travel_time())
        _, grad = objective.value_and_gradient(wrt=[q0])
        assert float(grad[q0]) == pytest.approx(g_raw, rel=1e-5)

    def test_overlapping_demands_gradients_separate(self):
        # Two demand declarations on the same OD with different time windows must get distinct gradients.
        W = build_merge_world()
        d3 = W.adddemand("orig1", "dest", 500, 800, flow=0.1)
        M = W.compile(backend="jax")
        q1 = M.parameter(W.DEMANDS[0], "flow")
        q3 = M.parameter(d3, "flow")
        objective = M.objective(lambda R: R.metrics.total_travel_time())
        _, grad = objective.value_and_gradient(wrt=[q1, q3])
        g1 = float(grad[q1])
        g3 = float(grad[q3])
        assert g1 > 0.0 and g3 > 0.0
        # The windows differ in length, so the sensitivities must differ.
        assert abs(g1 - g3) > 1e-3 * max(g1, g3)


# ================================================================
# Time profiles and interval separation
# ================================================================

class TestTimeProfile:
    def test_piecewise_constant(self):
        profile = PiecewiseConstant(breakpoints=[0, 300, 600, 900], values=[0, 20, 50])
        assert profile(0) == 0
        assert profile(299.9) == 0
        assert profile(300) == 20
        assert profile(600) == 50
        assert profile(900) == 0.0
        with pytest.raises(ValueError):
            PiecewiseConstant([0, 100], [1, 2])

    def test_set_toll_discretization(self):
        W = build_merge_world()
        W.ROUTE_CHOICE = "duo_logit"
        profile = PiecewiseConstant(breakpoints=[0, 300, 600, 900], values=[0, 20, 50])
        W.set_toll("link1", profile)
        M = W.compile(backend="jax", toll_interval=300)
        toll_row = np.asarray(M.raw.params.toll[0])
        expected = [profile(k * 300) for k in range(len(toll_row))]
        np.testing.assert_allclose(toll_row, expected)

    def test_toll_interval_independent_of_route_interval(self):
        W = build_merge_world()
        W.ROUTE_CHOICE = "duo_logit"
        M = W.compile(backend="jax", route_update_interval=300, toll_interval=60)
        assert M.raw.config.route_update_interval == 60
        assert M.raw.config.toll_step_size == 12
        assert M.raw.config.n_toll_steps == int(np.ceil(M.raw.config.tsize / 12))
        R = M.run()
        assert float(R.metrics.total_travel_time()) > 0.0

    def test_toll_variable_interval_mismatch(self):
        W = build_merge_world()
        W.ROUTE_CHOICE = "duo_logit"
        M = W.compile(backend="jax", toll_interval=300)
        with pytest.raises(ValueError, match="compile"):
            M.toll_variable(links=["link1"], interval=60)
        toll = M.toll_variable(links=["link1"], interval=300, lower=0.0)
        assert toll.shape == (1, int(M.raw.config.n_toll_steps))


# ================================================================
# Custom variables
# ================================================================

class TestCustomVariable:
    def test_gradient_matches_raw(self):
        # Composite variable: delta with u_link2 = u_link1 + delta.
        W = build_merge_world()
        params, config = world_to_jax(W)

        def raw_loss(delta):
            p = params._replace(u=params.u.at[1].set(params.u[0] + delta))
            return total_travel_time(simulate(p, config), config)

        g_raw = float(jax.grad(raw_loss)(jnp.float32(0.0)))

        M = build_merge_world().compile(backend="jax")
        v = M.variable(
            shape=(1,), initial=0.0, name="delta_u2",
            inject=lambda p, th: p._replace(u=p.u.at[1].set(p.u[0] + th[0])),
        )
        objective = M.objective(lambda R: R.metrics.total_travel_time())
        _, grad = objective.value_and_gradient(wrt=[v])
        assert grad[v].shape == (1,)
        assert float(grad[v][0]) == pytest.approx(g_raw, rel=1e-5)

    def test_equivalent_to_builtin_ref(self):
        # A custom variable replicating the built-in merge_priority ref must give the same gradient.
        M = build_merge_world().compile(backend="jax")
        mp1 = M.parameter("link1", "merge_priority")
        v = M.variable(
            shape=(1,), initial=float(M.raw.params.merge_priority[0]),
            inject=lambda p, th: p._replace(merge_priority=p.merge_priority.at[0].set(th[0])),
        )
        objective = M.objective(lambda R: R.metrics.total_travel_time())
        _, g_builtin = objective.value_and_gradient(wrt=[mp1])
        _, g_custom = objective.value_and_gradient(wrt=[v])
        assert float(g_custom[v][0]) == pytest.approx(float(g_builtin[mp1]), rel=1e-5)

    def test_minimize_with_custom_variable(self):
        # Shared log-factor on all free-flow speeds; larger speeds reduce TTT, so the factor should increase.
        M = build_merge_world().compile(backend="jax")
        v = M.variable(
            shape=(1,), initial=0.0, name="log_u_factor", lower=-0.5, upper=0.5,
            inject=lambda p, th: p._replace(u=p.u * jnp.exp(th[0])),
        )
        problem = M.minimize(variables=[v], objective=lambda R, x: R.metrics.total_travel_time())
        solution = problem.solve(steps=5, learning_rate=0.1)
        assert float(solution.value[v][0]) > 0.0
        assert solution.loss_history[-1] < solution.loss_history[0]

    def test_inject_type_check(self):
        M = build_merge_world().compile(backend="jax")
        v = M.variable(shape=(1,), initial=0.0, inject=lambda p, th: None)
        objective = M.objective(lambda R: R.metrics.total_travel_time())
        with pytest.raises(TypeError, match="Params"):
            objective.value_and_gradient(wrt=[v])

    def test_explain_mentions_custom(self):
        M = build_merge_world().compile(backend="jax")
        v = M.variable(shape=(2,), initial=0.0, name="my_var", inject=lambda p, th: p)
        objective = M.objective(lambda R: R.metrics.total_travel_time())
        text = objective.explain(wrt=[v])
        assert "my_var" in text
        assert "FD consistency not enforced" in text


# ================================================================
# Explain
# ================================================================

class TestExplain:
    def test_explain_contents(self):
        W = build_merge_world()
        M = W.compile(backend="jax")
        u1 = M.parameter("link1", "free_flow_speed")
        q1 = M.parameter(W.DEMANDS[0], "flow")
        objective = M.objective(lambda R: R.metrics.total_travel_time())
        text = objective.explain(wrt=[u1, q1])
        assert "FD parameterization: (u, kappa, tau)" in text
        assert "derived quantities: backward_wave_speed, capacity" in text
        assert "active interval: [0, 1000)" in text
        assert "gradient through discrete shortest-path index: no" in text


# ================================================================
# Parallel-routes gate (reference npz precomputed by the raw core)
# ================================================================

class TestParallelRoutesGate:
    @pytest.fixture(scope="class")
    def ref(self):
        assert os.path.exists(REFERENCE_NPZ), "Run tests/generate_parallel_routes_reference.py first"
        return np.load(REFERENCE_NPZ)

    @pytest.fixture(scope="class")
    def model(self):
        W, _ = create_world()
        return W.compile(backend="jax")

    def test_baseline_matches_reference(self, ref, model):
        R = model.run()
        assert float(R.metrics.total_travel_time()) == pytest.approx(float(ref["ttt0"]), rel=1e-5)
        assert float(R.metrics.completed_trips()) == pytest.approx(float(ref["trips0"]), rel=1e-5)
        np.testing.assert_allclose(np.asarray(R.state.cum_arrival), ref["cum_arrival"], rtol=1e-5, atol=1e-2)
        np.testing.assert_allclose(np.asarray(R.state.cum_departure), ref["cum_departure"], rtol=1e-5, atol=1e-2)

    def test_toll_optimization_matches_reference(self, ref, model):
        M = model
        reg = float(ref["reg_lambda"])
        toll = M.toll_variable(links=["highway23"], initial=0.0, lower=0.0)
        problem = M.minimize(
            variables=[toll],
            objective=lambda R, x: R.metrics.total_travel_time() + reg * jnp.sum(x[toll] ** 2),
        )
        solution = problem.solve(optimizer="adam", steps=int(ref["steps"]), learning_rate=float(ref["lr"]))
        np.testing.assert_allclose(solution.loss_history, ref["loss_history"], rtol=1e-5)
        np.testing.assert_allclose(np.asarray(solution.value[toll]).ravel(), ref["theta_opt"], rtol=1e-4, atol=1e-3)
        assert solution.final_loss < float(ref["ttt0"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
