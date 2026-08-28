"""
Object-oriented facade over the JAX differentiable core (unsim_diff).

This module lets users run differentiable simulations and compute gradients using traffic-model vocabulary (Link, Node, Demand objects and names) instead of raw JAX structures (Params, integer indices, _replace, jax.grad).

Usage
-----
>>> from unsim import World
>>> W = World(...); W.addNode(...); link1 = W.addLink(...); W.adddemand(...)
>>> M = W.compile(backend="jax")
>>> R = M.run()
>>> print(R.metrics.total_travel_time())
>>> u1 = M.parameter(link1, "free_flow_speed")
>>> objective = M.objective(lambda R: R.metrics.total_travel_time())
>>> value, grad = objective.value_and_gradient(wrt=[u1])
>>> print(grad[u1])
"""

import copy
from dataclasses import dataclass, field as dc_field
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp

from .unsim import Node, Link, Demand
from . import unsim_diff as core


# ================================================================
# Units and FD parameterization tables
# ================================================================

_UNITS = {
    "free_flow_speed": "m/s",
    "jam_density": "veh/m",
    "capacity": "veh/s",
    "backward_wave_speed": "m/s",
    "merge_priority": "-",
    "capacity_out": "veh/s",
    "capacity_in": "veh/s",
    "toll": "s",
    "flow_capacity": "veh/s",
    "absorption_ratio": "-",
    "flow": "veh/s",
}

# Independent FD fields per parameterization; all other FD fields are derived.
_FD_INDEPENDENT = {
    "u_kappa_tau": ("free_flow_speed", "jam_density"),
    "u_w_capacity": ("free_flow_speed", "backward_wave_speed", "capacity"),
    "u_w_tau": ("free_flow_speed", "backward_wave_speed"),
}

_FD_FIELDS = ("free_flow_speed", "jam_density", "capacity", "backward_wave_speed")

_LINK_PLAIN_FIELDS = ("merge_priority", "capacity_out", "capacity_in")

_NODE_FIELDS = ("flow_capacity", "absorption_ratio")

_FD_HUMAN_NAME = {
    "u_kappa_tau": "(u, kappa, tau)",
    "u_w_capacity": "(u, w, capacity)",
    "u_w_tau": "(u, w, tau)",
}

# Derived FD quantities per parameterization, for explain() output.
_FD_DERIVED = {
    "u_kappa_tau": ("backward_wave_speed", "capacity"),
    "u_w_capacity": ("jam_density",),
    "u_w_tau": ("jam_density", "capacity"),
}


def _fd_triplet(link, fd_param, varying_field, theta):
    """Compute (u, kappa, q_star) for one link with one field replaced by theta.

    Derived FD quantities are recomputed as pure functions of theta so that the gradient semantics match the chosen parameterization.
    Fixed quantities are taken from the authoring-layer Link object.

    Parameters
    ----------
    link : Link
        Authoring-layer link holding base FD values and tau.
    fd_param : str
        FD parameterization name ("u_kappa_tau", "u_w_capacity", "u_w_tau").
    varying_field : str
        The independent field replaced by theta.
    theta : jnp scalar
        Value of the varying field.

    Returns
    -------
    (u, kappa, q_star) : tuple of jnp scalars
    """
    tau = link.tau
    if fd_param == "u_kappa_tau":
        u = theta if varying_field == "free_flow_speed" else link.u
        kappa = theta if varying_field == "jam_density" else link.kappa
        w = 1.0 / (tau * kappa)
        q_star = u * w * kappa / (u + w)
        return u, kappa, q_star
    if fd_param == "u_w_capacity":
        u = theta if varying_field == "free_flow_speed" else link.u
        w = theta if varying_field == "backward_wave_speed" else link.w
        q_star = theta if varying_field == "capacity" else link.q_star
        kappa = q_star * (u + w) / (u * w)
        return u, kappa, q_star
    if fd_param == "u_w_tau":
        u = theta if varying_field == "free_flow_speed" else link.u
        w = theta if varying_field == "backward_wave_speed" else link.w
        kappa = 1.0 / (tau * w)
        q_star = u * w * kappa / (u + w)
        return u, kappa, q_star
    raise ValueError(f"Unknown FD parameterization: {fd_param}")


# ================================================================
# Time profiles
# ================================================================

class PiecewiseConstant:
    """Piecewise-constant time profile.

    Callable as ``profile(t)``, so it can be used anywhere a time function is expected, e.g. ``World.set_toll``.
    Returns 0 outside the covered range.

    Parameters
    ----------
    breakpoints : list of float
        Interval boundaries (s), length n+1 for n values.
    values : list of float
        Value on each interval [breakpoints[i], breakpoints[i+1]).
    """

    def __init__(self, breakpoints, values):
        if len(breakpoints) != len(values) + 1:
            raise ValueError("len(breakpoints) must be len(values) + 1")
        self.breakpoints = list(breakpoints)
        self.values = list(values)

    def __call__(self, t):
        """Evaluate the profile at time t (s)."""
        for i in range(len(self.values)):
            if self.breakpoints[i] <= t < self.breakpoints[i + 1]:
                return self.values[i]
        return 0.0

    def __repr__(self):
        return f"<PiecewiseConstant {self.breakpoints} -> {self.values}>"


# ================================================================
# References
# ================================================================

@dataclass(frozen=True)
class ParameterRef:
    """Semantic reference to one differentiable parameter.

    Equality and hashing use (kind, field, index) only, so two refs to the same parameter compare equal.

    Attributes
    ----------
    kind : str
        Entity kind: "link" or "node".
    field : str
        Physical quantity name (e.g. "free_flow_speed").
    index : int
        Entity index in the compiled arrays.
    name : str
        Entity name (for display).
    unit : str
        Physical unit of the quantity.
    fd_parameterization : str or None
        FD parameterization of the link, for FD fields only.
    shape : tuple
        Shape of the parameter value; () for scalars.
    """
    kind: str
    field: str
    index: int
    name: str = dc_field(compare=False, default="")
    unit: str = dc_field(compare=False, default="")
    fd_parameterization: str = dc_field(compare=False, default=None)
    shape: tuple = dc_field(compare=False, default=())

    @property
    def size(self):
        """Number of scalar entries in this parameter."""
        return int(np.prod(self.shape)) if self.shape else 1

    def __repr__(self):
        return f"<ParameterRef {self.name}.{self.field}>"


@dataclass(frozen=True)
class PathRef:
    """Semantic reference to an ordered path of links.

    Attributes
    ----------
    link_ids : tuple of int
        Ordered link indices from origin to destination.
    link_names : tuple of str
        Corresponding link names.
    """
    link_ids: tuple
    link_names: tuple = dc_field(compare=False, default=())

    def __repr__(self):
        return f"<PathRef {' -> '.join(self.link_names)}>"


@dataclass(frozen=True)
class TollVariable:
    """Optimization variable block: tolls on selected links over all toll steps.

    Created via ``DifferentiableWorld.toll_variable()``.

    Attributes
    ----------
    link_ids : tuple of int
        Selected link indices.
    n_steps : int
        Number of toll discretization steps.
    link_names : tuple of str
        Selected link names.
    initial : float
        Initial toll value (s).
    lower : float or None
        Lower bound (s), applied by projection during optimization.
    upper : float or None
        Upper bound (s).
    """
    link_ids: tuple
    n_steps: int
    link_names: tuple = dc_field(compare=False, default=())
    initial: float = dc_field(compare=False, default=0.0)
    lower: float = dc_field(compare=False, default=None)
    upper: float = dc_field(compare=False, default=None)

    @property
    def shape(self):
        """Shape of the variable block."""
        return (len(self.link_ids), self.n_steps)

    @property
    def size(self):
        """Number of scalar entries."""
        return len(self.link_ids) * self.n_steps

    def __repr__(self):
        return f"<TollVariable {len(self.link_ids)} links x {self.n_steps} steps>"


class CustomVariable:
    """User-defined variable with a custom injection into Params.

    Lets the user define composite variables such as differences, shared factors, or transformed parameters.
    The inject function maps (params, theta) to a new Params and must be a pure JAX-differentiable function; FD consistency across derived quantities is the user's responsibility here.
    Instances compare by identity, so use the same object when building and when reading gradients.

    Created via ``DifferentiableWorld.variable()``.

    Parameters
    ----------
    shape : tuple of int
        Shape of the variable value.
    initial : float or array_like
        Initial value, broadcast to ``shape``.
    inject : callable
        Function ``(params, theta) -> Params`` where ``theta`` has shape ``shape``.
    name : str, optional
        Display name.
    lower : float or None, optional
        Lower bound, applied by projection during optimization.
    upper : float or None, optional
        Upper bound.
    """

    def __init__(self, shape, initial, inject, name="custom", lower=None, upper=None):
        self.shape = tuple(int(n) for n in shape)
        self.initial = initial
        self.inject_fn = inject
        self.name = name
        self.lower = lower
        self.upper = upper

    @property
    def size(self):
        """Number of scalar entries."""
        return int(np.prod(self.shape)) if self.shape else 1

    def __repr__(self):
        return f"<CustomVariable '{self.name}' shape={self.shape}>"


# ================================================================
# VariableSet
# ================================================================

class VariableSet:
    """Maps selected ParameterRefs to a flat theta vector and back into Params.

    Centralizes flatten/unflatten, scatter into full-size arrays, and FD-consistent recomputation of derived quantities.

    Parameters
    ----------
    model : DifferentiableWorld
        Compiled model providing base params and entity lookup.
    refs : list of ParameterRef
        Selected variables, in order.
    """

    def __init__(self, model, refs):
        self._model = model
        self._refs = list(refs)
        self._offsets = []
        offset = 0
        for ref in self._refs:
            self._offsets.append(offset)
            offset += ref.size
        self._total_size = offset

    @property
    def size(self):
        """Total number of scalar variables."""
        return self._total_size

    @property
    def refs(self):
        """Selected ParameterRefs in order."""
        return list(self._refs)

    def initial_theta(self):
        """Build the initial flat theta vector from base parameter values.

        Returns
        -------
        jnp.ndarray, (size,)
        """
        pieces = []
        params = self._model._params
        for ref in self._refs:
            pieces.append(jnp.ravel(jnp.asarray(self._base_value(ref, params), dtype=jnp.float32)))
        return jnp.concatenate(pieces) if pieces else jnp.zeros(0, dtype=jnp.float32)

    def _base_value(self, ref, params):
        """Return the base value of one ref from params or the authoring layer."""
        if isinstance(ref, TollVariable):
            return jnp.full(ref.shape, ref.initial, dtype=jnp.float32)
        if isinstance(ref, CustomVariable):
            return jnp.broadcast_to(jnp.asarray(ref.initial, dtype=jnp.float32), ref.shape)
        i = ref.index
        if ref.kind == "demand":
            if ref.field == "flow":
                return self._model._snapshot_demand(i).flow
        if ref.kind == "link":
            if ref.field == "free_flow_speed":
                return params.u[i]
            if ref.field == "jam_density":
                return params.kappa[i]
            if ref.field == "capacity":
                return params.q_star[i]
            if ref.field == "backward_wave_speed":
                # w is not stored in Params; take it from the authoring-layer link.
                return self._model._snapshot_link(i).w
            if ref.field == "merge_priority":
                return params.merge_priority[i]
            if ref.field == "capacity_out":
                return params.capacity_out[i]
            if ref.field == "capacity_in":
                return params.capacity_in[i]
            if ref.field == "toll":
                return params.toll[i]
        if ref.kind == "node":
            if ref.field == "flow_capacity":
                return params.flow_capacity[i]
            if ref.field == "absorption_ratio":
                return params.absorption_ratio[i]
        raise ValueError(f"Unsupported parameter: {ref}")

    def inject(self, theta, params):
        """Build a full Params with selected variables replaced by theta.

        For FD fields, derived quantities are recomputed as functions of theta so that the differentiation semantics follow each link's FD parameterization.

        Parameters
        ----------
        theta : jnp.ndarray, (size,)
            Flat variable vector (may be a JAX tracer).
        params : Params
            Base parameters to update.

        Returns
        -------
        Params
        """
        for ref, offset in zip(self._refs, self._offsets):
            if isinstance(ref, CustomVariable):
                block = theta[offset:offset + ref.size].reshape(ref.shape)
                new_params = ref.inject_fn(params, block)
                if type(new_params) is not type(params):
                    raise TypeError(f"inject of {ref!r} must return a Params, got {type(new_params).__name__}")
                params = new_params
                continue
            sl = theta[offset] if ref.size == 1 else theta[offset:offset + ref.size].reshape(ref.shape)
            if isinstance(ref, TollVariable):
                ids = jnp.array(ref.link_ids, dtype=jnp.int32)
                params = params._replace(toll=params.toll.at[ids].set(sl))
                continue
            i = ref.index
            if ref.kind == "demand" and ref.field == "flow":
                d = self._model._snapshot_demand(i)
                oid, did, i0, i1 = self._model._demand_slots(d)
                delta = sl - d.flow
                params = params._replace(
                    demand_rate=params.demand_rate.at[oid, i0:i1].add(delta),
                    od_demand_rate=params.od_demand_rate.at[oid, did, i0:i1].add(delta),
                )
            elif ref.kind == "link" and ref.field in _FD_FIELDS:
                link = self._model._snapshot_link(i)
                u, kappa, q_star = _fd_triplet(link, ref.fd_parameterization, ref.field, sl)
                params = params._replace(
                    u=params.u.at[i].set(u),
                    kappa=params.kappa.at[i].set(kappa),
                    q_star=params.q_star.at[i].set(q_star),
                )
            elif ref.kind == "link" and ref.field == "merge_priority":
                params = params._replace(merge_priority=params.merge_priority.at[i].set(sl))
            elif ref.kind == "link" and ref.field == "capacity_out":
                params = params._replace(capacity_out=params.capacity_out.at[i].set(sl))
            elif ref.kind == "link" and ref.field == "capacity_in":
                params = params._replace(capacity_in=params.capacity_in.at[i].set(sl))
            elif ref.kind == "link" and ref.field == "toll":
                params = params._replace(toll=params.toll.at[i].set(sl))
            elif ref.kind == "node" and ref.field == "flow_capacity":
                params = params._replace(flow_capacity=params.flow_capacity.at[i].set(sl))
            elif ref.kind == "node" and ref.field == "absorption_ratio":
                params = params._replace(absorption_ratio=params.absorption_ratio.at[i].set(sl))
            else:
                raise ValueError(f"Unsupported parameter: {ref}")
        return params

    def unflatten(self, theta):
        """Split a flat vector into a dict keyed by ParameterRef.

        Parameters
        ----------
        theta : jnp.ndarray, (size,)

        Returns
        -------
        dict of {ParameterRef: jnp scalar or jnp.ndarray}
        """
        out = {}
        for ref, offset in zip(self._refs, self._offsets):
            if getattr(ref, "shape", ()):
                out[ref] = theta[offset:offset + ref.size].reshape(ref.shape)
            else:
                out[ref] = theta[offset]
        return out


class Gradients:
    """Read-only mapping from ParameterRef to gradient arrays."""

    def __init__(self, mapping):
        self._m = dict(mapping)

    def __getitem__(self, ref):
        return self._m[ref]

    def __iter__(self):
        return iter(self._m)

    def __len__(self):
        return len(self._m)

    def items(self):
        """Iterate over (ref, gradient) pairs."""
        return self._m.items()

    def keys(self):
        """Iterate over refs."""
        return self._m.keys()

    def values(self):
        """Iterate over gradient arrays."""
        return self._m.values()

    def __repr__(self):
        entries = ", ".join(f"{r.name}.{r.field}" for r in self._m)
        return f"<Gradients wrt [{entries}]>"


# ================================================================
# Result views
# ================================================================

class Metrics:
    """Differentiable aggregate traffic metrics of a simulation result.

    All methods return JAX scalars and are pure functions usable inside jax.jit and jax.grad.
    """

    def __init__(self, result):
        self._r = result

    def total_travel_time(self):
        """Total travel time (s)."""
        return core.total_travel_time(self._r.state, self._r.config)

    def completed_trips(self):
        """Total completed trips (veh)."""
        return core.trip_completed(self._r.state, self._r.config)

    def average_travel_time(self):
        """Average travel time per completed trip (s)."""
        return core.average_travel_time(self._r.state, self._r.config)


class LinkView:
    """Differentiable per-link queries on a simulation result."""

    def __init__(self, result, link_id, link_name):
        self._r = result
        self._i = link_id
        self._name = link_name

    def vehicle_count(self, t):
        """Number of vehicles on the link at time t (veh). Differentiable.

        Parameters
        ----------
        t : float
            Time (s).
        """
        dt = self._r.config.deltat
        n_arr = core.interp_1d(self._r.state.cum_arrival[self._i], t / dt)
        n_dep = core.interp_1d(self._r.state.cum_departure[self._i], t / dt)
        return jnp.maximum(n_arr - n_dep, 0.0)

    def density(self, t):
        """Average density on the link at time t (veh/m). Differentiable.

        Parameters
        ----------
        t : float
            Time (s).
        """
        return self.vehicle_count(t) / self._r.config.link_lengths[self._i]

    def cum_arrival(self):
        """Cumulative arrival curve (veh), shape (tsize+1,)."""
        return self._r.state.cum_arrival[self._i]

    def cum_departure(self):
        """Cumulative departure curve (veh), shape (tsize+1,)."""
        return self._r.state.cum_departure[self._i]

    def __repr__(self):
        return f"<LinkView '{self._name}'>"


class NodeView:
    """Per-node queries on a simulation result."""

    def __init__(self, result, node_id, node_name):
        self._r = result
        self._i = node_id
        self._name = node_name

    def queue(self, t):
        """Vertical queue length at an origin node at time t (veh). Differentiable in value.

        The timestep selection uses a floor index and is not differentiable with respect to t.

        Parameters
        ----------
        t : float
            Time (s).
        """
        config = self._r.config
        t_idx = int(np.clip(int(t / config.deltat), 0, config.tsize - 1))
        return self._r.state.demand_queue_history[self._i, t_idx]

    def __repr__(self):
        return f"<NodeView '{self._name}'>"


class PathView:
    """Differentiable path-level queries on a simulation result."""

    def __init__(self, result, path_ref):
        self._r = result
        self._path = path_ref

    def travel_time(self, departure_time):
        """Travel time of a virtual vehicle along the path (s). Differentiable.

        Parameters
        ----------
        departure_time : float
            Departure time from the path's first link (s).
        """
        return core.travel_time(list(self._path.link_ids), departure_time, self._r.state, self._r.params, self._r.config)


class ODView:
    """Differentiable OD-level queries on a simulation result."""

    def __init__(self, result, orig_id, dest_id):
        self._r = result
        self._o = orig_id
        self._d = dest_id

    def travel_time(self, departure_time, method="soft", temperature=None):
        """OD travel time (s).

        Parameters
        ----------
        departure_time : float
            Departure time (s).
        method : str, optional
            "soft" for fully differentiable soft route choice (default).
            "auto" for shortest-path chaining (route choice itself is not differentiated).
            "logsum" for the expected perceived cost under logit route choice.
        temperature : float or None, optional
            Logit temperature (s) for "soft" and "logsum". None uses the compiled default.
        """
        r = self._r
        if method == "soft":
            return core.travel_time_soft(self._o, self._d, departure_time, r.state, r.params, r.config, temperature=temperature)
        if method == "auto":
            return core.travel_time_auto(self._o, self._d, departure_time, r.state, r.params, r.config)
        if method == "logsum":
            return core.logsum_travel_time(self._o, self._d, departure_time, r.state, r.params, r.config, temperature=temperature)
        raise ValueError(f"Unknown method: {method}")


class SimResult:
    """Simulation result facade wrapping the raw SimState.

    Provides differentiable queries (metrics, link, node, path, od) and host-side analysis (analyzer).

    Attributes
    ----------
    state : SimState
        Raw JAX simulation state.
    params : Params
        Parameters used for this run.
    config : NetworkConfig
        Static network configuration.
    """

    def __init__(self, state, params, model):
        self.state = state
        self.params = params
        self._model = model
        self.config = model._config
        self.metrics = Metrics(self)

    def link(self, link):
        """Get a LinkView for a link (object, name, or index)."""
        i, name = self._model._resolve_link(link)
        return LinkView(self, i, name)

    def node(self, node):
        """Get a NodeView for a node (object, name, or index)."""
        i, name = self._model._resolve_node(node)
        return NodeView(self, i, name)

    def path(self, path):
        """Get a PathView for a PathRef or a list of links."""
        if not isinstance(path, PathRef):
            path = self._model.path(path)
        return PathView(self, path)

    def od(self, orig, dest):
        """Get an ODView for an origin and destination node."""
        oi, _ = self._model._resolve_node(orig)
        di, _ = self._model._resolve_node(dest)
        return ODView(self, oi, di)

    @property
    def analyzer(self):
        """Host-side Analyzer with this result written back into the snapshot World.

        Each access rewrites the snapshot World with this result's arrays, so the returned Analyzer reflects this result at access time.
        The write-back covers aggregate cumulative curves, origin queues, and absorbed counts; per-destination curves are not restored.
        """
        W = self._model._world
        ca = np.asarray(self.state.cum_arrival)
        cd = np.asarray(self.state.cum_departure)
        for i, link in enumerate(W.LINKS):
            link.cum_arrival = ca[i].tolist()
            link.cum_departure = cd[i].tolist()
        dqh = np.asarray(self.state.demand_queue_history)
        absorbed = np.asarray(self.state.absorbed_count)
        dq = np.asarray(self.state.demand_queue)
        for j, node in enumerate(W.NODES):
            node.demand_queue_history = dqh[j].tolist()
            node.absorbed_count = float(absorbed[j])
            node.demand_queue = float(dq[j])
        W.T = W.TSIZE
        return W.analyzer


# ================================================================
# Objective
# ================================================================

class Objective:
    """Objective function over simulation results with gradient support.

    Parameters
    ----------
    model : DifferentiableWorld
        Compiled model.
    fn : callable
        Function mapping a SimResult to a JAX scalar, e.g. ``lambda R: R.metrics.total_travel_time()``.
    checkpoint_every : int or None, optional
        Gradient checkpointing segment length passed to the core simulator.
    """

    def __init__(self, model, fn, checkpoint_every=None):
        self._model = model
        self._fn = fn
        self._checkpoint_every = checkpoint_every

    def value(self):
        """Evaluate the objective at the base parameters.

        Returns
        -------
        jnp scalar
        """
        return self._fn(self._model.run(checkpoint_every=self._checkpoint_every))

    def _make_loss(self, varset):
        """Build the scalar loss function theta -> objective value."""
        model = self._model
        fn = self._fn
        checkpoint_every = self._checkpoint_every

        def loss(theta):
            p = varset.inject(theta, model._params)
            state = model._simulate(p, model._config, differentiable=True, checkpoint_every=checkpoint_every)
            return fn(SimResult(state, p, model))

        return loss

    def value_and_gradient(self, wrt, jit=False):
        """Evaluate the objective and its gradient with respect to selected parameters.

        Parameters
        ----------
        wrt : list of ParameterRef
            Parameters to differentiate with respect to.
        jit : bool, optional
            If True, jit-compile the value-and-gradient function.

        Returns
        -------
        value : jnp scalar
        gradient : Gradients
            Mapping from each ref to its gradient (same shape as the parameter).
        """
        varset = VariableSet(self._model, wrt)
        f = jax.value_and_grad(self._make_loss(varset))
        if jit:
            f = jax.jit(f)
        value, g = f(varset.initial_theta())
        return value, Gradients(varset.unflatten(g))

    def gradient(self, wrt, jit=False):
        """Evaluate the gradient only. See ``value_and_gradient``."""
        return self.value_and_gradient(wrt, jit=jit)[1]

    def explain(self, wrt=None):
        """Describe what the gradient computation differentiates and what it holds fixed.

        Parameters
        ----------
        wrt : list of ParameterRef or None, optional
            Variables to describe. None describes only the objective and model.

        Returns
        -------
        str
            Human-readable description.
        """
        model = self._model
        config = model._config
        lines = []
        lines.append("Objective:")
        fn_name = getattr(self._fn, "__name__", None)
        lines.append(f"    {fn_name if fn_name and fn_name != '<lambda>' else 'custom function'}")
        lines.append("")
        lines.append("Variables:")
        for ref in (wrt or []):
            if isinstance(ref, TollVariable):
                lines.append(f"    toll on {len(ref.link_ids)} links [s]")
                lines.append(f"        toll steps: {ref.n_steps} x {float(config.toll_step_size) * config.deltat:.0f} s")
                continue
            if isinstance(ref, CustomVariable):
                lines.append(f"    {ref.name} (custom variable, shape {ref.shape})")
                lines.append("        injection: user-defined; FD consistency not enforced")
                continue
            lines.append(f"    {ref.name}.{ref.field} [{ref.unit}]")
            if ref.kind == "link" and ref.fd_parameterization is not None:
                lines.append(f"        FD parameterization: {_FD_HUMAN_NAME[ref.fd_parameterization]}")
                lines.append(f"        derived quantities: {', '.join(_FD_DERIVED[ref.fd_parameterization])}")
            if ref.kind == "demand":
                d = model._snapshot_demand(ref.index)
                lines.append(f"        active interval: [{d.t_start}, {d.t_end})")
        lines.append("")
        lines.append("Route choice:")
        lines.append(f"    {model.route_choice}")
        if config.use_logit:
            lines.append(f"    temperature: {config.logit_temperature:.0f} s")
        lines.append("    gradient through discrete shortest-path index: no")
        lines.append("")
        lines.append("Static quantities:")
        lines.append("    topology, time grid, link length")
        return "\n".join(lines)


# ================================================================
# Optimization
# ================================================================

class Solution:
    """Result of ``Problem.solve``.

    Attributes
    ----------
    value : Gradients
        Mapping from each variable to its optimized value array.
    theta : jnp.ndarray
        Final flat variable vector.
    loss_history : np.ndarray, (steps,)
        Loss at the start of each optimizer step.
    """

    def __init__(self, value_map, theta, loss_history):
        self.value = value_map
        self.theta = theta
        self.loss_history = loss_history

    @property
    def final_loss(self):
        """Loss at the start of the last optimizer step."""
        return float(self.loss_history[-1])

    def __repr__(self):
        return f"<Solution steps={len(self.loss_history)} final_loss={self.final_loss:.6g}>"


class Problem:
    """Optimization problem over selected variables of a compiled model.

    Created via ``DifferentiableWorld.minimize()``.

    Parameters
    ----------
    model : DifferentiableWorld
    variables : list of TollVariable, CustomVariable, or ParameterRef
        Decision variables.
    objective : callable
        Function ``(result, x) -> JAX scalar`` where ``x`` maps each variable to its current value array.
    checkpoint_every : int or None, optional
        Gradient checkpointing segment length.
    """

    def __init__(self, model, variables, objective, checkpoint_every=None):
        self._model = model
        self._variables = list(variables)
        self._objective = objective
        self._checkpoint_every = checkpoint_every
        self._varset = VariableSet(model, self._variables)

    def _make_loss(self):
        """Build the scalar loss function theta -> objective value."""
        model = self._model
        varset = self._varset
        objective = self._objective
        checkpoint_every = self._checkpoint_every

        def loss(theta):
            p = varset.inject(theta, model._params)
            state = model._simulate(p, model._config, differentiable=True, checkpoint_every=checkpoint_every)
            x = Gradients(varset.unflatten(theta))
            return objective(SimResult(state, p, model), x)

        return loss

    def _bound_vectors(self):
        """Build flat lower/upper bound vectors from per-variable bounds.

        Returns
        -------
        (lower, upper) : tuple of np.ndarray or (None, None)
            None if no variable declares a bound.
        """
        lower = np.full(self._varset.size, -np.inf, dtype=np.float32)
        upper = np.full(self._varset.size, np.inf, dtype=np.float32)
        has_bound = False
        offset = 0
        for ref in self._variables:
            size = ref.size
            lo = getattr(ref, "lower", None)
            hi = getattr(ref, "upper", None)
            if lo is not None:
                lower[offset:offset + size] = lo
                has_bound = True
            if hi is not None:
                upper[offset:offset + size] = hi
                has_bound = True
            offset += size
        return (jnp.array(lower), jnp.array(upper)) if has_bound else (None, None)

    def solve(self, optimizer="adam", steps=200, learning_rate=1.0, jit=True,
              b1=0.9, b2=0.999, eps=1e-8, verbose=False):
        """Minimize the objective with a first-order optimizer.

        Box bounds declared on variables are enforced by projection after each update.

        Parameters
        ----------
        optimizer : str, optional
            Only "adam" is supported.
        steps : int, optional
            Number of optimizer steps.
        learning_rate : float, optional
            Adam learning rate.
        jit : bool, optional
            If True (default), jit-compile the value-and-gradient function.
        b1, b2, eps : float, optional
            Adam hyperparameters.
        verbose : bool, optional
            If True, print the loss every 10 steps.

        Returns
        -------
        Solution
        """
        if optimizer != "adam":
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        vgrad = jax.value_and_grad(self._make_loss())
        if jit:
            vgrad = jax.jit(vgrad)

        theta = self._varset.initial_theta()
        lower, upper = self._bound_vectors()
        m = jnp.zeros_like(theta)
        v = jnp.zeros_like(theta)
        history = np.zeros(steps)
        for step in range(steps):
            value, g = vgrad(theta)
            history[step] = float(value)
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * g ** 2
            m_hat = m / (1.0 - b1 ** (step + 1))
            v_hat = v / (1.0 - b2 ** (step + 1))
            theta = theta - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
            if lower is not None:
                theta = jnp.clip(theta, lower, upper)
            if verbose and step % 10 == 0:
                print(f"  step {step}: loss={history[step]:.6g}")
        return Solution(Gradients(self._varset.unflatten(theta)), theta, history)


# ================================================================
# DifferentiableWorld
# ================================================================

class DifferentiableWorld:
    """Immutable compiled snapshot of a World for differentiable simulation.

    Create via ``World.compile(backend="jax")``.
    Holds the JAX parameter and configuration arrays plus the semantic index from Link/Node objects and names to integer indices.

    Parameters
    ----------
    W : World
        Source world. It is finalized if not already.
    backend : str, optional
        Only "jax" is supported.
    route_update_interval : float or None, optional
        DUO route update interval (s). None uses the built-in default (300 s).
    toll_interval : float or None, optional
        Toll discretization interval (s), independent of the route update interval. None couples it to the route update interval (current default behavior).
    """

    def __init__(self, W, backend="jax", route_update_interval=None, toll_interval=None):
        if backend != "jax":
            raise ValueError(f"Unsupported backend: {backend}")
        W.finalize_scenario()
        params, config = core.world_to_jax(W)
        if route_update_interval is not None or toll_interval is not None:
            params, config = self._apply_intervals(W, params, config, route_update_interval, toll_interval)
        self._params = params
        self._config = config
        self.route_choice = W.ROUTE_CHOICE
        # Snapshot the authoring-layer world so later user edits do not leak into this model.
        self._world = copy.deepcopy(W)
        self._link_index = {link.name: i for i, link in enumerate(self._world.LINKS)}
        self._node_index = {node.name: i for i, node in enumerate(self._world.NODES)}
        # Destination ordering must match world_to_jax (sorted destination names).
        dests = sorted(set(d for _, d, _, _, _ in self._world.demand_info))
        self._dest_index = {name: i for i, name in enumerate(dests)}
        if self.route_choice in ("duo", "duo_multipoint", "duo_logit"):
            self._simulate = core.simulate_duo
        elif self.route_choice == "aon":
            self._simulate = core.simulate_aon
        else:
            self._simulate = core.simulate

    @staticmethod
    def _apply_intervals(W, params, config, route_sec, toll_sec):
        """Rediscretize route updates and/or tolls at user-specified intervals.

        The toll array is rebuilt from each link's congestion_pricing function at the new cadence.
        """
        dt = config.deltat
        k_route = max(1, int(route_sec / dt)) if route_sec is not None else config.route_update_interval
        k_toll = max(1, int(toll_sec / dt)) if toll_sec is not None else k_route
        n_toll = max(1, int(np.ceil(config.tsize / k_toll)))
        toll_arr = np.zeros((config.n_links, n_toll), dtype=np.float32)
        for i, link in enumerate(W.LINKS):
            if link.congestion_pricing is not None:
                for j in range(n_toll):
                    toll_arr[i, j] = link.get_toll(j * k_toll * dt)
        config = config._replace(route_update_interval=k_route, n_toll_steps=n_toll, toll_step_size=k_toll)
        params = params._replace(toll=jnp.array(toll_arr))
        return params, config

    # ------------------------------------------------------------
    # Entity resolution
    # ------------------------------------------------------------

    def _resolve_link(self, link):
        """Resolve a Link object, name, or index to (index, name)."""
        if isinstance(link, Link):
            link = link.name
        if isinstance(link, str):
            if link not in self._link_index:
                raise KeyError(f"Unknown link: {link}")
            return self._link_index[link], link
        i = int(link)
        return i, self._world.LINKS[i].name

    def _resolve_node(self, node):
        """Resolve a Node object, name, or index to (index, name)."""
        if isinstance(node, Node):
            node = node.name
        if isinstance(node, str):
            if node not in self._node_index:
                raise KeyError(f"Unknown node: {node}")
            return self._node_index[node], node
        i = int(node)
        return i, self._world.NODES[i].name

    def _snapshot_link(self, i):
        """Get the snapshot Link object at index i."""
        return self._world.LINKS[i]

    def _snapshot_demand(self, i):
        """Get the snapshot Demand object at index i."""
        return self._world.DEMANDS[i]

    def _demand_slots(self, d):
        """Compute the array slots of one demand declaration.

        Returns
        -------
        (origin_node_id, dest_index, i_start, i_end) : tuple of int
            Indices into demand_rate / od_demand_rate; the time discretization matches world_to_jax.
        """
        dt = self._config.deltat
        tsize = self._config.tsize
        oid = self._node_index[d.orig]
        did = self._dest_index[d.dest]
        i_start = max(int(np.ceil(d.t_start / dt - 1e-10)), 0)
        i_end = min(int(np.ceil(d.t_end / dt - 1e-10)), tsize)
        return oid, did, i_start, i_end

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def run(self, differentiable=True, checkpoint_every=None):
        """Run the simulation with the compiled parameters.

        Parameters
        ----------
        differentiable : bool, optional
            If True (default), use the AD-compatible path.
            If False, use the faster forward-only path (not compatible with reverse-mode AD).
        checkpoint_every : int or None, optional
            Gradient checkpointing segment length (see core simulators).

        Returns
        -------
        SimResult
        """
        state = self._simulate(self._params, self._config, differentiable=differentiable, checkpoint_every=checkpoint_every)
        return SimResult(state, self._params, self)

    def parameter(self, entity, field):
        """Get a ParameterRef for an entity's physical quantity.

        Parameters
        ----------
        entity : Link, Node, or Demand (or a link/node name)
            Target entity.
        field : str
            Quantity name, e.g. "free_flow_speed", "capacity", "merge_priority", "toll", "flow_capacity", "flow".

        Returns
        -------
        ParameterRef

        Raises
        ------
        ValueError
            If the quantity is derived under the entity's FD parameterization, or the field is unknown.
        """
        if isinstance(entity, Demand):
            if field != "flow":
                raise ValueError(f"Unknown demand parameter: {field}")
            d = self._snapshot_demand(entity.id)
            name = f"{d.orig}->{d.dest}[{d.t_start},{d.t_end})"
            return ParameterRef(kind="demand", field="flow", index=d.id, name=name, unit=_UNITS["flow"])

        if isinstance(entity, Node) or (isinstance(entity, str) and entity in self._node_index and field in _NODE_FIELDS):
            i, name = self._resolve_node(entity)
            if field not in _NODE_FIELDS:
                raise ValueError(f"Unknown node parameter: {field}")
            return ParameterRef(kind="node", field=field, index=i, name=name, unit=_UNITS[field])

        i, name = self._resolve_link(entity)
        link = self._snapshot_link(i)
        if field in _FD_FIELDS:
            fd_param = link.fd_parameterization
            if field not in _FD_INDEPENDENT[fd_param]:
                raise ValueError(
                    f"{field} is derived from {_FD_HUMAN_NAME[fd_param]}. "
                    f"Choose another FD parameterization to vary {field} independently."
                )
            return ParameterRef(kind="link", field=field, index=i, name=name, unit=_UNITS[field], fd_parameterization=fd_param)
        if field in _LINK_PLAIN_FIELDS:
            return ParameterRef(kind="link", field=field, index=i, name=name, unit=_UNITS[field])
        if field == "toll":
            return ParameterRef(kind="link", field="toll", index=i, name=name, unit=_UNITS["toll"], shape=(int(self._config.n_toll_steps),))
        raise ValueError(f"Unknown link parameter: {field}")

    def parameter_fields(self, entity):
        """List the field names accepted by ``parameter()`` for an entity.

        For a link, only the FD quantities that are independent under the link's FD parameterization are included.

        Parameters
        ----------
        entity : Link, Node, or Demand (or a link/node name)
            Target entity.

        Returns
        -------
        list of str
        """
        if isinstance(entity, Demand):
            return ["flow"]
        if isinstance(entity, Node) or (isinstance(entity, str) and entity not in self._link_index):
            self._resolve_node(entity)
            return list(_NODE_FIELDS)
        i, _ = self._resolve_link(entity)
        fd_param = self._snapshot_link(i).fd_parameterization
        return list(_FD_INDEPENDENT[fd_param]) + list(_LINK_PLAIN_FIELDS) + ["toll"]

    def path(self, links):
        """Build a PathRef from an ordered list of links (objects or names).

        Parameters
        ----------
        links : list of Link or str
            Ordered links from origin to destination.

        Returns
        -------
        PathRef
        """
        ids = []
        names = []
        for link in links:
            i, name = self._resolve_link(link)
            ids.append(i)
            names.append(name)
        return PathRef(link_ids=tuple(ids), link_names=tuple(names))

    def toll_variable(self, links, interval=None, initial=0.0, lower=None, upper=None):
        """Build a toll optimization variable over selected links and all toll steps.

        Parameters
        ----------
        links : list of Link or str
            Links to toll.
        interval : float or None, optional
            Expected toll discretization interval (s). Must match the compiled toll interval (set via ``compile(toll_interval=...)``); a mismatch raises an error.
        initial : float, optional
            Initial toll value (s). Default 0.
        lower : float or None, optional
            Lower bound (s), applied by projection during optimization.
        upper : float or None, optional
            Upper bound (s).

        Returns
        -------
        TollVariable
        """
        if interval is not None:
            compiled_interval = float(self._config.toll_step_size) * float(self._config.deltat)
            if abs(interval - compiled_interval) > 1e-9:
                raise ValueError(
                    f"Toll interval {interval} s does not match the compiled interval {compiled_interval} s. "
                    f"Set it at compile time via compile(toll_interval={interval})."
                )
        ids = []
        names = []
        for link in links:
            i, name = self._resolve_link(link)
            ids.append(i)
            names.append(name)
        return TollVariable(link_ids=tuple(ids), n_steps=int(self._config.n_toll_steps),
                            link_names=tuple(names), initial=float(initial), lower=lower, upper=upper)

    def variable(self, shape, initial, inject, name="custom", lower=None, upper=None):
        """Build a user-defined variable with a custom injection into Params.

        Use this for composite variables that the built-in refs cannot express, e.g. a difference between two parameters, a factor shared across links, or a log-transformed parameter.
        The inject function must be a pure JAX-differentiable function; FD consistency across derived quantities is the caller's responsibility.

        Parameters
        ----------
        shape : tuple of int
            Shape of the variable value.
        initial : float or array_like
            Initial value, broadcast to ``shape``.
        inject : callable
            Function ``(params, theta) -> Params`` where ``theta`` has shape ``shape``.
        name : str, optional
            Display name.
        lower : float or None, optional
            Lower bound, applied by projection during optimization.
        upper : float or None, optional
            Upper bound.

        Returns
        -------
        CustomVariable
        """
        return CustomVariable(shape, initial, inject, name=name, lower=lower, upper=upper)

    def minimize(self, variables, objective, checkpoint_every=None):
        """Build an optimization problem over selected variables.

        Parameters
        ----------
        variables : list of TollVariable, CustomVariable, or ParameterRef
            Decision variables.
        objective : callable
            Function ``(result, x) -> JAX scalar`` where ``x`` maps each variable to its current value array.
        checkpoint_every : int or None, optional
            Gradient checkpointing segment length.

        Returns
        -------
        Problem
        """
        return Problem(self, variables, objective, checkpoint_every=checkpoint_every)

    def objective(self, fn, checkpoint_every=None):
        """Build an Objective from a function of a SimResult.

        Parameters
        ----------
        fn : callable
            Maps a SimResult to a JAX scalar.
        checkpoint_every : int or None, optional
            Gradient checkpointing segment length.

        Returns
        -------
        Objective
        """
        return Objective(self, fn, checkpoint_every=checkpoint_every)

    @property
    def raw(self):
        """Low-level access to the JAX core: ``raw.params``, ``raw.config``, ``raw.simulate``."""
        return SimpleNamespace(params=self._params, config=self._config, simulate=self._simulate)

    def __repr__(self):
        return f"<DifferentiableWorld '{self._world.NAME}' ({self._config.n_nodes} nodes, {self._config.n_links} links, route_choice={self.route_choice})>"
