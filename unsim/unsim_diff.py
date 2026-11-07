"""
UNsim-diff: JAX-based differentiable Network Link Transmission Model simulator.

All simulation internals are pure JAX functions, enabling jax.grad, jax.jit,
and jax.lax.scan for automatic differentiation and compilation.

Usage
-----
>>> from unsim import World
>>> from unsim.unsim_diff import world_to_jax, simulate, total_travel_time
>>> W = World(...); W.addNode(...); W.addLink(...); W.adddemand(...)
>>> params, config = world_to_jax(W)
>>> state = simulate(params, config)
>>> ttt = total_travel_time(state, config)
>>> grad_fn = jax.grad(lambda p: total_travel_time(simulate(p, config), config))
>>> grads = grad_fn(params)
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


# ================================================================
# Data structures (NamedTuple PyTrees)
# ================================================================

class NetworkConfig(NamedTuple):
    """Static network topology. Pass as static_argnums to jit.

    Attributes
    ----------
    n_nodes : int
    n_links : int
    tsize : int
        Number of simulation timesteps.
    deltat : float
        Timestep width (s).
    max_in : int
        Max inlinks per node (for padding).
    max_out : int
        Max outlinks per node (for padding).
    link_start_node : jnp.ndarray, (n_links,) int32
    link_end_node : jnp.ndarray, (n_links,) int32
    node_inlinks : jnp.ndarray, (n_nodes, max_in) int32, -1 padded
    node_outlinks : jnp.ndarray, (n_nodes, max_out) int32, -1 padded
    node_n_inlinks : jnp.ndarray, (n_nodes,) int32
    node_n_outlinks : jnp.ndarray, (n_nodes,) int32
    node_type : jnp.ndarray, (n_nodes,) int32
        0=origin, 1=destination, 2=dummy, 3=merge, 4=diverge
    link_lengths : jnp.ndarray, (n_links,) float
    """
    n_nodes: int
    n_links: int
    tsize: int
    deltat: float
    max_in: int
    max_out: int
    link_start_node: jnp.ndarray
    link_end_node: jnp.ndarray
    node_inlinks: jnp.ndarray
    node_outlinks: jnp.ndarray
    node_n_inlinks: jnp.ndarray
    node_n_outlinks: jnp.ndarray
    node_type: jnp.ndarray
    link_lengths: jnp.ndarray
    has_general_nodes: bool  # True if any node is type=5 (general/INM)
    # DUO extensions
    n_dests: int
    dest_node_ids: jnp.ndarray   # (n_dests,) int32
    tt_method: str  # "avg_density" or "multipoint"
    route_update_interval: int  # recompute shortest paths every N steps
    window_size: int  # sliding window size for scan carry (ceil(max_offset) + 2)
    n_toll_steps: int  # number of toll discretization steps
    toll_step_size: int  # timesteps per toll step (= route_update_interval)
    use_logit: bool  # True for "duo_logit" soft routing
    logit_temperature: float  # temperature for logit route choice (s)
    node_fifo: str  # "strict" (FIFO by cumulative-curve inverse mapping) or "coarse" (proportional)
    dtype: str = "float32"  # floating-point type of the simulation arrays ("float32" or "float64")


class Params(NamedTuple):
    """Differentiable simulation parameters.

    Attributes
    ----------
    u : jnp.ndarray, (n_links,)
        Free flow speed (m/s).
    kappa : jnp.ndarray, (n_links,)
        Jam density (veh/m).
    q_star : jnp.ndarray, (n_links,)
        Capacity (veh/s). w is derived as q_star*u/(u*kappa - q_star).
    capacity_out : jnp.ndarray, (n_links,)
        Outflow capacity (veh/s). INF for unconstrained.
    capacity_in : jnp.ndarray, (n_links,)
        Inflow capacity (veh/s). INF for unconstrained.
    flow_capacity : jnp.ndarray, (n_nodes,)
        Node flow capacity (veh/s). INF for unconstrained.
    absorption_ratio : jnp.ndarray, (n_nodes,)
    diverge_ratios : jnp.ndarray, (n_nodes, max_out)
    merge_priority : jnp.ndarray, (n_links,)
    demand_rate : jnp.ndarray, (n_nodes, tsize)
        External demand per timestep at each node.
    turning_fractions : jnp.ndarray, (n_nodes, max_in, max_out)
        Turning fraction matrix for INM general nodes.
    toll : jnp.ndarray, (n_links, n_toll_steps)
        Congestion pricing toll in seconds (time equivalent) per link
        per toll step. Zero for untolled links.
    route_bias : jnp.ndarray, (n_dests, n_nodes, max_out)
        Additive bias on the logit route-choice logits (dimensionless).
        Zero recovers the pure cost-based logit split. Only used for
        route_choice "duo_logit".
    """
    u: jnp.ndarray
    kappa: jnp.ndarray
    q_star: jnp.ndarray
    capacity_out: jnp.ndarray
    capacity_in: jnp.ndarray
    flow_capacity: jnp.ndarray
    absorption_ratio: jnp.ndarray
    diverge_ratios: jnp.ndarray
    merge_priority: jnp.ndarray
    demand_rate: jnp.ndarray
    turning_fractions: jnp.ndarray
    od_demand_rate: jnp.ndarray  # (n_nodes, n_dests, tsize) per-OD demand for DUO
    toll: jnp.ndarray  # (n_links, n_toll_steps) congestion pricing toll (s)
    route_bias: jnp.ndarray  # (n_dests, n_nodes, max_out) additive logit
    # bias on duo_logit route choice; zeros recover the pure cost-based
    # split


class LinkState(NamedTuple):
    """Derived link parameters (from Params + Config).

    Attributes
    ----------
    w : jnp.ndarray, (n_links,)
        Congestion wave speed.
    q_star : jnp.ndarray, (n_links,)
        Capacity.
    offset_u : jnp.ndarray, (n_links,)
        Free-flow offset in timesteps.
    offset_w : jnp.ndarray, (n_links,)
        Congestion wave offset in timesteps.
    """
    w: jnp.ndarray
    q_star: jnp.ndarray
    offset_u: jnp.ndarray
    offset_w: jnp.ndarray


class SimState(NamedTuple):
    """Mutable simulation state carried through jax.lax.scan.

    Attributes
    ----------
    cum_arrival : jnp.ndarray, (n_links, tsize+1)
    cum_departure : jnp.ndarray, (n_links, tsize+1)
    demand_queue : jnp.ndarray, (n_nodes,)
    absorbed_count : jnp.ndarray, (n_nodes,)
    demand_queue_history : jnp.ndarray, (n_nodes, tsize)
    """
    cum_arrival: jnp.ndarray
    cum_departure: jnp.ndarray
    demand_queue: jnp.ndarray
    absorbed_count: jnp.ndarray
    demand_queue_history: jnp.ndarray
    # DUO per-destination tracking
    cum_arrival_d: jnp.ndarray    # (n_links, n_dests, tsize+1)
    cum_departure_d: jnp.ndarray  # (n_links, n_dests, tsize+1)
    prev_next_link_ids: jnp.ndarray  # (n_dests, n_nodes) cached shortest path trees


class ScanCarry(NamedTuple):
    """Small carry for jax.lax.scan with sliding window.

    Replaces the full-history SimState as scan carry to reduce
    reverse-mode AD memory from O(tsize^2 * n_links) to
    O(tsize * n_links * window_size).

    Attributes
    ----------
    cum_arrival_w : jnp.ndarray, (n_links, window_size+1)
        Sliding window of cumulative arrivals. Last column = current time.
    cum_departure_w : jnp.ndarray, (n_links, window_size+1)
        Sliding window of cumulative departures. Last column = current time.
    demand_queue : jnp.ndarray, (n_nodes,)
    absorbed_count : jnp.ndarray, (n_nodes,)
    cum_arrival_full : jnp.ndarray, (n_links, tsize+1)
        Full-history aggregate cumulative arrivals (needed by the FIFO inverse mapping, which may look back beyond the window).
    cum_arrival_d : jnp.ndarray, (tsize+1, n_links, n_dests)
        Full-history per-destination cumulative arrivals (time-major, see ``_fifo_lookup_batch``).
    cum_departure_d_cur : jnp.ndarray, (n_links, n_dests)
        Per-destination cumulative departures at current time only.
    orig_out_d : jnp.ndarray, (n_origins, n_dests)
        Per-destination cumulative departures from the origin vertical queues at current time.
    prev_next_link_ids : jnp.ndarray, (n_dests, n_nodes)
    prev_dists : jnp.ndarray, (n_dests, n_nodes)
        Cached shortest-path distances from last route update.
    prev_link_cost : jnp.ndarray, (n_links,)
        Cached generalized link costs from last route update (logit routing uses them until the next update).
    """
    cum_arrival_w: jnp.ndarray
    cum_departure_w: jnp.ndarray
    demand_queue: jnp.ndarray
    absorbed_count: jnp.ndarray
    cum_arrival_full: jnp.ndarray
    cum_arrival_d: jnp.ndarray
    cum_departure_d_cur: jnp.ndarray
    orig_out_d: jnp.ndarray
    prev_next_link_ids: jnp.ndarray
    prev_dists: jnp.ndarray
    prev_link_cost: jnp.ndarray


class FwdCarry(NamedTuple):
    """Forward-only carry for jax.lax.scan (no AD tape).

    Uses full-history arrays instead of sliding windows.
    When used without jax.grad, XLA does not retain intermediate carries,
    eliminating reverse-mode AD memory overhead.

    Attributes
    ----------
    cum_arrival : jnp.ndarray, (n_links, tsize+1)
        Full cumulative arrivals history.
    cum_departure : jnp.ndarray, (n_links, tsize+1)
        Full cumulative departures history.
    demand_queue : jnp.ndarray, (n_nodes,)
    absorbed_count : jnp.ndarray, (n_nodes,)
    cum_arrival_d : jnp.ndarray, (tsize+1, n_links, n_dests)
        Full-history per-destination cumulative arrivals (time-major, see ``_fifo_lookup_batch``).
    cum_departure_d_cur : jnp.ndarray, (n_links, n_dests)
        Per-destination cumulative departures at current time only.
    orig_out_d : jnp.ndarray, (n_origins, n_dests)
        Per-destination cumulative departures from the origin vertical queues at current time.
    prev_next_link_ids : jnp.ndarray, (n_dests, n_nodes)
    prev_dists : jnp.ndarray, (n_dests, n_nodes)
        Cached shortest-path distances from last route update.
    prev_link_cost : jnp.ndarray, (n_links,)
        Cached generalized link costs from last route update.
    """
    cum_arrival: jnp.ndarray
    cum_departure: jnp.ndarray
    demand_queue: jnp.ndarray
    absorbed_count: jnp.ndarray
    cum_arrival_d: jnp.ndarray
    cum_departure_d_cur: jnp.ndarray
    orig_out_d: jnp.ndarray
    prev_next_link_ids: jnp.ndarray
    prev_dists: jnp.ndarray
    prev_link_cost: jnp.ndarray


class StepOutput(NamedTuple):
    """Per-step output from scan for SimState reconstruction.

    Attributes
    ----------
    inflow_rates : jnp.ndarray, (n_links,)
    outflow_rates : jnp.ndarray, (n_links,)
    demand_queue : jnp.ndarray, (n_nodes,)
    inflow_d : jnp.ndarray, (n_links, n_dests)
    outflow_d : jnp.ndarray, (n_links, n_dests)
    """
    inflow_rates: jnp.ndarray
    outflow_rates: jnp.ndarray
    demand_queue: jnp.ndarray
    inflow_d: jnp.ndarray
    outflow_d: jnp.ndarray


class AoNFwdCarry(NamedTuple):
    """Forward-only carry for AoN scan (minimal, no BF cache).

    Routes are precomputed before the scan loop, so no need to carry
    prev_next_link_ids or prev_dists. Eliminates per-destination
    arrays from stacked outputs.

    demand_queue_history is kept in the carry (not stacked outputs) so
    that the scan output is empty, avoiding all stacked allocations.

    Attributes
    ----------
    cum_arrival : jnp.ndarray, (n_links, tsize+1)
    cum_departure : jnp.ndarray, (n_links, tsize+1)
    demand_queue : jnp.ndarray, (n_nodes,)
    absorbed_count : jnp.ndarray, (n_nodes,)
    cum_arrival_d : jnp.ndarray, (tsize+1, n_links, n_dests)
        Full-history per-destination cumulative arrivals (time-major; needed by the FIFO inverse mapping).
    cum_departure_d_cur : jnp.ndarray, (n_links, n_dests)
        Per-destination cumulative departures at current time only.
    orig_out_d : jnp.ndarray, (n_origins, n_dests)
        Per-destination cumulative departures from the origin vertical queues at current time.
    demand_queue_history : jnp.ndarray, (n_nodes, tsize)
        Full history of demand queues (for total_travel_time).
    """
    cum_arrival: jnp.ndarray
    cum_departure: jnp.ndarray
    demand_queue: jnp.ndarray
    absorbed_count: jnp.ndarray
    cum_arrival_d: jnp.ndarray
    cum_departure_d_cur: jnp.ndarray
    orig_out_d: jnp.ndarray
    demand_queue_history: jnp.ndarray


# ================================================================
# Helper functions
# ================================================================

def compute_link_state(params, config):
    """Derive FD quantities and offsets from params.

    Parameters
    ----------
    params : Params
    config : NetworkConfig

    Returns
    -------
    LinkState
    """
    q_star = params.q_star
    denom = jnp.where(params.u * params.kappa - q_star > 0.0, params.u * params.kappa - q_star, 1e-10)
    w = q_star * params.u / denom
    safe_w = jnp.where(w > 0.0, w, 1e-10)
    offset_u = config.link_lengths / (params.u * config.deltat)
    offset_w = config.link_lengths / (safe_w * config.deltat)
    return LinkState(w=w, q_star=q_star, offset_u=offset_u, offset_w=offset_w)


def _make_fake_state(carry, config):
    """Build a lightweight SimState from ScanCarry for existing compute functions.

    The returned SimState uses the sliding window as cum_arrival / cum_departure.
    Functions that only access ``[:, t_index]`` or ``interp_batch`` work correctly
    when called with ``t_index = config.window_size`` (the "current" position).
    """
    n_dests = max(config.n_dests, 1)
    return SimState(
        cum_arrival=carry.cum_arrival_w,
        cum_departure=carry.cum_departure_w,
        demand_queue=carry.demand_queue,
        absorbed_count=carry.absorbed_count,
        demand_queue_history=jnp.zeros((config.n_nodes, 1)),
        cum_arrival_d=jnp.zeros((config.n_links, n_dests, 1)),
        cum_departure_d=jnp.zeros((config.n_links, n_dests, 1)),
        prev_next_link_ids=carry.prev_next_link_ids,
    )


def _make_state_from_fwd_carry(carry, config):
    """Build a lightweight SimState from FwdCarry for compute functions.

    Unlike ``_make_fake_state``, the arrays here are full-history so
    ``t_index`` passed to compute functions is the absolute timestep.
    """
    n_dests = max(config.n_dests, 1)
    return SimState(
        cum_arrival=carry.cum_arrival,
        cum_departure=carry.cum_departure,
        demand_queue=carry.demand_queue,
        absorbed_count=carry.absorbed_count,
        demand_queue_history=jnp.zeros((config.n_nodes, 1)),
        cum_arrival_d=jnp.zeros((config.n_links, n_dests, 1)),
        cum_departure_d=jnp.zeros((config.n_links, n_dests, 1)),
        prev_next_link_ids=carry.prev_next_link_ids,
    )


# ================================================================
# Derivatives at ties
# ================================================================
# The model is piecewise smooth: min, max, comparisons and interpolation indices switch between branches.
# Where two branches meet (a tie), the one-sided derivatives can differ.
# JAX's own rules differ between operations (0.5/0.5 for jnp.minimum, jnp.maximum and jnp.clip, one branch for jnp.where).
# Mixing them can give a derivative outside every one-sided derivative, e.g. a nonzero derivative of total travel time with respect to a merge priority although total travel time does not depend on it.
# The functions below apply one convention to all ties:
# - at a tie between flow regimes, the derivative follows the branch that is active when every demand is infinitesimally smaller (free flow, demand-limited flow, queues ending, a sending cohort that ends before a grid value of the arrival curve, INM inlinks that reach their demand just before an outlink fills, each with the derivative of its own demand);
# - bounds that the model never crosses (non-negative flows, queues and remaining capacities, the index range of the interpolation) pass the derivative through;
# - operands that differ only by float32 rounding count as tied, so that the rounding of the operands does not decide the branch of the derivative (the float32 tolerances are kept in float64 simulations).
# Forward values are unchanged.

# Relative gap below which two operands count as tied when the derivative branch is chosen.
_TIE_RTOL = 1e-5
# Rounding error of a link demand or supply, in float32 ulps of the cumulative counts it is computed from.
_COUNT_ULPS = 8.0


def _near(a, b):
    """Whether ``a`` and ``b`` are equal up to the relative gap ``_TIE_RTOL``.

    Parameters
    ----------
    a, b : jnp.ndarray
        Broadcastable operands.

    Returns
    -------
    jnp.ndarray of bool
    """
    return jnp.abs(a - b) <= _TIE_RTOL * jnp.maximum(jnp.abs(a), jnp.abs(b))


def _tie_where(take_x, tied, x, y, exact):
    """Value ``exact`` with the derivative of ``x`` where ``take_x`` or ``tied`` holds, else of ``y``.

    Parameters
    ----------
    take_x : jnp.ndarray of bool
        Where the exact computation selects ``x``.
    tied : jnp.ndarray of bool
        Where the selection is a tie whose derivative follows ``x``.
    x, y : jnp.ndarray
        Branches.
    exact : jnp.ndarray
        Forward value (the exact selection between ``x`` and ``y``).

    Returns
    -------
    jnp.ndarray
    """
    sel = jnp.where(jnp.logical_or(take_x, tied), x, y)
    return sel + jax.lax.stop_gradient(exact - sel)


def _tie_min(a, b):
    """``min(a, b)`` whose derivative at a tie is that of ``a``.

    A tie is ``|a - b| <= _TIE_RTOL * max(|a|, |b|)``.

    Parameters
    ----------
    a, b : jnp.ndarray
        Broadcastable operands; ``a`` is the branch whose derivative is used at a tie.

    Returns
    -------
    jnp.ndarray
        ``jnp.minimum(a, b)``.
    """
    return _tie_where(a <= b, _near(a, b), a, b, jnp.minimum(a, b))


def _tie_min_atol(a, b, atol):
    """``min(a, b)`` whose derivative at a tie is that of ``a``, with an absolute tie tolerance.

    A tie is ``|a - b| <= atol`` or ``|a - b| <= _TIE_RTOL * max(|a|, |b|)``.

    Parameters
    ----------
    a, b : jnp.ndarray
        Broadcastable operands; ``a`` is the branch whose derivative is used at a tie.
    atol : jnp.ndarray
        Absolute tolerance, e.g. the rounding error of an operand that is a difference of large counts.

    Returns
    -------
    jnp.ndarray
        ``jnp.minimum(a, b)``.
    """
    tied = jnp.logical_or(_near(a, b), jnp.abs(a - b) <= atol)
    return _tie_where(a <= b, tied, a, b, jnp.minimum(a, b))


def _tie_max(a, b):
    """``max(a, b)`` whose derivative at a tie is that of ``a``.

    A tie is ``|a - b| <= _TIE_RTOL * max(|a|, |b|)``.

    Parameters
    ----------
    a, b : jnp.ndarray
        Broadcastable operands; ``a`` is the branch whose derivative is used at a tie.

    Returns
    -------
    jnp.ndarray
        ``jnp.maximum(a, b)``.
    """
    return _tie_where(a >= b, _near(a, b), a, b, jnp.maximum(a, b))


def _nonneg(x):
    """``max(x, 0)`` for a quantity that is non-negative in exact arithmetic; the derivative is that of ``x``.

    The clamp only removes rounding errors, so the derivative passes through everywhere, also where rounding makes ``x`` slightly negative.

    Parameters
    ----------
    x : jnp.ndarray

    Returns
    -------
    jnp.ndarray
        ``jnp.maximum(x, 0.0)``.
    """
    return x + jax.lax.stop_gradient(jnp.maximum(x, 0.0) - x)


def _inm_tie_corrections(theta, phi_in, phi_out, B, rem_dem, rem_sup, d_up, d_down, theta_in, theta_out):
    """Zero-valued terms that give the INM flow update the derivatives of "Derivatives at ties".

    The update is ``qi + theta * phi_in + c_in`` and ``qo + theta * phi_out + c_out``, where ``c_in`` and ``c_out`` are 0 in value.
    Inlinks whose demand is reached at ``theta`` take the derivative of their own remaining demand, also when several are reached together.
    They are reached just before the outlinks that fill at the same ``theta``: the other active inlinks take the derivative of the step at which such an outlink fills with the supply left after those inlinks.
    The outlink increments have the derivative of the turning-fraction sums of the inlink increments, so that the linearization conserves flow at the node.
    Without ties the terms have derivative 0.

    Parameters
    ----------
    theta : jnp scalar
        Step of the iteration.
    phi_in : jnp.ndarray, (max_in,)
        Priorities of the active inlinks, 0 for the others.
    phi_out : jnp.ndarray, (max_out,)
        Turning-fraction sums of ``phi_in``.
    B : jnp.ndarray, (max_in, max_out)
        Turning fractions.
    rem_dem : jnp.ndarray, (max_in,)
        Remaining demand of the inlinks.
    rem_sup : jnp.ndarray, (max_out,)
        Remaining supply of the outlinks.
    d_up : jnp.ndarray of bool, (max_in,)
        Active inlinks.
    d_down : jnp.ndarray of bool, (max_out,)
        Active outlinks.
    theta_in : jnp.ndarray, (max_in,)
        Step at which each active inlink reaches its demand (inf for the others).
    theta_out : jnp.ndarray, (max_out,)
        Step at which each active outlink fills (inf for the others).

    Returns
    -------
    c_in : jnp.ndarray, (max_in,)
    c_out : jnp.ndarray, (max_out,)
    """
    reached = d_up & jnp.isfinite(theta_in) & _near(theta_in, theta)
    filled = d_down & jnp.isfinite(theta_out) & _near(theta_out, theta)
    phi_rest = jnp.sum(B * jnp.where(reached, 0.0, phi_in)[:, None], axis=0)
    fills = filled & (phi_rest > 1e-15)
    sup_rest = rem_sup - jnp.sum(B * jnp.where(reached, rem_dem, 0.0)[:, None], axis=0)
    theta_fill = jnp.min(jnp.where(fills, sup_rest / jnp.where(fills, phi_rest, 1.0), jnp.inf))
    theta_rest = jnp.where(jnp.any(fills), theta_fill, theta)
    d_in = jnp.where(reached, rem_dem, theta_rest * phi_in)
    d_out = jnp.sum(B * d_in[:, None], axis=0)
    inc_in = theta * phi_in
    inc_out = theta * phi_out
    c_in = (d_in - jax.lax.stop_gradient(d_in)) - (inc_in - jax.lax.stop_gradient(inc_in))
    c_out = (d_out - jax.lax.stop_gradient(d_out)) - (inc_out - jax.lax.stop_gradient(inc_out))
    return c_in, c_out


def interp_batch(arrays_2d, frac_indices):
    """Linearly interpolate each row at its fractional index.

    Parameters
    ----------
    arrays_2d : jnp.ndarray, (n, m)
    frac_indices : jnp.ndarray, (n,)

    Returns
    -------
    jnp.ndarray, (n,)
    """
    max_idx = arrays_2d.shape[1] - 1
    # Index range clamp; at the bounds the derivative passes through (see "Derivatives at ties").
    fi = _tie_max(_tie_min(frac_indices, float(max_idx)), 0.0)
    i_low = jnp.floor(fi).astype(jnp.int32)
    i_low = jnp.clip(i_low, 0, max_idx - 1)
    frac = fi - i_low.astype(fi.dtype)
    row_ids = jnp.arange(arrays_2d.shape[0])
    return arrays_2d[row_ids, i_low] * (1.0 - frac) + arrays_2d[row_ids, i_low + 1] * frac


def interp_1d(array_1d, frac_index):
    """Linearly interpolate a 1D array at a fractional index.

    Parameters
    ----------
    array_1d : jnp.ndarray, (m,)
    frac_index : float or jnp scalar

    Returns
    -------
    jnp scalar
    """
    max_idx = array_1d.shape[0] - 1
    # Index range clamp; at the bounds the derivative passes through (see "Derivatives at ties").
    fi = _tie_max(_tie_min(frac_index, float(max_idx)), 0.0)
    i_low = jnp.floor(fi).astype(jnp.int32)
    i_low = jnp.clip(i_low, 0, max_idx - 1)
    frac = fi - i_low.astype(fi.dtype)
    return array_1d[i_low] * (1.0 - frac) + array_1d[i_low + 1] * frac


def differentiable_mid(a, b, c):
    """Mid-value of three scalars: sum - min - max. Fully differentiable.

    Parameters
    ----------
    a, b, c : jnp scalar

    Returns
    -------
    jnp scalar
    """
    return a + b + c - jnp.minimum(a, jnp.minimum(b, c)) - jnp.maximum(a, jnp.maximum(b, c))


# ================================================================
# Demand / Supply computation (vectorized over all links)
# ================================================================

def compute_demands(t_index, state, link_state, params, config):
    """Compute demand D_l(t) for all links.

    D_l = min{(N_U(t+1-offset_u) - N_D(t)) / dt, q*, capacity_out}

    Parameters
    ----------
    t_index : int
    state : SimState
    link_state : LinkState
    params : Params
    config : NetworkConfig

    Returns
    -------
    jnp.ndarray, (n_links,)
    """
    dt = config.deltat
    frac_indices = (t_index + 1.0) - link_state.offset_u
    N_U_past = interp_batch(state.cum_arrival, frac_indices)
    N_D_now = state.cum_departure[:, t_index]
    D = (N_U_past - N_D_now) / dt
    # At D == q* the derivative follows the uncongested branch D, and D >= 0 by conservation (see "Derivatives at ties").
    # The tie tolerance covers the float32 rounding of the count difference.
    atol = _COUNT_ULPS * jnp.finfo(jnp.float32).eps * (jnp.abs(N_U_past) + jnp.abs(N_D_now)) / dt
    D = _tie_min_atol(D, link_state.q_star, atol)
    D = _tie_min_atol(D, params.capacity_out, atol)
    D = _nonneg(D)
    return D


def compute_supplies(t_index, state, link_state, params, config):
    """Compute supply S_l(t) for all links.

    S_l = min{(N_D(t+1-offset_w) + kappa*d - N_U(t)) / dt, q*, capacity_in}

    Parameters
    ----------
    t_index : int
    state : SimState
    link_state : LinkState
    params : Params
    config : NetworkConfig

    Returns
    -------
    jnp.ndarray, (n_links,)
    """
    dt = config.deltat
    frac_indices = (t_index + 1.0) - link_state.offset_w
    N_D_past = interp_batch(state.cum_departure, frac_indices)
    N_U_now = state.cum_arrival[:, t_index]
    S = (N_D_past + params.kappa * config.link_lengths - N_U_now) / dt
    # At S == q* the derivative follows the uncongested branch q*, and S >= 0 by the storage limit (see "Derivatives at ties").
    # The tie tolerance covers the float32 rounding of the count difference.
    atol = _COUNT_ULPS * jnp.finfo(jnp.float32).eps * (
        jnp.abs(N_D_past) + params.kappa * config.link_lengths + jnp.abs(N_U_now)) / dt
    S = _tie_min_atol(link_state.q_star, S, atol)
    S = _tie_min_atol(params.capacity_in, S, atol)
    S = _nonneg(S)
    return S


# ================================================================
# Node transfer models
# ================================================================

def _safe_idx(link_ids):
    """Clamp link indices to >= 0 for safe array indexing (padded -1 -> 0)."""
    return jnp.maximum(link_ids, 0)


def compute_node_transfers(t_index, demands, supplies, state, params, config):
    """Compute flow rates at all nodes for one timestep (vectorized).

    All node types are computed simultaneously over all nodes using batched
    operations, then results are scattered to link-indexed arrays.
    This avoids the sequential fori_loop and enables efficient GPU execution.

    Parameters
    ----------
    t_index : int
    demands : jnp.ndarray, (n_links,)
    supplies : jnp.ndarray, (n_links,)
    state : SimState
    params : Params
    config : NetworkConfig

    Returns
    -------
    inflow_rates : jnp.ndarray, (n_links,)
    outflow_rates : jnp.ndarray, (n_links,)
    new_demand_queue : jnp.ndarray, (n_nodes,)
    new_absorbed : jnp.ndarray, (n_nodes,)
    """
    dt = config.deltat
    n_links = config.n_links
    n_nodes = config.n_nodes
    ntype = config.node_type  # (n_nodes,)
    dq = state.demand_queue   # (n_nodes,)

    # ---- Batch-gather inputs for all nodes ----
    in_lids = config.node_inlinks    # (n_nodes, max_in)
    out_lids = config.node_outlinks  # (n_nodes, max_out)
    safe_in = _safe_idx(in_lids)     # (n_nodes, max_in)
    safe_out = _safe_idx(out_lids)   # (n_nodes, max_out)
    n_in = config.node_n_inlinks     # (n_nodes,)
    n_out = config.node_n_outlinks   # (n_nodes,)

    in_dem = demands[safe_in]        # (n_nodes, max_in)
    out_sup = supplies[safe_out]     # (n_nodes, max_out)
    in_valid = jnp.arange(config.max_in)[None, :] < n_in[:, None]    # (n_nodes, max_in)
    out_valid = jnp.arange(config.max_out)[None, :] < n_out[:, None]  # (n_nodes, max_out)

    fc = params.flow_capacity  # (n_nodes,)

    # ---- Origin (type=0) ----
    ext_demand = params.demand_rate[:, t_index]  # (n_nodes,)
    eff_demand = ext_demand + dq / dt            # (n_nodes,)

    # Single outlink.
    # At a tie the derivative follows the effective demand (no queue forms), and the flow is non-negative by construction (see "Derivatives at ties").
    s0 = out_sup[:, 0]  # (n_nodes,)
    origin_flow_single = _nonneg(_tie_min(_tie_min(eff_demand, s0), fc))

    # Multi outlink
    betas = params.diverge_ratios  # (n_nodes, max_out)
    s_over_b = jnp.where(betas > 1e-10,
                          out_sup / jnp.maximum(betas, 1e-10), jnp.inf)
    s_over_b = jnp.where(out_valid, s_over_b, jnp.inf)
    min_s = jnp.min(s_over_b, axis=1)  # (n_nodes,)
    origin_flow_multi = _nonneg(_tie_min(_tie_min(eff_demand, min_s), fc))

    origin_flow = jnp.where(n_out == 1, origin_flow_single, origin_flow_multi)  # (n_nodes,)
    # The queue update is non-negative because origin_flow <= eff_demand.
    origin_dq = _nonneg(dq + (ext_demand - origin_flow) * dt)          # (n_nodes,)

    # Origin inflows per outlink
    j_out = jnp.arange(config.max_out)[None, :]  # (1, max_out)
    origin_inflows_single = jnp.where(j_out == 0, origin_flow[:, None], 0.0)
    origin_inflows_multi = betas * origin_flow[:, None]
    origin_inflows = jnp.where(
        (n_out == 1)[:, None], origin_inflows_single, origin_inflows_multi)  # (n_nodes, max_out)

    # ---- Destination (type=1) ----
    dest_outflows = in_dem  # (n_nodes, max_in)
    dest_absorbed = jnp.sum(jnp.where(in_valid, in_dem, 0.0), axis=1) * dt  # (n_nodes,)

    # ---- Dummy (type=2) ----
    D_dummy = in_dem[:, 0]  # (n_nodes,)
    pass_ratio = 1.0 - params.absorption_ratio  # (n_nodes,)
    dummy_flow = jnp.where(
        pass_ratio > 1e-10,
        _tie_min(D_dummy, out_sup[:, 0] / jnp.maximum(pass_ratio, 1e-10)),
        D_dummy)
    dummy_flow = _nonneg(_tie_min(dummy_flow, fc))  # (n_nodes,)
    dummy_outflow = dummy_flow
    dummy_inflow = dummy_flow * pass_ratio
    dummy_absorbed = dummy_flow * params.absorption_ratio * dt  # (n_nodes,)

    # ---- Merge (type=3) ----
    S_merge = jnp.minimum(out_sup[:, 0], fc)  # (n_nodes,)
    total_D = jnp.sum(jnp.where(in_valid, in_dem, 0.0), axis=1)  # (n_nodes,)
    p = jnp.where(in_valid, params.merge_priority[safe_in], 0.0)  # (n_nodes, max_in)
    total_p = jnp.sum(p, axis=1, keepdims=True)
    has_p = total_p > 0.0
    safe_total_p = jnp.where(has_p, total_p, 1.0)
    alphas = jnp.where(has_p, p / safe_total_p, 0.0)

    # Ties (see "Derivatives at ties"): base and extra flows follow the inlink's own demand, the remaining capacity
    # follows its unclamped value, and at total_D == S_merge the uncongested formula is used.
    # rem_S_raw >= 0 because sum(alphas) == 1.
    base_q = _tie_min(in_dem, alphas * S_merge[:, None])
    rem_S_raw = S_merge - jnp.sum(jnp.where(in_valid, base_q, 0.0), axis=1)
    rem_S = _nonneg(rem_S_raw)
    surplus_cap = jnp.where(in_valid, in_dem - base_q, 0.0)
    cum_surplus = jnp.cumsum(surplus_cap, axis=1)
    consumed = cum_surplus - surplus_cap
    avail_rem_raw = rem_S[:, None] - consumed
    avail_rem = _tie_where(avail_rem_raw >= 0.0, _near(rem_S[:, None], consumed), avail_rem_raw, 0.0,
                           jnp.where(avail_rem_raw > 0.0, avail_rem_raw, 0.0))
    extra_q = _tie_min(surplus_cap, avail_rem)
    q_raw = base_q + extra_q
    sum_q_raw = jnp.sum(jnp.where(in_valid, q_raw, 0.0), axis=1, keepdims=True)
    q_cong = q_raw + alphas * (S_merge[:, None] - sum_q_raw)
    q_cong = jnp.where(in_valid, q_cong, 0.0)
    uncongested = total_D[:, None] <= S_merge[:, None]
    merge_q_all = _tie_where(uncongested, _near(total_D[:, None], S_merge[:, None]), in_dem, q_cong,
                             jnp.where(uncongested, in_dem, q_cong))  # (n_nodes, max_in)
    merge_q_out = jnp.sum(jnp.where(in_valid, merge_q_all, 0.0), axis=1)  # (n_nodes,)

    # ---- Diverge (type=4) ----
    D_div = in_dem[:, 0]  # (n_nodes,)
    div_betas = params.diverge_ratios  # (n_nodes, max_out)
    div_s_over_b = jnp.where(div_betas > 1e-10,
                              out_sup / jnp.maximum(div_betas, 1e-10), jnp.inf)
    div_s_over_b = jnp.where(out_valid, div_s_over_b, jnp.inf)
    div_min_s = jnp.min(div_s_over_b, axis=1)  # (n_nodes,)
    div_flow = _nonneg(_tie_min(_tie_min(D_div, div_min_s), fc))
    div_outflows_per_link = div_betas * div_flow[:, None]  # (n_nodes, max_out)

    # ---- General / INM (type=5) ----
    if config.has_general_nodes:
        inm_alpha_all = params.merge_priority[safe_in]  # (n_nodes, max_in)
        EPS_INM = 1e-10
        # Turning fractions below EPS_B do not count as "sending to j" in the FIFO
        # blocking rule (same rule as Node._transfer_general).
        EPS_B = 1e-6
        max_iter_inm = config.max_in + config.max_out + 1

        def _inm_single(B_i, alpha_i, dem_i, sup_i, iv, ov):
            """INM solver for one node."""
            def inm_body(carry_iter, _):
                qi, qo = carry_iter
                demand_ok = qi < (dem_i - EPS_INM)
                out_full = qo >= (sup_i - EPS_INM)
                blocked = jnp.any(jnp.logical_and(B_i > EPS_B, out_full[None, :]), axis=1)
                d_up = jnp.logical_and(demand_ok, ~blocked)
                d_up = jnp.logical_and(d_up, iv)

                supply_ok = qo < (sup_i - EPS_INM)
                has_active_up = jnp.any(jnp.logical_and(d_up[:, None], B_i > EPS_B), axis=0)
                d_down = jnp.logical_and(supply_ok, has_active_up)
                d_down = jnp.logical_and(d_down, ov)

                any_active = jnp.any(d_up)
                phi_in = jnp.where(d_up, alpha_i, 0.0)
                phi_out = jnp.sum(B_i * phi_in[:, None], axis=0)

                theta_in = jnp.where(
                    jnp.logical_and(d_up, phi_in > 1e-15),
                    (dem_i - qi) / jnp.maximum(phi_in, 1e-15), jnp.inf)
                theta_out = jnp.where(
                    jnp.logical_and(d_down, phi_out > 1e-15),
                    (sup_i - qo) / jnp.maximum(phi_out, 1e-15), jnp.inf)
                theta = jnp.minimum(jnp.min(theta_in), jnp.min(theta_out))
                theta = jnp.where(any_active, theta, 0.0)
                theta = jnp.where(theta > 1e17, 0.0, theta)
                theta = jnp.maximum(theta, 0.0)
                c_in, c_out = _inm_tie_corrections(theta, phi_in, phi_out, B_i, dem_i - qi, sup_i - qo,
                                                   d_up, d_down, theta_in, theta_out)
                return (qi + theta * phi_in + c_in, qo + theta * phi_out + c_out), None

            qi_init = jnp.zeros(config.max_in, dtype=config.dtype)
            qo_init = jnp.zeros(config.max_out, dtype=config.dtype)
            # Unrolled: the loop is short, and unrolling lets the compiler fuse across iterations.
            (inm_q_in, inm_q_out), _ = jax.lax.scan(
                inm_body, (qi_init, qo_init), None, length=max_iter_inm, unroll=True)
            return inm_q_in, inm_q_out

        inm_q_in_all, inm_q_out_all = jax.vmap(_inm_single)(
            params.turning_fractions, inm_alpha_all,
            in_dem, out_sup, in_valid, out_valid)  # (n_nodes, max_in), (n_nodes, max_out)
    else:
        inm_q_in_all = jnp.zeros((n_nodes, config.max_in), dtype=config.dtype)
        inm_q_out_all = jnp.zeros((n_nodes, config.max_out), dtype=config.dtype)

    # ---- Dispatch by node type: outflows for inlinks (n_nodes, max_in) ----
    j_in = jnp.arange(config.max_in)[None, :]  # (1, max_in)
    nt = ntype[:, None]  # (n_nodes, 1)

    out_val_dest = dest_outflows  # (n_nodes, max_in)
    out_val_dummy = jnp.where(j_in == 0, dummy_outflow[:, None], 0.0)
    out_val_merge = merge_q_all
    out_val_diverge = jnp.where(j_in == 0, div_flow[:, None], 0.0)
    out_val_general = inm_q_in_all

    outflow_per_inlink = jnp.where(nt == 1, out_val_dest,
                         jnp.where(nt == 2, out_val_dummy,
                         jnp.where(nt == 3, out_val_merge,
                         jnp.where(nt == 4, out_val_diverge,
                         jnp.where(nt == 5, out_val_general, 0.0)))))  # (n_nodes, max_in)

    # ---- Dispatch by node type: inflows for outlinks (n_nodes, max_out) ----
    in_val_origin = origin_inflows  # (n_nodes, max_out)
    in_val_dummy = jnp.where(j_out == 0, dummy_inflow[:, None], 0.0)
    in_val_merge = jnp.where(j_out == 0, merge_q_out[:, None], 0.0)
    in_val_diverge = div_outflows_per_link  # (n_nodes, max_out)
    in_val_general = inm_q_out_all

    inflow_per_outlink = jnp.where(nt == 0, in_val_origin,
                         jnp.where(nt == 2, in_val_dummy,
                         jnp.where(nt == 3, in_val_merge,
                         jnp.where(nt == 4, in_val_diverge,
                         jnp.where(nt == 5, in_val_general, 0.0)))))  # (n_nodes, max_out)

    # ---- Scatter to link-indexed arrays ----
    # Each link is inlink of exactly one node and outlink of exactly one node,
    # so there are no write conflicts among valid entries.
    # Mask invalid entries to 0 so .add() on padded index 0 is harmless.
    flat_in_lids = safe_in.ravel()       # (n_nodes * max_in,)
    flat_out_vals = (outflow_per_inlink * in_valid).ravel()  # (n_nodes * max_in,)
    outflows = jnp.zeros(n_links, dtype=config.dtype).at[flat_in_lids].add(flat_out_vals)

    flat_out_lids = safe_out.ravel()     # (n_nodes * max_out,)
    flat_in_vals = (inflow_per_outlink * out_valid).ravel()  # (n_nodes * max_out,)
    inflows = jnp.zeros(n_links, dtype=config.dtype).at[flat_out_lids].add(flat_in_vals)

    # ---- Update demand queue (origin only) ----
    new_dq = jnp.where(ntype == 0, origin_dq, dq)

    # ---- Update absorbed count ----
    abs_inc = jnp.where(ntype == 1, dest_absorbed,
              jnp.where(ntype == 2, dummy_absorbed, 0.0))
    new_abs = state.absorbed_count + abs_inc

    return inflows, outflows, new_dq, new_abs


# ================================================================
# Simulation step and loop
# ================================================================

def simulation_step(carry, t_index, params, link_state, config):
    """One LTM timestep for jax.lax.scan (windowed carry).

    Parameters
    ----------
    carry : ScanCarry
    t_index : jnp scalar int (absolute timestep index)
    params : Params
    link_state : LinkState
    config : NetworkConfig

    Returns
    -------
    (new_carry, StepOutput)
    """
    dt = config.deltat
    W = config.window_size

    # Build fake SimState backed by the sliding window
    fake_state = _make_fake_state(carry, config)

    # Compute demands / supplies using window (W = "current time" in window coords)
    demands = compute_demands(W, fake_state, link_state, params, config)
    supplies = compute_supplies(W, fake_state, link_state, params, config)

    # Node transfers (t_index is absolute -- used for demand_rate lookup)
    inflow_rates, outflow_rates, new_dq, new_abs = compute_node_transfers(
        t_index, demands, supplies, fake_state, params, config)

    # Update sliding window: shift left, append new cumulative column
    new_cum_arr = carry.cum_arrival_w[:, -1] + dt * inflow_rates
    new_cum_dep = carry.cum_departure_w[:, -1] + dt * outflow_rates
    new_arr_w = jnp.concatenate(
        [carry.cum_arrival_w[:, 1:], new_cum_arr[:, None]], axis=1)
    new_dep_w = jnp.concatenate(
        [carry.cum_departure_w[:, 1:], new_cum_dep[:, None]], axis=1)

    n_dests = max(config.n_dests, 1)
    new_carry = ScanCarry(
        cum_arrival_w=new_arr_w,
        cum_departure_w=new_dep_w,
        demand_queue=new_dq,
        absorbed_count=new_abs,
        cum_arrival_full=carry.cum_arrival_full,
        cum_arrival_d=carry.cum_arrival_d,
        cum_departure_d_cur=carry.cum_departure_d_cur,
        orig_out_d=carry.orig_out_d,
        prev_next_link_ids=carry.prev_next_link_ids,
        prev_dists=carry.prev_dists,
        prev_link_cost=carry.prev_link_cost,
    )

    output = StepOutput(
        inflow_rates=inflow_rates,
        outflow_rates=outflow_rates,
        demand_queue=new_dq,
        inflow_d=jnp.zeros((config.n_links, n_dests), dtype=config.dtype),
        outflow_d=jnp.zeros((config.n_links, n_dests), dtype=config.dtype),
    )
    return new_carry, output


def simulation_step_fwd(carry, t_index, params, link_state, config):
    """One LTM timestep for forward-only jax.lax.scan (full-array carry).

    Uses absolute ``t_index`` into full-history arrays instead of a sliding
    window.  Avoids the per-step concatenate/shift of the windowed carry.

    Parameters
    ----------
    carry : FwdCarry
    t_index : jnp scalar int (absolute timestep index)
    params : Params
    link_state : LinkState
    config : NetworkConfig

    Returns
    -------
    (new_carry, StepOutput)
    """
    dt = config.deltat
    n_dests = max(config.n_dests, 1)

    # Build SimState view backed by full-history arrays
    state = _make_state_from_fwd_carry(carry, config)

    # Compute demands / supplies using absolute t_index
    demands = compute_demands(t_index, state, link_state, params, config)
    supplies = compute_supplies(t_index, state, link_state, params, config)

    # Node transfers
    inflow_rates, outflow_rates, new_dq, new_abs = compute_node_transfers(
        t_index, demands, supplies, state, params, config)

    # Update full arrays at t_index + 1
    new_cum_arr = carry.cum_arrival[:, t_index] + dt * inflow_rates
    new_cum_dep = carry.cum_departure[:, t_index] + dt * outflow_rates

    new_carry = FwdCarry(
        cum_arrival=carry.cum_arrival.at[:, t_index + 1].set(new_cum_arr),
        cum_departure=carry.cum_departure.at[:, t_index + 1].set(new_cum_dep),
        demand_queue=new_dq,
        absorbed_count=new_abs,
        cum_arrival_d=carry.cum_arrival_d,
        cum_departure_d_cur=carry.cum_departure_d_cur,
        orig_out_d=carry.orig_out_d,
        prev_next_link_ids=carry.prev_next_link_ids,
        prev_dists=carry.prev_dists,
        prev_link_cost=carry.prev_link_cost,
    )

    output = StepOutput(
        inflow_rates=inflow_rates,
        outflow_rates=outflow_rates,
        demand_queue=new_dq,
        inflow_d=jnp.zeros((config.n_links, n_dests), dtype=config.dtype),
        outflow_d=jnp.zeros((config.n_links, n_dests), dtype=config.dtype),
    )
    return new_carry, output


def _scan_with_checkpoint(step_fn, init_carry, tsize, checkpoint_every):
    """Run a time-loop scan with gradient checkpointing (rematerialization).

    Splits the ``tsize``-step loop into segments of ``checkpoint_every``
    steps and wraps each segment in ``jax.checkpoint``. Reverse-mode AD
    then stores only the scan carry at segment boundaries and recomputes
    intra-segment intermediates during the backward pass. Peak residual
    memory drops from O(tsize) to O(tsize / checkpoint_every +
    checkpoint_every) per-step residuals, at the cost of roughly one
    extra forward pass during the backward sweep.

    Parameters
    ----------
    step_fn : callable
        Scan body ``(carry, t_index) -> (new_carry, output)`` where
        ``t_index`` is the absolute timestep index.
    init_carry : PyTree
        Initial scan carry.
    tsize : int
        Total number of timesteps.
    checkpoint_every : int
        Number of timesteps per checkpointed segment. Clamped to
        ``[1, tsize]``. Smaller values save more memory but increase
        recomputation overhead per segment boundary.

    Returns
    -------
    (final_carry, outputs)
        Identical to ``jax.lax.scan(step_fn, init_carry,
        jnp.arange(tsize))``.
    """
    checkpoint_every = int(checkpoint_every)
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be a positive integer")
    checkpoint_every = min(checkpoint_every, tsize)
    n_seg = tsize // checkpoint_every
    n_main = n_seg * checkpoint_every

    @jax.checkpoint
    def _segment(carry, t_start):
        t_indices = t_start + jnp.arange(checkpoint_every)
        return jax.lax.scan(step_fn, carry, t_indices)

    t_starts = jnp.arange(n_seg) * checkpoint_every
    carry, outputs = jax.lax.scan(_segment, init_carry, t_starts)
    # (n_seg, checkpoint_every, ...) -> (n_seg * checkpoint_every, ...)
    outputs = jax.tree_util.tree_map(
        lambda x: x.reshape((n_main,) + x.shape[2:]), outputs)

    if n_main < tsize:
        # Remainder steps (not checkpointed; at most checkpoint_every - 1)
        carry, out_rem = jax.lax.scan(step_fn, carry, jnp.arange(n_main, tsize))
        outputs = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate([a, b], axis=0), outputs, out_rem)

    return carry, outputs


def _reconstruct_state(final_carry, outputs, config):
    """Reconstruct full SimState from scan outputs via cumulative sum."""
    dt = config.deltat
    n_dests = max(config.n_dests, 1)

    # outputs arrays have shape (tsize, ...) from scan stacking
    cum_arrival = jnp.zeros((config.n_links, config.tsize + 1), dtype=config.dtype)
    cum_arrival = cum_arrival.at[:, 1:].set(
        jnp.cumsum(outputs.inflow_rates, axis=0).T * dt)

    cum_departure = jnp.zeros((config.n_links, config.tsize + 1), dtype=config.dtype)
    cum_departure = cum_departure.at[:, 1:].set(
        jnp.cumsum(outputs.outflow_rates, axis=0).T * dt)

    demand_queue_history = outputs.demand_queue.T  # (tsize, n_nodes) -> (n_nodes, tsize)

    cum_arrival_d = jnp.zeros((config.n_links, n_dests, config.tsize + 1), dtype=config.dtype)
    cum_arrival_d = cum_arrival_d.at[:, :, 1:].set(
        jnp.cumsum(outputs.inflow_d, axis=0).transpose(1, 2, 0) * dt)

    cum_departure_d = jnp.zeros((config.n_links, n_dests, config.tsize + 1), dtype=config.dtype)
    cum_departure_d = cum_departure_d.at[:, :, 1:].set(
        jnp.cumsum(outputs.outflow_d, axis=0).transpose(1, 2, 0) * dt)

    return SimState(
        cum_arrival=cum_arrival,
        cum_departure=cum_departure,
        demand_queue=final_carry.demand_queue,
        absorbed_count=final_carry.absorbed_count,
        demand_queue_history=demand_queue_history,
        cum_arrival_d=cum_arrival_d,
        cum_departure_d=cum_departure_d,
        prev_next_link_ids=final_carry.prev_next_link_ids,
    )


def _build_state_fwd(final_carry, outputs, config):
    """Build SimState from forward-only scan results.

    Uses cum_arrival / cum_departure directly from the carry (already
    full-history) instead of re-integrating flow rates.
    Per-destination cumulative arrays are still reconstructed from outputs.
    """
    n_dests = max(config.n_dests, 1)
    dt = config.deltat

    demand_queue_history = outputs.demand_queue.T  # (tsize, n_nodes) -> (n_nodes, tsize)

    cum_arrival_d = jnp.zeros((config.n_links, n_dests, config.tsize + 1), dtype=config.dtype)
    cum_arrival_d = cum_arrival_d.at[:, :, 1:].set(
        jnp.cumsum(outputs.inflow_d, axis=0).transpose(1, 2, 0) * dt)

    cum_departure_d = jnp.zeros((config.n_links, n_dests, config.tsize + 1), dtype=config.dtype)
    cum_departure_d = cum_departure_d.at[:, :, 1:].set(
        jnp.cumsum(outputs.outflow_d, axis=0).transpose(1, 2, 0) * dt)

    return SimState(
        cum_arrival=final_carry.cum_arrival,
        cum_departure=final_carry.cum_departure,
        demand_queue=final_carry.demand_queue,
        absorbed_count=final_carry.absorbed_count,
        demand_queue_history=demand_queue_history,
        cum_arrival_d=cum_arrival_d,
        cum_departure_d=cum_departure_d,
        prev_next_link_ids=final_carry.prev_next_link_ids,
    )


def simulate(params, config, differentiable=True, checkpoint_every=None):
    """Run full LTM simulation.

    Parameters
    ----------
    params : Params
    config : NetworkConfig
    differentiable : bool, optional
        If True (default), use windowed carry suitable for jax.grad.
        If False, use full-array carry for faster forward-only evaluation
        (not compatible with reverse-mode AD).
    checkpoint_every : int or None, optional
        If None (default), no gradient checkpointing (current behavior).
        If a positive integer, split the time loop into segments of this
        many steps and apply jax.checkpoint to each, reducing peak GPU
        memory of reverse-mode AD at the cost of extra recomputation in
        the backward pass. Smaller values save more memory. Only effective
        when ``differentiable=True``.

    Returns
    -------
    SimState
        Final simulation state.
    """
    link_state = compute_link_state(params, config)
    n_dests = max(config.n_dests, 1)

    if differentiable:
        W = config.window_size
        init_carry = ScanCarry(
            cum_arrival_w=jnp.zeros((config.n_links, W + 1), dtype=config.dtype),
            cum_departure_w=jnp.zeros((config.n_links, W + 1), dtype=config.dtype),
            demand_queue=jnp.zeros(config.n_nodes, dtype=config.dtype),
            absorbed_count=jnp.zeros(config.n_nodes, dtype=config.dtype),
            # Per-destination arrays are unused in the aggregate LTM: size-1 placeholders
            cum_arrival_full=jnp.zeros((config.n_links, 1), dtype=config.dtype),
            cum_arrival_d=jnp.zeros((config.n_links, n_dests, 1), dtype=config.dtype),
            cum_departure_d_cur=jnp.zeros((config.n_links, n_dests), dtype=config.dtype),
            orig_out_d=jnp.zeros((0, n_dests), dtype=config.dtype),
            prev_next_link_ids=jnp.full((n_dests, config.n_nodes), -1, dtype=jnp.int32),
            prev_dists=jnp.full((n_dests, config.n_nodes), 1e15, dtype=config.dtype),
            prev_link_cost=jnp.zeros(config.n_links, dtype=config.dtype),
        )
        step_fn = functools.partial(simulation_step,
                                    params=params, link_state=link_state, config=config)
        if checkpoint_every is None:
            final_carry, outputs = jax.lax.scan(
                step_fn, init_carry, jnp.arange(config.tsize))
        else:
            final_carry, outputs = _scan_with_checkpoint(
                step_fn, init_carry, config.tsize, checkpoint_every)
        return _reconstruct_state(final_carry, outputs, config)
    else:
        init_carry = FwdCarry(
            cum_arrival=jnp.zeros((config.n_links, config.tsize + 1), dtype=config.dtype),
            cum_departure=jnp.zeros((config.n_links, config.tsize + 1), dtype=config.dtype),
            demand_queue=jnp.zeros(config.n_nodes, dtype=config.dtype),
            absorbed_count=jnp.zeros(config.n_nodes, dtype=config.dtype),
            cum_arrival_d=jnp.zeros((config.n_links, n_dests, 1), dtype=config.dtype),
            cum_departure_d_cur=jnp.zeros((config.n_links, n_dests), dtype=config.dtype),
            orig_out_d=jnp.zeros((0, n_dests), dtype=config.dtype),
            prev_next_link_ids=jnp.full((n_dests, config.n_nodes), -1, dtype=jnp.int32),
            prev_dists=jnp.full((n_dests, config.n_nodes), 1e15, dtype=config.dtype),
            prev_link_cost=jnp.zeros(config.n_links, dtype=config.dtype),
        )
        step_fn = functools.partial(simulation_step_fwd,
                                    params=params, link_state=link_state, config=config)
        final_carry, outputs = jax.lax.scan(step_fn, init_carry, jnp.arange(config.tsize))
        return _build_state_fwd(final_carry, outputs, config)


# ================================================================
# Objective functions (differentiable)
# ================================================================

def total_travel_time(state, config):
    """Total travel time (s). Differentiable.

    TTT = sum of (N_U(t) - N_D(t)) * dt over all links and timesteps
        + sum of demand_queue * dt over all origins and timesteps.

    Parameters
    ----------
    state : SimState
    config : NetworkConfig

    Returns
    -------
    jnp scalar
    """
    dt = config.deltat
    # n_on_link = N_U - N_D >= 0 by model invariant (departures cannot
    # exceed arrivals).  Tiny negatives (~ -1e-5) are float32 artifacts.
    # Omitting jnp.maximum avoids the 0.5-gradient artifact at exact-zero
    # timesteps, which otherwise leaks phantom gradients to unrelated params.
    n_on_link = state.cum_arrival[:, :config.tsize] - state.cum_departure[:, :config.tsize]
    link_ttt = jnp.sum(n_on_link) * dt
    # demand_queue_history is already clamped >= 0 at each simulation step.
    queue_ttt = jnp.sum(state.demand_queue_history) * dt
    return link_ttt + queue_ttt


def trip_completed(state, config):
    """Total completed trips (veh). Differentiable.

    Parameters
    ----------
    state : SimState
    config : NetworkConfig

    Returns
    -------
    jnp scalar
    """
    return jnp.sum(state.absorbed_count)


def average_travel_time(state, config):
    """Average travel time per completed trip (s). Differentiable.

    Parameters
    ----------
    state : SimState
    config : NetworkConfig

    Returns
    -------
    jnp scalar
    """
    ttt = total_travel_time(state, config)
    completed = jnp.maximum(trip_completed(state, config), 1e-10)
    return ttt / completed


# ================================================================
# Newell state query (post-simulation, differentiable)
# ================================================================

def compute_N(t_seconds, x, link_id, state, params, config):
    """Cumulative count N(t, x) for a link using Newell's formula. Differentiable.

    Parameters
    ----------
    t_seconds : float or jnp scalar
    x : float or jnp scalar
        Position from upstream (m).
    link_id : int
    state : SimState
    params : Params
    config : NetworkConfig

    Returns
    -------
    jnp scalar
    """
    dt = config.deltat
    u = params.u[link_id]
    ls = compute_link_state(params, config)
    w = ls.w[link_id]
    d = config.link_lengths[link_id]
    kappa = params.kappa[link_id]

    idx_free = (t_seconds - x / u) / dt
    N_free = interp_1d(state.cum_arrival[link_id], idx_free)

    idx_cong = (t_seconds - (d - x) / w) / dt
    N_cong = interp_1d(state.cum_departure[link_id], idx_cong) + kappa * (d - x)

    # At a tie the derivative follows the free-flow branch (see "Derivatives at ties").
    return _tie_min(N_free, N_cong)


def invert_interp_1d(array_1d, value):
    """Find fractional index where a monotonically non-decreasing array reaches *value*.

    This is the inverse of ``interp_1d``: given a target value, find the
    fractional index *t* such that ``interp_1d(array_1d, t) ~ value``.
    Uses ``searchsorted`` for segment selection and explicit linear
    interpolation for the differentiable part.

    Parameters
    ----------
    array_1d : jnp.ndarray, (m,)
        Monotonically non-decreasing 1-D array (e.g. cumulative departures).
    value : float or jnp scalar
        Target cumulative count.

    Returns
    -------
    jnp scalar
        Fractional index (multiply by ``dt`` to obtain time in seconds).
    """
    n = array_1d.shape[0]
    # Segment selection (non-differentiable, integer index only).
    # side='right' ensures that when value exactly matches array[k],
    # frac lands at 0 (not 1), avoiding the jnp.clip upper-boundary
    # gradient cutoff.
    idx = jnp.searchsorted(array_1d, value, side='right')
    idx = jnp.clip(idx, 1, n - 1)
    lo = idx - 1
    # Differentiable linear interpolation within the segment.
    # Straight-through clip: forward value is clamped for safety,
    # but gradient flows through unclipped to avoid 0.5 attenuation
    # at exact grid-point matches (frac=0 or frac=1).
    a_lo = array_1d[lo]
    a_hi = array_1d[idx]
    slope = jnp.maximum(a_hi - a_lo, 1e-10)
    raw_frac = (value - a_lo) / slope
    frac = raw_frac + jax.lax.stop_gradient(
        jnp.clip(raw_frac, 0.0, 1.0) - raw_frac)
    result = (lo + frac).astype(array_1d.dtype)
    return result + jax.lax.stop_gradient(
        jnp.clip(result, 0.0, n - 1.0) - result)


def link_exit_time(link_id, t_enter, state, params, config):
    """Compute the time a virtual vehicle exits a link. Differentiable.

    Uses Newell's formula: t_exit = max(t_enter + d/u, invert(cum_departure, N))
    where N = cum_arrival(t_enter).

    Parameters
    ----------
    link_id : int
        Link index.
    t_enter : float or jnp scalar
        Time entering the link's upstream end (s).
    state : SimState
    params : Params
    config : NetworkConfig

    Returns
    -------
    jnp scalar
        Exit time from the link's downstream end (s).
    """
    dt = config.deltat
    d = config.link_lengths[link_id]
    u = params.u[link_id]

    # Cumulative count at upstream at t_enter
    N = interp_1d(state.cum_arrival[link_id], t_enter / dt)

    # Free-flow constraint
    t_freeflow = t_enter + d / u

    # Queuing constraint
    t_queue = invert_interp_1d(state.cum_departure[link_id], N) * dt

    # At a tie the derivative follows the free-flow branch (see "Derivatives at ties").
    return _tie_max(t_freeflow, t_queue)


def travel_time(path_link_ids, t_depart, state, params, config):
    """Compute travel time of a virtual vehicle along a fixed path. Differentiable.

    Chains ``link_exit_time`` over an ordered sequence of links.
    The path must be specified explicitly (no shortest-path search).

    Parameters
    ----------
    path_link_ids : list[int] or tuple[int]
        Ordered link indices forming the path from origin to destination.
    t_depart : float or jnp scalar
        Departure time (s).
    state : SimState
    params : Params
    config : NetworkConfig

    Returns
    -------
    jnp scalar
        Travel time (s).
    """
    t_current = jnp.asarray(t_depart, dtype=config.dtype)
    for lid in path_link_ids:
        t_current = link_exit_time(lid, t_current, state, params, config)
    return t_current - t_depart


# ================================================================
# Autodiff-compatible travel time (soft route choice)
# ================================================================

def invert_interp_batch(arrays_2d, values):
    """Find fractional indices where rows of a 2D array reach target values.

    Batch version of ``invert_interp_1d``: for each row *i*, find the
    fractional index *t_i* such that ``arrays_2d[i, t_i] ~ values[i]``.

    Parameters
    ----------
    arrays_2d : jnp.ndarray, (K, m)
        Each row is monotonically non-decreasing (e.g. cumulative departures).
    values : jnp.ndarray, (K,)
        Target cumulative counts per row.

    Returns
    -------
    jnp.ndarray, (K,)
        Fractional indices (multiply by ``dt`` to obtain time in seconds).
    """
    n = arrays_2d.shape[1]
    # Segment selection: vmap searchsorted over rows (non-differentiable part)
    idx = jax.vmap(lambda row, v: jnp.searchsorted(row, v, side='right'))(
        arrays_2d, values)
    idx = jnp.clip(idx, 1, n - 1)
    lo = idx - 1
    # Differentiable linear interpolation within the segment.
    # Straight-through clip (see invert_interp_1d for rationale).
    row_ids = jnp.arange(arrays_2d.shape[0])
    a_lo = arrays_2d[row_ids, lo]
    a_hi = arrays_2d[row_ids, idx]
    slope = jnp.maximum(a_hi - a_lo, 1e-10)
    raw_frac = (values - a_lo) / slope
    frac = raw_frac + jax.lax.stop_gradient(
        jnp.clip(raw_frac, 0.0, 1.0) - raw_frac)
    result = (lo + frac).astype(arrays_2d.dtype)
    return result + jax.lax.stop_gradient(
        jnp.clip(result, 0.0, n - 1.0) - result)


def link_exit_time_batch(link_ids, t_enters, state, params, config):
    """Compute exit times for multiple links simultaneously. Differentiable.

    Batch version of ``link_exit_time`` using vectorized operations.

    Parameters
    ----------
    link_ids : jnp.ndarray, (K,) int32
        Link indices.
    t_enters : jnp.ndarray, (K,) float
        Entry times per link (s).
    state : SimState
    params : Params
    config : NetworkConfig

    Returns
    -------
    jnp.ndarray, (K,) float
        Exit times (s).
    """
    dt = config.deltat
    d = config.link_lengths[link_ids]
    u = params.u[link_ids]

    # Cumulative count at upstream at t_enter
    N = interp_batch(state.cum_arrival[link_ids], t_enters / dt)

    # Free-flow constraint
    t_freeflow = t_enters + d / u

    # Queuing constraint
    t_queue = invert_interp_batch(state.cum_departure[link_ids], N) * dt

    # At a tie the derivative follows the free-flow branch (see "Derivatives at ties").
    return _tie_max(t_freeflow, t_queue)


def compute_link_cost_from_state(t_depart, state, params, config):
    """Compute generalized link costs from post-simulation state. Differentiable.

    Recomputes a ``LinkState`` and instantaneous travel times at
    ``t_depart``, then adds the toll component.

    Parameters
    ----------
    t_depart : float or jnp scalar
        Departure time (s).
    state : SimState
    params : Params
    config : NetworkConfig

    Returns
    -------
    link_cost : jnp.ndarray, (n_links,)
        Generalized link cost (travel time + toll, in seconds).
    link_tt : jnp.ndarray, (n_links,)
        Instantaneous link travel time (s) without toll.
    """
    link_state = compute_link_state(params, config)
    t_index = jnp.clip(
        jnp.floor(t_depart / config.deltat).astype(jnp.int32),
        0, config.tsize - 1)
    link_tt = compute_instantaneous_tt(
        t_index, state, link_state, params, config, method=config.tt_method)
    toll_step = jnp.minimum(
        t_index // config.toll_step_size, config.n_toll_steps - 1)
    current_toll = params.toll[:, toll_step]
    return link_tt + current_toll, link_tt


def travel_time_soft(origin_node, dest_node, t_depart, state, params, config,
                     temperature=None, max_hops=None):
    """Expected travel time with soft route choice. Fully differentiable.

    Propagates a probability distribution over nodes through the network
    using logit softmax routing probabilities and Newell-based congestion-
    aware link exit times.  The gradient captures both congestion effects
    and route-switching effects.

    Parameters
    ----------
    origin_node : int
        Origin node ID.
    dest_node : int
        Destination node ID.
    t_depart : float or jnp scalar
        Departure time (s).
    state : SimState
        Post-simulation state (from ``simulate``, ``simulate_duo``, etc.).
    params : Params
    config : NetworkConfig
    temperature : float or None, optional
        Logit temperature (s).  Lower values approach deterministic
        shortest-path routing.  Defaults to ``config.logit_temperature``.
    max_hops : int or None, optional
        Maximum propagation hops.  Defaults to ``config.n_nodes``.

    Returns
    -------
    jnp scalar
        Expected travel time (s).
    """
    tau = temperature if temperature is not None else config.logit_temperature
    n_hops = max_hops if max_hops is not None else config.n_nodes
    n_nodes = config.n_nodes
    max_out = config.max_out

    # --- 1. Link costs at departure time ---
    link_cost, _ = compute_link_cost_from_state(t_depart, state, params, config)

    # --- 2. Bellman-Ford shortest distances to dest_node ---
    dists, _ = bellman_ford_reverse(link_cost, dest_node, config)

    # --- 3. Logit routing probabilities at all nodes ---
    safe_outlinks = jnp.maximum(config.node_outlinks, 0)   # (n_nodes, max_out)
    outlink_valid = config.node_outlinks >= 0               # (n_nodes, max_out)
    outlink_end = config.link_end_node[safe_outlinks]       # (n_nodes, max_out)
    safe_end = jnp.maximum(outlink_end, 0)

    dist_from_end = dists[safe_end]                         # (n_nodes, max_out)
    cost_via = link_cost[safe_outlinks] + dist_from_end     # (n_nodes, max_out)

    INF = 1e15
    has_out = config.node_n_outlinks > 0                    # (n_nodes,)
    logits = jnp.where(outlink_valid, -cost_via / tau, -INF)
    logits = jnp.where(has_out[:, None], logits, 0.0)
    route_probs = jax.nn.softmax(logits, axis=-1)           # (n_nodes, max_out)

    # --- 4. Node probability propagation ---
    prob_init = jnp.zeros(n_nodes, dtype=config.dtype).at[origin_node].set(1.0)
    tw_init = jnp.zeros(n_nodes, dtype=config.dtype).at[origin_node].set(
        jnp.asarray(t_depart, dtype=config.dtype))

    # Precompute downstream node ids for scatter-add
    flat_end = safe_end.reshape(-1)                         # (n_nodes * max_out,)
    flat_valid = outlink_valid.reshape(-1)                  # (n_nodes * max_out,)
    flat_lids = safe_outlinks.reshape(-1)                   # (n_nodes * max_out,)

    def hop_fn(_, carry):
        prob, t_weighted, acc_prob, acc_time = carry

        # Expected arrival time at each node.
        # Use safe divisor: replace near-zero prob with 1.0 to avoid
        # 1/eps gradient explosion in the backward pass.
        safe_prob = jnp.where(prob > 1e-8, prob, 1.0)
        t_arr = t_weighted / safe_prob                      # (n_nodes,)

        # Batch link_exit_time for all (node, outlink) pairs
        flat_t_enter = jnp.repeat(t_arr, max_out)          # (n_nodes * max_out,)
        flat_exit = link_exit_time_batch(
            flat_lids, flat_t_enter, state, params, config)
        exit_times = flat_exit.reshape(n_nodes, max_out)    # (n_nodes, max_out)

        # Stop gradient on exit_times for near-zero-probability nodes
        # to prevent NaN from large invert_interp gradients on flat
        # cumulative curves at irrelevant (node, time) pairs.
        active = (prob > 1e-8)[:, None]                     # (n_nodes, 1)
        exit_times = jnp.where(
            active, exit_times, jax.lax.stop_gradient(exit_times))

        # Transfer probability and weighted time along each outlink
        transfer_p = prob[:, None] * route_probs            # (n_nodes, max_out)
        transfer_p = jnp.where(outlink_valid, transfer_p, 0.0)
        transfer_t = transfer_p * exit_times                # (n_nodes, max_out)

        # Scatter-add to downstream nodes
        flat_tp = transfer_p.reshape(-1)
        flat_tt = transfer_t.reshape(-1)

        new_prob = jnp.zeros(n_nodes, dtype=config.dtype).at[flat_end].add(
            jnp.where(flat_valid, flat_tp, 0.0))
        new_tw = jnp.zeros(n_nodes, dtype=config.dtype).at[flat_end].add(
            jnp.where(flat_valid, flat_tt, 0.0))

        # Absorb at destination
        dest_p = new_prob[dest_node]
        dest_t = new_tw[dest_node]
        acc_prob = acc_prob + dest_p
        acc_time = acc_time + dest_t
        new_prob = new_prob.at[dest_node].set(0.0)
        new_tw = new_tw.at[dest_node].set(0.0)

        return (new_prob, new_tw, acc_prob, acc_time)

    init = (prob_init, tw_init,
            jnp.zeros((), dtype=config.dtype), jnp.zeros((), dtype=config.dtype))
    _, _, final_acc_prob, final_acc_time = jax.lax.fori_loop(
        0, n_hops, hop_fn, init)

    return final_acc_time / jnp.maximum(final_acc_prob, 1e-30) - t_depart


def logsum_travel_time(origin_node, dest_node, t_depart, state, params, config,
                       temperature=None):
    """Expected minimum travel time via the logsum Bellman equation. Differentiable.

    Uses the logsum (log-sum-exp) formula from random utility theory::

        V(dest) = 0
        V(n) = -tau * log( sum_o exp(-(c_o + V(end_o)) / tau) )

    where ``c_o`` is the generalized cost of outlink *o*.  This is
    equivalent to "soft Bellman-Ford" and gives the expected perceived
    cost under the logit route choice model.

    Parameters
    ----------
    origin_node : int
        Origin node ID.
    dest_node : int
        Destination node ID.
    t_depart : float or jnp scalar
        Departure time (s), used to evaluate instantaneous link costs.
    state : SimState
    params : Params
    config : NetworkConfig
    temperature : float or None, optional
        Logit temperature (s).  Defaults to ``config.logit_temperature``.

    Returns
    -------
    jnp scalar
        Expected perceived travel time (s).
    """
    tau = temperature if temperature is not None else config.logit_temperature
    n_nodes = config.n_nodes
    INF = 1e15

    link_cost, _ = compute_link_cost_from_state(t_depart, state, params, config)

    safe_outlinks = jnp.maximum(config.node_outlinks, 0)   # (n_nodes, max_out)
    outlink_valid = config.node_outlinks >= 0
    outlink_end = config.link_end_node[safe_outlinks]
    safe_end = jnp.maximum(outlink_end, 0)
    has_out = config.node_n_outlinks > 0                    # (n_nodes,)

    V = jnp.full(n_nodes, INF, dtype=config.dtype).at[dest_node].set(0.0)

    def relax(V, _):
        cost_via = link_cost[safe_outlinks] + V[safe_end]   # (n_nodes, max_out)
        logits = jnp.where(outlink_valid, -cost_via / tau, -INF)
        logits = jnp.where(has_out[:, None], logits, -INF)
        # V_new(n) = -tau * logsumexp(-cost_via / tau)
        V_new = -tau * jax.nn.logsumexp(logits, axis=-1)    # (n_nodes,)
        V_new = jnp.where(has_out, V_new, INF)
        V_new = V_new.at[dest_node].set(0.0)
        return jnp.minimum(V, V_new), None

    max_iter = int(n_nodes**0.5) + 2
    V_final, _ = jax.lax.scan(relax, V, None, length=max_iter)
    return V_final[origin_node]


def travel_time_auto(origin_node, dest_node, t_depart, state, params, config):
    """Travel time along the auto-derived shortest path. Partially differentiable.

    Extracts the shortest path from a Bellman-Ford tree computed on
    instantaneous link costs at ``t_depart``, then chains
    ``link_exit_time`` along that path for a congestion-aware travel time.

    The path selection itself is **not** differentiable (discrete argmin),
    so the gradient only captures congestion effects along the fixed route,
    not route-switching effects.  For full differentiability use
    ``travel_time_soft``.

    Parameters
    ----------
    origin_node : int
        Origin node ID.
    dest_node : int
        Destination node ID.
    t_depart : float or jnp scalar
        Departure time (s).
    state : SimState
    params : Params
    config : NetworkConfig

    Returns
    -------
    jnp scalar
        Travel time (s).
    """
    link_cost, _ = compute_link_cost_from_state(t_depart, state, params, config)
    _, next_link_ids = bellman_ford_reverse(link_cost, dest_node, config)
    # Stop gradient on the discrete route choice
    next_link_ids = jax.lax.stop_gradient(next_link_ids)

    max_path = config.n_nodes

    # Extract path via fori_loop (JIT compatible)
    def extract_step(i, carry):
        path, current_node, done = carry
        lid = next_link_ids[current_node]
        safe_lid = jnp.maximum(lid, 0)
        next_nd = config.link_end_node[safe_lid]
        valid = (~done) & (lid >= 0)
        path = jnp.where(valid, path.at[i].set(lid), path)
        current_node = jnp.where(valid, next_nd, current_node)
        done = done | (~valid) | (next_nd == dest_node)
        return (path, current_node, done)

    path_init = jnp.full(max_path, -1, dtype=jnp.int32)
    path, _, _ = jax.lax.fori_loop(
        0, max_path, extract_step,
        (path_init, jnp.int32(origin_node), jnp.bool_(False)))

    # Chain link_exit_time over valid path entries
    def chain_step(i, t_current):
        lid = path[i]
        safe_lid = jnp.maximum(lid, 0)
        t_next = link_exit_time(safe_lid, t_current, state, params, config)
        return jnp.where(lid >= 0, t_next, t_current)

    t_final = jax.lax.fori_loop(0, max_path, chain_step, jnp.asarray(t_depart, dtype=config.dtype))
    return t_final - t_depart


# ================================================================
# Multi-commodity (per-destination) FIFO step shared by DUO / AoN
# ================================================================

_FLAT_TOL = 1e-6  # veh; segments of the cumulative curve narrower than this are treated as flat


def _fifo_lookup_batch(cum_agg, cum_d, values, t_max, follow_rank=True):
    """Per-destination cumulative counts at a given rank of each aggregate cumulative curve.

    JAX counterpart of ``unsim._fifo_lookup``, vectorized over rows.
    Inverts the piecewise linear curve cum_agg[i] to find tau <= t_max with cum_agg[i](tau) = values[i] and returns cum_d[i, :, tau].
    Entries of ``cum_agg`` beyond ``t_max`` are not yet written and are ignored in the search.

    Parameters
    ----------
    cum_agg : jnp.ndarray, (n, T+1)
        Aggregate cumulative curves, non-decreasing up to t_max.
    cum_d : jnp.ndarray, (T+1, n, k)
        Per-destination cumulative curves whose sum over k equals cum_agg.
        Time-major layout: the scan carry holding this array is then updated
        and scatter-added in place by XLA, in the forward and the backward pass.
        With the time axis last, XLA assigns a non-default layout to the carry and the
        backward pass copies and transposes the whole array at every timestep.
    values : jnp.ndarray, (n,)
        Rank N on each aggregate curve.
    t_max : jnp scalar int
        Largest timestep index allowed for tau.
    follow_rank : bool, optional
        If True (default), the derivative summed over destinations equals the derivative of the rank, also where the rank hits a grid value, reaches the end of the written range or lies on a flat segment (used for departures).
        If False, the derivative of the interpolation on the segment of the rank is used, and a rank on a grid value where the curve rises takes the segment before it (used for the sending cohort, whose composition only sets turning fractions).

    Returns
    -------
    jnp.ndarray, (n, k)
    """
    n, T1 = cum_agg.shape
    masked = jnp.where(jnp.arange(T1)[None, :] <= t_max, cum_agg, jnp.inf)
    j = jnp.sum(masked <= values[:, None], axis=1) - 1  # bisect_right - 1
    j = jnp.clip(j, 0, jnp.maximum(t_max - 1, 0))
    rows = jnp.arange(n)
    if not follow_rank:
        # A cohort that ends exactly at a grid value where the curve rises takes the segment before it, where it ends
        # when every demand is slightly smaller (see "Derivatives at ties"). The value is the same array entry.
        # Plateaus keep segment j: the per-destination curves can differ along a float32 plateau of the aggregate.
        j_prev = jnp.maximum(j - 1, 0)
        use_left = (j > 0) & (cum_agg[rows, j] == values) & (cum_agg[rows, j] - cum_agg[rows, j_prev] > _FLAT_TOL)
        j = jnp.where(use_left, j_prev, j)
    lo = cum_agg[rows, j]
    hi = cum_agg[rows, j + 1]
    # Segments narrower than FLAT_TOL vehicles are treated as flat: the interpolation
    # error is below FLAT_TOL, and 1/(hi - lo) in the backward pass stays finite in float32
    # (denormal-sized segments arise from underflowing logit probabilities).
    valid = (hi - lo) > _FLAT_TOL
    frac_raw = jnp.where(valid, (values - lo) / jnp.where(valid, hi - lo, 1.0), 0.0)[:, None]
    frac = jnp.clip(frac_raw, 0.0, 1.0)
    # Exact endpoints and a + frac*(b-a) form, exact where a == b (see unsim._fifo_lookup).
    # Columns j and j+1 are fetched by one gather: the VJP of each gather from the
    # full-history array materializes an array of that size per scan step.
    both = cum_d[jnp.stack([j, j + 1], axis=-1), rows[:, None], :]  # (n, 2, k)
    lo_d = both[:, 0, :]
    hi_d = both[:, 1, :]
    out = jnp.where(frac >= 1.0, hi_d, lo_d + frac * (hi_d - lo_d))
    if not follow_rank:
        # The value is out; the derivative is that of the interpolation on segment j with the unclipped fraction.
        lin = lo_d + frac_raw * (hi_d - lo_d)
        return jax.lax.stop_gradient(out) + (lin - jax.lax.stop_gradient(lin))
    # The value is out; the derivative is that of lin (see "Derivatives at ties").
    # lin interpolates on segment j with the unclipped fraction.
    # On a flat segment it adds the change of the rank, split by the composition of the segment p-1..p through which the curve reached the plateau (the last vehicles in FIFO order).
    # The derivative of lin summed over destinations is the derivative of the rank, whereas the clip of out halves it at a grid value and the endpoint and flat branches take it from hi_d or lo_d alone.
    p = jnp.maximum(jnp.sum(masked < (lo - _FLAT_TOL)[:, None], axis=1), 1)
    last = jax.lax.stop_gradient(cum_d[jnp.stack([p - 1, p], axis=-1), rows[:, None], :])  # (n, 2, k)
    rise = last[:, 1, :] - last[:, 0, :]
    total = jnp.sum(rise, axis=1, keepdims=True)
    share = jnp.where(total > _FLAT_TOL, rise / jnp.where(total > _FLAT_TOL, total, 1.0), 1.0 / rise.shape[1])
    lin = lo_d + frac_raw * (hi_d - lo_d) + jnp.where(valid[:, None], 0.0, (values - lo)[:, None] * share)
    return jax.lax.stop_gradient(out) + (lin - jax.lax.stop_gradient(lin))


def _n_hist(config):
    """Length of the per-destination arrival history kept in the carry: full (strict) or current only (coarse)."""
    return config.tsize + 1 if config.node_fifo == "strict" else 1


def _hist_indices(config, t_index):
    """Columns of the per-destination arrival history for the current and next time."""
    if config.node_fifo == "strict":
        return t_index, t_index + 1
    return 0, 0


def _origin_ids(config):
    """Indices of origin-type nodes as a static int32 array."""
    return jnp.array(np.nonzero(np.asarray(config.node_type) == 0)[0], dtype=jnp.int32)


def _origin_cumulative_demand(params, config, origin_ids):
    """Cumulative OD demand curves of the origin nodes.

    N_dem^s(t) = dt * sum_{t' < t} od_demand_rate[o, s, t'], the demand that has arrived at the origin's vertical queue by timestep t.

    Returns
    -------
    od_cum : jnp.ndarray, (tsize+1, n_origins, n_dests)
        Time-major, see ``_fifo_lookup_batch``.
    od_cum_agg : jnp.ndarray, (n_origins, tsize+1)
    """
    od = params.od_demand_rate[origin_ids, :config.n_dests, :].transpose(2, 0, 1)  # (tsize, n_origins, n_dests)
    od_cum = jnp.concatenate(
        [jnp.zeros((1, od.shape[1], od.shape[2]), dtype=config.dtype),
         jnp.cumsum(od, axis=0) * config.deltat], axis=0)
    return od_cum, jnp.sum(od_cum, axis=2).T


def _duo_commodity_step(t_index, demands, supplies, state, cum_dep_cur, cum_arr_cur, ca_d_cur, params,
                        config, route_match, cum_arrival_full, cum_arrival_d,
                        cum_departure_d_cur, orig_out_d, od_cum, od_cum_agg, origin_ids):
    """Node models and per-destination transfer of one DUO timestep.

    Mirrors Steps 4-7 of ``World._exec_duo``:
    turning fractions from the destination composition of each inlink's sending cohort,
    aggregate node models, per-destination departures on links and on the origin vertical queues,
    and routing of the per-destination outflows to the downstream links.
    With ``config.node_fifo == "strict"`` departures follow FIFO, N_D^s(t+1) = N_U^s(tau) with N_U(tau) = N_D(t+1);
    with ``"coarse"`` they are proportional to the composition of all vehicles that have entered the link,
    turning fractions use the vehicles on the inlinks (shared by all inlinks), and origin queues depart in proportion to the queued composition.

    Parameters
    ----------
    t_index : jnp scalar int
        Absolute timestep index.
    demands, supplies : jnp.ndarray, (n_links,)
    state : SimState
        View used by ``compute_node_transfers`` (demand queue and absorbed counts).
    cum_dep_cur : jnp.ndarray, (n_links,)
        Aggregate cumulative departures at t_index.
    cum_arr_cur : jnp.ndarray, (n_links,)
        Aggregate cumulative arrivals at t_index.
    ca_d_cur : jnp.ndarray, (n_links, n_dests)
        Per-destination cumulative arrivals at t_index.
    params : Params
        ``demand_rate`` must be the per-origin sum of ``od_demand_rate``.
    config : NetworkConfig
    route_match : jnp.ndarray, (n_dests, n_nodes, max_out)
        Fraction of destination-d flow leaving each node through each outlink.
    cum_arrival_full : jnp.ndarray, (n_links, tsize+1)
    cum_arrival_d : jnp.ndarray, (tsize+1, n_links, n_dests)
    cum_departure_d_cur : jnp.ndarray, (n_links, n_dests)
    orig_out_d : jnp.ndarray, (n_origins, n_dests)
    od_cum : jnp.ndarray, (tsize+1, n_origins, n_dests)
    od_cum_agg : jnp.ndarray, (n_origins, tsize+1)
    origin_ids : jnp.ndarray, (n_origins,) int32

    Returns
    -------
    inflow_d : jnp.ndarray, (n_links, n_dests)
    outflow_d : jnp.ndarray, (n_links, n_dests)
    outflows : jnp.ndarray, (n_links,)
        Aggregate outflow rates from the node models.
    new_dq : jnp.ndarray, (n_nodes,)
    new_absorbed : jnp.ndarray, (n_nodes,)
    new_cum_departure_d_cur : jnp.ndarray, (n_links, n_dests)
    new_orig_out_d : jnp.ndarray, (n_origins, n_dests)
    """
    dt = config.deltat
    n_links = config.n_links
    n_nodes = config.n_nodes
    n_dests = config.n_dests

    safe_inlinks = jnp.maximum(config.node_inlinks, 0)  # (n_nodes, max_in)
    inlink_valid = config.node_inlinks >= 0
    safe_outlinks = jnp.maximum(config.node_outlinks, 0)  # (n_nodes, max_out)
    outlink_valid = config.node_outlinks >= 0
    n_out = config.node_n_outlinks[:, None]
    equal_split = jnp.where(
        jnp.arange(config.max_out)[None, :] < n_out,
        1.0 / jnp.maximum(n_out, 1), 0.0)  # (n_nodes, max_out)

    strict = config.node_fifo == "strict"
    if strict:
        # Sending cohort composition of each link: the vehicles that can leave in this step,
        # ranks N_D(t) to N_D(t) + D*dt: S^s = N_U^s(tau_D) - N_D^s(t), N_U(tau_D) = N_D(t) + D*dt
        # The cohort is non-negative by FIFO; the clamp passes the derivative through (see "Derivatives at ties").
        send_d = _nonneg(
            _fifo_lookup_batch(cum_arrival_full, cum_arrival_d, cum_dep_cur + dt * demands, t_index,
                               follow_rank=False)
            - cum_departure_d_cur)
    else:
        # coarse: composition of all vehicles on the link
        send_d = _nonneg(ca_d_cur - cum_departure_d_cur)
    # Origin vertical queues: queued vehicles plus the demand of this step
    send_orig = _nonneg(od_cum[t_index + 1] - orig_out_d)  # (n_origins, n_dests)
    send_orig_dense = jnp.zeros((n_nodes, n_dests), dtype=config.dtype).at[origin_ids].set(send_orig)

    # Per-inlink turning fractions and aggregate diverge ratios
    inlink_send = send_d[safe_inlinks] * inlink_valid[:, :, None]  # (n_nodes, max_in, n_dests)
    to_out = jnp.einsum("nid,dno->nio", inlink_send, route_match)  # (n_nodes, max_in, max_out)
    agg_to_out = jnp.sum(to_out, axis=1) + jnp.einsum("nd,dno->no", send_orig_dense, route_match)
    total = jnp.sum(agg_to_out, axis=1, keepdims=True)
    duo_diverge = jnp.where(total > 1e-10, agg_to_out / jnp.maximum(total, 1e-10), equal_split)
    if strict:
        total_i = jnp.sum(to_out, axis=2, keepdims=True)
        duo_tf = jnp.where(total_i > 1e-10, to_out / jnp.maximum(total_i, 1e-10),
                           equal_split[:, None, :])
    else:
        # coarse: all inlinks share the node-level ratios
        duo_tf = jnp.broadcast_to(duo_diverge[:, None, :], (n_nodes, config.max_in, config.max_out))
    params_duo = params._replace(diverge_ratios=duo_diverge, turning_fractions=duo_tf)

    # Aggregate node models
    inflows, outflows, new_dq, _ = compute_node_transfers(
        t_index, demands, supplies, state, params_duo, config)

    # Per-destination departures of links
    if strict:
        N_D_new = cum_dep_cur + dt * outflows
        cd_new = _fifo_lookup_batch(cum_arrival_full, cum_arrival_d, N_D_new, t_index)
        outflow_d = (cd_new - cum_departure_d_cur) / dt
    else:
        ratio = jnp.where(cum_arr_cur[:, None] > 1e-10,
                          ca_d_cur / jnp.maximum(cum_arr_cur[:, None], 1e-10), 0.0)
        outflow_d = outflows[:, None] * ratio
        cd_new = cum_departure_d_cur + dt * outflow_d

    # Per-destination departures of the origin vertical queues
    orig_flow = jnp.sum(inflows[safe_outlinks[origin_ids]] * outlink_valid[origin_ids], axis=1)
    if strict:
        N_out_new = jnp.sum(orig_out_d, axis=1) + dt * orig_flow
        out_new = _fifo_lookup_batch(od_cum_agg, od_cum, N_out_new, t_index + 1)
    else:
        send_total = jnp.sum(send_orig, axis=1, keepdims=True)
        share = jnp.where(send_total > 1e-15, send_orig / jnp.maximum(send_total, 1e-15), 0.0)
        # Empty queue and no demand in this step (send_total = 0): orig_flow * share would drop the derivative of orig_flow (share = 0).
        # The departures stay 0 and take the derivative of the queue, i.e. the whole queue departs, as in the origin model whose ties follow the uncongested branch (see "Derivatives at ties").
        send_raw = od_cum[t_index + 1] - orig_out_d
        out_new = orig_out_d + jnp.where(send_total > 1e-15, dt * orig_flow[:, None] * share,
                                         send_raw - jax.lax.stop_gradient(send_raw))
    orig_out_rate = (out_new - orig_out_d) / dt  # (n_origins, n_dests)

    # Route per-destination flows to the downstream links
    inlink_outflow_d = outflow_d[safe_inlinks] * inlink_valid[:, :, None]  # (n_nodes, max_in, n_dests)
    node_arriving = jnp.sum(inlink_outflow_d, axis=1).at[origin_ids].add(orig_out_rate)

    dest_mask = (config.dest_node_ids[None, :] == jnp.arange(n_nodes)[:, None])  # (n_nodes, n_dests)
    absorb_per_node = jnp.sum(jnp.where(dest_mask, node_arriving, 0.0), axis=1)
    new_absorbed = state.absorbed_count + absorb_per_node * dt
    node_arriving = jnp.where(dest_mask, 0.0, node_arriving)

    routed = node_arriving[:, :, None] * route_match.transpose(1, 0, 2)  # (n_nodes, n_dests, max_out)
    flat_routed = routed.transpose(0, 2, 1).reshape(-1, n_dests) * outlink_valid.reshape(-1)[:, None]
    inflow_d = jnp.zeros((n_links, n_dests), dtype=config.dtype).at[safe_outlinks.reshape(-1)].add(flat_routed)

    return inflow_d, outflow_d, outflows, new_dq, new_absorbed, cd_new, out_new


# ================================================================
# AON (All-or-Nothing) -> fixed route simulation
# ================================================================

def _aon_simulation_step_fwd(carry, t_index, params, link_state, config,
                              route_match, origin_ids, od_cum, od_cum_agg):
    """One AoN timestep for forward-only jax.lax.scan (minimal carry).

    Uses precomputed ``route_match`` instead of running Bellman-Ford or
    computing travel times at each step.  Does not output per-destination
    arrays.

    Parameters
    ----------
    carry : AoNFwdCarry
    t_index : jnp scalar int (absolute timestep index)
    params : Params
        Slim params with od_demand_rate zeroed out and demand_rate set to the per-origin OD demand sum.
    link_state : LinkState
    config : NetworkConfig
    route_match : jnp.ndarray, (n_dests, n_nodes, max_out)
        Precomputed routing indicator (hard 0/1 for AoN).
    origin_ids : jnp.ndarray, (n_origins,) int32
        Indices of origin nodes.
    od_cum : jnp.ndarray, (tsize+1, n_origins, n_dests)
    od_cum_agg : jnp.ndarray, (n_origins, tsize+1)
        Cumulative OD demand curves from ``_origin_cumulative_demand``.

    Returns
    -------
    (AoNFwdCarry, None)
    """
    dt = config.deltat
    n_links = config.n_links
    n_nodes = config.n_nodes
    n_dests = config.n_dests

    # Build SimState view backed by full-history arrays
    state = SimState(
        cum_arrival=carry.cum_arrival,
        cum_departure=carry.cum_departure,
        demand_queue=carry.demand_queue,
        absorbed_count=carry.absorbed_count,
        demand_queue_history=jnp.zeros((n_nodes, 1)),
        cum_arrival_d=jnp.zeros((n_links, n_dests, 1)),
        cum_departure_d=jnp.zeros((n_links, n_dests, 1)),
        prev_next_link_ids=jnp.zeros((n_dests, n_nodes), dtype=jnp.int32),
    )

    demands = compute_demands(t_index, state, link_state, params, config)
    supplies = compute_supplies(t_index, state, link_state, params, config)

    ht, ht1 = _hist_indices(config, t_index)
    ca_d_cur = carry.cum_arrival_d[ht]
    inflow_d, outflow_d, outflows, new_dq, new_absorbed, cd_new, out_new = _duo_commodity_step(
        t_index, demands, supplies, state, carry.cum_departure[:, t_index],
        carry.cum_arrival[:, t_index], ca_d_cur, params,
        config, route_match, carry.cum_arrival, carry.cum_arrival_d,
        carry.cum_departure_d_cur, carry.orig_out_d, od_cum, od_cum_agg, origin_ids)

    new_cum_arr_d = ca_d_cur + dt * inflow_d
    new_cum_arr = carry.cum_arrival[:, t_index] + dt * jnp.sum(inflow_d, axis=1)
    new_cum_dep = carry.cum_departure[:, t_index] + dt * outflows

    new_carry = AoNFwdCarry(
        cum_arrival=carry.cum_arrival.at[:, t_index + 1].set(new_cum_arr),
        cum_departure=carry.cum_departure.at[:, t_index + 1].set(new_cum_dep),
        demand_queue=new_dq,
        absorbed_count=new_absorbed,
        cum_arrival_d=carry.cum_arrival_d.at[ht1].set(new_cum_arr_d),
        cum_departure_d_cur=cd_new,
        orig_out_d=out_new,
        demand_queue_history=carry.demand_queue_history.at[:, t_index].set(new_dq),
    )

    # No per-step output -- everything is in the carry
    return new_carry, None


def _build_state_aon_fwd(final_carry, config, next_link_ids):
    """Build SimState from AoN forward-only scan results.

    Per-destination cumulative arrays are set to minimal dummies since
    they are not needed for total_travel_time / trip_completed and
    would consume O(n_links * n_dests * tsize) memory.

    Parameters
    ----------
    final_carry : AoNFwdCarry
    config : NetworkConfig
    next_link_ids : jnp.ndarray, (n_dests, n_nodes)

    Returns
    -------
    SimState
    """
    n_dests = max(config.n_dests, 1)

    return SimState(
        cum_arrival=final_carry.cum_arrival,
        cum_departure=final_carry.cum_departure,
        demand_queue=final_carry.demand_queue,
        absorbed_count=final_carry.absorbed_count,
        demand_queue_history=final_carry.demand_queue_history,
        cum_arrival_d=jnp.zeros((config.n_links, n_dests, 1), dtype=config.dtype),
        cum_departure_d=jnp.zeros((config.n_links, n_dests, 1), dtype=config.dtype),
        prev_next_link_ids=next_link_ids,
    )


def simulate_aon(params, config, differentiable=True, checkpoint_every=None):
    """Run AON simulation with routes fixed at free-flow shortest paths.

    Per-destination tracking is maintained internally so that traffic is
    correctly absorbed at destination nodes.

    When ``differentiable=False``, uses a forward-only path that
    eliminates per-destination arrays from scan outputs. The carry still
    holds the per-destination cumulative arrival history
    (tsize+1 x n_links x n_dests), which the FIFO inverse mapping needs.
    The returned SimState has minimal dummy arrays for
    ``cum_arrival_d`` / ``cum_departure_d``.

    When ``differentiable=True``, falls back to ``simulate_duo`` with
    routes frozen at t=0.

    Parameters
    ----------
    params : Params
    config : NetworkConfig
    differentiable : bool, optional
        If True (default), supports jax.grad w.r.t. params.
        If False, use memory-optimized forward-only evaluation.
    checkpoint_every : int or None, optional
        Gradient checkpointing segment length passed to ``simulate_duo``
        when ``differentiable=True``. See ``simulate_duo``. Ignored when
        ``differentiable=False``.

    Returns
    -------
    SimState
    """
    if differentiable:
        # Fall back to DUO with routes frozen at t=0.
        config_aon = config._replace(route_update_interval=config.tsize + 1)
        return simulate_duo(params, config_aon, differentiable=True,
                            checkpoint_every=checkpoint_every)

    # --- Memory-optimized forward-only path ---
    link_state = compute_link_state(params, config)
    n_dests = config.n_dests

    # Precompute free-flow travel times and run Bellman-Ford once
    ff_tt = config.link_lengths / params.u  # free-flow travel time
    _, bf_preds = bellman_ford_all_dests(ff_tt, config)
    next_link_ids = bf_preds  # (n_dests, n_nodes) int32

    # Precompute route_match (static for entire simulation)
    safe_outlinks = jnp.maximum(config.node_outlinks, 0)  # (n_nodes, max_out)
    outlink_valid = config.node_outlinks >= 0  # (n_nodes, max_out)
    route_match = (
        (next_link_ids[:, :, None] == safe_outlinks[None, :, :])
        & outlink_valid[None, :, :]
    ).astype(config.dtype)  # (n_dests, n_nodes, max_out)

    # Compact cumulative OD demand: only origin nodes to save memory
    origin_ids = _origin_ids(config)
    od_cum, od_cum_agg = _origin_cumulative_demand(params, config, origin_ids)

    # Slim params: od_demand_rate zeroed to free memory, demand_rate = per-origin OD demand sum
    slim_params = params._replace(
        demand_rate=jnp.sum(params.od_demand_rate[:, :n_dests, :], axis=1),
        od_demand_rate=jnp.zeros((config.n_nodes, 1, 1), dtype=config.dtype))

    # Initialize minimal carry (demand_queue_history in carry, not output)
    init_carry = AoNFwdCarry(
        cum_arrival=jnp.zeros((config.n_links, config.tsize + 1),
                               dtype=config.dtype),
        cum_departure=jnp.zeros((config.n_links, config.tsize + 1),
                                 dtype=config.dtype),
        demand_queue=jnp.zeros(config.n_nodes, dtype=config.dtype),
        absorbed_count=jnp.zeros(config.n_nodes, dtype=config.dtype),
        cum_arrival_d=jnp.zeros((_n_hist(config), config.n_links, n_dests),
                                 dtype=config.dtype),
        cum_departure_d_cur=jnp.zeros((config.n_links, n_dests),
                                       dtype=config.dtype),
        orig_out_d=jnp.zeros((origin_ids.shape[0], n_dests), dtype=config.dtype),
        demand_queue_history=jnp.zeros((config.n_nodes, config.tsize),
                                        dtype=config.dtype),
    )

    step_fn = functools.partial(
        _aon_simulation_step_fwd,
        params=slim_params, link_state=link_state, config=config,
        route_match=route_match, origin_ids=origin_ids,
        od_cum=od_cum, od_cum_agg=od_cum_agg)
    final_carry, _ = jax.lax.scan(step_fn, init_carry, jnp.arange(config.tsize))

    return _build_state_aon_fwd(final_carry, config, next_link_ids)


# ================================================================
# DUO (Dynamic User Optimal) -- JAX differentiable
# ================================================================

def _compute_tt_avg_density(t_index, state, link_state, params, config):
    """Average-density method: k_avg = (N_U-N_D)/L -> FD -> tau = L/v."""
    n_on_link = state.cum_arrival[:, t_index] - state.cum_departure[:, t_index]
    k_avg = _nonneg(n_on_link) / config.link_lengths
    k_star = link_state.q_star / params.u
    # Use k_safe to prevent division-by-zero in backward pass.
    # jnp.where evaluates both branches, so 0 * grad(w*kappa/0) = NaN.
    k_safe = jnp.maximum(k_avg, 1e-10)
    v_cong = link_state.w * (params.kappa - k_safe) / k_safe
    v = jnp.where(k_avg <= k_star, params.u, v_cong)
    v = jnp.clip(v, 0.01, params.u)
    return config.link_lengths / v


def _compute_tt_multipoint(t_index, state, link_state, params, config,
                            n_segments=10):
    """Multi-point spatial integration: tau = sum(dx / v_i) from N(t,x) curves."""
    L = config.link_lengths
    dx = L / n_segments
    k_star = link_state.q_star / params.u

    total_tt = jnp.zeros(config.n_links, dtype=config.dtype)

    for i in range(n_segments):
        frac0 = i / n_segments
        frac1 = (i + 1) / n_segments

        t_free0 = t_index * 1.0 - frac0 * link_state.offset_u
        N_free0 = interp_batch(state.cum_arrival, jnp.clip(t_free0, 0.0, config.tsize))
        t_cong0 = t_index * 1.0 - (1.0 - frac0) * link_state.offset_w
        N_cong0 = interp_batch(state.cum_departure, jnp.clip(t_cong0, 0.0, config.tsize)) + params.kappa * L * (1.0 - frac0)
        N0 = _tie_min(N_free0, N_cong0)

        t_free1 = t_index * 1.0 - frac1 * link_state.offset_u
        N_free1 = interp_batch(state.cum_arrival, jnp.clip(t_free1, 0.0, config.tsize))
        t_cong1 = t_index * 1.0 - (1.0 - frac1) * link_state.offset_w
        N_cong1 = interp_batch(state.cum_departure, jnp.clip(t_cong1, 0.0, config.tsize)) + params.kappa * L * (1.0 - frac1)
        N1 = _tie_min(N_free1, N_cong1)

        k_seg = _nonneg((N0 - N1) / dx)

        is_free = k_seg <= k_star
        v_cong = jnp.where(k_seg > 1e-10,
                            link_state.w * (params.kappa - k_seg) / k_seg, params.u)
        v = jnp.where(is_free, params.u, v_cong)
        v = jnp.clip(v, 0.01, params.u)

        total_tt = total_tt + dx / v

    return total_tt


def compute_instantaneous_tt(t_index, state, link_state, params, config,
                              method="multipoint"):
    """Compute instantaneous travel time for all links.

    Parameters
    ----------
    t_index : int
    state : SimState
    link_state : LinkState
    params : Params
    config : NetworkConfig
    method : str
        "avg_density" -- k_avg=(N_U-N_D)/L -> FD -> tau=L/v.
        "multipoint" -- spatial integration with N(t,x) curves (default).

    Returns
    -------
    jnp.ndarray, (n_links,)
        Instantaneous travel time per link (s).
    """
    if method == "avg_density":
        return _compute_tt_avg_density(t_index, state, link_state, params, config)
    return _compute_tt_multipoint(t_index, state, link_state, params, config)


def bellman_ford_reverse(link_tt, dest_id, config):
    """Reverse Bellman-Ford from a single destination. Fully vectorized JAX.

    Uses scatter-min for parallel edge relaxation (no sequential loops).

    Parameters
    ----------
    link_tt : jnp.ndarray, (n_links,)
        Link travel times.
    dest_id : int
        Destination node ID.
    config : NetworkConfig

    Returns
    -------
    dist : jnp.ndarray, (n_nodes,) float
        Shortest distance from each node to destination.
    next_link_id : jnp.ndarray, (n_nodes,) int32
        For each node, the link ID to take toward dest. -1 if unreachable.
    """
    INF = 1e15
    dist = jnp.full(config.n_nodes, INF, dtype=config.dtype).at[dest_id].set(0.0)
    pred = jnp.full(config.n_nodes, -1, dtype=jnp.int32)

    edge_from = config.link_start_node  # (n_links,)
    edge_to = config.link_end_node      # (n_links,)
    link_ids = jnp.arange(config.n_links, dtype=jnp.int32)

    def relax_iteration(carry, _):
        dist, pred = carry
        # All candidate new distances via reverse edges (vectorized)
        new_dist = dist[edge_to] + link_tt  # (n_links,)

        # Scatter-min: for each from_node, find minimum new_dist across all its edges
        best_dist = jnp.full(config.n_nodes, INF, dtype=config.dtype).at[edge_from].min(new_dist)

        # Which nodes improved?
        improved = best_dist < dist

        # Update pred: for improved nodes, pick an edge that achieved the min
        # achieved[e] = this edge matches the best distance for its from_node
        achieved = (new_dist <= best_dist[edge_from] + 1e-12) & improved[edge_from]
        # Scatter link_ids where achieved; for multiple achievers, any one is correct
        # Use large sentinel for non-achieved so .min picks the real one
        pred_candidates = jnp.where(achieved, link_ids, config.n_links)
        best_pred = jnp.full(config.n_nodes, config.n_links, dtype=jnp.int32)
        best_pred = best_pred.at[edge_from].min(pred_candidates)
        best_pred = jnp.where(best_pred >= config.n_links, -1, best_pred)
        pred = jnp.where(improved, best_pred, pred)

        # Update dist
        dist = jnp.where(improved, best_dist, dist)

        return (dist, pred), None

    # Road networks are roughly planar: diameter ~ sqrt(n), so limit iterations
    # (same bound as World._duo_shortest_trees)
    max_iter = int(config.n_nodes**0.5) + 10
    (dist, pred), _ = jax.lax.scan(relax_iteration, (dist, pred), None, length=max_iter)
    return dist, pred


def bellman_ford_all_dests(link_tt, config):
    """Compute shortest path trees for all destinations.

    Parameters
    ----------
    link_tt : jnp.ndarray, (n_links,)
    config : NetworkConfig

    Returns
    -------
    dists : jnp.ndarray, (n_dests, n_nodes) float
        Shortest distances from each node to each destination.
    next_link_ids : jnp.ndarray, (n_dests, n_nodes) int32
    """
    def bf_single(dest_id):
        return bellman_ford_reverse(link_tt, dest_id, config)

    return jax.vmap(bf_single)(config.dest_node_ids)


def duo_simulation_step(carry, t_index, params, link_state, config,
                        origin_ids, od_cum, od_cum_agg):
    """One DUO timestep for jax.lax.scan (windowed carry).

    Parameters
    ----------
    carry : ScanCarry
    t_index : jnp scalar int (absolute timestep index)
    params : Params
        ``demand_rate`` must be the per-origin sum of ``od_demand_rate``.
    link_state : LinkState
    config : NetworkConfig
    origin_ids : jnp.ndarray, (n_origins,) int32
    od_cum : jnp.ndarray, (tsize+1, n_origins, n_dests)
    od_cum_agg : jnp.ndarray, (n_origins, tsize+1)
        Cumulative OD demand curves from ``_origin_cumulative_demand``.

    Returns
    -------
    (new_carry, StepOutput)
    """
    dt = config.deltat
    W = config.window_size

    # Build fake SimState backed by the sliding window
    fake_state = _make_fake_state(carry, config)

    # Step 1: LTM demands/supplies (W = "current time" in window coords)
    demands = compute_demands(W, fake_state, link_state, params, config)
    supplies = compute_supplies(W, fake_state, link_state, params, config)

    # Step 2: Instantaneous travel times (window-relative)
    link_tt = compute_instantaneous_tt(W, fake_state, link_state, params, config,
                                       method=config.tt_method)

    # Step 2b: Add congestion pricing toll for generalized cost
    toll_step = jnp.minimum(
        t_index // config.toll_step_size,
        config.n_toll_steps - 1
    )
    current_toll = params.toll[:, toll_step]
    link_cost = link_tt + current_toll

    # Step 3: Bellman-Ford shortest paths (recomputed every route_update_interval steps)
    should_update = (t_index % config.route_update_interval) == 0
    bf_dists, bf_preds, route_cost = jax.lax.cond(
        should_update,
        lambda _: bellman_ford_all_dests(link_cost, config) + (link_cost,),
        lambda _: (carry.prev_dists, carry.prev_next_link_ids, carry.prev_link_cost),
        None,
    )
    next_link_ids = bf_preds  # (n_dests, n_nodes) int32
    dists = bf_dists           # (n_dests, n_nodes) float
    if not config.use_logit:
        # Stop gradient on distances when not used for logit routing
        dists = jax.lax.stop_gradient(dists)

    # Step 4: Routing matrices
    route_match = _duo_route_match(next_link_ids, dists, route_cost, params, config)

    # Steps 5-7: Node models, FIFO departures by destination, routing
    ht, ht1 = _hist_indices(config, t_index)
    ca_d_cur = carry.cum_arrival_d[ht]
    inflow_d, outflow_d, outflows, new_dq, new_absorbed, cd_new, out_new = _duo_commodity_step(
        t_index, demands, supplies, fake_state, carry.cum_departure_w[:, -1],
        carry.cum_arrival_w[:, -1], ca_d_cur, params,
        config, route_match, carry.cum_arrival_full, carry.cum_arrival_d,
        carry.cum_departure_d_cur, carry.orig_out_d, od_cum, od_cum_agg, origin_ids)

    # Step 8: Update cumulative counts (sliding window and full history)
    new_cum_arr_d = ca_d_cur + dt * inflow_d
    new_cum_arr = carry.cum_arrival_w[:, -1] + dt * jnp.sum(inflow_d, axis=1)
    new_cum_dep = carry.cum_departure_w[:, -1] + dt * outflows

    new_arr_w = jnp.concatenate(
        [carry.cum_arrival_w[:, 1:], new_cum_arr[:, None]], axis=1)
    new_dep_w = jnp.concatenate(
        [carry.cum_departure_w[:, 1:], new_cum_dep[:, None]], axis=1)

    new_carry = ScanCarry(
        cum_arrival_w=new_arr_w,
        cum_departure_w=new_dep_w,
        demand_queue=new_dq,
        absorbed_count=new_absorbed,
        cum_arrival_full=carry.cum_arrival_full.at[:, ht1].set(new_cum_arr),
        cum_arrival_d=carry.cum_arrival_d.at[ht1].set(new_cum_arr_d),
        cum_departure_d_cur=cd_new,
        orig_out_d=out_new,
        prev_next_link_ids=next_link_ids,
        prev_dists=dists,
        prev_link_cost=route_cost,
    )

    output = StepOutput(
        inflow_rates=jnp.sum(inflow_d, axis=1),
        outflow_rates=outflows,
        demand_queue=new_dq,
        inflow_d=inflow_d,
        outflow_d=outflow_d,
    )
    return new_carry, output


def _duo_route_match(next_link_ids, dists, link_cost, params, config):
    """Routing matrices route_match[d, n, o]: fraction of destination-d flow leaving node n through outlink o.

    Hard routing follows the shortest path trees. Logit routing uses softmax of
    -(link cost + distance from the outlink's end node) / temperature plus ``params.route_bias``.

    Parameters
    ----------
    next_link_ids : jnp.ndarray, (n_dests, n_nodes) int32
    dists : jnp.ndarray, (n_dests, n_nodes)
    link_cost : jnp.ndarray, (n_links,)
    params : Params
    config : NetworkConfig

    Returns
    -------
    jnp.ndarray, (n_dests, n_nodes, max_out)
    """
    safe_outlinks = jnp.maximum(config.node_outlinks, 0)  # (n_nodes, max_out)
    outlink_valid = config.node_outlinks >= 0  # (n_nodes, max_out)

    if config.use_logit:
        # Soft logit routing: cost_via_outlink = link_cost + dist_to_dest_from_end_node
        outlink_end_nodes = config.link_end_node[safe_outlinks]  # (n_nodes, max_out)
        safe_end_nodes = jnp.maximum(outlink_end_nodes, 0)
        dist_from_end = dists[:, safe_end_nodes]  # (n_dests, n_nodes, max_out)
        cost_via = link_cost[safe_outlinks][None, :, :] + dist_from_end  # (n_dests, n_nodes, max_out)
        # Mask invalid outlinks; use 0 logit for nodes with no outlinks
        # to avoid softmax([-inf,...]) = NaN
        INF = 1e15
        cost_via = jnp.where(outlink_valid[None, :, :], cost_via, INF)
        logits = -cost_via / config.logit_temperature \
            + params.route_bias[:, :, :cost_via.shape[2]]
        has_outlinks = config.node_n_outlinks > 0  # (n_nodes,)
        logits = jnp.where(
            ~has_outlinks[None, :, None],
            0.0,  # no outlinks: uniform (value irrelevant, absorbed before routing)
            jnp.where(outlink_valid[None, :, :], logits, -INF))
        return jax.nn.softmax(logits, axis=-1)  # (n_dests, n_nodes, max_out)

    # Hard routing: route_match[d, n, o] = (next_link_ids[d, n] == outlinks[n, o])
    route_match = (next_link_ids[:, :, None] == safe_outlinks[None, :, :]) & outlink_valid[None, :, :]
    return route_match.astype(config.dtype)


def duo_simulation_step_fwd(carry, t_index, params, link_state, config,
                            origin_ids, od_cum, od_cum_agg):
    """One DUO timestep for forward-only jax.lax.scan (full-array carry).

    Uses absolute ``t_index`` into full-history arrays instead of a sliding
    window.  Otherwise identical to ``duo_simulation_step``.

    Parameters
    ----------
    carry : FwdCarry
    t_index : jnp scalar int (absolute timestep index)
    params : Params
        ``demand_rate`` must be the per-origin sum of ``od_demand_rate``.
    link_state : LinkState
    config : NetworkConfig
    origin_ids : jnp.ndarray, (n_origins,) int32
    od_cum : jnp.ndarray, (tsize+1, n_origins, n_dests)
    od_cum_agg : jnp.ndarray, (n_origins, tsize+1)
        Cumulative OD demand curves from ``_origin_cumulative_demand``.

    Returns
    -------
    (new_carry, StepOutput)
    """
    dt = config.deltat

    # Build SimState view backed by full-history arrays
    state = _make_state_from_fwd_carry(carry, config)

    # Step 1: LTM demands/supplies (absolute t_index)
    demands = compute_demands(t_index, state, link_state, params, config)
    supplies = compute_supplies(t_index, state, link_state, params, config)

    # Step 2: Instantaneous travel times (absolute t_index)
    link_tt = compute_instantaneous_tt(t_index, state, link_state, params, config,
                                       method=config.tt_method)

    # Step 2b: Add congestion pricing toll for generalized cost
    toll_step = jnp.minimum(
        t_index // config.toll_step_size,
        config.n_toll_steps - 1
    )
    current_toll = params.toll[:, toll_step]
    link_cost = link_tt + current_toll

    # Step 3: Bellman-Ford shortest paths (recomputed every route_update_interval steps)
    should_update = (t_index % config.route_update_interval) == 0
    bf_dists, bf_preds, route_cost = jax.lax.cond(
        should_update,
        lambda _: bellman_ford_all_dests(link_cost, config) + (link_cost,),
        lambda _: (carry.prev_dists, carry.prev_next_link_ids, carry.prev_link_cost),
        None,
    )
    next_link_ids = bf_preds  # (n_dests, n_nodes) int32
    dists = bf_dists           # (n_dests, n_nodes) float
    if not config.use_logit:
        dists = jax.lax.stop_gradient(dists)

    # Step 4: Routing matrices
    route_match = _duo_route_match(next_link_ids, dists, route_cost, params, config)

    # Steps 5-7: Node models, FIFO departures by destination, routing
    ht, ht1 = _hist_indices(config, t_index)
    ca_d_cur = carry.cum_arrival_d[ht]
    inflow_d, outflow_d, outflows, new_dq, new_absorbed, cd_new, out_new = _duo_commodity_step(
        t_index, demands, supplies, state, carry.cum_departure[:, t_index],
        carry.cum_arrival[:, t_index], ca_d_cur, params,
        config, route_match, carry.cum_arrival, carry.cum_arrival_d,
        carry.cum_departure_d_cur, carry.orig_out_d, od_cum, od_cum_agg, origin_ids)

    # Step 8: Update cumulative counts at t_index + 1
    new_cum_arr_d = ca_d_cur + dt * inflow_d
    new_cum_arr = carry.cum_arrival[:, t_index] + dt * jnp.sum(inflow_d, axis=1)
    new_cum_dep = carry.cum_departure[:, t_index] + dt * outflows

    new_carry = FwdCarry(
        cum_arrival=carry.cum_arrival.at[:, t_index + 1].set(new_cum_arr),
        cum_departure=carry.cum_departure.at[:, t_index + 1].set(new_cum_dep),
        demand_queue=new_dq,
        absorbed_count=new_absorbed,
        cum_arrival_d=carry.cum_arrival_d.at[ht1].set(new_cum_arr_d),
        cum_departure_d_cur=cd_new,
        orig_out_d=out_new,
        prev_next_link_ids=next_link_ids,
        prev_dists=dists,
        prev_link_cost=route_cost,
    )

    output = StepOutput(
        inflow_rates=jnp.sum(inflow_d, axis=1),
        outflow_rates=outflows,
        demand_queue=new_dq,
        inflow_d=inflow_d,
        outflow_d=outflow_d,
    )
    return new_carry, output


def simulate_duo(params, config, differentiable=True, checkpoint_every=None):
    """Run DUO simulation.

    Parameters
    ----------
    params : Params
    config : NetworkConfig
    differentiable : bool, optional
        If True (default), use windowed carry suitable for jax.grad.
        If False, use full-array carry for faster forward-only evaluation
        (not compatible with reverse-mode AD).
    checkpoint_every : int or None, optional
        If None (default), no gradient checkpointing (current behavior).
        If a positive integer, split the time loop into segments of this
        many steps and apply jax.checkpoint to each, reducing peak GPU
        memory of reverse-mode AD at the cost of extra recomputation in
        the backward pass. Smaller values save more memory. Only effective
        when ``differentiable=True``.

    Returns
    -------
    SimState
    """
    link_state = compute_link_state(params, config)
    n_dests = config.n_dests

    # Origin vertical queues: cumulative OD demand curves; the aggregate demand
    # seen by the origin node model is the per-origin sum of the OD demand.
    origin_ids = _origin_ids(config)
    od_cum, od_cum_agg = _origin_cumulative_demand(params, config, origin_ids)
    params = params._replace(
        demand_rate=jnp.sum(params.od_demand_rate[:, :n_dests, :], axis=1))
    n_origins = origin_ids.shape[0]
    n_hist = _n_hist(config)

    if differentiable:
        W = config.window_size
        init_carry = ScanCarry(
            cum_arrival_w=jnp.zeros((config.n_links, W + 1), dtype=config.dtype),
            cum_departure_w=jnp.zeros((config.n_links, W + 1), dtype=config.dtype),
            demand_queue=jnp.zeros(config.n_nodes, dtype=config.dtype),
            absorbed_count=jnp.zeros(config.n_nodes, dtype=config.dtype),
            cum_arrival_full=jnp.zeros((config.n_links, n_hist), dtype=config.dtype),
            cum_arrival_d=jnp.zeros((n_hist, config.n_links, n_dests), dtype=config.dtype),
            cum_departure_d_cur=jnp.zeros((config.n_links, n_dests), dtype=config.dtype),
            orig_out_d=jnp.zeros((n_origins, n_dests), dtype=config.dtype),
            prev_next_link_ids=jnp.full((n_dests, config.n_nodes), -1, dtype=jnp.int32),
            prev_dists=jnp.full((n_dests, config.n_nodes), 1e15, dtype=config.dtype),
            prev_link_cost=jnp.zeros(config.n_links, dtype=config.dtype),
        )
        step_fn = functools.partial(duo_simulation_step,
                                    params=params, link_state=link_state, config=config,
                                    origin_ids=origin_ids, od_cum=od_cum, od_cum_agg=od_cum_agg)
        if checkpoint_every is None:
            final_carry, outputs = jax.lax.scan(
                step_fn, init_carry, jnp.arange(config.tsize))
        else:
            final_carry, outputs = _scan_with_checkpoint(
                step_fn, init_carry, config.tsize, checkpoint_every)
        return _reconstruct_state(final_carry, outputs, config)
    else:
        init_carry = FwdCarry(
            cum_arrival=jnp.zeros((config.n_links, config.tsize + 1), dtype=config.dtype),
            cum_departure=jnp.zeros((config.n_links, config.tsize + 1), dtype=config.dtype),
            demand_queue=jnp.zeros(config.n_nodes, dtype=config.dtype),
            absorbed_count=jnp.zeros(config.n_nodes, dtype=config.dtype),
            cum_arrival_d=jnp.zeros((n_hist, config.n_links, n_dests), dtype=config.dtype),
            cum_departure_d_cur=jnp.zeros((config.n_links, n_dests), dtype=config.dtype),
            orig_out_d=jnp.zeros((n_origins, n_dests), dtype=config.dtype),
            prev_next_link_ids=jnp.full((n_dests, config.n_nodes), -1, dtype=jnp.int32),
            prev_dists=jnp.full((n_dests, config.n_nodes), 1e15, dtype=config.dtype),
            prev_link_cost=jnp.zeros(config.n_links, dtype=config.dtype),
        )
        step_fn = functools.partial(duo_simulation_step_fwd,
                                    params=params, link_state=link_state, config=config,
                                    origin_ids=origin_ids, od_cum=od_cum, od_cum_agg=od_cum_agg)
        final_carry, outputs = jax.lax.scan(step_fn, init_carry, jnp.arange(config.tsize))
        return _build_state_fwd(final_carry, outputs, config)


# ================================================================
# World -> JAX conversion
# ================================================================

def world_to_jax(W, dtype="float32"):
    """Convert a finalized World to JAX PyTree structures.

    Parameters
    ----------
    W : World
        Must have finalize_scenario() called.
    dtype : str, optional
        Floating-point type of the simulation, "float32" (default) or "float64".
        "float64" turns on ``jax_enable_x64`` for the whole process; float32 simulations still run in float32 afterwards.

    Returns
    -------
    params : Params
    config : NetworkConfig
    """
    if dtype not in ("float32", "float64"):
        raise ValueError(f"dtype must be 'float32' or 'float64', got {dtype!r}")
    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    np_dtype = np.dtype(dtype)
    if not W.finalized:
        W.finalize_scenario()

    n_nodes = len(W.NODES)
    n_links = len(W.LINKS)
    tsize = W.TSIZE
    dt = W.DELTAT

    max_in = max((len(n.inlinks) for n in W.NODES), default=1)
    max_out = max((len(n.outlinks) for n in W.NODES), default=1)
    max_in = max(max_in, 1)
    max_out = max(max_out, 1)

    link_start = np.array([l.start_node.id for l in W.LINKS], dtype=np.int32)
    link_end = np.array([l.end_node.id for l in W.LINKS], dtype=np.int32)

    node_inlinks = np.full((n_nodes, max_in), -1, dtype=np.int32)
    node_outlinks = np.full((n_nodes, max_out), -1, dtype=np.int32)
    node_n_in = np.zeros(n_nodes, dtype=np.int32)
    node_n_out = np.zeros(n_nodes, dtype=np.int32)

    for node in W.NODES:
        for j, link in enumerate(node.inlinks.values()):
            node_inlinks[node.id, j] = link.id
        node_n_in[node.id] = len(node.inlinks)
        for j, link in enumerate(node.outlinks.values()):
            node_outlinks[node.id, j] = link.id
        node_n_out[node.id] = len(node.outlinks)

    type_map = {"origin": 0, "destination": 1, "dummy": 2, "merge": 3, "diverge": 4, "general": 5}
    node_type = np.array([type_map[n.node_type()] for n in W.NODES], dtype=np.int32)

    # Demand rate per timestep
    # Match Python's check: t_start <= t_index*dt < t_end
    # i_start: first t_index where t_start <= t_index*dt -> ceil(t_start/dt)
    # i_end: first t_index where t_index*dt >= t_end -> ceil(t_end/dt)
    demand_rate = np.zeros((n_nodes, tsize), dtype=np_dtype)
    for node in W.NODES:
        for t_start, t_end, flow in node.demand_table:
            i_start = max(int(np.ceil(t_start / dt - 1e-10)), 0)
            i_end = min(int(np.ceil(t_end / dt - 1e-10)), tsize)
            demand_rate[node.id, i_start:i_end] += flow

    # Diverge ratios
    diverge_ratios = np.zeros((n_nodes, max_out), dtype=np_dtype)
    for node in W.NODES:
        for j, link_name in enumerate(node.outlinks):
            diverge_ratios[node.id, j] = node.diverge_ratios.get(link_name, 0.0)

    # Turning fraction matrix (n_nodes, max_in, max_out)
    tf = np.zeros((n_nodes, max_in, max_out), dtype=np_dtype)
    for node in W.NODES:
        if node.turning_fraction_matrix is not None:
            # General node with explicit turning fractions
            for i_row in range(len(node.turning_fraction_matrix)):
                for j_col in range(len(node.turning_fraction_matrix[i_row])):
                    tf[node.id, i_row, j_col] = node.turning_fraction_matrix[i_row][j_col]
        else:
            # Build from node type: merge->B[i][0]=1, diverge->B[0][j]=ratio, dummy->B[0][0]=pass
            ntype = node.node_type()
            inlinks_list = list(node.inlinks.values())
            outlinks_list = list(node.outlinks.values())
            I = len(inlinks_list)
            J = len(outlinks_list)
            if ntype == "merge":
                for ii in range(I):
                    tf[node.id, ii, 0] = 1.0
            elif ntype == "diverge":
                for jj, ln in enumerate(outlinks_list):
                    tf[node.id, 0, jj] = node.diverge_ratios.get(ln.name, 0.0)
            elif ntype == "dummy":
                tf[node.id, 0, 0] = 1.0 - node.absorption_ratio
            elif J > 0:
                eq = 1.0 / J if J > 0 else 0.0
                for ii in range(I):
                    for jj in range(J):
                        tf[node.id, ii, jj] = eq

    # Link parameters
    u = np.array([l.u for l in W.LINKS], dtype=np_dtype)
    kappa = np.array([l.kappa for l in W.LINKS], dtype=np_dtype)
    merge_priority = np.array([l.merge_priority for l in W.LINKS], dtype=np_dtype)
    capacity_out = np.array([l.capacity_out for l in W.LINKS], dtype=np_dtype)
    capacity_in = np.array([l.capacity_in for l in W.LINKS], dtype=np_dtype)
    flow_capacity = np.array([n.flow_capacity for n in W.NODES], dtype=np_dtype)
    absorption_ratio = np.array([n.absorption_ratio for n in W.NODES], dtype=np_dtype)
    lengths = np.array([l.length for l in W.LINKS], dtype=np_dtype)

    # DUO: destinations and per-OD demand (sorted for a stable ordering across runs)
    destinations = sorted(set(dest for _, dest, _, _, _ in W.demand_info)) if W.demand_info else []
    dest_ids = np.array([W.NODES_NAME_DICT[d].id for d in destinations], dtype=np.int32) if destinations else np.array([0], dtype=np.int32)
    n_dests = max(len(destinations), 1)

    # Per-OD demand rate: (n_nodes, n_dests, tsize)
    od_demand_rate = np.zeros((n_nodes, n_dests, tsize), dtype=np_dtype)
    dest_idx_map = {d: i for i, d in enumerate(destinations)}
    for orig, dest, t_start, t_end, flow in W.demand_info:
        oid = W.NODES_NAME_DICT[orig].id
        did = dest_idx_map.get(dest)
        if did is not None:
            i_start = max(int(np.ceil(t_start / dt - 1e-10)), 0)
            i_end = min(int(np.ceil(t_end / dt - 1e-10)), tsize)
            od_demand_rate[oid, did, i_start:i_end] += flow

    # Travel time method depends on route_choice
    tt_method = "multipoint" if W.ROUTE_CHOICE == "duo_multipoint" else "avg_density"

    # Sliding window size: max lookback for interp_batch + safety margin
    w_speeds = np.array([l.w for l in W.LINKS], dtype=np_dtype)
    offset_u_vals = lengths / (u * dt)
    offset_w_vals = lengths / (w_speeds * dt)
    max_offset = max(float(np.max(offset_u_vals)), float(np.max(offset_w_vals)))
    window_size = min(int(np.ceil(max_offset)) + 2, tsize)

    # Discretize congestion pricing toll at route_update_interval cadence
    route_update_interval = max(1, int(300 / float(dt)))
    n_toll_steps = max(1, int(np.ceil(tsize / route_update_interval)))
    toll_arr = np.zeros((n_links, n_toll_steps), dtype=np_dtype)
    for i, link in enumerate(W.LINKS):
        if link.congestion_pricing is not None:
            for k in range(n_toll_steps):
                t_sec = k * route_update_interval * dt
                toll_arr[i, k] = link.get_toll(t_sec)

    config = NetworkConfig(
        n_nodes=n_nodes, n_links=n_links, tsize=tsize, deltat=float(dt),
        max_in=max_in, max_out=max_out,
        link_start_node=jnp.array(link_start),
        link_end_node=jnp.array(link_end),
        node_inlinks=jnp.array(node_inlinks),
        node_outlinks=jnp.array(node_outlinks),
        node_n_inlinks=jnp.array(node_n_in),
        node_n_outlinks=jnp.array(node_n_out),
        node_type=jnp.array(node_type),
        link_lengths=jnp.array(lengths),
        has_general_nodes=bool(5 in node_type),
        n_dests=n_dests,
        dest_node_ids=jnp.array(dest_ids),
        tt_method=tt_method,
        route_update_interval=route_update_interval,
        window_size=window_size,
        n_toll_steps=n_toll_steps,
        toll_step_size=route_update_interval,
        use_logit=(W.ROUTE_CHOICE == "duo_logit"),
        logit_temperature=float(getattr(W, 'LOGIT_TEMPERATURE', 60.0)),
        node_fifo=getattr(W, "NODE_FIFO", "strict"),
        dtype=dtype,
    )

    params = Params(
        u=jnp.array(u),
        kappa=jnp.array(kappa),
        q_star=jnp.array(np.array([l.q_star for l in W.LINKS], dtype=np_dtype)),
        capacity_out=jnp.array(capacity_out),
        capacity_in=jnp.array(capacity_in),
        flow_capacity=jnp.array(flow_capacity),
        absorption_ratio=jnp.array(absorption_ratio),
        diverge_ratios=jnp.array(diverge_ratios),
        merge_priority=jnp.array(merge_priority),
        demand_rate=jnp.array(demand_rate),
        turning_fractions=jnp.array(tf),
        od_demand_rate=jnp.array(od_demand_rate),
        toll=jnp.array(toll_arr),
        route_bias=jnp.zeros((od_demand_rate.shape[1],
                              od_demand_rate.shape[0],
                              tf.shape[2]), dtype=config.dtype),
    )

    return params, config
