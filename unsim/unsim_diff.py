"""
UNsim-diff: JAX-based differentiable Network Link Transmission Model simulator.

All simulation internals are pure JAX functions, enabling jax.grad, jax.jit,
and jax.lax.scan for automatic differentiation and compilation.

Usage
-----
>>> from unsim import World
>>> from unsim.unsim_diff import world_to_jax, simulate, total_travel_time
>>> W = World(...); W.addNode(...); W.addLink(...); W.adddemand(...)
>>> params, config, lengths = world_to_jax(W)
>>> state = simulate(params, config, lengths)
>>> ttt = total_travel_time(state, config)
>>> grad_fn = jax.grad(lambda p: total_travel_time(simulate(p, config, lengths), config))
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
    link_lengths : jnp.ndarray, (n_links,) float32
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
    cum_arrival_d_cur : jnp.ndarray, (n_links, n_dests)
        Per-destination cumulative arrivals at current time only.
    cum_departure_d_cur : jnp.ndarray, (n_links, n_dests)
        Per-destination cumulative departures at current time only.
    prev_next_link_ids : jnp.ndarray, (n_dests, n_nodes)
    prev_dists : jnp.ndarray, (n_dests, n_nodes)
        Cached shortest-path distances from last route update.
    """
    cum_arrival_w: jnp.ndarray
    cum_departure_w: jnp.ndarray
    demand_queue: jnp.ndarray
    absorbed_count: jnp.ndarray
    cum_arrival_d_cur: jnp.ndarray
    cum_departure_d_cur: jnp.ndarray
    prev_next_link_ids: jnp.ndarray
    prev_dists: jnp.ndarray


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
    cum_arrival_d_cur : jnp.ndarray, (n_links, n_dests)
        Per-destination cumulative arrivals at current time only.
    cum_departure_d_cur : jnp.ndarray, (n_links, n_dests)
        Per-destination cumulative departures at current time only.
    prev_next_link_ids : jnp.ndarray, (n_dests, n_nodes)
    prev_dists : jnp.ndarray, (n_dests, n_nodes)
        Cached shortest-path distances from last route update.
    """
    cum_arrival: jnp.ndarray
    cum_departure: jnp.ndarray
    demand_queue: jnp.ndarray
    absorbed_count: jnp.ndarray
    cum_arrival_d_cur: jnp.ndarray
    cum_departure_d_cur: jnp.ndarray
    prev_next_link_ids: jnp.ndarray
    prev_dists: jnp.ndarray


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
    arrays from stacked outputs to dramatically reduce memory.

    demand_queue_history is kept in the carry (not stacked outputs) so
    that the scan output is empty, avoiding all stacked allocations.

    Attributes
    ----------
    cum_arrival : jnp.ndarray, (n_links, tsize+1)
    cum_departure : jnp.ndarray, (n_links, tsize+1)
    demand_queue : jnp.ndarray, (n_nodes,)
    absorbed_count : jnp.ndarray, (n_nodes,)
    cum_arrival_d_cur : jnp.ndarray, (n_links, n_dests)
        Per-destination cumulative arrivals at current time only.
    cum_departure_d_cur : jnp.ndarray, (n_links, n_dests)
        Per-destination cumulative departures at current time only.
    demand_queue_history : jnp.ndarray, (n_nodes, tsize)
        Full history of demand queues (for total_travel_time).
    """
    cum_arrival: jnp.ndarray
    cum_departure: jnp.ndarray
    demand_queue: jnp.ndarray
    absorbed_count: jnp.ndarray
    cum_arrival_d_cur: jnp.ndarray
    cum_departure_d_cur: jnp.ndarray
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
    w = q_star * params.u / (params.u * params.kappa - q_star)
    offset_u = config.link_lengths / (params.u * config.deltat)
    offset_w = config.link_lengths / (w * config.deltat)
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
    fi = jnp.clip(frac_indices, 0.0, max_idx)
    i_low = jnp.floor(fi).astype(jnp.int32)
    i_low = jnp.clip(i_low, 0, max_idx - 1)
    frac = fi - i_low.astype(jnp.float32)
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
    fi = jnp.clip(frac_index, 0.0, max_idx)
    i_low = jnp.floor(fi).astype(jnp.int32)
    i_low = jnp.clip(i_low, 0, max_idx - 1)
    frac = fi - i_low.astype(jnp.float32)
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
    D = jnp.minimum(D, link_state.q_star)
    D = jnp.minimum(D, params.capacity_out)
    D = jnp.maximum(D, 0.0)
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
    S = jnp.minimum(S, link_state.q_star)
    S = jnp.minimum(S, params.capacity_in)
    S = jnp.maximum(S, 0.0)
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

    # Single outlink
    s0 = out_sup[:, 0]  # (n_nodes,)
    origin_flow_single = jnp.maximum(jnp.minimum(jnp.minimum(eff_demand, s0), fc), 0.0)

    # Multi outlink
    betas = params.diverge_ratios  # (n_nodes, max_out)
    s_over_b = jnp.where(betas > 1e-10,
                          out_sup / jnp.maximum(betas, 1e-10), jnp.inf)
    s_over_b = jnp.where(out_valid, s_over_b, jnp.inf)
    min_s = jnp.min(s_over_b, axis=1)  # (n_nodes,)
    origin_flow_multi = jnp.maximum(jnp.minimum(jnp.minimum(eff_demand, min_s), fc), 0.0)

    origin_flow = jnp.where(n_out == 1, origin_flow_single, origin_flow_multi)  # (n_nodes,)
    origin_dq = jnp.maximum(dq + (ext_demand - origin_flow) * dt, 0.0)          # (n_nodes,)

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
        jnp.minimum(D_dummy, out_sup[:, 0] / jnp.maximum(pass_ratio, 1e-10)),
        D_dummy)
    dummy_flow = jnp.maximum(jnp.minimum(dummy_flow, fc), 0.0)  # (n_nodes,)
    dummy_outflow = dummy_flow
    dummy_inflow = dummy_flow * pass_ratio
    dummy_absorbed = dummy_flow * params.absorption_ratio * dt  # (n_nodes,)

    # ---- Merge (type=3): 2-input mid-value formula ----
    D1 = in_dem[:, 0]  # (n_nodes,)
    if config.max_in >= 2:
        D2 = jnp.where(n_in >= 2, in_dem[:, 1], 0.0)  # (n_nodes,)
        safe_in1 = safe_in[:, 1]
    else:
        D2 = jnp.zeros(n_nodes, dtype=jnp.float32)
        safe_in1 = jnp.zeros(n_nodes, dtype=jnp.int32)
    S_merge = jnp.minimum(out_sup[:, 0], fc)  # (n_nodes,)
    p1 = params.merge_priority[safe_in[:, 0]]  # (n_nodes,)
    p2 = params.merge_priority[safe_in1]        # (n_nodes,)
    total_p = p1 + p2
    a1 = p1 / jnp.maximum(total_p, 1e-10)
    a2 = p2 / jnp.maximum(total_p, 1e-10)

    total_D = D1 + D2
    q1_cong = jnp.maximum(differentiable_mid(D1, S_merge - D2, a1 * S_merge), 0.0)
    q2_cong = jnp.maximum(differentiable_mid(D2, S_merge - D1, a2 * S_merge), 0.0)
    merge_q1 = jnp.where(total_D <= S_merge, D1, q1_cong)  # (n_nodes,)
    merge_q2 = jnp.where(total_D <= S_merge, D2, q2_cong)  # (n_nodes,)
    merge_q3 = merge_q1 + merge_q2  # (n_nodes,)

    # ---- Diverge (type=4) ----
    D_div = in_dem[:, 0]  # (n_nodes,)
    div_betas = params.diverge_ratios  # (n_nodes, max_out)
    div_s_over_b = jnp.where(div_betas > 1e-10,
                              out_sup / jnp.maximum(div_betas, 1e-10), jnp.inf)
    div_s_over_b = jnp.where(out_valid, div_s_over_b, jnp.inf)
    div_min_s = jnp.min(div_s_over_b, axis=1)  # (n_nodes,)
    div_flow = jnp.maximum(jnp.minimum(jnp.minimum(D_div, div_min_s), fc), 0.0)
    div_outflows_per_link = div_betas * div_flow[:, None]  # (n_nodes, max_out)

    # ---- General / INM (type=5) ----
    if config.has_general_nodes:
        inm_alpha_all = params.merge_priority[safe_in]  # (n_nodes, max_in)
        EPS_INM = 1e-10
        max_iter_inm = config.max_in + config.max_out + 1

        def _inm_single(B_i, alpha_i, dem_i, sup_i, iv, ov):
            """INM solver for one node."""
            def inm_body(carry_iter, _):
                qi, qo = carry_iter
                demand_ok = qi < (dem_i - EPS_INM)
                out_full = qo >= (sup_i - EPS_INM)
                blocked = jnp.any(jnp.logical_and(B_i > 0, out_full[None, :]), axis=1)
                d_up = jnp.logical_and(demand_ok, ~blocked)
                d_up = jnp.logical_and(d_up, iv)

                supply_ok = qo < (sup_i - EPS_INM)
                has_active_up = jnp.any(jnp.logical_and(d_up[:, None], B_i > 0), axis=0)
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
                return (qi + theta * phi_in, qo + theta * phi_out), None

            qi_init = jnp.zeros(config.max_in, dtype=jnp.float32)
            qo_init = jnp.zeros(config.max_out, dtype=jnp.float32)
            (inm_q_in, inm_q_out), _ = jax.lax.scan(
                inm_body, (qi_init, qo_init), None, length=max_iter_inm)
            return inm_q_in, inm_q_out

        inm_q_in_all, inm_q_out_all = jax.vmap(_inm_single)(
            params.turning_fractions, inm_alpha_all,
            in_dem, out_sup, in_valid, out_valid)  # (n_nodes, max_in), (n_nodes, max_out)
    else:
        inm_q_in_all = jnp.zeros((n_nodes, config.max_in), dtype=jnp.float32)
        inm_q_out_all = jnp.zeros((n_nodes, config.max_out), dtype=jnp.float32)

    # ---- Dispatch by node type: outflows for inlinks (n_nodes, max_in) ----
    j_in = jnp.arange(config.max_in)[None, :]  # (1, max_in)
    nt = ntype[:, None]  # (n_nodes, 1)

    out_val_dest = dest_outflows  # (n_nodes, max_in)
    out_val_dummy = jnp.where(j_in == 0, dummy_outflow[:, None], 0.0)
    if config.max_in >= 2:
        out_val_merge = jnp.where(j_in == 0, merge_q1[:, None],
                        jnp.where(j_in == 1, merge_q2[:, None], 0.0))
    else:
        out_val_merge = jnp.where(j_in == 0, merge_q1[:, None], 0.0)
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
    in_val_merge = jnp.where(j_out == 0, merge_q3[:, None], 0.0)
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
    outflows = jnp.zeros(n_links, dtype=jnp.float32).at[flat_in_lids].add(flat_out_vals)

    flat_out_lids = safe_out.ravel()     # (n_nodes * max_out,)
    flat_in_vals = (inflow_per_outlink * out_valid).ravel()  # (n_nodes * max_out,)
    inflows = jnp.zeros(n_links, dtype=jnp.float32).at[flat_out_lids].add(flat_in_vals)

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
        cum_arrival_d_cur=carry.cum_arrival_d_cur,
        cum_departure_d_cur=carry.cum_departure_d_cur,
        prev_next_link_ids=carry.prev_next_link_ids,
        prev_dists=carry.prev_dists,
    )

    output = StepOutput(
        inflow_rates=inflow_rates,
        outflow_rates=outflow_rates,
        demand_queue=new_dq,
        inflow_d=jnp.zeros((config.n_links, n_dests), dtype=jnp.float32),
        outflow_d=jnp.zeros((config.n_links, n_dests), dtype=jnp.float32),
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
        cum_arrival_d_cur=carry.cum_arrival_d_cur,
        cum_departure_d_cur=carry.cum_departure_d_cur,
        prev_next_link_ids=carry.prev_next_link_ids,
        prev_dists=carry.prev_dists,
    )

    output = StepOutput(
        inflow_rates=inflow_rates,
        outflow_rates=outflow_rates,
        demand_queue=new_dq,
        inflow_d=jnp.zeros((config.n_links, n_dests), dtype=jnp.float32),
        outflow_d=jnp.zeros((config.n_links, n_dests), dtype=jnp.float32),
    )
    return new_carry, output


def _reconstruct_state(final_carry, outputs, config):
    """Reconstruct full SimState from scan outputs via cumulative sum."""
    dt = config.deltat
    n_dests = max(config.n_dests, 1)

    # outputs arrays have shape (tsize, ...) from scan stacking
    cum_arrival = jnp.zeros((config.n_links, config.tsize + 1), dtype=jnp.float32)
    cum_arrival = cum_arrival.at[:, 1:].set(
        jnp.cumsum(outputs.inflow_rates, axis=0).T * dt)

    cum_departure = jnp.zeros((config.n_links, config.tsize + 1), dtype=jnp.float32)
    cum_departure = cum_departure.at[:, 1:].set(
        jnp.cumsum(outputs.outflow_rates, axis=0).T * dt)

    demand_queue_history = outputs.demand_queue.T  # (tsize, n_nodes) -> (n_nodes, tsize)

    cum_arrival_d = jnp.zeros((config.n_links, n_dests, config.tsize + 1), dtype=jnp.float32)
    cum_arrival_d = cum_arrival_d.at[:, :, 1:].set(
        jnp.cumsum(outputs.inflow_d, axis=0).transpose(1, 2, 0) * dt)

    cum_departure_d = jnp.zeros((config.n_links, n_dests, config.tsize + 1), dtype=jnp.float32)
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

    cum_arrival_d = jnp.zeros((config.n_links, n_dests, config.tsize + 1), dtype=jnp.float32)
    cum_arrival_d = cum_arrival_d.at[:, :, 1:].set(
        jnp.cumsum(outputs.inflow_d, axis=0).transpose(1, 2, 0) * dt)

    cum_departure_d = jnp.zeros((config.n_links, n_dests, config.tsize + 1), dtype=jnp.float32)
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


def simulate(params, config, differentiable=True):
    """Run full LTM simulation.

    Parameters
    ----------
    params : Params
    config : NetworkConfig
    differentiable : bool, optional
        If True (default), use windowed carry suitable for jax.grad.
        If False, use full-array carry for faster forward-only evaluation
        (not compatible with reverse-mode AD).

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
            cum_arrival_w=jnp.zeros((config.n_links, W + 1), dtype=jnp.float32),
            cum_departure_w=jnp.zeros((config.n_links, W + 1), dtype=jnp.float32),
            demand_queue=jnp.zeros(config.n_nodes, dtype=jnp.float32),
            absorbed_count=jnp.zeros(config.n_nodes, dtype=jnp.float32),
            cum_arrival_d_cur=jnp.zeros((config.n_links, n_dests), dtype=jnp.float32),
            cum_departure_d_cur=jnp.zeros((config.n_links, n_dests), dtype=jnp.float32),
            prev_next_link_ids=jnp.full((n_dests, config.n_nodes), -1, dtype=jnp.int32),
            prev_dists=jnp.full((n_dests, config.n_nodes), 1e15, dtype=jnp.float32),
        )
        step_fn = functools.partial(simulation_step,
                                    params=params, link_state=link_state, config=config)
        final_carry, outputs = jax.lax.scan(step_fn, init_carry, jnp.arange(config.tsize))
        return _reconstruct_state(final_carry, outputs, config)
    else:
        init_carry = FwdCarry(
            cum_arrival=jnp.zeros((config.n_links, config.tsize + 1), dtype=jnp.float32),
            cum_departure=jnp.zeros((config.n_links, config.tsize + 1), dtype=jnp.float32),
            demand_queue=jnp.zeros(config.n_nodes, dtype=jnp.float32),
            absorbed_count=jnp.zeros(config.n_nodes, dtype=jnp.float32),
            cum_arrival_d_cur=jnp.zeros((config.n_links, n_dests), dtype=jnp.float32),
            cum_departure_d_cur=jnp.zeros((config.n_links, n_dests), dtype=jnp.float32),
            prev_next_link_ids=jnp.full((n_dests, config.n_nodes), -1, dtype=jnp.int32),
            prev_dists=jnp.full((n_dests, config.n_nodes), 1e15, dtype=jnp.float32),
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

    return jnp.minimum(N_free, N_cong)


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
    result = (lo + frac).astype(jnp.float32)
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

    return jnp.maximum(t_freeflow, t_queue)


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
    t_current = jnp.float32(t_depart)
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
    result = (lo + frac).astype(jnp.float32)
    return result + jax.lax.stop_gradient(
        jnp.clip(result, 0.0, n - 1.0) - result)


def link_exit_time_batch(link_ids, t_enters, state, params, config):
    """Compute exit times for multiple links simultaneously. Differentiable.

    Batch version of ``link_exit_time`` using vectorized operations.

    Parameters
    ----------
    link_ids : jnp.ndarray, (K,) int32
        Link indices.
    t_enters : jnp.ndarray, (K,) float32
        Entry times per link (s).
    state : SimState
    params : Params
    config : NetworkConfig

    Returns
    -------
    jnp.ndarray, (K,) float32
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

    return jnp.maximum(t_freeflow, t_queue)


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
    prob_init = jnp.zeros(n_nodes, dtype=jnp.float32).at[origin_node].set(1.0)
    tw_init = jnp.zeros(n_nodes, dtype=jnp.float32).at[origin_node].set(
        jnp.float32(t_depart))

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

        new_prob = jnp.zeros(n_nodes, dtype=jnp.float32).at[flat_end].add(
            jnp.where(flat_valid, flat_tp, 0.0))
        new_tw = jnp.zeros(n_nodes, dtype=jnp.float32).at[flat_end].add(
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
            jnp.float32(0.0), jnp.float32(0.0))
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

    V = jnp.full(n_nodes, INF, dtype=jnp.float32).at[dest_node].set(0.0)

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

    t_final = jax.lax.fori_loop(0, max_path, chain_step, jnp.float32(t_depart))
    return t_final - t_depart


# ================================================================
# AON (All-or-Nothing) -> fixed route simulation
# ================================================================

def _aon_simulation_step_fwd(carry, scan_input, params, link_state, config,
                              route_match, origin_ids):
    """One AoN timestep for forward-only jax.lax.scan (minimal carry).

    Uses precomputed ``route_match`` instead of running Bellman-Ford or
    computing travel times at each step.  Does not output per-destination
    arrays, saving O(tsize * n_links * n_dests) memory.

    Parameters
    ----------
    carry : AoNFwdCarry
    scan_input : tuple (t_index, od_weights_compact)
        t_index : jnp scalar int (absolute timestep index)
        od_weights_compact : jnp.ndarray, (n_origins, n_dests)
            Per-OD demand rates for origin nodes at this timestep.
    params : Params
        Slim params with od_demand_rate zeroed out.
    link_state : LinkState
    config : NetworkConfig
    route_match : jnp.ndarray, (n_dests, n_nodes, max_out)
        Precomputed routing indicator (hard 0/1 for AoN).
    origin_ids : jnp.ndarray, (n_origins,) int32
        Indices of origin nodes.

    Returns
    -------
    (AoNFwdCarry, None)
    """
    t_index, od_weights_compact = scan_input
    dt = config.deltat
    n_links = config.n_links
    n_nodes = config.n_nodes
    n_dests = config.n_dests

    # Build SimState view backed by full-history arrays
    n_dests_safe = max(n_dests, 1)
    state = SimState(
        cum_arrival=carry.cum_arrival,
        cum_departure=carry.cum_departure,
        demand_queue=carry.demand_queue,
        absorbed_count=carry.absorbed_count,
        demand_queue_history=jnp.zeros((n_nodes, 1)),
        cum_arrival_d=jnp.zeros((n_links, n_dests_safe, 1)),
        cum_departure_d=jnp.zeros((n_links, n_dests_safe, 1)),
        prev_next_link_ids=jnp.zeros((n_dests_safe, n_nodes), dtype=jnp.int32),
    )

    # Step 1: LTM demands/supplies
    demands = compute_demands(t_index, state, link_state, params, config)
    supplies = compute_supplies(t_index, state, link_state, params, config)

    # Step 2: Dynamic diverge ratios from precomputed route_match
    n_on_d = carry.cum_arrival_d_cur - carry.cum_departure_d_cur  # (n_links, n_dests)
    n_on_d = jnp.maximum(n_on_d, 0.0)

    is_origin = config.node_type == 0  # (n_nodes,)

    safe_inlinks = jnp.maximum(config.node_inlinks, 0)  # (n_nodes, max_in)
    inlink_valid = config.node_inlinks >= 0  # (n_nodes, max_in)
    inlink_veh = n_on_d[safe_inlinks]  # (n_nodes, max_in, n_dests)
    inlink_veh = inlink_veh * inlink_valid[:, :, None]
    node_inlink_total = jnp.sum(inlink_veh, axis=1)  # (n_nodes, n_dests)

    # Expand compact OD demand to full node array
    od_weights = jnp.zeros((n_nodes, n_dests), dtype=jnp.float32)
    od_weights = od_weights.at[origin_ids].set(od_weights_compact)
    dest_weight = jnp.where(is_origin[:, None], od_weights, node_inlink_total)

    safe_outlinks = jnp.maximum(config.node_outlinks, 0)  # (n_nodes, max_out)
    outlink_valid = config.node_outlinks >= 0  # (n_nodes, max_out)

    outlink_flow = jnp.einsum("nd,dno->no", dest_weight, route_match)

    total = jnp.sum(outlink_flow, axis=1, keepdims=True)  # (n_nodes, 1)
    n_out = config.node_n_outlinks[:, None]
    equal_split = jnp.where(
        jnp.arange(config.max_out)[None, :] < n_out,
        1.0 / jnp.maximum(n_out, 1), 0.0)
    duo_diverge = jnp.where(total > 1e-10,
                             outlink_flow / jnp.maximum(total, 1e-10),
                             equal_split)

    duo_tf = jnp.broadcast_to(duo_diverge[:, None, :],
                               (n_nodes, config.max_in, config.max_out))
    params_duo = params._replace(diverge_ratios=duo_diverge,
                                  turning_fractions=duo_tf)

    # Step 3: Node models
    inflows, outflows, new_dq, _ = compute_node_transfers(
        t_index, demands, supplies, state, params_duo, config)

    # Step 4: FIFO allocation of departures by destination
    total_arr = carry.cum_arrival[:, t_index]  # (n_links,)
    arr_d = carry.cum_arrival_d_cur  # (n_links, n_dests)
    fifo_ratio = jnp.where(
        total_arr[:, None] > 1e-10,
        arr_d / jnp.maximum(total_arr[:, None], 1e-10),
        0.0)
    outflow_d = outflows[:, None] * fifo_ratio  # (n_links, n_dests)

    # Step 5: Route per-dest traffic
    new_absorbed = carry.absorbed_count

    inlink_outflow_d = outflow_d[safe_inlinks]  # (n_nodes, max_in, n_dests)
    inlink_outflow_d = inlink_outflow_d * inlink_valid[:, :, None]
    node_arriving = jnp.sum(inlink_outflow_d, axis=1)  # (n_nodes, n_dests)

    node_arriving = node_arriving + jnp.where(
        is_origin[:, None], od_weights, 0.0)

    dest_mask = (config.dest_node_ids[None, :] == jnp.arange(n_nodes)[:, None])
    absorb_per_node = jnp.sum(jnp.where(dest_mask, node_arriving, 0.0), axis=1)
    new_absorbed = new_absorbed + absorb_per_node * dt
    node_arriving = jnp.where(dest_mask, 0.0, node_arriving)

    route_match_ndo = route_match.transpose(1, 0, 2)  # (n_nodes, n_dests, max_out)
    routed = node_arriving[:, :, None] * route_match_ndo

    flat_lids = safe_outlinks.reshape(-1)
    flat_valid = outlink_valid.reshape(-1)
    flat_routed = routed.transpose(0, 2, 1).reshape(-1, n_dests)
    flat_routed = flat_routed * flat_valid[:, None]

    inflow_d = jnp.zeros((n_links, n_dests), dtype=jnp.float32)
    inflow_d = inflow_d.at[flat_lids].add(flat_routed)

    # Step 6: Update per-dest current cumulative counts
    new_cum_arr_d_cur = carry.cum_arrival_d_cur + dt * inflow_d
    new_cum_dep_d_cur = carry.cum_departure_d_cur + dt * outflow_d

    # Aggregate from per-dest -> update full arrays
    new_cum_arr = jnp.sum(new_cum_arr_d_cur, axis=1)
    new_cum_dep = jnp.sum(new_cum_dep_d_cur, axis=1)

    new_carry = AoNFwdCarry(
        cum_arrival=carry.cum_arrival.at[:, t_index + 1].set(new_cum_arr),
        cum_departure=carry.cum_departure.at[:, t_index + 1].set(new_cum_dep),
        demand_queue=new_dq,
        absorbed_count=new_absorbed,
        cum_arrival_d_cur=new_cum_arr_d_cur,
        cum_departure_d_cur=new_cum_dep_d_cur,
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
        cum_arrival_d=jnp.zeros((config.n_links, n_dests, 1), dtype=jnp.float32),
        cum_departure_d=jnp.zeros((config.n_links, n_dests, 1), dtype=jnp.float32),
        prev_next_link_ids=next_link_ids,
    )


def simulate_aon(params, config, differentiable=True):
    """Run AON simulation with routes fixed at free-flow shortest paths.

    Per-destination tracking is maintained internally so that traffic is
    correctly absorbed at destination nodes.

    When ``differentiable=False``, uses a memory-optimized forward-only
    path that eliminates per-destination arrays from scan outputs,
    reducing memory from O(tsize * n_links * n_dests) to
    O(tsize * n_links).  The returned SimState has minimal dummy arrays
    for ``cum_arrival_d`` / ``cum_departure_d``.

    When ``differentiable=True``, falls back to ``simulate_duo`` with
    routes frozen at t=0.

    Parameters
    ----------
    params : Params
    config : NetworkConfig
    differentiable : bool, optional
        If True (default), supports jax.grad w.r.t. params.
        If False, use memory-optimized forward-only evaluation.

    Returns
    -------
    SimState
    """
    if differentiable:
        # Fall back to DUO with routes frozen at t=0.
        config_aon = config._replace(route_update_interval=config.tsize + 1)
        return simulate_duo(params, config_aon, differentiable=True)

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
    ).astype(jnp.float32)  # (n_dests, n_nodes, max_out)

    # Compact OD demand: only origin nodes to save memory
    # od_demand_rate (n_nodes, n_dests, tsize) -> od_seq (tsize, n_origins, n_dests)
    is_origin_np = np.asarray(config.node_type) == 0
    origin_ids_np = np.nonzero(is_origin_np)[0]
    n_origins = len(origin_ids_np)
    origin_ids = jnp.array(origin_ids_np, dtype=jnp.int32)
    od_compact = params.od_demand_rate[origin_ids, :n_dests, :]
    # Transpose to (tsize, n_origins, n_dests) for sequential scan input
    od_seq = jnp.transpose(od_compact, (2, 0, 1))
    del od_compact  # free intermediate

    # Create slim params with od_demand_rate zeroed to free memory
    slim_params = params._replace(
        od_demand_rate=jnp.zeros((config.n_nodes, 1, 1), dtype=jnp.float32))

    # Initialize minimal carry (demand_queue_history in carry, not output)
    init_carry = AoNFwdCarry(
        cum_arrival=jnp.zeros((config.n_links, config.tsize + 1),
                               dtype=jnp.float32),
        cum_departure=jnp.zeros((config.n_links, config.tsize + 1),
                                 dtype=jnp.float32),
        demand_queue=jnp.zeros(config.n_nodes, dtype=jnp.float32),
        absorbed_count=jnp.zeros(config.n_nodes, dtype=jnp.float32),
        cum_arrival_d_cur=jnp.zeros((config.n_links, n_dests),
                                     dtype=jnp.float32),
        cum_departure_d_cur=jnp.zeros((config.n_links, n_dests),
                                       dtype=jnp.float32),
        demand_queue_history=jnp.zeros((config.n_nodes, config.tsize),
                                        dtype=jnp.float32),
    )

    step_fn = functools.partial(
        _aon_simulation_step_fwd,
        params=slim_params, link_state=link_state, config=config,
        route_match=route_match, origin_ids=origin_ids)
    scan_inputs = (jnp.arange(config.tsize), od_seq)
    final_carry, _ = jax.lax.scan(step_fn, init_carry, scan_inputs)

    return _build_state_aon_fwd(final_carry, config, next_link_ids)


# ================================================================
# DUO (Dynamic User Optimal) -- JAX differentiable
# ================================================================

def _compute_tt_avg_density(t_index, state, link_state, params, config):
    """Average-density method: k_avg = (N_U-N_D)/L -> FD -> tau = L/v."""
    n_on_link = state.cum_arrival[:, t_index] - state.cum_departure[:, t_index]
    k_avg = jnp.maximum(n_on_link, 0.0) / config.link_lengths
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

    total_tt = jnp.zeros(config.n_links, dtype=jnp.float32)

    for i in range(n_segments):
        frac0 = i / n_segments
        frac1 = (i + 1) / n_segments

        t_free0 = t_index * 1.0 - frac0 * link_state.offset_u
        N_free0 = interp_batch(state.cum_arrival, jnp.clip(t_free0, 0.0, config.tsize))
        t_cong0 = t_index * 1.0 - (1.0 - frac0) * link_state.offset_w
        N_cong0 = interp_batch(state.cum_departure, jnp.clip(t_cong0, 0.0, config.tsize)) + params.kappa * L * (1.0 - frac0)
        N0 = jnp.minimum(N_free0, N_cong0)

        t_free1 = t_index * 1.0 - frac1 * link_state.offset_u
        N_free1 = interp_batch(state.cum_arrival, jnp.clip(t_free1, 0.0, config.tsize))
        t_cong1 = t_index * 1.0 - (1.0 - frac1) * link_state.offset_w
        N_cong1 = interp_batch(state.cum_departure, jnp.clip(t_cong1, 0.0, config.tsize)) + params.kappa * L * (1.0 - frac1)
        N1 = jnp.minimum(N_free1, N_cong1)

        k_seg = jnp.maximum((N0 - N1) / dx, 0.0)

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
    dist : jnp.ndarray, (n_nodes,) float32
        Shortest distance from each node to destination.
    next_link_id : jnp.ndarray, (n_nodes,) int32
        For each node, the link ID to take toward dest. -1 if unreachable.
    """
    INF = 1e15
    dist = jnp.full(config.n_nodes, INF, dtype=jnp.float32).at[dest_id].set(0.0)
    pred = jnp.full(config.n_nodes, -1, dtype=jnp.int32)

    edge_from = config.link_start_node  # (n_links,)
    edge_to = config.link_end_node      # (n_links,)
    link_ids = jnp.arange(config.n_links, dtype=jnp.int32)

    def relax_iteration(carry, _):
        dist, pred = carry
        # All candidate new distances via reverse edges (vectorized)
        new_dist = dist[edge_to] + link_tt  # (n_links,)

        # Scatter-min: for each from_node, find minimum new_dist across all its edges
        best_dist = jnp.full(config.n_nodes, INF, dtype=jnp.float32).at[edge_from].min(new_dist)

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
    max_iter = int(config.n_nodes**0.5) + 2
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
    dists : jnp.ndarray, (n_dests, n_nodes) float32
        Shortest distances from each node to each destination.
    next_link_ids : jnp.ndarray, (n_dests, n_nodes) int32
    """
    def bf_single(dest_id):
        return bellman_ford_reverse(link_tt, dest_id, config)

    return jax.vmap(bf_single)(config.dest_node_ids)


def duo_simulation_step(carry, t_index, params, link_state, config):
    """One DUO timestep for jax.lax.scan (windowed carry).

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
    n_links = config.n_links
    n_nodes = config.n_nodes
    n_dests = config.n_dests

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
    bf_dists, bf_preds = jax.lax.cond(
        should_update,
        lambda _: bellman_ford_all_dests(link_cost, config),
        lambda _: (carry.prev_dists, carry.prev_next_link_ids),
        None,
    )
    next_link_ids = bf_preds  # (n_dests, n_nodes) int32
    dists = bf_dists           # (n_dests, n_nodes) float32
    if not config.use_logit:
        # Stop gradient on distances when not used for logit routing
        dists = jax.lax.stop_gradient(dists)

    # Step 4: Dynamic diverge ratios -- vectorized tensor operations
    n_on_d = carry.cum_arrival_d_cur - carry.cum_departure_d_cur  # (n_links, n_dests)
    n_on_d = jnp.maximum(n_on_d, 0.0)

    is_origin = config.node_type == 0  # (n_nodes,)

    # Gather inlink vehicle counts per node: (n_nodes, max_in, n_dests)
    safe_inlinks = jnp.maximum(config.node_inlinks, 0)  # (n_nodes, max_in)
    inlink_valid = config.node_inlinks >= 0  # (n_nodes, max_in)
    inlink_veh = n_on_d[safe_inlinks]  # (n_nodes, max_in, n_dests)
    inlink_veh = inlink_veh * inlink_valid[:, :, None]  # mask invalid
    node_inlink_total = jnp.sum(inlink_veh, axis=1)  # (n_nodes, n_dests)

    # For origins: use OD demand as weight
    od_weights = params.od_demand_rate[:, :n_dests, t_index]  # (n_nodes, n_dests)
    dest_weight = jnp.where(is_origin[:, None], od_weights, node_inlink_total)  # (n_nodes, n_dests)

    # Build route indicator
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
        logits = -cost_via / config.logit_temperature
        has_outlinks = config.node_n_outlinks > 0  # (n_nodes,)
        logits = jnp.where(
            ~has_outlinks[None, :, None],
            0.0,  # no outlinks: uniform (value irrelevant, absorbed before routing)
            jnp.where(outlink_valid[None, :, :], logits, -INF))
        route_match = jax.nn.softmax(logits, axis=-1)  # (n_dests, n_nodes, max_out)
    else:
        # Hard routing: route_match[d, n, o] = (next_link_ids[d, n] == outlinks[n, o])
        route_match = (next_link_ids[:, :, None] == safe_outlinks[None, :, :]) & outlink_valid[None, :, :]
        route_match = route_match.astype(jnp.float32)

    # outlink_flow[n, o] = sum_d dest_weight[n, d] * route_match[d, n, o]
    outlink_flow = jnp.einsum("nd,dno->no", dest_weight, route_match)

    total = jnp.sum(outlink_flow, axis=1, keepdims=True)  # (n_nodes, 1)
    n_out = config.node_n_outlinks[:, None]  # (n_nodes, 1)
    equal_split = jnp.where(
        jnp.arange(config.max_out)[None, :] < n_out,
        1.0 / jnp.maximum(n_out, 1), 0.0)  # (n_nodes, max_out)
    duo_diverge = jnp.where(total > 1e-10, outlink_flow / jnp.maximum(total, 1e-10), equal_split)

    # Override diverge_ratios in params for this step
    duo_tf = jnp.broadcast_to(duo_diverge[:, None, :],
                               (n_nodes, config.max_in, config.max_out))
    params_duo = params._replace(diverge_ratios=duo_diverge, turning_fractions=duo_tf)

    # Step 5: Node models -> aggregate outflow/inflow (absolute t_index for demand_rate)
    inflows, outflows, new_dq, _ = compute_node_transfers(
        t_index, demands, supplies, fake_state, params_duo, config)

    # Step 6: FIFO allocation of departures by destination
    total_arr = carry.cum_arrival_w[:, -1]  # (n_links,)
    arr_d = carry.cum_arrival_d_cur  # (n_links, n_dests)
    fifo_ratio = jnp.where(
        total_arr[:, None] > 1e-10,
        arr_d / jnp.maximum(total_arr[:, None], 1e-10),
        0.0)  # (n_links, n_dests)
    outflow_d = outflows[:, None] * fifo_ratio  # (n_links, n_dests)

    # Step 7: Route per-dest traffic -- vectorized tensor operations
    new_absorbed = carry.absorbed_count

    inlink_outflow_d = outflow_d[safe_inlinks]  # (n_nodes, max_in, n_dests)
    inlink_outflow_d = inlink_outflow_d * inlink_valid[:, :, None]
    node_arriving = jnp.sum(inlink_outflow_d, axis=1)  # (n_nodes, n_dests)

    # Add OD demand at origins
    node_arriving = node_arriving + jnp.where(
        is_origin[:, None], od_weights, 0.0)

    # Absorb at destinations
    dest_mask = (config.dest_node_ids[None, :] == jnp.arange(n_nodes)[:, None])  # (n_nodes, n_dests)
    absorb_per_node = jnp.sum(jnp.where(dest_mask, node_arriving, 0.0), axis=1)  # (n_nodes,)
    new_absorbed = new_absorbed + absorb_per_node * dt
    node_arriving = jnp.where(dest_mask, 0.0, node_arriving)

    # Route to outlinks via routing probabilities
    route_match_ndo = route_match.transpose(1, 0, 2)  # (n_nodes, n_dests, max_out)
    routed = node_arriving[:, :, None] * route_match_ndo

    flat_lids = safe_outlinks.reshape(-1)
    flat_valid = outlink_valid.reshape(-1)
    flat_routed = routed.transpose(0, 2, 1).reshape(-1, n_dests)
    flat_routed = flat_routed * flat_valid[:, None]

    inflow_d = jnp.zeros((n_links, n_dests), dtype=jnp.float32)
    inflow_d = inflow_d.at[flat_lids].add(flat_routed)

    # Step 8: Update per-dest current cumulative counts
    new_cum_arr_d_cur = carry.cum_arrival_d_cur + dt * inflow_d
    new_cum_dep_d_cur = carry.cum_departure_d_cur + dt * outflow_d

    # Aggregate from per-dest -> update sliding window
    new_cum_arr = jnp.sum(new_cum_arr_d_cur, axis=1)
    new_cum_dep = jnp.sum(new_cum_dep_d_cur, axis=1)

    new_arr_w = jnp.concatenate(
        [carry.cum_arrival_w[:, 1:], new_cum_arr[:, None]], axis=1)
    new_dep_w = jnp.concatenate(
        [carry.cum_departure_w[:, 1:], new_cum_dep[:, None]], axis=1)

    new_carry = ScanCarry(
        cum_arrival_w=new_arr_w,
        cum_departure_w=new_dep_w,
        demand_queue=new_dq,
        absorbed_count=new_absorbed,
        cum_arrival_d_cur=new_cum_arr_d_cur,
        cum_departure_d_cur=new_cum_dep_d_cur,
        prev_next_link_ids=next_link_ids,
        prev_dists=dists,
    )

    # DUO: per-dest aggregate is authoritative (may differ from node model aggregate)
    output = StepOutput(
        inflow_rates=jnp.sum(inflow_d, axis=1),
        outflow_rates=jnp.sum(outflow_d, axis=1),
        demand_queue=new_dq,
        inflow_d=inflow_d,
        outflow_d=outflow_d,
    )
    return new_carry, output


def duo_simulation_step_fwd(carry, t_index, params, link_state, config):
    """One DUO timestep for forward-only jax.lax.scan (full-array carry).

    Uses absolute ``t_index`` into full-history arrays instead of a sliding
    window.  Otherwise identical to ``duo_simulation_step``.

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
    n_links = config.n_links
    n_nodes = config.n_nodes
    n_dests = config.n_dests

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
    bf_dists, bf_preds = jax.lax.cond(
        should_update,
        lambda _: bellman_ford_all_dests(link_cost, config),
        lambda _: (carry.prev_dists, carry.prev_next_link_ids),
        None,
    )
    next_link_ids = bf_preds  # (n_dests, n_nodes) int32
    dists = bf_dists           # (n_dests, n_nodes) float32
    if not config.use_logit:
        dists = jax.lax.stop_gradient(dists)

    # Step 4: Dynamic diverge ratios -- vectorized tensor operations
    n_on_d = carry.cum_arrival_d_cur - carry.cum_departure_d_cur  # (n_links, n_dests)
    n_on_d = jnp.maximum(n_on_d, 0.0)

    is_origin = config.node_type == 0  # (n_nodes,)

    # Gather inlink vehicle counts per node: (n_nodes, max_in, n_dests)
    safe_inlinks = jnp.maximum(config.node_inlinks, 0)  # (n_nodes, max_in)
    inlink_valid = config.node_inlinks >= 0  # (n_nodes, max_in)
    inlink_veh = n_on_d[safe_inlinks]  # (n_nodes, max_in, n_dests)
    inlink_veh = inlink_veh * inlink_valid[:, :, None]  # mask invalid
    node_inlink_total = jnp.sum(inlink_veh, axis=1)  # (n_nodes, n_dests)

    # For origins: use OD demand as weight
    od_weights = params.od_demand_rate[:, :n_dests, t_index]  # (n_nodes, n_dests)
    dest_weight = jnp.where(is_origin[:, None], od_weights, node_inlink_total)  # (n_nodes, n_dests)

    # Build route indicator
    safe_outlinks = jnp.maximum(config.node_outlinks, 0)  # (n_nodes, max_out)
    outlink_valid = config.node_outlinks >= 0  # (n_nodes, max_out)

    if config.use_logit:
        # Soft logit routing: cost_via_outlink = link_cost + dist_to_dest_from_end_node
        outlink_end_nodes = config.link_end_node[safe_outlinks]  # (n_nodes, max_out)
        safe_end_nodes = jnp.maximum(outlink_end_nodes, 0)
        dist_from_end = dists[:, safe_end_nodes]  # (n_dests, n_nodes, max_out)
        cost_via = link_cost[safe_outlinks][None, :, :] + dist_from_end  # (n_dests, n_nodes, max_out)
        INF = 1e15
        cost_via = jnp.where(outlink_valid[None, :, :], cost_via, INF)
        logits = -cost_via / config.logit_temperature
        has_outlinks = config.node_n_outlinks > 0  # (n_nodes,)
        logits = jnp.where(
            ~has_outlinks[None, :, None],
            0.0,
            jnp.where(outlink_valid[None, :, :], logits, -INF))
        route_match = jax.nn.softmax(logits, axis=-1)  # (n_dests, n_nodes, max_out)
    else:
        # Hard routing: route_match[d, n, o] = (next_link_ids[d, n] == outlinks[n, o])
        route_match = (next_link_ids[:, :, None] == safe_outlinks[None, :, :]) & outlink_valid[None, :, :]
        route_match = route_match.astype(jnp.float32)

    # outlink_flow[n, o] = sum_d dest_weight[n, d] * route_match[d, n, o]
    outlink_flow = jnp.einsum("nd,dno->no", dest_weight, route_match)

    total = jnp.sum(outlink_flow, axis=1, keepdims=True)  # (n_nodes, 1)
    n_out = config.node_n_outlinks[:, None]  # (n_nodes, 1)
    equal_split = jnp.where(
        jnp.arange(config.max_out)[None, :] < n_out,
        1.0 / jnp.maximum(n_out, 1), 0.0)  # (n_nodes, max_out)
    duo_diverge = jnp.where(total > 1e-10, outlink_flow / jnp.maximum(total, 1e-10), equal_split)

    # Override diverge_ratios in params for this step
    duo_tf = jnp.broadcast_to(duo_diverge[:, None, :],
                               (n_nodes, config.max_in, config.max_out))
    params_duo = params._replace(diverge_ratios=duo_diverge, turning_fractions=duo_tf)

    # Step 5: Node models -> aggregate outflow/inflow (absolute t_index for demand_rate)
    inflows, outflows, new_dq, _ = compute_node_transfers(
        t_index, demands, supplies, state, params_duo, config)

    # Step 6: FIFO allocation of departures by destination
    total_arr = carry.cum_arrival[:, t_index]  # (n_links,)
    arr_d = carry.cum_arrival_d_cur  # (n_links, n_dests)
    fifo_ratio = jnp.where(
        total_arr[:, None] > 1e-10,
        arr_d / jnp.maximum(total_arr[:, None], 1e-10),
        0.0)  # (n_links, n_dests)
    outflow_d = outflows[:, None] * fifo_ratio  # (n_links, n_dests)

    # Step 7: Route per-dest traffic -- vectorized tensor operations
    new_absorbed = carry.absorbed_count

    inlink_outflow_d = outflow_d[safe_inlinks]  # (n_nodes, max_in, n_dests)
    inlink_outflow_d = inlink_outflow_d * inlink_valid[:, :, None]
    node_arriving = jnp.sum(inlink_outflow_d, axis=1)  # (n_nodes, n_dests)

    # Add OD demand at origins
    node_arriving = node_arriving + jnp.where(
        is_origin[:, None], od_weights, 0.0)

    # Absorb at destinations
    dest_mask = (config.dest_node_ids[None, :] == jnp.arange(n_nodes)[:, None])  # (n_nodes, n_dests)
    absorb_per_node = jnp.sum(jnp.where(dest_mask, node_arriving, 0.0), axis=1)  # (n_nodes,)
    new_absorbed = new_absorbed + absorb_per_node * dt
    node_arriving = jnp.where(dest_mask, 0.0, node_arriving)

    # Route to outlinks via routing probabilities
    route_match_ndo = route_match.transpose(1, 0, 2)  # (n_nodes, n_dests, max_out)
    routed = node_arriving[:, :, None] * route_match_ndo

    flat_lids = safe_outlinks.reshape(-1)
    flat_valid = outlink_valid.reshape(-1)
    flat_routed = routed.transpose(0, 2, 1).reshape(-1, n_dests)
    flat_routed = flat_routed * flat_valid[:, None]

    inflow_d = jnp.zeros((n_links, n_dests), dtype=jnp.float32)
    inflow_d = inflow_d.at[flat_lids].add(flat_routed)

    # Step 8: Update per-dest current cumulative counts
    new_cum_arr_d_cur = carry.cum_arrival_d_cur + dt * inflow_d
    new_cum_dep_d_cur = carry.cum_departure_d_cur + dt * outflow_d

    # Aggregate from per-dest -> update full arrays at t_index + 1
    new_cum_arr = jnp.sum(new_cum_arr_d_cur, axis=1)
    new_cum_dep = jnp.sum(new_cum_dep_d_cur, axis=1)

    new_carry = FwdCarry(
        cum_arrival=carry.cum_arrival.at[:, t_index + 1].set(new_cum_arr),
        cum_departure=carry.cum_departure.at[:, t_index + 1].set(new_cum_dep),
        demand_queue=new_dq,
        absorbed_count=new_absorbed,
        cum_arrival_d_cur=new_cum_arr_d_cur,
        cum_departure_d_cur=new_cum_dep_d_cur,
        prev_next_link_ids=next_link_ids,
        prev_dists=dists,
    )

    # DUO: per-dest aggregate is authoritative (may differ from node model aggregate)
    output = StepOutput(
        inflow_rates=jnp.sum(inflow_d, axis=1),
        outflow_rates=jnp.sum(outflow_d, axis=1),
        demand_queue=new_dq,
        inflow_d=inflow_d,
        outflow_d=outflow_d,
    )
    return new_carry, output


def simulate_duo(params, config, differentiable=True):
    """Run DUO simulation.

    Parameters
    ----------
    params : Params
    config : NetworkConfig
    differentiable : bool, optional
        If True (default), use windowed carry suitable for jax.grad.
        If False, use full-array carry for faster forward-only evaluation
        (not compatible with reverse-mode AD).

    Returns
    -------
    SimState
    """
    link_state = compute_link_state(params, config)
    n_dests = config.n_dests

    if differentiable:
        W = config.window_size
        init_carry = ScanCarry(
            cum_arrival_w=jnp.zeros((config.n_links, W + 1), dtype=jnp.float32),
            cum_departure_w=jnp.zeros((config.n_links, W + 1), dtype=jnp.float32),
            demand_queue=jnp.zeros(config.n_nodes, dtype=jnp.float32),
            absorbed_count=jnp.zeros(config.n_nodes, dtype=jnp.float32),
            cum_arrival_d_cur=jnp.zeros((config.n_links, n_dests), dtype=jnp.float32),
            cum_departure_d_cur=jnp.zeros((config.n_links, n_dests), dtype=jnp.float32),
            prev_next_link_ids=jnp.full((n_dests, config.n_nodes), -1, dtype=jnp.int32),
            prev_dists=jnp.full((n_dests, config.n_nodes), 1e15, dtype=jnp.float32),
        )
        step_fn = functools.partial(duo_simulation_step,
                                    params=params, link_state=link_state, config=config)
        final_carry, outputs = jax.lax.scan(step_fn, init_carry, jnp.arange(config.tsize))
        return _reconstruct_state(final_carry, outputs, config)
    else:
        init_carry = FwdCarry(
            cum_arrival=jnp.zeros((config.n_links, config.tsize + 1), dtype=jnp.float32),
            cum_departure=jnp.zeros((config.n_links, config.tsize + 1), dtype=jnp.float32),
            demand_queue=jnp.zeros(config.n_nodes, dtype=jnp.float32),
            absorbed_count=jnp.zeros(config.n_nodes, dtype=jnp.float32),
            cum_arrival_d_cur=jnp.zeros((config.n_links, n_dests), dtype=jnp.float32),
            cum_departure_d_cur=jnp.zeros((config.n_links, n_dests), dtype=jnp.float32),
            prev_next_link_ids=jnp.full((n_dests, config.n_nodes), -1, dtype=jnp.int32),
            prev_dists=jnp.full((n_dests, config.n_nodes), 1e15, dtype=jnp.float32),
        )
        step_fn = functools.partial(duo_simulation_step_fwd,
                                    params=params, link_state=link_state, config=config)
        final_carry, outputs = jax.lax.scan(step_fn, init_carry, jnp.arange(config.tsize))
        return _build_state_fwd(final_carry, outputs, config)


# ================================================================
# World -> JAX conversion
# ================================================================

def world_to_jax(W):
    """Convert a finalized World to JAX PyTree structures.

    Parameters
    ----------
    W : World
        Must have finalize_scenario() called.

    Returns
    -------
    params : Params
    config : NetworkConfig
    """
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
    demand_rate = np.zeros((n_nodes, tsize), dtype=np.float32)
    for node in W.NODES:
        for t_start, t_end, flow in node.demand_table:
            i_start = max(int(np.ceil(t_start / dt - 1e-10)), 0)
            i_end = min(int(np.ceil(t_end / dt - 1e-10)), tsize)
            demand_rate[node.id, i_start:i_end] += flow

    # Diverge ratios
    diverge_ratios = np.zeros((n_nodes, max_out), dtype=np.float32)
    for node in W.NODES:
        for j, link_name in enumerate(node.outlinks):
            diverge_ratios[node.id, j] = node.diverge_ratios.get(link_name, 0.0)

    # Turning fraction matrix (n_nodes, max_in, max_out)
    tf = np.zeros((n_nodes, max_in, max_out), dtype=np.float32)
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
    u = np.array([l.u for l in W.LINKS], dtype=np.float32)
    kappa = np.array([l.kappa for l in W.LINKS], dtype=np.float32)
    merge_priority = np.array([l.merge_priority for l in W.LINKS], dtype=np.float32)
    capacity_out = np.array([l.capacity_out for l in W.LINKS], dtype=np.float32)
    capacity_in = np.array([l.capacity_in for l in W.LINKS], dtype=np.float32)
    flow_capacity = np.array([n.flow_capacity for n in W.NODES], dtype=np.float32)
    absorption_ratio = np.array([n.absorption_ratio for n in W.NODES], dtype=np.float32)
    lengths = np.array([l.length for l in W.LINKS], dtype=np.float32)

    # DUO: destinations and per-OD demand
    destinations = list(set(dest for _, dest, _, _, _ in W.demand_info)) if W.demand_info else []
    dest_ids = np.array([W.NODES_NAME_DICT[d].id for d in destinations], dtype=np.int32) if destinations else np.array([0], dtype=np.int32)
    n_dests = max(len(destinations), 1)

    # Per-OD demand rate: (n_nodes, n_dests, tsize)
    od_demand_rate = np.zeros((n_nodes, n_dests, tsize), dtype=np.float32)
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
    w_speeds = np.array([l.w for l in W.LINKS], dtype=np.float32)
    offset_u_vals = lengths / (u * dt)
    offset_w_vals = lengths / (w_speeds * dt)
    max_offset = max(float(np.max(offset_u_vals)), float(np.max(offset_w_vals)))
    window_size = min(int(np.ceil(max_offset)) + 2, tsize)

    # Discretize congestion pricing toll at route_update_interval cadence
    route_update_interval = max(1, int(300 / float(dt)))
    n_toll_steps = max(1, int(np.ceil(tsize / route_update_interval)))
    toll_arr = np.zeros((n_links, n_toll_steps), dtype=np.float32)
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
    )

    params = Params(
        u=jnp.array(u),
        kappa=jnp.array(kappa),
        q_star=jnp.array(np.array([l.q_star for l in W.LINKS], dtype=np.float32)),
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
    )

    return params, config
