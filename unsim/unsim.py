"""
UNsim: Network Link Transmission Model (LTM) simulator.
"""

import numpy as np
from collections import defaultdict
from heapq import heappush, heappop

from .analyzer import Analyzer


def equal_tolerance(val, check, rel_tol=0.1, abs_tol=0.0):
    """Check if a value is within tolerance of expected value.

    Parameters
    ----------
    val : float
        Actual value.
    check : float
        Expected value.
    rel_tol : float, optional
        Relative tolerance. Default 0.1.
    abs_tol : float, optional
        Absolute tolerance. Default 0.0.

    Returns
    -------
    bool
        True if within tolerance.
    """
    if check == 0 and abs_tol == 0:
        abs_tol = 0.1
    return abs(val - check) <= abs(check * rel_tol) + abs_tol


def _interp(array, fractional_index):
    """Linearly interpolate array at a fractional index.

    Parameters
    ----------
    array : list[float]
        Array to interpolate.
    fractional_index : float
        Possibly non-integer index.

    Returns
    -------
    float
        Interpolated value.
    """
    if fractional_index <= 0:
        return array[0]
    max_idx = len(array) - 1
    if fractional_index >= max_idx:
        return array[max_idx]
    i_low = int(fractional_index)
    frac = fractional_index - i_low
    return array[i_low] * (1 - frac) + array[i_low + 1] * frac


class Node:
    """Network node.

    Parameters
    ----------
    W : World
        Parent world.
    name : str
        Node name.
    x : float
        X coordinate for visualization.
    y : float
        Y coordinate for visualization.
    flow_capacity : float, optional
        Maximum flow rate through this node (veh/s). Default 1e10 (inactive).
    diverge_ratio : dict[str, float] or None, optional
        Diverge ratios {outlink_name: ratio}. Used when auto_diverge=False.
        Ratios for pass-through flow should sum to 1.0. None for equal split.
    absorption_ratio : float or None, optional
        Fraction of inflow absorbed (destined for this node). 0 for pass-through.
        Used when auto_diverge=False. None defaults to 0.
    turning_fractions : dict[str, dict[str, float]] or None, optional
        Turning fraction matrix for general nodes (>=2 inlinks, >=2 outlinks).
        Format: {inlink_name: {outlink_name: fraction}}.
        Used by the INM (Incremental Node Model).

    Attributes
    ----------
    inlinks : dict[str, Link]
        Incoming links keyed by name.
    outlinks : dict[str, Link]
        Outgoing links keyed by name.
    demand_queue : float
        Vertical queue length at origin nodes (veh).
    diverge_ratios : dict[str, float]
        Fraction of pass-through flow going to each outlink.
    absorption_ratio : float
        Fraction of inflow destined for this node (0 for pass-through).
    absorbed_count : float
        Cumulative absorbed vehicles (veh).
    """

    def __init__(s, W, name, x, y, flow_capacity=1e10,
                 diverge_ratio=None, absorption_ratio=None,
                 turning_fractions=None):
        s.W = W
        s.name = name
        s.x = x
        s.y = y
        s.flow_capacity = flow_capacity
        s.id = len(W.NODES)
        s.inlinks = {}
        s.outlinks = {}
        s.demand_queue = 0.0
        s.demand_queue_history = []
        s.demand_table = []
        s._user_diverge_ratio = diverge_ratio      # {outlink_name: ratio} or None
        s._user_absorption_ratio = absorption_ratio  # float or None
        s._user_turning_fractions = turning_fractions  # {inlink: {outlink: beta}} or None
        s.diverge_ratios = {}
        s.absorption_ratio = 0.0
        s.absorbed_count = 0.0
        s.turning_fraction_matrix = None  # built by build_turning_fraction_matrix
        s._node_type = None

        W.NODES.append(s)
        W.NODES_NAME_DICT[name] = s

    def node_type(s):
        """Classify node type from link topology.

        Returns
        -------
        str
            One of "origin", "destination", "dummy", "merge", "diverge", "general".
        """
        if s._node_type is not None:
            return s._node_type
        has_in = len(s.inlinks) > 0
        has_out = len(s.outlinks) > 0
        if not has_in and has_out:
            s._node_type = "origin"
        elif has_in and not has_out:
            s._node_type = "destination"
        elif len(s.inlinks) == 1 and len(s.outlinks) == 1:
            s._node_type = "dummy"
        elif len(s.inlinks) >= 2 and len(s.outlinks) == 1:
            s._node_type = "merge"
        elif len(s.inlinks) == 1 and len(s.outlinks) >= 2:
            s._node_type = "diverge"
        else:
            s._node_type = "general"
        return s._node_type

    def get_demand_at(s, t):
        """Get external demand flow rate at time t.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Total demand flow rate (veh/s).
        """
        total = 0.0
        for t_start, t_end, flow in s.demand_table:
            if t_start <= t < t_end:
                total += flow
        return total

    def compute_transfer(s, t_index, demands, supplies):
        """Compute flow rates at this node for one timestep.

        Dispatches to the appropriate node model based on node type.

        Parameters
        ----------
        t_index : int
            Current timestep index.
        demands : dict[Link, float]
            Demand D_l for each link.
        supplies : dict[Link, float]
            Supply S_l for each link.
        """
        ntype = s.node_type()
        if ntype == "origin":
            s._transfer_origin(t_index, demands, supplies)
        elif ntype == "destination":
            s._transfer_destination(t_index, demands, supplies)
        elif ntype == "dummy":
            s._transfer_dummy(t_index, demands, supplies)
        elif ntype == "merge":
            s._transfer_merge(t_index, demands, supplies)
        elif ntype == "diverge":
            s._transfer_diverge(t_index, demands, supplies)
        elif ntype == "general":
            s._transfer_general(t_index, demands, supplies)

    def _transfer_origin(s, t_index, demands, supplies):
        """Origin node model with vertical queue.

        Parameters
        ----------
        t_index : int
            Current timestep index.
        demands : dict[Link, float]
            Demand D_l for each link.
        supplies : dict[Link, float]
            Supply S_l for each link.
        """
        dt = s.W.DELTAT
        t = t_index * dt
        external_demand = s.get_demand_at(t)

        if len(s.outlinks) == 1:
            outlink = list(s.outlinks.values())[0]
            effective_demand = external_demand + s.demand_queue / dt
            supply = supplies[outlink]
            flow = min(effective_demand, supply)
            flow = min(flow, s.flow_capacity)
            flow = max(flow, 0)
            outlink.inflow_rate = flow
            s.demand_queue += (external_demand - flow) * dt
            s.demand_queue = max(0.0, s.demand_queue)
        else:
            effective_demand = external_demand + s.demand_queue / dt
            min_flow = effective_demand
            for outlink in s.outlinks.values():
                beta = s.diverge_ratios.get(outlink.name, 0)
                if beta > 0:
                    min_flow = min(min_flow, supplies[outlink] / beta)
            min_flow = min(min_flow, s.flow_capacity)
            flow = max(min(min_flow, effective_demand), 0)
            for outlink in s.outlinks.values():
                beta = s.diverge_ratios.get(outlink.name, 0)
                outlink.inflow_rate = beta * flow
            s.demand_queue += (external_demand - flow) * dt
            s.demand_queue = max(0.0, s.demand_queue)

        s.demand_queue_history.append(s.demand_queue)

    def _transfer_destination(s, t_index, demands, supplies):
        """Destination node model. Accepts all incoming demand.

        Parameters
        ----------
        t_index : int
            Current timestep index.
        demands : dict[Link, float]
            Demand D_l for each link.
        supplies : dict[Link, float]
            Supply S_l for each link.
        """
        for inlink in s.inlinks.values():
            inlink.outflow_rate = demands[inlink]
            s.absorbed_count += demands[inlink] * s.W.DELTAT

    def _transfer_dummy(s, t_index, demands, supplies):
        """Dummy node model (1 inlink, 1 outlink) with optional absorption.

        Parameters
        ----------
        t_index : int
            Current timestep index.
        demands : dict[Link, float]
            Demand D_l for each link.
        supplies : dict[Link, float]
            Supply S_l for each link.
        """
        inlink = list(s.inlinks.values())[0]
        outlink = list(s.outlinks.values())[0]
        D = demands[inlink]
        pass_ratio = 1 - s.absorption_ratio

        if pass_ratio > 1e-10:
            flow = min(D, supplies[outlink] / pass_ratio)
        else:
            flow = D
        flow = min(flow, s.flow_capacity)
        flow = max(flow, 0)
        inlink.outflow_rate = flow
        outlink.inflow_rate = flow * pass_ratio
        s.absorbed_count += flow * s.absorption_ratio * s.W.DELTAT

    def _transfer_merge(s, t_index, demands, supplies):
        """Merge node model. Uses mid-value formula for 2-link case.

        Parameters
        ----------
        t_index : int
            Current timestep index.
        demands : dict[Link, float]
            Demand D_l for each link.
        supplies : dict[Link, float]
            Supply S_l for each link.
        """
        inlinks = list(s.inlinks.values())
        outlink = list(s.outlinks.values())[0]

        S = supplies[outlink]
        S = min(S, s.flow_capacity)

        Ds = [demands[l] for l in inlinks]
        total_D = sum(Ds)

        if total_D <= S:
            for i, l in enumerate(inlinks):
                l.outflow_rate = Ds[i]
            outlink.inflow_rate = total_D
        else:
            priorities = [l.merge_priority for l in inlinks]
            total_p = sum(priorities)
            alphas = [p / total_p for p in priorities]

            if len(inlinks) == 2:
                D1, D2 = Ds[0], Ds[1]
                a1, a2 = alphas[0], alphas[1]
                q1 = sorted([D1, S - D2, a1 * S])[1]
                q2 = sorted([D2, S - D1, a2 * S])[1]
                q1 = max(q1, 0)
                q2 = max(q2, 0)
                inlinks[0].outflow_rate = q1
                inlinks[1].outflow_rate = q2
                outlink.inflow_rate = q1 + q2
            else:
                flows = []
                for i in range(len(inlinks)):
                    flows.append(min(Ds[i], alphas[i] * S))
                total_flow = sum(flows)
                if total_flow < S:
                    remaining = S - total_flow
                    for i in range(len(inlinks)):
                        extra = min(Ds[i] - flows[i], remaining)
                        if extra > 0:
                            flows[i] += extra
                            remaining -= extra
                for i, l in enumerate(inlinks):
                    l.outflow_rate = max(flows[i], 0)
                outlink.inflow_rate = sum(flows)

    def _transfer_diverge(s, t_index, demands, supplies):
        """Diverge node model. Splits flow by diverge ratios.

        Parameters
        ----------
        t_index : int
            Current timestep index.
        demands : dict[Link, float]
            Demand D_l for each link.
        supplies : dict[Link, float]
            Supply S_l for each link.
        """
        inlink = list(s.inlinks.values())[0]
        outlinks = list(s.outlinks.values())

        D = demands[inlink]
        min_flow = D
        for outlink in outlinks:
            beta = s.diverge_ratios.get(outlink.name, 0)
            if beta > 0:
                min_flow = min(min_flow, supplies[outlink] / beta)
        if s.flow_capacity is not None:
            min_flow = min(min_flow, s.flow_capacity)
        flow = max(min(min_flow, D), 0)

        inlink.outflow_rate = flow
        for outlink in outlinks:
            beta = s.diverge_ratios.get(outlink.name, 0)
            outlink.inflow_rate = beta * flow

    def build_turning_fraction_matrix(s):
        """Build the I x J turning fraction matrix from user specification.

        Called by ``World.finalize_scenario()`` for general nodes.
        If turning_fractions is not specified, defaults to equal split 1/J.
        """
        inlinks_list = list(s.inlinks.values())
        outlinks_list = list(s.outlinks.values())
        I = len(inlinks_list)
        J = len(outlinks_list)

        B = [[0.0] * J for _ in range(I)]

        if s._user_turning_fractions:
            in_idx = {l.name: i for i, l in enumerate(inlinks_list)}
            out_idx = {l.name: j for j, l in enumerate(outlinks_list)}
            for in_name, out_dict in s._user_turning_fractions.items():
                i = in_idx.get(in_name)
                if i is None:
                    continue
                for out_name, beta in out_dict.items():
                    j = out_idx.get(out_name)
                    if j is not None:
                        B[i][j] = beta
        else:
            equal_share = 1.0 / J if J > 0 else 0.0
            for i in range(I):
                for j in range(J):
                    B[i][j] = equal_share

        s.turning_fraction_matrix = B

    def _transfer_general(s, t_index, demands, supplies):
        """General node model using the Incremental Node Model (Floetteroed 2011).

        Handles arbitrary m-input x n-output nodes with turning fraction
        matrix B[i][j] and merge priority alpha[i].

        Parameters
        ----------
        t_index : int
            Current timestep index.
        demands : dict[Link, float]
            Demand D_l for each link.
        supplies : dict[Link, float]
            Supply S_l for each link.
        """
        inlinks_list = list(s.inlinks.values())
        outlinks_list = list(s.outlinks.values())
        I = len(inlinks_list)
        J = len(outlinks_list)

        Delta = [demands[l] for l in inlinks_list]
        Sigma = [supplies[l] for l in outlinks_list]
        alpha = [l.merge_priority for l in inlinks_list]
        B = s.turning_fraction_matrix

        q_in = [0.0] * I
        q_out = [0.0] * J
        EPS = 1e-10

        for _ in range(I + J + 1):
            # D_up: inlinks that can still send flow (FIFO condition)
            D_up = [False] * I
            for i in range(I):
                if q_in[i] >= Delta[i] - EPS:
                    continue
                blocked = False
                for j in range(J):
                    if B[i][j] > 0 and q_out[j] >= Sigma[j] - EPS:
                        blocked = True
                        break
                D_up[i] = not blocked

            # D_down: outlinks that can still receive flow from active inlinks
            D_down = [False] * J
            for j in range(J):
                if q_out[j] >= Sigma[j] - EPS:
                    continue
                for i in range(I):
                    if D_up[i] and B[i][j] > 0:
                        D_down[j] = True
                        break

            if not any(D_up):
                break

            # Compute phi (priority-weighted direction)
            phi_in = [alpha[i] if D_up[i] else 0.0 for i in range(I)]
            phi_out = [sum(B[i][j] * phi_in[i] for i in range(I)) for j in range(J)]

            # Compute theta (maximum step size)
            theta = float('inf')
            for i in range(I):
                if D_up[i] and phi_in[i] > 1e-15:
                    theta = min(theta, (Delta[i] - q_in[i]) / phi_in[i])
            for j in range(J):
                if D_down[j] and phi_out[j] > 1e-15:
                    theta = min(theta, (Sigma[j] - q_out[j]) / phi_out[j])

            if theta <= 0 or theta > 1e17:
                break

            # Update flows
            for i in range(I):
                q_in[i] += theta * phi_in[i]
            for j in range(J):
                q_out[j] += theta * phi_out[j]

        # Store results
        for i, link in enumerate(inlinks_list):
            link.outflow_rate = q_in[i]
        for j, link in enumerate(outlinks_list):
            link.inflow_rate = q_out[j]

    def __repr__(s):
        return f"<Node '{s.name}'>"


class Link:
    """Network link with triangular fundamental diagram.

    Parameters
    ----------
    W : World
        Parent world.
    name : str
        Link name.
    start_node : Node or str
        Upstream node (or its name).
    end_node : Node or str
        Downstream node (or its name).
    length : float
        Link length (m).
    free_flow_speed : float, optional
        Free flow speed u (m/s). Default 20.
    jam_density : float, optional
        Jam density kappa (veh/m). Default 0.2.
    merge_priority : float, optional
        Merge priority weight at downstream merge nodes. Default 1.
    capacity_out : float, optional
        Maximum outflow rate (veh/s). Default 1e10 (inactive).
    capacity_in : float, optional
        Maximum inflow rate (veh/s). Default 1e10 (inactive).
    capacity : float or None, optional
        Link capacity q* (veh/s). Overrides default derivation.
    backward_wave_speed : float or None, optional
        Congestion wave speed w (m/s). Overrides default derivation.
    congestion_pricing : callable or None, optional
        Dynamic congestion pricing function. Takes time (s) as input,
        returns toll value in seconds (time equivalent). Used as
        additional cost in DUO route choice. Default None (no toll).

    Attributes
    ----------
    u : float
        Free flow speed (m/s).
    w : float
        Congestion wave speed (m/s).
    kappa : float
        Jam density (veh/m).
    q_star : float
        Capacity (veh/s).
    k_star : float
        Critical density (veh/m).
    cum_arrival : list[float]
        Cumulative inflow count N_U at each timestep.
    cum_departure : list[float]
        Cumulative outflow count N_D at each timestep.
    """

    def __init__(s, W, name, start_node, end_node, length,
                 free_flow_speed=20, jam_density=0.2,
                 merge_priority=1, capacity_out=1e10, capacity_in=1e10,
                 capacity=None, backward_wave_speed=None,
                 congestion_pricing=None):
        s.W = W
        s.name = name
        s.length = length
        s.merge_priority = merge_priority
        s.capacity_out = capacity_out
        s.capacity_in = capacity_in

        if isinstance(start_node, str):
            start_node = W.get_node(start_node)
        if isinstance(end_node, str):
            end_node = W.get_node(end_node)
        s.start_node = start_node
        s.end_node = end_node

        start_node.outlinks[name] = s
        end_node.inlinks[name] = s

        s.u = free_flow_speed
        s.tau = W.REACTION_TIME

        if backward_wave_speed is not None:
            s.w = backward_wave_speed
            s.kappa = 1 / (s.tau * s.w) if capacity is None else None
        else:
            s.kappa = jam_density
            s.w = 1 / (s.tau * s.kappa)

        if capacity is not None:
            s.q_star = capacity
            s.capacity = capacity
            # Derive kappa from u, w, q*: q* = u*w*kappa/(u+w) -> kappa = q*(u+w)/(u*w)
            s.kappa = capacity * (s.u + s.w) / (s.u * s.w)
            s.k_star = capacity / s.u
        else:
            s.capacity = s.u * s.w * s.kappa / (s.u + s.w)
            s.q_star = s.capacity
            s.k_star = s.capacity / s.u

        s.offset_u = None
        s.offset_w = None
        s.cum_arrival = None
        s.cum_departure = None
        s.inflow_rate = 0.0
        s.outflow_rate = 0.0
        # Congestion pricing
        s.congestion_pricing = congestion_pricing

        # DUO per-destination tracking
        s.cum_arrival_d = {}   # {dest_name: [float]}
        s.cum_departure_d = {}
        s.inflow_rate_d = {}   # {dest_name: float} current timestep
        s.outflow_rate_d = {}  # {dest_name: float}

        s.id = len(W.LINKS)
        W.LINKS.append(s)
        W.LINKS_NAME_DICT[name] = s

    def get_toll(s, t):
        """Get congestion toll at time t.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Toll value in seconds (time equivalent).
        """
        if s.congestion_pricing is not None:
            return s.congestion_pricing(t)
        return 0.0

    def init_after_scenario(s):
        """Initialize cumulative count arrays and precompute offsets.

        Called by ``World.finalize_scenario()``.
        """
        dt = s.W.DELTAT
        tsize = s.W.TSIZE
        s.offset_u = s.length / (s.u * dt)
        s.offset_w = s.length / (s.w * dt)
        s.cum_arrival = [0.0] * (tsize + 1)
        s.cum_departure = [0.0] * (tsize + 1)

    def compute_demand(s, t_index):
        """Compute sending flow (demand) D_l(t).

        D_l(t) = min{(N_U(t+dt-d/u) - N_D(t)) / dt, q*, capacity_out}

        Parameters
        ----------
        t_index : int
            Current timestep index.

        Returns
        -------
        float
            Demand flow rate (veh/s).
        """
        dt = s.W.DELTAT
        N_U_past = _interp(s.cum_arrival, t_index + 1 - s.offset_u)
        N_D_now = s.cum_departure[t_index]

        D = min((N_U_past - N_D_now) / dt, s.q_star)
        D = max(D, 0)

        D = min(D, s.capacity_out)
        return D

    def compute_supply(s, t_index):
        """Compute receiving flow (supply) S_l(t).

        S_l(t) = min{(N_D(t+dt-d/w) + kappa*d - N_U(t)) / dt, q*, capacity_in}

        Parameters
        ----------
        t_index : int
            Current timestep index.

        Returns
        -------
        float
            Supply flow rate (veh/s).
        """
        dt = s.W.DELTAT
        N_D_past = _interp(s.cum_departure, t_index + 1 - s.offset_w)
        N_U_now = s.cum_arrival[t_index]

        S = min((N_D_past + s.kappa * s.length - N_U_now) / dt, s.q_star)
        S = max(S, 0)

        S = min(S, s.capacity_in)
        return S

    def compute_N(s, t_seconds, x):
        """Compute cumulative count N(t, x) using Newell's simplified KW theory.

        N(t,x) = min{N_U(t - x/u), N_D(t - (d-x)/w) + kappa*(d-x)}

        Parameters
        ----------
        t_seconds : float
            Time in seconds.
        x : float
            Position from upstream end (m). 0 <= x <= length.

        Returns
        -------
        float
            Cumulative vehicle count.
        """
        dt = s.W.DELTAT
        t_free = t_seconds - x / s.u
        idx_free = t_free / dt
        N_free = _interp(s.cum_arrival, idx_free)

        t_cong = t_seconds - (s.length - x) / s.w
        idx_cong = t_cong / dt
        N_cong = _interp(s.cum_departure, idx_cong) + s.kappa * (s.length - x)

        return min(N_free, N_cong)

    def _compute_state_point(s, t, x):
        """Compute (q, k, v) at a point analytically from Newell's formula.

        Determines free-flow or congested regime, then computes q from the
        derivative of the active boundary curve. k and v follow from the FD.
        This guarantees FD consistency (no numerical differentiation artifacts).

        Parameters
        ----------
        t : float
            Time (s).
        x : float
            Position from upstream end (m).

        Returns
        -------
        q : float
            Flow (veh/s).
        k : float
            Density (veh/m).
        v : float
            Speed (m/s).
        """
        dt = s.W.DELTAT

        # Free-flow branch
        t_free = t - x / s.u
        idx_free = t_free / dt
        N_free = _interp(s.cum_arrival, idx_free)

        # Congested branch
        t_cong = t - (s.length - x) / s.w
        idx_cong = t_cong / dt
        N_cong = _interp(s.cum_departure, idx_cong) + s.kappa * (s.length - x)

        if N_free <= N_cong:
            # Free-flow regime: q = dN_U/dt at the reference time
            i = max(0, min(int(idx_free), len(s.cum_arrival) - 2))
            q = max((s.cum_arrival[i + 1] - s.cum_arrival[i]) / dt, 0)
            q = min(q, s.q_star)
            k = q / s.u
            v = s.u
        else:
            # Congested regime: q = dN_D/dt at the reference time
            i = max(0, min(int(idx_cong), len(s.cum_departure) - 2))
            q = max((s.cum_departure[i + 1] - s.cum_departure[i]) / dt, 0)
            q = min(q, s.q_star)
            k = s.kappa - q / s.w
            v = q / k if k > 1e-10 else 0
        return q, k, v

    @staticmethod
    def _parse_arg(val, default):
        """Parse a scalar or 2-element range [lo, hi].

        Returns
        -------
        is_range : bool
        lo : float
        hi : float
            For scalars, lo == hi == val.
        """
        if val is None:
            return False, default, default
        a = np.asarray(val, dtype=float)
        if a.ndim == 0:
            v = a.item()
            return False, v, v
        if a.shape == (2,):
            return True, float(a[0]), float(a[1])
        raise ValueError(f"Expected scalar or 2-element range, got shape {a.shape}")

    def q(s, t, x=None):
        """Compute flow rate q = dN/dt (veh/s) using Newell's formula.

        Averages are computed directly from N differences:
        - q([t0,t1], x)         = (N(t1,x) - N(t0,x)) / (t1-t0)
        - q([t0,t1], [x0,x1])   = Edie's q from 4-corner N values.

        Parameters
        ----------
        t : float or list[float]
            Time (s). Scalar for a point, [t0, t1] for time-range average.
        x : float or list[float] or None, optional
            Position from upstream end (m). Scalar for a point,
            [x0, x1] for space-range average. None for link midpoint.

        Returns
        -------
        float
            Flow rate, or average flow rate over the specified range.
        """
        t_rng, t0, t1 = s._parse_arg(t, None)
        x_rng, x0, x1 = s._parse_arg(x, s.length / 2)

        if t_rng and x_rng:
            # Edie's q = total distance / area, trapezoidal in x
            dt = t1 - t0
            return max(((s.compute_N(t1, x0) - s.compute_N(t0, x0))
                      + (s.compute_N(t1, x1) - s.compute_N(t0, x1))) / (2 * dt), 0)
        elif t_rng:
            # Exact: (N(t1,x) - N(t0,x)) / dt
            return max((s.compute_N(t1, x0) - s.compute_N(t0, x0)) / (t1 - t0), 0)
        elif x_rng:
            # Average of point values at boundaries
            q0 = s._compute_state_point(t0, x0)[0]
            q1 = s._compute_state_point(t0, x1)[0]
            return (q0 + q1) / 2
        else:
            return s._compute_state_point(t0, x0)[0]

    def k(s, t, x=None):
        """Compute density k = -dN/dx (veh/m) using Newell's formula.

        Averages are computed directly from N differences:
        - k(t, [x0,x1])         = (N(t,x0) - N(t,x1)) / (x1-x0)
        - k([t0,t1], [x0,x1])   = Edie's k from 4-corner N values.

        Parameters
        ----------
        t : float or list[float]
            Time (s). Scalar for a point, [t0, t1] for time-range average.
        x : float or list[float] or None, optional
            Position from upstream end (m). Scalar for a point,
            [x0, x1] for space-range average. None for link midpoint.

        Returns
        -------
        float
            Density, or average density over the specified range.
        """
        t_rng, t0, t1 = s._parse_arg(t, None)
        x_rng, x0, x1 = s._parse_arg(x, s.length / 2)

        if t_rng and x_rng:
            # Edie's k = total time / area, trapezoidal in t
            dx = x1 - x0
            return max(((s.compute_N(t0, x0) - s.compute_N(t0, x1))
                      + (s.compute_N(t1, x0) - s.compute_N(t1, x1))) / (2 * dx), 0)
        elif x_rng:
            # Exact: (N(t,x0) - N(t,x1)) / dx
            return max((s.compute_N(t0, x0) - s.compute_N(t0, x1)) / (x1 - x0), 0)
        elif t_rng:
            # Average of point values at boundaries
            k0 = s._compute_state_point(t0, x0)[1]
            k1 = s._compute_state_point(t1, x0)[1]
            return (k0 + k1) / 2
        else:
            return s._compute_state_point(t0, x0)[1]

    def v(s, t, x=None):
        """Compute speed v = q/k (m/s) using Newell's formula.

        For point queries, uses analytical FD-consistent computation.
        For range queries, v = q_range / k_range (space-mean speed).

        Parameters
        ----------
        t : float or list[float]
            Time (s). Scalar for a point, [t0, t1] for time-range average.
        x : float or list[float] or None, optional
            Position from upstream end (m). Scalar for a point,
            [x0, x1] for space-range average. None for link midpoint.

        Returns
        -------
        float
            Speed, or average speed over the specified range.
        """
        t_rng, t0, t1 = s._parse_arg(t, None)
        x_rng, x0, x1 = s._parse_arg(x, s.length / 2)
        if not t_rng and not x_rng:
            return s._compute_state_point(t0, x0)[2]
        q_val = s.q(t, x)
        k_val = s.k(t, x)
        return q_val / k_val if k_val > 1e-10 else float(s.u)

    def instantaneous_travel_time(s, t_seconds, method="multipoint", n_segments=10):
        """Compute instantaneous link travel time.

        Parameters
        ----------
        t_seconds : float
            Current time (s).
        method : str, optional
            "avg_density" -- single average density k_avg=(N_U-N_D)/L -> FD -> tau=L/v.
            "multipoint" -- spatial integration with N(t,x) curves (default).
        n_segments : int, optional
            Number of segments for multipoint method. Default 10.

        Returns
        -------
        float
            Instantaneous travel time (s).
        """
        if method == "avg_density":
            dt = s.W.DELTAT
            t_idx = int(t_seconds / dt)
            t_idx = max(0, min(t_idx, len(s.cum_arrival) - 1))
            n_on = max(s.cum_arrival[t_idx] - s.cum_departure[t_idx], 0)
            k_avg = n_on / s.length
            k_star = s.q_star / s.u
            if k_avg <= k_star:
                v = s.u
            else:
                v = s.w * (s.kappa - k_avg) / max(k_avg, 1e-10)
            v = max(min(v, s.u), 0.01)
            return s.length / v

        # multipoint: spatial integration
        L = s.length
        dx = L / n_segments
        k_star = s.q_star / s.u
        total_tt = 0.0

        for i in range(n_segments):
            x0 = i * dx
            x1 = (i + 1) * dx
            N0 = s.compute_N(t_seconds, x0)
            N1 = s.compute_N(t_seconds, x1)
            k_seg = max((N0 - N1) / dx, 0)

            if k_seg <= k_star:
                v = s.u
            else:
                v = s.w * (s.kappa - k_seg) / max(k_seg, 1e-10)
            v = max(min(v, s.u), 0.01)
            total_tt += dx / v

        return total_tt

    def __repr__(s):
        return f"<Link '{s.name}'>"


class World:
    """Simulation world managing nodes, links, demand, and execution.

    Parameters
    ----------
    name : str, optional
        Scenario name.
    deltat : float or None, optional
        Simulation timestep width (s). None (default) for automatic setting
        as min(d_l/u_l) across all links.
    tmax : float, optional
        Total simulation duration (s).
    print_mode : int, optional
        Print progress messages. Default 1.
    save_mode : int, optional
        Save results. Default 1.
    reaction_time : float, optional
        Reaction time for FD computation (s). Default 1.
    auto_diverge : bool, optional
        If True, automatically compute diverge/absorption ratios from OD demand
        composition and shortest paths. If False (default), use ratios specified
        in addNode (diverge_ratio, absorption_ratio).

    Attributes
    ----------
    NODES : list[Node]
        All nodes.
    LINKS : list[Link]
        All links.
    analyzer : Analyzer
        Result analyzer (created after finalize_scenario).
    """

    def __init__(s, name="", deltat=None, tmax=2000, print_mode=1,
                 save_mode=1, show_mode=0, reaction_time=1,
                 random_seed=None, auto_diverge=False,
                 route_choice=None, logit_temperature=60.0, **kwargs):
        s.NAME = name
        s.DELTAT = deltat  # None = auto-compute in finalize_scenario
        s.TMAX = tmax
        s.REACTION_TIME = reaction_time
        s.AUTO_DIVERGE = auto_diverge
        # Route choice: "fix" (default), "aon", "duo", "duo_multipoint", "duo_logit"
        if route_choice is None:
            route_choice = "fix"
        s.ROUTE_CHOICE = route_choice
        s.LOGIT_TEMPERATURE = logit_temperature
        s.print_mode = print_mode
        s.save_mode = save_mode

        s.NODES = []
        s.LINKS = []
        s.NODES_NAME_DICT = {}
        s.LINKS_NAME_DICT = {}
        s.demand_info = []

        s.TSIZE = 0
        s.T = 0
        s.TIME = 0.0
        s.finalized = False
        s.analyzer = None

        s._shortest_paths = {}
        s._ff_travel_times = {}

    def get_link(s, name):
        """Get a link by name.

        Parameters
        ----------
        name : str or Link
            Link name, or Link object (returned as-is).

        Returns
        -------
        Link
        """
        if isinstance(name, Link):
            return name
        return s.LINKS_NAME_DICT[name]

    def get_node(s, name):
        """Get a node by name.

        Parameters
        ----------
        name : str or Node
            Node name, or Node object (returned as-is).

        Returns
        -------
        Node
        """
        if isinstance(name, Node):
            return name
        return s.NODES_NAME_DICT[name]

    def addNode(s, name, x=0, y=0, flow_capacity=1e10,
                diverge_ratio=None, absorption_ratio=None,
                turning_fractions=None, **kwargs):
        """Add a node to the network.

        Parameters
        ----------
        name : str
            Node name.
        x : float, optional
            X coordinate for visualization.
        y : float, optional
            Y coordinate for visualization.
        flow_capacity : float, optional
            Maximum flow rate through this node (veh/s). Default 1e10 (inactive).
        diverge_ratio : dict[str, float] or None, optional
            Diverge ratios {outlink_name: ratio}. Used when auto_diverge=False.
        absorption_ratio : float or None, optional
            Fraction of inflow absorbed at this node. Used when auto_diverge=False.
        turning_fractions : dict[str, dict[str, float]] or None, optional
            Turning fraction matrix for general nodes.
            Format: {inlink_name: {outlink_name: fraction}}.

        Returns
        -------
        Node
            The created node.
        """
        node = Node(s, name, x, y, flow_capacity=flow_capacity,
                    diverge_ratio=diverge_ratio, absorption_ratio=absorption_ratio,
                    turning_fractions=turning_fractions)
        return node

    def addLink(s, name, start_node, end_node, length,
                free_flow_speed=20, jam_density=0.2,
                merge_priority=1, capacity_out=1e10, capacity_in=1e10,
                capacity=None, backward_wave_speed=None,
                congestion_pricing=None,
                **kwargs):
        """Add a link to the network.

        FD is triangular. By default, derived from free_flow_speed, jam_density,
        and reaction_time. Optionally override capacity and/or backward_wave_speed
        directly; jam_density is then recomputed for consistency.

        Parameters
        ----------
        name : str
            Link name.
        start_node : Node or str
            Upstream node (or its name).
        end_node : Node or str
            Downstream node (or its name).
        length : float
            Link length (m).
        free_flow_speed : float, optional
            Free flow speed u (m/s). Default 20.
        jam_density : float, optional
            Jam density kappa (veh/m). Default 0.2.
        merge_priority : float, optional
            Merge priority weight. Default 1.
        capacity_out : float, optional
            Maximum outflow rate (veh/s). Default 1e10 (inactive).
        capacity_in : float, optional
            Maximum inflow rate (veh/s). Default 1e10 (inactive).
        capacity : float or None, optional
            Link capacity q* (veh/s). Overrides the default q*=u*w*kappa/(u+w).
        backward_wave_speed : float or None, optional
            Congestion wave speed w (m/s). Overrides w=1/(tau*kappa).
        congestion_pricing : callable or None, optional
            Dynamic congestion pricing function. Takes time (s) as input,
            returns toll value in seconds (time equivalent). Used as
            additional cost in DUO route choice. Default None (no toll).

        Returns
        -------
        Link
            The created link.
        """
        link = Link(s, name, start_node, end_node, length,
                    free_flow_speed=free_flow_speed,
                    jam_density=jam_density,
                    merge_priority=merge_priority,
                    capacity_out=capacity_out,
                    capacity_in=capacity_in,
                    capacity=capacity,
                    backward_wave_speed=backward_wave_speed,
                    congestion_pricing=congestion_pricing)
        return link

    def adddemand(s, orig, dest, t_start, t_end, flow=-1, volume=-1, **kwargs):
        """Add OD traffic demand.

        Parameters
        ----------
        orig : str or Node
            Origin node (name or object).
        dest : str or Node
            Destination node (name or object).
        t_start : float
            Demand start time (s).
        t_end : float
            Demand end time (s).
        flow : float, optional
            Demand flow rate (veh/s). Specify either flow or volume.
        volume : float, optional
            Total demand volume (veh). Converted to flow internally.
        """
        if not isinstance(orig, str):
            orig = orig.name
        if not isinstance(dest, str):
            dest = dest.name
        if volume > 0 and flow < 0:
            flow = volume / (t_end - t_start)
        s.demand_info.append((orig, dest, t_start, t_end, flow))

    def _compute_shortest_paths(s):
        """Compute shortest paths by free-flow time for all OD pairs using Dijkstra."""
        adj = defaultdict(list)
        for link in s.LINKS:
            ff_time = link.length / link.u
            adj[link.start_node.name].append((link.end_node.name, link.name, ff_time))

        od_pairs = set()
        for orig, dest, _, _, _ in s.demand_info:
            od_pairs.add((orig, dest))

        for orig_name, dest_name in od_pairs:
            dist = {orig_name: 0}
            prev = {}
            heap = [(0, orig_name)]
            visited = set()

            while heap:
                d, u = heappop(heap)
                if u in visited:
                    continue
                visited.add(u)
                if u == dest_name:
                    break
                for v, link_name, ff_time in adj[u]:
                    nd = d + ff_time
                    if v not in dist or nd < dist[v]:
                        dist[v] = nd
                        prev[v] = (u, link_name)
                        heappush(heap, (nd, v))

            if dest_name in prev or orig_name == dest_name:
                path = []
                current = dest_name
                while current != orig_name:
                    prev_node, link_name = prev[current]
                    path.append(link_name)
                    current = prev_node
                path.reverse()
                s._shortest_paths[(orig_name, dest_name)] = path
                s._ff_travel_times[(orig_name, dest_name)] = dist.get(dest_name, 0)

    def _get_free_flow_travel_time(s, orig_name, dest_name):
        """Get free-flow travel time for an OD pair.

        Parameters
        ----------
        orig_name : str
            Origin node name.
        dest_name : str
            Destination node name.

        Returns
        -------
        float
            Free-flow travel time (s).
        """
        return s._ff_travel_times.get((orig_name, dest_name), 0)

    def _compute_diverge_and_absorption_ratios(s):
        """Compute diverge ratios and absorption ratios from demand composition.

        For each node with outlinks, determines what fraction of flow is absorbed
        (destined for this node) vs. passes through to each outlink.
        """
        for node in s.NODES:
            if len(node.outlinks) == 0:
                continue

            outlink_flow = defaultdict(float)
            absorbed_flow = 0.0
            total_flow = 0.0

            for orig, dest, t_start, t_end, flow in s.demand_info:
                path = s._shortest_paths.get((orig, dest), [])
                vol = flow * (t_end - t_start)

                for i, link_name in enumerate(path):
                    link = s.LINKS_NAME_DICT[link_name]
                    if link.end_node == node:
                        total_flow += vol
                        if i + 1 < len(path):
                            next_link_name = path[i + 1]
                            outlink_flow[next_link_name] += vol
                        else:
                            absorbed_flow += vol
                        break

                if orig == node.name:
                    total_flow += vol
                    if path:
                        outlink_flow[path[0]] += vol
                    else:
                        absorbed_flow += vol

            if total_flow > 0:
                node.absorption_ratio = absorbed_flow / total_flow
                pass_through = total_flow - absorbed_flow
                if pass_through > 0 and len(node.outlinks) > 0:
                    for outlink_name in node.outlinks:
                        node.diverge_ratios[outlink_name] = outlink_flow[outlink_name] / pass_through
                else:
                    for outlink_name in node.outlinks:
                        node.diverge_ratios[outlink_name] = 0.0
            elif len(node.outlinks) > 0:
                n_out = len(node.outlinks)
                for outlink_name in node.outlinks:
                    node.diverge_ratios[outlink_name] = 1.0 / n_out

    def _apply_user_diverge_ratios(s):
        """Apply user-specified diverge ratios and absorption ratios from addNode.

        For nodes without user specification, defaults to equal split (diverge)
        and 0 absorption.
        """
        for node in s.NODES:
            # Absorption ratio
            if node._user_absorption_ratio is not None:
                node.absorption_ratio = node._user_absorption_ratio
            else:
                node.absorption_ratio = 0.0

            # Diverge ratios
            if len(node.outlinks) == 0:
                continue
            if node._user_diverge_ratio is not None:
                for outlink_name in node.outlinks:
                    node.diverge_ratios[outlink_name] = node._user_diverge_ratio.get(outlink_name, 0.0)
            else:
                # Default: equal split
                n_out = len(node.outlinks)
                for outlink_name in node.outlinks:
                    node.diverge_ratios[outlink_name] = 1.0 / n_out

    def _register_demands_to_origins(s):
        """Register demand entries to origin nodes' demand_table."""
        for orig, dest, t_start, t_end, flow in s.demand_info:
            orig_node = s.NODES_NAME_DICT[orig]
            orig_node.demand_table.append((t_start, t_end, flow))

    def finalize_scenario(s):
        """Finalize scenario setup before simulation.

        If deltat is None, automatically sets it to min(d_l/u_l) across all links.
        Validates deltat constraint, initializes arrays, computes shortest paths,
        diverge/absorption ratios, and creates Analyzer.

        Raises
        ------
        ValueError
            If deltat exceeds d_l/u_l for any link.
        """
        if s.finalized:
            return

        # Auto-compute deltat if not specified
        if s.DELTAT is None:
            if not s.LINKS:
                s.DELTAT = 1.0
            else:
                s.DELTAT = min(link.length / link.u for link in s.LINKS)

        s.TSIZE = int(s.TMAX / s.DELTAT)

        for link in s.LINKS:
            max_dt = link.length / link.u
            if s.DELTAT > max_dt + 1e-10:
                raise ValueError(
                    f"DELTAT={s.DELTAT} exceeds d/u={max_dt:.2f} for link '{link.name}'. "
                    f"Reduce DELTAT to at most {max_dt:.2f}."
                )

        for link in s.LINKS:
            link.init_after_scenario()

        for node in s.NODES:
            node.node_type()

        # Build turning fraction matrices for general nodes
        for node in s.NODES:
            if node.node_type() == "general":
                node.build_turning_fraction_matrix()

        # Shortest paths always computed (for free-flow travel time / delay)
        s._compute_shortest_paths()

        if s.AUTO_DIVERGE:
            s._compute_diverge_and_absorption_ratios()
        else:
            s._apply_user_diverge_ratios()
        s._register_demands_to_origins()

        # DUO / AON initialization (per-destination tracking)
        if s.ROUTE_CHOICE in ("duo", "duo_multipoint", "duo_logit", "aon"):
            s._destinations = list(set(dest for _, dest, _, _, _ in s.demand_info))
            s._od_demands = {}  # {(orig, dest): [(t_start, t_end, flow)]}
            for orig, dest, t_start, t_end, flow in s.demand_info:
                s._od_demands.setdefault((orig, dest), []).append((t_start, t_end, flow))
            # Init per-destination cumulative curves for each link
            for link in s.LINKS:
                for dest in s._destinations:
                    link.cum_arrival_d[dest] = np.zeros(s.TSIZE + 1)
                    link.cum_departure_d[dest] = np.zeros(s.TSIZE + 1)

            # Pre-cache node types and classify nodes for DUO dispatch
            s._duo_origin_nodes = []
            s._duo_destination_nodes = []
            s._duo_other_nodes = []
            s._duo_multi_out_nodes = []  # non-origin/dest nodes with >1 outlinks
            for node in s.NODES:
                nt = node.node_type()
                if nt == "origin":
                    s._duo_origin_nodes.append(node)
                elif nt == "destination":
                    s._duo_destination_nodes.append(node)
                else:
                    s._duo_other_nodes.append(node)
                    if len(node.outlinks) > 1:
                        s._duo_multi_out_nodes.append(node)

            # Build reverse adjacency list for Bellman-Ford
            # For each node, list of (link, from_node_name) where link.end_node == node
            s._duo_reverse_adj = {}  # {to_node_name: [(link, from_node_name)]}
            for node in s.NODES:
                s._duo_reverse_adj[node.name] = []
            for link in s.LINKS:
                s._duo_reverse_adj[link.end_node.name].append(
                    (link, link.start_node.name))

            # Node name to index mapping for numpy-based Bellman-Ford
            s._duo_node_names = [n.name for n in s.NODES]
            s._duo_node_idx = {name: i for i, name in enumerate(s._duo_node_names)}
            n_nodes = len(s.NODES)
            n_links = len(s.LINKS)
            # Pre-build link arrays for vectorized BF
            s._duo_link_from_idx = np.array([s._duo_node_idx[l.start_node.name] for l in s.LINKS], dtype=np.int32)
            s._duo_link_to_idx = np.array([s._duo_node_idx[l.end_node.name] for l in s.LINKS], dtype=np.int32)
            s._duo_dest_idx = {d: s._duo_node_idx[d] for d in s._destinations}

            # Pre-compute link property arrays for vectorized travel time
            n_links = len(s.LINKS)
            s._duo_length_arr = np.array([l.length for l in s.LINKS])
            s._duo_inv_lengths = 1.0 / s._duo_length_arr
            s._duo_u_arr = np.array([l.u for l in s.LINKS])
            s._duo_w_arr = np.array([l.w for l in s.LINKS])
            s._duo_kappa_arr = np.array([l.kappa for l in s.LINKS])
            s._duo_k_stars = np.array([l.k_star for l in s.LINKS])
            s._duo_cum_arr_snapshot = np.empty(n_links)
            s._duo_cum_dep_snapshot = np.empty(n_links)

            # Assign link indices for fast lookup
            link_name_to_idx = {l.name: i for i, l in enumerate(s.LINKS)}
            for i, link in enumerate(s.LINKS):
                link._duo_idx = i

            # Per-node outlink index sets (for fast membership check in routing)
            s._duo_node_outlink_idxs = []
            for node in s.NODES:
                s._duo_node_outlink_idxs.append(
                    set(link_name_to_idx[ln] for ln in node.outlinks))

            # Node index for classified node lists
            node_name_to_idx = s._duo_node_idx
            s._duo_origin_node_idxs = [node_name_to_idx[n.name] for n in s._duo_origin_nodes]
            s._duo_other_node_idxs = [node_name_to_idx[n.name] for n in s._duo_other_nodes]
            s._duo_multi_out_node_idxs = [node_name_to_idx[n.name] for n in s._duo_multi_out_nodes]

            # Pre-compute demand lookup table
            s._duo_orig_demands = {}
            for (orig, dest), entries in s._od_demands.items():
                if orig not in s._duo_orig_demands:
                    s._duo_orig_demands[orig] = {}
                s._duo_orig_demands[orig][dest] = entries

            # Destination name set and index for fast lookup
            s._duo_dest_set = set(s._destinations)
            n_dests = len(s._destinations)
            s._duo_dest_to_idx = {d: i for i, d in enumerate(s._destinations)}
            # Destination node indices for vectorized absorption
            node_name_to_idx_map = {n.name: i for i, n in enumerate(s.NODES)}
            s._duo_dest_node_idxs = np.array(
                [node_name_to_idx_map[d] for d in s._destinations], dtype=np.int32)
            # Absorbed count array (written back to nodes after simulation)
            s._duo_absorbed_arr = np.zeros(len(s.NODES))

            # Global arrays for all links' per-destination tracking
            n_links_total = len(s.LINKS)
            s._duo_all_ca_d = np.zeros((n_links_total, n_dests, s.TSIZE + 1))
            s._duo_all_cd_d = np.zeros((n_links_total, n_dests, s.TSIZE + 1))
            s._duo_all_ifd = np.zeros((n_links_total, n_dests))
            s._duo_all_ofd = np.zeros((n_links_total, n_dests))
            # Give each link a view into the global arrays
            for i, link in enumerate(s.LINKS):
                link._ca_d = s._duo_all_ca_d[i]
                link._cd_d = s._duo_all_cd_d[i]
                link._ifd = s._duo_all_ifd[i]
                link._ofd = s._duo_all_ofd[i]

            # Pre-allocate work arrays for FIFO step
            s._duo_all_cum_arr = np.empty(n_links_total)
            s._duo_all_outflow = np.empty(n_links_total)

            # Pre-build per-origin demand arrays: {orig_name: np.array of shape (n_dests,)}
            # for each distinct time window
            s._duo_demand_windows = {}  # {orig_name: [(t_start, t_end, demand_arr)]}
            for orig_name, dest_dict in s._duo_orig_demands.items():
                windows = {}  # (t_start, t_end) -> demand_arr
                for dest_name, entries in dest_dict.items():
                    di = s._duo_dest_to_idx.get(dest_name)
                    if di is None:
                        continue
                    for t_start, t_end_d, flow in entries:
                        key = (t_start, t_end_d)
                        if key not in windows:
                            windows[key] = np.zeros(n_dests)
                        windows[key][di] += flow
                s._duo_demand_windows[orig_name] = list(windows.items())

        s.analyzer = Analyzer(s)

        s.finalized = True

    def check_simulation_ongoing(s):
        """Check if there are remaining timesteps to simulate.

        Returns
        -------
        bool
            True if simulation has not reached TMAX.
        """
        if not s.finalized:
            s.finalize_scenario()
        return s.T < s.TSIZE

    def exec_simulation(s, duration_t=None):
        """Run the LTM simulation.

        Executes the main loop: compute demand/supply, apply node models,
        update cumulative counts. For route_choice="duo", runs the DUO
        reactive assignment loop.

        Parameters
        ----------
        duration_t : float or None, optional
            Duration to simulate (s). None runs to completion.
        """
        if not s.finalized:
            s.finalize_scenario()

        if s.ROUTE_CHOICE in ("duo", "duo_multipoint", "duo_logit", "aon"):
            s._exec_duo(duration_t)
            return

        if duration_t is not None:
            t_end = min(s.T + int(duration_t / s.DELTAT), s.TSIZE)
        else:
            t_end = s.TSIZE

        while s.T < t_end:
            s.TIME = s.T * s.DELTAT

            demands = {}
            supplies = {}
            for link in s.LINKS:
                demands[link] = link.compute_demand(s.T)
                supplies[link] = link.compute_supply(s.T)

            for link in s.LINKS:
                link.inflow_rate = 0.0
                link.outflow_rate = 0.0

            for node in s.NODES:
                node.compute_transfer(s.T, demands, supplies)

            for link in s.LINKS:
                link.cum_arrival[s.T + 1] = link.cum_arrival[s.T] + s.DELTAT * link.inflow_rate
                link.cum_departure[s.T + 1] = link.cum_departure[s.T] + s.DELTAT * link.outflow_rate

            s.T += 1

        if s.T >= s.TSIZE and s.print_mode:
            print(f" Simulation completed. {s.NAME}")

    def _duo_shortest_trees(s, t_seconds):
        """Compute shortest path trees for all destinations.

        Returns
        -------
        tree_arr : np.ndarray, shape (n_dests, n_nodes), int32
            tree_arr[di, ni] = link index of next hop toward destination di
            from node ni. -1 if unreachable.
        dist_arr : np.ndarray, shape (n_dests, n_nodes), float64
            dist_arr[di, ni] = shortest distance from node ni to dest di.
            Only returned when route_choice is "duo_logit".
        link_cost_arr : np.ndarray, shape (n_links,), float64
            Generalized link cost (travel time + toll).
            Only returned when route_choice is "duo_logit".
        """
        tt_method = "multipoint" if s.ROUTE_CHOICE == "duo_multipoint" else "avg_density"

        # Compute link travel times
        n_links = len(s.LINKS)
        link_tt_arr = np.empty(n_links)
        if tt_method == "avg_density":
            t_idx = max(0, min(int(t_seconds / s.DELTAT), s.TSIZE - 1))
            # Vectorized travel time computation
            cum_arr = s._duo_cum_arr_snapshot
            cum_dep = s._duo_cum_dep_snapshot
            for i in range(n_links):
                cum_arr[i] = s.LINKS[i].cum_arrival[t_idx]
                cum_dep[i] = s.LINKS[i].cum_departure[t_idx]
            n_on = np.maximum(cum_arr - cum_dep, 0.0)
            k_avg = n_on * s._duo_inv_lengths
            v = np.where(k_avg <= s._duo_k_stars, s._duo_u_arr,
                         s._duo_w_arr * (s._duo_kappa_arr - k_avg) / np.maximum(k_avg, 1e-10))
            v = np.clip(v, 0.01, s._duo_u_arr)
            link_tt_arr = s._duo_length_arr / v
        else:
            for i, link in enumerate(s.LINKS):
                link_tt_arr[i] = link.instantaneous_travel_time(t_seconds, method=tt_method)

        # Add congestion pricing toll for generalized cost
        toll_arr = np.array([link.get_toll(t_seconds) for link in s.LINKS])
        link_cost_arr = link_tt_arr + toll_arr

        n_nodes = len(s.NODES)
        n_dests = len(s._destinations)
        from_idx = s._duo_link_from_idx
        to_idx = s._duo_link_to_idx
        max_iter = int(n_nodes**0.5) + 10

        tree_arr = np.full((n_dests, n_nodes), -1, dtype=np.int32)
        dist_arr = np.full((n_dests, n_nodes), 1e15)
        INF = 1e15

        for di in range(n_dests):
            dest_i = s._duo_dest_idx[s._destinations[di]]
            dist = np.full(n_nodes, INF)
            dist[dest_i] = 0.0
            prev_link_idx = np.full(n_nodes, -1, dtype=np.int32)

            for _ in range(max_iter):
                old_dist = dist.copy()
                new_dist = old_dist[to_idx] + link_cost_arr
                improved = new_dist < dist[from_idx]
                if not improved.any():
                    break
                imp_indices = np.where(improved)[0]
                for j in imp_indices:
                    fi = from_idx[j]
                    if new_dist[j] < dist[fi]:
                        dist[fi] = new_dist[j]
                        prev_link_idx[fi] = j

            tree_arr[di] = prev_link_idx
            dist_arr[di] = dist

        if s.ROUTE_CHOICE == "duo_logit":
            return tree_arr, dist_arr, link_cost_arr
        return tree_arr

    def _get_od_demand_at(s, orig_name, dest_name, t):
        """Get OD demand flow rate at time t.

        Parameters
        ----------
        orig_name : str
        dest_name : str
        t : float
            Time (s).

        Returns
        -------
        float
            Demand flow rate (veh/s).
        """
        total = 0.0
        for t_start, t_end, flow in s._od_demands.get((orig_name, dest_name), []):
            if t_start <= t < t_end:
                total += flow
        return total

    def _exec_duo(s, duration_t=None, route_update_interval=None):
        """Run DUO (Dynamic User Optimal) reactive assignment.

        Parameters
        ----------
        duration_t : float or None
            Duration to simulate (s). None runs to completion.
        route_update_interval : int or None
            Update shortest paths every N timesteps. None defaults to
            max(1, int(120 / DELTAT)) (~2 minute interval).
        """
        if duration_t is not None:
            t_end = min(s.T + int(duration_t / s.DELTAT), s.TSIZE)
        else:
            t_end = s.TSIZE

        dt = s.DELTAT

        # Route update frequency: default 5 minute interval
        if route_update_interval is None:
            route_update_interval = max(1, int(300 / dt))

        # Pre-cache local references for inner loop performance
        LINKS = s.LINKS
        destinations = s._destinations
        origin_nodes = s._duo_origin_nodes
        dest_nodes = s._duo_destination_nodes
        other_nodes = s._duo_other_nodes
        multi_out_nodes = s._duo_multi_out_nodes
        dest_set = s._duo_dest_set
        n_dests = len(destinations)
        d2i = s._duo_dest_to_idx
        demand_windows = s._duo_demand_windows
        outlink_idx_sets = s._duo_node_outlink_idxs
        node_name_to_idx = s._duo_node_idx

        # Pre-build inlink lists
        for node in s.NODES:
            node._inlink_list = list(node.inlinks.values())

        tree_arr = None  # (n_dests, n_nodes) int32
        dist_arr = None  # (n_dests, n_nodes) float, for duo_logit
        link_cost_arr = None  # (n_links,) float, for duo_logit
        use_logit = (s.ROUTE_CHOICE == "duo_logit")
        logit_temp = s.LOGIT_TEMPERATURE
        link_to_idx = s._duo_link_to_idx  # end-node index for each link

        while s.T < t_end:
            s.TIME = s.T * dt
            t_sec = s.TIME

            if s.print_mode and s.T % max(1, t_end // 20) == 0:
                print(f"  DUO step {s.T}/{t_end}, t={s.TIME:.0f}s", flush=True)

            # Step 1: Compute aggregate demands/supplies (vectorized LTM)
            demands = {}
            supplies = {}
            for link in LINKS:
                demands[link] = link.compute_demand(s.T)
                supplies[link] = link.compute_supply(s.T)
            # Also build arrays for Step 6
            n_links_t = len(LINKS)

            # Step 2-3: Update shortest path trees periodically
            if tree_arr is None or s.T % route_update_interval == 0:
                result = s._duo_shortest_trees(t_sec)
                if use_logit:
                    tree_arr, dist_arr, link_cost_arr = result
                else:
                    tree_arr = result

            # Step 4: Compute dynamic diverge ratios for multi-output nodes only
            T = s.T
            for node in multi_out_nodes:
                outlink_flow = defaultdict(float)
                total_flow = 0.0
                ni = node_name_to_idx[node.name]
                inlinks = node._inlink_list
                outlink_names = node.outlinks
                oidx_set = outlink_idx_sets[ni]
                # Gather inlink vehicle counts: vectorize over dests
                if len(inlinks) == 1:
                    n_d_all = inlinks[0]._ca_d[:, T] - inlinks[0]._cd_d[:, T]
                else:
                    n_d_all = np.zeros(n_dests)
                    for inlink in inlinks:
                        n_d_all += inlink._ca_d[:, T] - inlink._cd_d[:, T]
                # Only process dests with vehicles
                active = np.where(n_d_all > 1e-10)[0]
                if use_logit:
                    # Logit-based soft routing
                    outlinks_list = list(outlink_names.values())
                    outlink_ids = [ol._duo_idx for ol in outlinks_list]
                    for di in active:
                        costs = np.array([
                            link_cost_arr[li] + dist_arr[di, link_to_idx[li]]
                            for li in outlink_ids])
                        costs_shifted = costs - costs.min()
                        weights = np.exp(-costs_shifted / logit_temp)
                        probs = weights / weights.sum()
                        for j, li in enumerate(outlink_ids):
                            outlink_flow[li] += n_d_all[di] * probs[j]
                        total_flow += n_d_all[di]
                else:
                    for di in active:
                        li = tree_arr[di, ni]
                        if li >= 0 and li in oidx_set:
                            outlink_flow[li] = outlink_flow.get(li, 0.0) + n_d_all[di]
                            total_flow += n_d_all[di]

                if total_flow > 1e-10:
                    inv_total = 1.0 / total_flow
                    for outlink_name, outlink in outlink_names.items():
                        node.diverge_ratios[outlink_name] = outlink_flow.get(outlink._duo_idx, 0.0) * inv_total
                else:
                    n_out = len(outlink_names)
                    eq = 1.0 / n_out
                    for outlink_name in outlink_names:
                        node.diverge_ratios[outlink_name] = eq

            # Step 4b: Sync turning_fraction_matrix for general nodes
            for node in other_nodes:
                if node.turning_fraction_matrix is not None:
                    outlinks_list = list(node.outlinks.values())
                    dr = node.diverge_ratios
                    for i in range(len(node.turning_fraction_matrix)):
                        for j, ol in enumerate(outlinks_list):
                            node.turning_fraction_matrix[i][j] = dr.get(ol.name, 0.0)

            # Step 5: Node models -> aggregate inflow/outflow rates
            for link in LINKS:
                link.inflow_rate = 0.0
                link.outflow_rate = 0.0

            for node in origin_nodes:
                s._duo_transfer_origin(node, T, demands, supplies, tree_arr)
            for node in dest_nodes:
                for inlink in node._inlink_list:
                    inlink.outflow_rate = demands[inlink]
            for node in other_nodes:
                node.compute_transfer(T, demands, supplies)

            # Step 6: FIFO - fully vectorized across all links
            all_ca_d = s._duo_all_ca_d
            all_cd_d = s._duo_all_cd_d
            all_ifd = s._duo_all_ifd
            all_ofd = s._duo_all_ofd
            cum_arr_t = s._duo_all_cum_arr
            outflow_t = s._duo_all_outflow
            for i in range(n_links_t):
                cum_arr_t[i] = LINKS[i].cum_arrival[T]
                outflow_t[i] = LINKS[i].outflow_rate
            ratio = np.where(cum_arr_t > 1e-10,
                             outflow_t / np.maximum(cum_arr_t, 1e-10), 0.0)
            all_ofd[:] = all_ca_d[:, :, T] * ratio[:, None]

            # Step 7: Per-destination inflow routing + absorption
            all_ifd[:] = 0.0
            n_nodes_t = len(s.NODES)
            link_end_idx = s._duo_link_to_idx

            # 7a: Gather inlink outflows to nodes (one scatter operation)
            node_arriving = np.zeros((n_nodes_t, n_dests))
            np.add.at(node_arriving, link_end_idx, all_ofd)

            # 7b: Add origin demand
            for node in origin_nodes:
                ni = node_name_to_idx[node.name]
                windows = demand_windows.get(node.name, [])
                for (t_start, t_end_d), darr in windows:
                    if t_start <= t_sec < t_end_d:
                        node_arriving[ni] += darr

            # 7c: Absorb at destinations
            absorbed_counts = s._duo_absorbed_arr
            for di in range(n_dests):
                ni = s._duo_dest_node_idxs[di]
                val = node_arriving[ni, di]
                if val > 1e-15:
                    absorbed_counts[ni] += val * dt
                    node_arriving[ni, di] = 0.0

            # 7d: Route active traffic via shortest path trees (or logit)
            active_ni, active_di = np.where(node_arriving > 1e-15)
            if use_logit:
                for k in range(len(active_ni)):
                    ni = active_ni[k]
                    di = active_di[k]
                    oidx = outlink_idx_sets[ni]
                    if not oidx:
                        continue
                    oidx_list = list(oidx)
                    costs = np.array([
                        link_cost_arr[li] + dist_arr[di, link_to_idx[li]]
                        for li in oidx_list])
                    costs_shifted = costs - costs.min()
                    weights = np.exp(-costs_shifted / logit_temp)
                    probs = weights / weights.sum()
                    flow = node_arriving[ni, di]
                    for j, li in enumerate(oidx_list):
                        all_ifd[li, di] += flow * probs[j]
            else:
                for k in range(len(active_ni)):
                    ni = active_ni[k]
                    di = active_di[k]
                    li = tree_arr[di, ni]
                    if li >= 0 and li in outlink_idx_sets[ni]:
                        all_ifd[li, di] += node_arriving[ni, di]

            # Step 8: Update cumulative counts - fully vectorized across all links
            T1 = T + 1
            all_ca_d[:, :, T1] = all_ca_d[:, :, T] + dt * all_ifd
            all_cd_d[:, :, T1] = all_cd_d[:, :, T] + dt * all_ofd
            agg_arr = all_ifd.sum(axis=1) * dt  # (n_links,)
            agg_dep = all_ofd.sum(axis=1) * dt
            for i in range(n_links_t):
                LINKS[i].cum_arrival[T1] = LINKS[i].cum_arrival[T] + agg_arr[i]
                LINKS[i].cum_departure[T1] = LINKS[i].cum_departure[T] + agg_dep[i]

            s.T += 1

        # Write back absorbed counts to nodes
        for i, node in enumerate(s.NODES):
            node.absorbed_count += s._duo_absorbed_arr[i]

        # Copy numpy arrays back to dict format for analyzer compatibility
        for link in LINKS:
            for di, dest in enumerate(destinations):
                link.cum_arrival_d[dest] = link._ca_d[di, :]
                link.cum_departure_d[dest] = link._cd_d[di, :]

        if s.T >= s.TSIZE and s.print_mode:
            print(f" Simulation completed (DUO). {s.NAME}")

    def _duo_transfer_origin(s, node, t_index, demands, supplies, tree_arr):
        """Origin node transfer for DUO: aggregate demand from all OD pairs.

        Parameters
        ----------
        node : Node
        t_index : int
        demands, supplies : dict
        tree_arr : np.ndarray (n_dests, n_nodes) int32
        """
        dt = s.DELTAT
        t = t_index * dt

        orig_dests = s._duo_orig_demands.get(node.name, {})
        total_demand = 0.0
        for dest, entries in orig_dests.items():
            for t_start, t_end_d, flow in entries:
                if t_start <= t < t_end_d:
                    total_demand += flow

        effective_demand = total_demand + node.demand_queue / dt

        if len(node.outlinks) == 1:
            outlink = list(node.outlinks.values())[0]
            supply = supplies[outlink]
            flow = min(effective_demand, supply)
            if node.flow_capacity is not None:
                flow = min(flow, node.flow_capacity)
            flow = max(flow, 0)
            outlink.inflow_rate = flow
        else:
            dest_flow = {}
            for dest, entries in orig_dests.items():
                for t_start, t_end_d, fl in entries:
                    if t_start <= t < t_end_d:
                        dest_flow[dest] = dest_flow.get(dest, 0.0) + fl

            total_q = sum(dest_flow.values()) + node.demand_queue / dt
            if total_q < 1e-15:
                for outlink in node.outlinks.values():
                    outlink.inflow_rate = 0.0
                node.demand_queue_history.append(node.demand_queue)
                return

            ni = s._duo_node_idx[node.name]
            oidx_set = s._duo_node_outlink_idxs[ni]
            outlink_demand = defaultdict(float)
            for dest, q in dest_flow.items():
                di = s._duo_dest_to_idx.get(dest)
                if di is not None:
                    li = tree_arr[di, ni]
                    if li >= 0 and li in oidx_set:
                        outlink_demand[s.LINKS[li].name] += q
            if node.demand_queue > 0:
                denom = max(sum(dest_flow.values()), 1e-15)
                q_per_dt = node.demand_queue / dt
                for outlink_name in outlink_demand:
                    outlink_demand[outlink_name] += (
                        outlink_demand[outlink_name] / denom * q_per_dt)

            min_ratio = 1.0
            for outlink_name, od_q in outlink_demand.items():
                if od_q > 1e-15:
                    ratio = supplies[node.outlinks[outlink_name]] / od_q
                    if ratio < min_ratio:
                        min_ratio = ratio
            if node.flow_capacity is not None:
                total_out = sum(outlink_demand.values())
                if total_out > 1e-15:
                    r = node.flow_capacity / total_out
                    if r < min_ratio:
                        min_ratio = r
            if min_ratio < 0:
                min_ratio = 0.0
            elif min_ratio > 1.0:
                min_ratio = 1.0

            actual_total = 0.0
            for outlink_name in node.outlinks:
                outlink = node.outlinks[outlink_name]
                outlink.inflow_rate = outlink_demand.get(outlink_name, 0.0) * min_ratio
                actual_total += outlink.inflow_rate

            flow = actual_total

        node.demand_queue += (total_demand - flow) * dt
        node.demand_queue = max(0.0, node.demand_queue)
        node.demand_queue_history.append(node.demand_queue)

    def __repr__(s):
        return f"<World '{s.NAME}'>"
