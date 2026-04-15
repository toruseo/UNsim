"""
UNsim Analyzer: simulation result analysis and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io


def _interp_for_analyzer(array, fractional_index):
    """Linear interpolation (same logic as unsim._interp)."""
    if fractional_index <= 0:
        return array[0]
    max_idx = len(array) - 1
    if fractional_index >= max_idx:
        return array[max_idx]
    i_low = int(fractional_index)
    frac = fractional_index - i_low
    return array[i_low] * (1 - frac) + array[i_low + 1] * frac


def _invert_cum(cum_array, N, dt, tmax):
    """Find earliest time t such that cum_array(t) >= N.

    Uses binary search on the cumulative departure array.

    Parameters
    ----------
    cum_array : list[float]
        Monotonically non-decreasing cumulative counts.
    N : float
        Target cumulative count.
    dt : float
        Timestep width.
    tmax : float
        Maximum simulation time.

    Returns
    -------
    float
        Time (s) when cum_array first reaches N. Linear interpolation
        within timesteps for sub-step accuracy.
    """
    if N <= cum_array[0]:
        return 0.0

    n = len(cum_array)
    if N > cum_array[-1]:
        return tmax

    # Binary search for the first index where cum_array[i] >= N
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cum_array[mid] < N:
            lo = mid + 1
        else:
            hi = mid

    # Linear interpolation between lo-1 and lo
    if lo > 0 and cum_array[lo - 1] < N:
        frac = (N - cum_array[lo - 1]) / max(cum_array[lo] - cum_array[lo - 1], 1e-10)
        return (lo - 1 + frac) * dt
    return lo * dt


class Analyzer:
    """Analyzer for UNsim simulation results.

    Parameters
    ----------
    W : World
        Parent world.

    Attributes
    ----------
    trip_all : float
        Total demand volume (veh).
    trip_completed : float
        Total completed trips (veh).
    total_travel_time : float
        Total travel time of all vehicles (s).
    average_travel_time : float
        Average travel time per completed trip (s).
    average_delay : float
        Average delay per completed trip (s).
    """

    def __init__(s, W):
        s.W = W
        s.trip_all = 0
        s.trip_completed = 0
        s.total_travel_time = 0
        s.average_travel_time = 0
        s.average_delay = 0

    # ================================================================
    # Analysis
    # ================================================================

    def basic_analysis(s):
        """Compute basic aggregate statistics.

        Calculates trip_all, trip_completed, total_travel_time,
        average_travel_time, and average_delay.
        """
        W = s.W
        dt = W.DELTAT

        s.trip_all = 0
        for orig, dest, t_start, t_end, flow in W.demand_info:
            s.trip_all += flow * (t_end - t_start)

        s.trip_completed = 0
        if W.ROUTE_CHOICE in ("duo", "duo_multipoint", "duo_logit", "aon"):
            # DUO absorbs traffic via node.absorbed_count in step 7
            for node in W.NODES:
                s.trip_completed += node.absorbed_count
        else:
            for node in W.NODES:
                if node.node_type() == "destination":
                    for inlink in node.inlinks.values():
                        s.trip_completed += inlink.cum_departure[-1]
                elif node.absorption_ratio > 0:
                    s.trip_completed += node.absorbed_count

        s.total_travel_time = 0
        for link in W.LINKS:
            for t in range(W.TSIZE):
                n_on_link = link.cum_arrival[t] - link.cum_departure[t]
                s.total_travel_time += max(n_on_link, 0) * dt
        for node in W.NODES:
            if node.node_type() == "origin":
                for q_val in node.demand_queue_history:
                    s.total_travel_time += max(q_val, 0) * dt

        if s.trip_completed > 0:
            s.average_travel_time = s.total_travel_time / s.trip_completed
        else:
            s.average_travel_time = 0

        total_ff_tt = 0
        total_demand = 0
        for orig_name, dest_name, t_start, t_end, flow in W.demand_info:
            vol = flow * (t_end - t_start)
            ff_tt = W._get_free_flow_travel_time(orig_name, dest_name)
            total_ff_tt += ff_tt * vol
            total_demand += vol
        if total_demand > 0:
            avg_ff_tt = total_ff_tt / total_demand
        else:
            avg_ff_tt = 0
        s.average_delay = s.average_travel_time - avg_ff_tt

    def print_simple_stats(s):
        """Print basic simulation statistics to stdout."""
        s.basic_analysis()
        print(f"  Simulation Results:")
        print(f"    Total trips:     {s.trip_all:.1f}")
        print(f"    Completed trips: {s.trip_completed:.1f}")
        print(f"    Total travel time: {s.total_travel_time:.1f} s")
        print(f"    Avg travel time: {s.average_travel_time:.1f} s")
        print(f"    Avg delay:       {s.average_delay:.1f} s")

    def link_to_pandas(s):
        """Get link-level statistics as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            Columns: link, start_node, end_node, traffic_volume,
            vehicles_remain, free_travel_time, average_travel_time.
        """
        W = s.W
        dt = W.DELTAT
        rows = []
        for link in W.LINKS:
            traffic_volume = link.cum_departure[-1]
            vehicles_remain = link.cum_arrival[-1] - link.cum_departure[-1]
            free_travel_time = link.length / link.u

            total_veh_time = 0
            for t in range(W.TSIZE):
                n_on_link = link.cum_arrival[t] - link.cum_departure[t]
                total_veh_time += max(n_on_link, 0) * dt

            if traffic_volume > 0:
                avg_tt = total_veh_time / traffic_volume
            else:
                avg_tt = free_travel_time

            rows.append({
                "link": link.name,
                "start_node": link.start_node.name,
                "end_node": link.end_node.name,
                "traffic_volume": traffic_volume,
                "vehicles_remain": vehicles_remain,
                "free_travel_time": free_travel_time,
                "average_travel_time": avg_tt,
            })
        return pd.DataFrame(rows)

    def _link_exit_time(s, link, t_enter):
        """Compute exit time from a link's downstream end, entering at t_enter.

        Uses the Newell formula: the N-th vehicle (N = cum_arrival(t_enter))
        exits at t_exit = max(t_enter + d/u, invert(cum_departure, N)).
        The first term is the free-flow constraint (cannot travel faster than u),
        the second is the queuing constraint (must wait for preceding vehicles).

        Parameters
        ----------
        link : Link
        t_enter : float
            Time entering the link's upstream end (s).

        Returns
        -------
        float
            Time exiting the link's downstream end (s). inf if not within TMAX.
        """
        W = s.W
        dt = W.DELTAT
        idx = t_enter / dt
        N = _interp_for_analyzer(link.cum_arrival, idx)

        # Free-flow constraint: cannot arrive before t_enter + d/u
        t_freeflow = t_enter + link.length / link.u

        # Queuing constraint: must wait for cum_departure to reach N
        t_queue = _invert_cum(link.cum_departure, N, dt, W.TMAX)

        return max(t_freeflow, t_queue)

    def travel_time(s, orig, dest, t_depart, path=None):
        """Compute travel time from node orig to node dest departing at t_depart.

        By default, finds the time-dependent shortest path using Dijkstra with
        actual (congestion-dependent) link travel times from cumulative counts.
        LTM satisfies FIFO, so Dijkstra yields the optimal solution.

        Parameters
        ----------
        orig : str or Node
            Origin node.
        dest : str or Node
            Destination node.
        t_depart : float
            Departure time (s).
        path : list[str|Link] or None, optional
            Explicit path as ordered list of links. None for time-dependent
            shortest path (default).

        Returns
        -------
        float
            Travel time (s). Returns inf if the vehicle does not arrive
            within the simulation period.
        """
        W = s.W
        orig = W.get_node(orig)
        dest = W.get_node(dest)

        if orig is dest:
            return 0.0

        if path is not None:
            # Trace along explicit path
            t_current = t_depart
            for l in path:
                link = W.get_link(l)
                t_current = s._link_exit_time(link, t_current)
                if t_current >= W.TMAX:
                    return float('inf')
            return t_current - t_depart

        # Time-dependent Dijkstra: minimize arrival time at dest
        # arrival[node_id] = earliest arrival time at that node
        from heapq import heappush, heappop
        n_nodes = len(W.NODES)
        arrival = [float('inf')] * n_nodes
        arrival[orig.id] = t_depart
        visited = [False] * n_nodes
        heap = [(t_depart, orig.id)]

        while heap:
            t_arr, nid = heappop(heap)
            if visited[nid]:
                continue
            visited[nid] = True
            if nid == dest.id:
                return t_arr - t_depart

            node = W.NODES[nid]
            for link in node.outlinks.values():
                next_node = link.end_node
                if visited[next_node.id]:
                    continue
                t_exit = s._link_exit_time(link, t_arr)
                if t_exit < arrival[next_node.id]:
                    arrival[next_node.id] = t_exit
                    heappush(heap, (t_exit, next_node.id))

        return float('inf')

    # ================================================================
    # Time-space diagram
    # ================================================================

    # Mode aliases for time_space_diagram
    _MODE_ALIASES = {
        "k": "density", "density": "density",
        "q": "flow", "flow": "flow",
        "v": "speed", "speed": "speed",
        "k_norm": "k_norm", "q_norm": "q_norm", "v_norm": "v_norm",
        "N": "N",
    }

    def time_space_diagram(s, links=None, mode="density", figsize=(12, 4),
                           xlim=None, ylim=None, cmap=None, n_contours=20,
                           nt=100, nx=50, vmin=None, vmax=None):
        """Draw a time-space diagram for one or more links.

        Parameters
        ----------
        links : Link or str or list[Link|str] or None, optional
            Link(s) to plot. Accepts Link objects or names. None for all links.
        mode : str, optional
            "density" or "k" for density, "flow" or "q" for flow,
            "speed" or "v" for speed, "N" for cumulative count contours.
            "k_norm" / "q_norm" / "v_norm" for normalized values
            (k/kappa, q/q*, v/u per link).
        figsize : tuple, optional
            Figure size.
        xlim : tuple or None, optional
            Time axis limits (s).
        ylim : tuple or None, optional
            Space axis limits (m).
        cmap : str or None, optional
            Colormap name.
        n_contours : int, optional
            Number of contour lines for mode="N".
        nt : int, optional
            Number of time grid points.
        nx : int, optional
            Number of space grid points per link.
        vmin : float or None, optional
            Minimum value for colormap. None for auto.
        vmax : float or None, optional
            Maximum value for colormap. None for auto.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        mode = s._MODE_ALIASES.get(mode, mode)

        if links is None:
            links = s.W.LINKS
        elif not isinstance(links, list):
            links = [links]
        links = [s.W.get_link(l) for l in links]

        fig, ax = plt.subplots(figsize=figsize)
        ts = np.linspace(0, s.W.TMAX, nt)

        x_offset = 0
        link_boundaries = [0]
        for link in links:
            x_offset += link.length
            link_boundaries.append(x_offset)

        default_cmaps = {
            "density": "viridis_r", "flow": "Blues", "speed": "viridis",
            "k_norm": "viridis_r", "q_norm": "Blues", "v_norm": "viridis",
        }
        cm = cmap or default_cmaps.get(mode, "viridis")
        pcm_last = None

        for li, link in enumerate(links):
            xs = np.linspace(0, link.length, nx)
            T, X = np.meshgrid(ts, xs, indexing='ij')

            Z = np.empty_like(T)
            for i in range(nt):
                for j in range(nx):
                    q, k, v = link._compute_state_point(ts[i], xs[j])
                    if mode == "N":
                        Z[i, j] = link.compute_N(ts[i], xs[j])
                    elif mode == "density":
                        Z[i, j] = k
                    elif mode == "flow":
                        Z[i, j] = q
                    elif mode == "speed":
                        Z[i, j] = v
                    elif mode == "k_norm":
                        Z[i, j] = k / link.kappa
                    elif mode == "q_norm":
                        Z[i, j] = q / link.q_star if link.q_star > 0 else 0
                    elif mode == "v_norm":
                        Z[i, j] = v / link.u if link.u > 0 else 0

            X_plot = X + link_boundaries[li]

            if mode == "N":
                ax.contour(T, X_plot, Z, levels=n_contours, colors="k", linewidths=0.5)
            else:
                pcm_last = ax.pcolormesh(T, X_plot, Z, cmap=cm, shading="gouraud",
                                         vmin=vmin, vmax=vmax)

        label_map = {
            "density": "k (veh/m)", "flow": "q (veh/s)", "speed": "v (m/s)",
            "k_norm": "k / kappa", "q_norm": "q / q*", "v_norm": "v / u",
        }
        if pcm_last is not None:
            plt.colorbar(pcm_last, ax=ax, label=label_map.get(mode, mode))

        # Link boundaries
        for b in link_boundaries[1:-1]:
            ax.axhline(b, color="gray", linestyle="--", linewidth=0.5)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (m)")
        ax.set_title(f"Time-Space Diagram ({mode})")
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        plt.tight_layout()
        return fig

    # ================================================================
    # Network visualization helpers
    # ================================================================

    def _network_extents(s):
        """Compute coordinate extents for network plots.

        Returns
        -------
        minx, maxx, miny, maxy : float
        """
        xs = [n.x for n in s.W.NODES]
        ys = [n.y for n in s.W.NODES]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        dx = maxx - minx or 1
        dy = maxy - miny or 1
        buf = max(dx, dy) * 0.1
        return minx - buf, maxx + buf, miny - buf, maxy + buf

    def _link_offset(s, link, left_handed=True):
        """Compute perpendicular offset for link drawing (left/right-handed traffic).

        Parameters
        ----------
        link : Link
        left_handed : bool

        Returns
        -------
        ox, oy : float
            Offset in x and y.
        """
        x1, y1 = link.start_node.x, link.start_node.y
        x2, y2 = link.end_node.x, link.end_node.y
        sign = 1 if left_handed else -1
        return sign * (y1 - y2) * 0.05, sign * (x2 - x1) * 0.05

    # ================================================================
    # Network plot (single time)
    # ================================================================

    def network(s, t=None, figsize=(6, 6), left_handed=True,
                minwidth=0.5, maxwidth=12, node_size=4,
                legend=True):
        """Draw network state at time t.

        Links are colored by speed (viridis) and sized by density.

        Parameters
        ----------
        t : float or None, optional
            Time (s). None for TMAX/2.
        figsize : tuple, optional
            Figure size.
        left_handed : bool, optional
            True for left-handed traffic (Japan/UK).
        minwidth : float, optional
            Minimum link width.
        maxwidth : float, optional
            Maximum link width.
        node_size : float, optional
            Node marker size.
        legend : bool, optional
            Show colorbar legend.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        if t is None:
            t = s.W.TMAX / 2

        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.colormaps["viridis"]

        for link in s.W.LINKS:
            ox, oy = s._link_offset(link, left_handed)
            x1, y1 = link.start_node.x + ox, link.start_node.y + oy
            x2, y2 = link.end_node.x + ox, link.end_node.y + oy

            # Compute state at segments along the link
            n_seg = 10
            seg_xs = np.linspace(0, link.length, n_seg + 1)
            for i in range(n_seg):
                x_mid = (seg_xs[i] + seg_xs[i + 1]) / 2
                qi = link.q(t, x_mid)
                ki = link.k(t, x_mid)
                vi = qi / ki if ki > 1e-10 else link.u

                alpha_seg = (i + 0.5) / n_seg
                sx = x1 * (1 - alpha_seg) + x2 * alpha_seg
                sy = y1 * (1 - alpha_seg) + y2 * alpha_seg
                alpha_next = (i + 1.5) / n_seg
                ex = x1 * (1 - alpha_next) + x2 * alpha_next
                ey = y1 * (1 - alpha_next) + y2 * alpha_next

                lw = ki / link.kappa * (maxwidth - minwidth) + minwidth
                color = cmap(vi / link.u)
                ax.plot([sx, ex], [sy, ey], color=color, linewidth=lw,
                        solid_capstyle="butt", zorder=6)

            # Base line
            ax.plot([x1, x2], [y1, y2], "k--", linewidth=0.25, zorder=5)

        # Nodes
        for node in s.W.NODES:
            ax.plot(node.x, node.y, "ko", markersize=node_size, zorder=10)
            ax.annotate(node.name, (node.x, node.y), fontsize=6,
                        ha="center", va="bottom", zorder=11)

        # Legend (colorbar for speed)
        if legend:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
            cbar.set_label("v / u (speed ratio)")

        ax.set_aspect("equal")
        ax.set_title(f"Network (t={t:.0f}s)")
        plt.tight_layout()
        return fig

    # ================================================================
    # Network average
    # ================================================================

    def network_average(s, figsize=(6, 6), left_handed=True,
                        minwidth=0.5, maxwidth=12, node_size=4,
                        legend=True, show_labels=True):
        """Draw network with time-averaged traffic state.

        Links are colored by delay ratio (jet) and sized by traffic volume.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
        left_handed : bool, optional
            True for left-handed traffic.
        minwidth : float, optional
            Minimum link width.
        maxwidth : float, optional
            Maximum link width.
        node_size : float, optional
            Node marker size.
        legend : bool, optional
            Show colorbar legend.
        show_labels : bool, optional
            Show node name labels. Default True.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        df = s.link_to_pandas()
        max_vol = df["traffic_volume"].max() if df["traffic_volume"].max() > 0 else 1

        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.colormaps["jet"]

        for _, row in df.iterrows():
            link = s.W.LINKS_NAME_DICT[row["link"]]
            ox, oy = s._link_offset(link, left_handed)
            x1, y1 = link.start_node.x + ox, link.start_node.y + oy
            x2, y2 = link.end_node.x + ox, link.end_node.y + oy

            ff_tt = row["free_travel_time"]
            avg_tt = row["average_travel_time"]
            vol = row["traffic_volume"]

            # Color by delay ratio
            pace_ratio = avg_tt / ff_tt if ff_tt > 0 else 1
            color_coef = np.clip((pace_ratio - 1) / 1, 0.1, 0.9)
            color = cmap(color_coef)

            # Width by volume
            lw = vol / max_vol * (maxwidth - minwidth) + minwidth

            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw,
                    solid_capstyle="butt", zorder=6)
            ax.plot([x1, x2], [y1, y2], "k--", linewidth=0.25, zorder=5)

        for node in s.W.NODES:
            ax.plot(node.x, node.y, "ko", markersize=node_size, zorder=10)
            if show_labels:
                ax.annotate(node.name, (node.x, node.y), fontsize=6,
                            ha="center", va="bottom", zorder=11)

        if legend:
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                       norm=plt.Normalize(0.1, 0.9))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6,
                                ticks=[0.1, 0.5, 0.9])
            cbar.set_ticklabels(["free flow", "delay 50%", "delay 100%+"])
            cbar.set_label("Delay")

        ax.set_aspect("equal")
        ax.set_title("Network Average")
        plt.tight_layout()
        return fig

    # ================================================================
    # Network animation
    # ================================================================

    def network_anim(s, figsize=(6, 6), left_handed=True,
                     minwidth=0.5, maxwidth=12, node_size=4,
                     timestep_skip=10, duration=100, dpi=80,
                     file_name=None):
        """Create animated GIF of network state over time.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
        left_handed : bool, optional
            True for left-handed traffic.
        minwidth : float, optional
            Minimum link width.
        maxwidth : float, optional
            Maximum link width.
        node_size : float, optional
            Node marker size.
        timestep_skip : int, optional
            Render every N-th timestep.
        duration : int, optional
            Frame duration in ms.
        dpi : int, optional
            Image resolution.
        file_name : str or None, optional
            Output GIF path. None for "out_{name}.gif".

        Returns
        -------
        str
            Path to the saved GIF file.
        """
        W = s.W
        if file_name is None:
            file_name = f"out_{W.NAME or 'unsim'}.gif"

        cmap = plt.colormaps["viridis"]
        minx, maxx, miny, maxy = s._network_extents()

        frames = []
        times = range(0, W.TMAX, W.DELTAT * timestep_skip)

        for t in times:
            fig, ax = plt.subplots(figsize=figsize)

            for link in W.LINKS:
                ox, oy = s._link_offset(link, left_handed)
                x1, y1 = link.start_node.x + ox, link.start_node.y + oy
                x2, y2 = link.end_node.x + ox, link.end_node.y + oy

                n_seg = 10
                seg_xs = np.linspace(0, link.length, n_seg + 1)
                for i in range(n_seg):
                    x_mid = (seg_xs[i] + seg_xs[i + 1]) / 2
                    qi = link.q(t, x_mid)
                    ki = link.k(t, x_mid)
                    vi = qi / ki if ki > 1e-10 else link.u

                    a0 = (i + 0.5) / n_seg
                    a1 = min((i + 1.5) / n_seg, 1)
                    sx = x1 + (x2 - x1) * a0
                    sy = y1 + (y2 - y1) * a0
                    ex = x1 + (x2 - x1) * a1
                    ey = y1 + (y2 - y1) * a1

                    lw = ki / link.kappa * (maxwidth - minwidth) + minwidth
                    ax.plot([sx, ex], [sy, ey], color=cmap(vi / link.u),
                            linewidth=lw, solid_capstyle="butt", zorder=6)

                ax.plot([x1, x2], [y1, y2], "k--", linewidth=0.25, zorder=5)

            for node in W.NODES:
                ax.plot(node.x, node.y, "ko", markersize=node_size, zorder=10)

            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            ax.set_aspect("equal")
            ax.set_title(f"t={t:.0f}s")
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            plt.close(fig)
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            buf.close()

        if frames:
            frames[0].save(file_name, save_all=True, append_images=frames[1:],
                           optimize=False, duration=duration, loop=0)

        return file_name
