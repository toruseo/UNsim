"""
Chicago-Sketch congested scenario builder.

Creates a 3-period demand pattern:
  Period 1 (0 ~ T/3):   original demand
  Period 2 (T/3 ~ 2T/3): N_FACTOR * original demand (peak)
  Period 3 (2T/3 ~ T):  original demand

Usage:
  import example12a_chicago_congested as scenario
  W, params, config, congested_idx = scenario.build(N_FACTOR=1.5)

Environment variables:
  PEAK_FACTOR : peak demand multiplier (default 1.5)
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def build(N_FACTOR=None):
    """Build congested Chicago scenario.

    Parameters
    ----------
    N_FACTOR : float, optional
        Peak period demand multiplier. Default from PEAK_FACTOR env or 1.5.

    Returns
    -------
    W : World
    params : Params
    config : NetworkConfig
    congested_idx : np.ndarray of int
        Indices of links where max vehicles > 1.1x free-flow capacity level.
    """
    import jax.numpy as jnp
    from unsim import World
    from unsim.unsim_diff import world_to_jax, simulate_duo

    if N_FACTOR is None:
        N_FACTOR = float(os.environ.get("PEAK_FACTOR", "1.5"))

    # Build base Chicago network
    code = open(os.path.join(os.path.dirname(__file__),
                             "example06b_chicago_calibrated.py")).read()
    parts = code.split("W.exec_simulation()")
    g = {"__file__": os.path.join(os.path.dirname(__file__),
                                   "example06b_chicago_calibrated.py")}
    exec(parts[0], g)
    W = g["W"]

    # 3-period demand: period 2 = N_FACTOR * base
    original_demands = list(W.demand_info)
    W.demand_info = []
    for orig, dest, ts, te, flow in original_demands:
        dur = te - ts
        t1 = ts + dur / 3
        t2 = ts + 2 * dur / 3
        W.demand_info.append((orig, dest, ts, t1, flow))
        W.demand_info.append((orig, dest, t1, t2, flow * N_FACTOR))
        W.demand_info.append((orig, dest, t2, te, flow))

    W.finalize_scenario()
    params, config = world_to_jax(W)

    # Identify congested links via true simulation
    state = simulate_duo(params, config)
    import jax
    jax.block_until_ready(state.cum_departure)

    n_links = int(config.n_links)
    cum_arr = np.asarray(state.cum_arrival[:, :config.tsize])
    cum_dep = np.asarray(state.cum_departure[:, :config.tsize])
    max_count = (cum_arr - cum_dep).max(axis=1)

    u_np = np.asarray(params.u)
    kappa_np = np.asarray(params.kappa)
    q_star = np.asarray(params.q_star)
    lengths_np = np.asarray(config.link_lengths)
    ff_vehicles = q_star * lengths_np / u_np
    congested_idx = np.where(max_count > ff_vehicles * 1.1)[0]

    return W, params, config, congested_idx


if __name__ == "__main__":
    import jax
    from unsim.unsim_diff import simulate_duo, total_travel_time, trip_completed

    W, params, config, congested_idx = build()
    n_links = int(config.n_links)

    state = simulate_duo(params, config)
    jax.block_until_ready(state.cum_departure)

    ttt = float(total_travel_time(state, config))
    lengths = np.asarray(config.link_lengths)
    u_vals = np.asarray(params.u)
    total_vol = np.asarray(state.cum_departure[:, -1])
    ff_ttt = np.sum(total_vol * lengths / u_vals)
    delay = ttt - ff_ttt

    total_demand = sum(f * (te - ts) for _, _, ts, te, f in W.demand_info)
    trips = float(trip_completed(state, config))

    print(f"Network: {config.n_nodes} nodes, {n_links} links")
    print(f"Peak factor: {float(os.environ.get('PEAK_FACTOR', '1.5'))}")
    print(f"Total demand: {total_demand:.0f} veh")
    print(f"Trips completed: {trips:.0f} ({trips/total_demand*100:.1f}%)")
    print(f"TTT: {ttt/3600:.0f} veh-hr, FF: {ff_ttt/3600:.0f} veh-hr")
    print(f"Delay ratio: {delay/ttt*100:.1f}%")
    print(f"Congested links: {len(congested_idx)}/{n_links} "
          f"({len(congested_idx)/n_links*100:.1f}%)")
