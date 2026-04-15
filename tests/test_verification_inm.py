"""
Verification tests for the Incremental Node Model (INM).
Based on Floetteroed & Rohde (2011) Table 1 and Table 2.

Network: 3 upstream links (P_S, P_E, P_N) -> intersection -> 3 downstream links (S_N, S_W, S_S)
Turning fractions:
    P_S -> {S_N: 0.5, S_W: 0.5}
    P_E -> {S_W: 1.0}
    P_N -> {S_W: 0.5, S_S: 0.5}
Merge priorities:
    P_N = 10, P_S = 1, P_E = 0.1

FD: u=20, w=5, kappa=0.2 -> q*=0.8 veh/s, k*=0.04 veh/m
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unsim import *


def avg_outflow(link, t1, t2):
    dt = link.W.DELTAT
    i1 = max(0, min(int(t1 / dt), len(link.cum_departure) - 1))
    i2 = max(0, min(int(t2 / dt), len(link.cum_departure) - 1))
    if i2 <= i1:
        return 0
    return (link.cum_departure[i2] - link.cum_departure[i1]) / ((i2 - i1) * dt)


def avg_inflow(link, t1, t2):
    dt = link.W.DELTAT
    i1 = max(0, min(int(t1 / dt), len(link.cum_arrival) - 1))
    i2 = max(0, min(int(t2 / dt), len(link.cum_arrival) - 1))
    if i2 <= i1:
        return 0
    return (link.cum_arrival[i2] - link.cum_arrival[i1]) / ((i2 - i1) * dt)


def build_flotterod_scenario(W, sw_capacity=None):
    """Build the Floetteroed 3x3 intersection scenario.

    Parameters
    ----------
    W : World
    sw_capacity : float or None
        Override capacity for S_W link. None for default (0.8 veh/s).
    """
    # Origins
    W.addNode("O_S", 0, 0)
    W.addNode("O_E", 2, 0)
    W.addNode("O_N", 1, 2)

    # Intersection with turning fractions
    W.addNode("intersection", 1, 1,
              turning_fractions={
                  "P_S": {"S_N": 0.5, "S_W": 0.5},
                  "P_E": {"S_W": 1.0},
                  "P_N": {"S_W": 0.5, "S_S": 0.5},
              })

    # Destinations
    W.addNode("D_N", 1, 3)
    W.addNode("D_W", -1, 1)
    W.addNode("D_S", 1, -1)

    # Upstream links (with merge priorities)
    W.addLink("P_S", "O_S", "intersection",
              length=1000, free_flow_speed=20, backward_wave_speed=5,
              jam_density=0.2, merge_priority=1.0)
    W.addLink("P_E", "O_E", "intersection",
              length=1000, free_flow_speed=20, backward_wave_speed=5,
              jam_density=0.2, merge_priority=0.1)
    W.addLink("P_N", "O_N", "intersection",
              length=1000, free_flow_speed=20, backward_wave_speed=5,
              jam_density=0.2, merge_priority=10.0)

    # Downstream links
    W.addLink("S_N", "intersection", "D_N",
              length=1000, free_flow_speed=20, backward_wave_speed=5,
              jam_density=0.2)
    if sw_capacity is not None:
        W.addLink("S_W", "intersection", "D_W",
                  length=1000, free_flow_speed=20, backward_wave_speed=5,
                  capacity=sw_capacity)
    else:
        W.addLink("S_W", "intersection", "D_W",
                  length=1000, free_flow_speed=20, backward_wave_speed=5,
                  jam_density=0.2)
    W.addLink("S_S", "intersection", "D_S",
              length=1000, free_flow_speed=20, backward_wave_speed=5,
              jam_density=0.2)


def test_flotterod_table1_uncongested():
    """Floetteroed Table 1: uncongested 3x3 intersection.

    All demands well below capacity -> free flow everywhere.

    Expected outflows at intersection:
        q_in: P_S=600/3600, P_E=100/3600, P_N=600/3600
        q_out: S_N=300/3600, S_W=700/3600, S_S=300/3600
    """
    W = World(deltat=5, tmax=3000, print_mode=0)
    build_flotterod_scenario(W)

    d_PS = 600 / 3600
    d_PE = 100 / 3600
    d_PN = 600 / 3600
    W.adddemand("O_S", "D_N", 0, 2000, d_PS)
    W.adddemand("O_E", "D_W", 0, 2000, d_PE)
    W.adddemand("O_N", "D_S", 0, 2000, d_PN)
    W.exec_simulation()

    # Steady-state upstream outflows (= inflows to intersection)
    assert equal_tolerance(avg_outflow(W.get_link("P_S"), 200, 1500), d_PS)
    assert equal_tolerance(avg_outflow(W.get_link("P_E"), 200, 1500), d_PE)
    assert equal_tolerance(avg_outflow(W.get_link("P_N"), 200, 1500), d_PN)

    # Downstream inflows
    q_SN = 300 / 3600  # P_S*0.5
    q_SW = 700 / 3600  # P_S*0.5 + P_E*1.0 + P_N*0.5
    q_SS = 300 / 3600  # P_N*0.5
    assert equal_tolerance(avg_inflow(W.get_link("S_N"), 200, 1500), q_SN)
    assert equal_tolerance(avg_inflow(W.get_link("S_W"), 200, 1500), q_SW)
    assert equal_tolerance(avg_inflow(W.get_link("S_S"), 200, 1500), q_SS)

    # All links free flow
    for name in ["P_S", "P_E", "P_N", "S_N", "S_W", "S_S"]:
        assert equal_tolerance(W.get_link(name).v(1000, 500), 20, rel_tol=0.1)


def test_flotterod_table2_congested():
    """Floetteroed Table 2: congested 3x3 intersection.

    S_W has limited capacity = 400/3600 veh/s.
    P_N has highest priority (alpha=10) -> all demand passes through.
    P_S and P_E throttled by S_W supply constraint.

    Expected steady-state flows (veh/s):
        q_in: P_S=166.67/3600, P_E=16.67/3600, P_N=600/3600
        q_out: S_N=83.34/3600, S_W=400/3600, S_S=300/3600
    """
    q_star_sw = 400 / 3600

    W = World(deltat=5, tmax=3000, print_mode=0)
    build_flotterod_scenario(W, sw_capacity=q_star_sw)

    d_PS = 600 / 3600
    d_PE = 100 / 3600
    d_PN = 600 / 3600
    W.adddemand("O_S", "D_N", 0, 2000, d_PS)
    W.adddemand("O_E", "D_W", 0, 2000, d_PE)
    W.adddemand("O_N", "D_S", 0, 2000, d_PN)
    W.exec_simulation()

    q_in_PS = 166.67 / 3600
    q_in_PE = 16.67 / 3600
    q_in_PN = 600 / 3600
    q_out_SN = 83.34 / 3600
    q_out_SW = 400 / 3600
    q_out_SS = 300 / 3600

    # Upstream outflows
    assert equal_tolerance(avg_outflow(W.get_link("P_N"), 500, 2000), q_in_PN)
    assert equal_tolerance(avg_outflow(W.get_link("P_S"), 500, 2000), q_in_PS, rel_tol=0.15)
    assert equal_tolerance(avg_outflow(W.get_link("P_E"), 500, 2000), q_in_PE, rel_tol=0.15)

    # Downstream inflows
    assert equal_tolerance(avg_inflow(W.get_link("S_N"), 500, 2000), q_out_SN, rel_tol=0.15)
    assert equal_tolerance(avg_inflow(W.get_link("S_W"), 500, 2000), q_out_SW)
    assert equal_tolerance(avg_inflow(W.get_link("S_S"), 500, 2000), q_out_SS)

    # P_N: free flow (high priority, full demand passes)
    assert equal_tolerance(W.get_link("P_N").v(1500, 500), 20, rel_tol=0.1)

    # Flow conservation: sum(q_in) ~ sum(q_out)
    assert equal_tolerance(q_in_PS + q_in_PE + q_in_PN,
                           q_out_SN + q_out_SW + q_out_SS, rel_tol=0.01)


def test_inm_flow_conservation():
    """Flow conservation at INM node: sum(cum_departure of inlinks) == sum(cum_arrival of outlinks)."""
    W = World(deltat=5, tmax=3000, print_mode=0)
    build_flotterod_scenario(W, sw_capacity=400 / 3600)

    W.adddemand("O_S", "D_N", 0, 2000, 600 / 3600)
    W.adddemand("O_E", "D_W", 0, 2000, 100 / 3600)
    W.adddemand("O_N", "D_S", 0, 2000, 600 / 3600)
    W.exec_simulation()

    PS = W.get_link("P_S")
    PE = W.get_link("P_E")
    PN = W.get_link("P_N")
    SN = W.get_link("S_N")
    SW = W.get_link("S_W")
    SS = W.get_link("S_S")

    for t in range(W.TSIZE + 1):
        total_in = PS.cum_departure[t] + PE.cum_departure[t] + PN.cum_departure[t]
        total_out = SN.cum_arrival[t] + SW.cum_arrival[t] + SS.cum_arrival[t]
        assert abs(total_in - total_out) < 1e-6, \
            f"Flow conservation violated at t={t * W.DELTAT}s: in={total_in:.4f} out={total_out:.4f}"
