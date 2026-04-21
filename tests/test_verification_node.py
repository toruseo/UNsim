"""
Verifies UNsim outputs for merging/diverging/INM nodes in various configurations.
Tests check theoretical flow rates using equal_tolerance against cumulative
departure rates and Edie state.

Also covers manual diverge_ratio / absorption_ratio specification
and Incremental Node Model (INM) based on Floetteroed & Rohde (2011).
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unsim import *

"""
default FD:
    u = 20, kappa = 0.2, tau = 1, w = 5
    k* = 0.04, q* = 0.8

Congested branch: k = (kappa*w - q) / w, v = q / k
Free flow branch: k = q / u, v = u
"""

def q2k_cong(q, kappa=0.2, w=5):
    """Density on congested branch of triangular FD."""
    return (kappa * w - q) / w

def q2k_free(q, u=20):
    """Density on free-flow branch of triangular FD."""
    return q / u

def avg_outflow(link, t1_sec, t2_sec):
    """Average outflow rate of a link over [t1, t2] in seconds."""
    dt = link.W.DELTAT
    i1 = int(t1_sec / dt)
    i2 = int(t2_sec / dt)
    i1 = max(0, min(i1, len(link.cum_departure) - 1))
    i2 = max(0, min(i2, len(link.cum_departure) - 1))
    if i2 <= i1:
        return 0
    return (link.cum_departure[i2] - link.cum_departure[i1]) / ((i2 - i1) * dt)

def avg_inflow(link, t1_sec, t2_sec):
    """Average inflow rate of a link over [t1, t2] in seconds."""
    dt = link.W.DELTAT
    i1 = int(t1_sec / dt)
    i2 = int(t2_sec / dt)
    i1 = max(0, min(i1, len(link.cum_arrival) - 1))
    i2 = max(0, min(i2, len(link.cum_arrival) - 1))
    if i2 <= i1:
        return 0
    return (link.cum_arrival[i2] - link.cum_arrival[i1]) / ((i2 - i1) * dt)
# ============================================================
# Merge tests
# ============================================================

def test_merge_fair_nocongestion():
    """Merge, equal priority, D1+D2 < S.
    Theory: q1=0.3, q2=0.3, q3=0.6. All free flow.
    """
    W = World(name="", deltat=5, tmax=1200, print_mode=0, save_mode=1)

    W.addNode("orig1", 0, 0)
    W.addNode("orig2", 0, 2)
    W.addNode("merge", 1, 1)
    W.addNode("dest", 2, 1)
    link1 = W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
    link2 = W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
    link3 = W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig1", "dest", 0, 1000, 0.3)
    W.adddemand("orig2", "dest", 0, 1000, 0.3)

    W.exec_simulation()

    # Steady-state outflow rates (t=200-800s)
    assert equal_tolerance(avg_outflow(link1, 200, 800), 0.3)
    assert equal_tolerance(avg_outflow(link2, 200, 800), 0.3)
    assert equal_tolerance(avg_outflow(link3, 200, 800), 0.6)

    # Edie state: all free flow during demand period
    
    assert equal_tolerance(link1.q(540, 550), 0.3)
    assert equal_tolerance(link1.k(540, 550), q2k_free(0.3))
    assert equal_tolerance(link1.v(540, 550), 20)
    assert equal_tolerance(link3.q(540, 550), 0.6)
    assert equal_tolerance(link3.k(540, 550), q2k_free(0.6))
    assert equal_tolerance(link3.v(540, 550), 20)

    # Travel time and volume
    df = W.analyzer.link_to_pandas()
    assert equal_tolerance(df[df["link"]=="link1"]["average_travel_time"].values[0], 50)
    assert equal_tolerance(df[df["link"]=="link1"]["traffic_volume"].values[0], 300)
    assert equal_tolerance(df[df["link"]=="link3"]["traffic_volume"].values[0], 600)
def test_merge_fair_congestion():
    """Merge, equal priority, D1+D2 > S.
    Theory: alpha1=alpha2=0.5, S=0.8
    q1 = mid{0.5, 0.3, 0.4} = 0.4
    q2 = mid{0.5, 0.3, 0.4} = 0.4
    q3 = 0.8
    """
    W = World(name="", deltat=5, tmax=1200, print_mode=0, save_mode=1)

    W.addNode("orig1", 0, 0)
    W.addNode("orig2", 0, 2)
    W.addNode("merge", 1, 1)
    W.addNode("dest", 2, 1)
    link1 = W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
    link2 = W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
    link3 = W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig1", "dest", 0, 1000, 0.5)
    W.adddemand("orig2", "dest", 0, 1000, 0.5)

    W.exec_simulation()

    # Steady-state outflow: q1=q2=0.4, q3=0.8
    assert equal_tolerance(avg_outflow(link1, 200, 800), 0.4)
    assert equal_tolerance(avg_outflow(link2, 200, 800), 0.4)
    assert equal_tolerance(avg_outflow(link3, 200, 800), 0.8)
    # Flow conservation
    assert equal_tolerance(
        avg_outflow(link1, 200, 800) + avg_outflow(link2, 200, 800),
        avg_inflow(link3, 200, 800))
    # link3: free flow at capacity
    
    assert equal_tolerance(link3.q(540, 550), 0.8)
    assert equal_tolerance(link3.k(540, 550), q2k_free(0.8))
    assert equal_tolerance(link3.v(540, 550), 20)
def test_merge_unfair():
    """Merge, priority 1:1.5 (alpha1=0.4, alpha2=0.6), D1=D2=0.5, S=0.8.
    Theory:
    q1 = mid{0.5, 0.3, 0.32} = 0.32
    q2 = mid{0.5, 0.3, 0.48} = 0.48
    q3 = 0.8
    """
    W = World(name="", deltat=5, tmax=1200, print_mode=0, save_mode=1)

    W.addNode("orig1", 0, 0)
    W.addNode("orig2", 0, 2)
    W.addNode("merge", 1, 1)
    W.addNode("dest", 2, 1)
    link1 = W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
    link2 = W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1.5)
    link3 = W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig1", "dest", 0, 1000, 0.5)
    W.adddemand("orig2", "dest", 0, 1000, 0.5)

    W.exec_simulation()

    # Steady-state outflow: q1=0.32, q2=0.48
    assert equal_tolerance(avg_outflow(link1, 200, 800), 0.32)
    assert equal_tolerance(avg_outflow(link2, 200, 800), 0.48)
    assert equal_tolerance(avg_outflow(link3, 200, 800), 0.8)
    # Priority ratio: q2/q1 = 1.5
    assert equal_tolerance(avg_outflow(link2, 200, 800) / avg_outflow(link1, 200, 800), 1.5)
    # Flow conservation
    assert equal_tolerance(
        avg_outflow(link1, 200, 800) + avg_outflow(link2, 200, 800),
        avg_inflow(link3, 200, 800))
def test_merge_veryunfair():
    """Merge, priority 1:100, D1=D2=0.5, S=0.8.
    Theory:
    q1 = mid{0.5, 0.3, 0.00792} = 0.3
    q2 = mid{0.5, 0.3, 0.7921} = 0.5  (=D2, link2 free flow)
    q3 = 0.8
    """
    W = World(name="", deltat=5, tmax=1200, print_mode=0, save_mode=1)

    W.addNode("orig1", 0, 0)
    W.addNode("orig2", 0, 2)
    W.addNode("merge", 1, 1)
    W.addNode("dest", 2, 1)
    link1 = W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
    link2 = W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=100)
    link3 = W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig1", "dest", 0, 1000, 0.5)
    W.adddemand("orig2", "dest", 0, 1000, 0.5)

    W.exec_simulation()

    # Steady-state outflow
    assert equal_tolerance(avg_outflow(link1, 200, 800), 0.3)
    assert equal_tolerance(avg_outflow(link2, 200, 800), 0.5)
    assert equal_tolerance(avg_outflow(link3, 200, 800), 0.8)
    # link2 free flow (q=D=0.5)
    
    assert equal_tolerance(link2.q(540, 550), 0.5)
    assert equal_tolerance(link2.k(540, 550), q2k_free(0.5))
    assert equal_tolerance(link2.v(540, 550), 20)
def test_merge_flowcheck_priority_1to2():
    """Merge, priority 1:2, both oversaturated (D1=D2=0.8), S=0.8.
    Theory: alpha1=1/3, alpha2=2/3
    q1 = mid{0.8, 0, 0.2667} = 0.2667
    q2 = mid{0.8, 0, 0.5333} = 0.5333
    q3 = 0.8
    """
    W = World(name="", deltat=5, tmax=1200, print_mode=0, save_mode=1)

    W.addNode("orig1", 0, 0)
    W.addNode("orig2", 0, 2)
    W.addNode("merge", 1, 1)
    W.addNode("dest", 2, 1)
    link1 = W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
    link2 = W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=2)
    link3 = W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig1", "dest", 0, 1000, 0.8)
    W.adddemand("orig2", "dest", 0, 1000, 0.8)

    W.exec_simulation()

    q1_theory = 0.8 / 3
    q2_theory = 0.8 * 2 / 3

    assert equal_tolerance(avg_outflow(link1, 200, 800), q1_theory)
    assert equal_tolerance(avg_outflow(link2, 200, 800), q2_theory)
    assert equal_tolerance(avg_outflow(link3, 200, 800), 0.8)
    # Priority ratio: q2/q1 = 2
    assert equal_tolerance(avg_outflow(link2, 200, 800) / avg_outflow(link1, 200, 800), 2.0)
def test_merge_flowcheck_priority_1to5():
    """Merge, priority 1:5, both oversaturated (D1=D2=0.8), S=0.8.
    Theory: alpha1=1/6, alpha2=5/6
    q1 = 0.8/6 = 0.1333
    q2 = 0.8*5/6 = 0.6667
    q3 = 0.8
    """
    W = World(name="", deltat=5, tmax=1200, print_mode=0, save_mode=1)

    W.addNode("orig1", 0, 0)
    W.addNode("orig2", 0, 2)
    W.addNode("merge", 1, 1)
    W.addNode("dest", 2, 1)
    link1 = W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
    link2 = W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=5)
    link3 = W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig1", "dest", 0, 1000, 0.8)
    W.adddemand("orig2", "dest", 0, 1000, 0.8)

    W.exec_simulation()

    q1_theory = 0.8 / 6
    q2_theory = 0.8 * 5 / 6

    assert equal_tolerance(avg_outflow(link1, 200, 800), q1_theory)
    assert equal_tolerance(avg_outflow(link2, 200, 800), q2_theory)
    assert equal_tolerance(avg_outflow(link3, 200, 800), 0.8)
    assert equal_tolerance(avg_outflow(link2, 200, 800) / avg_outflow(link1, 200, 800), 5.0)
def test_merge_one_free_one_congested():
    """Merge, equal priority, D1=0.3, D2=0.7, S=0.8.
    Theory: D1+D2=1.0 > 0.8
    q1 = mid{0.3, 0.1, 0.4} = 0.3  (=D1, link1 free flow)
    q2 = mid{0.7, 0.5, 0.4} = 0.5  (link2 congested)
    q3 = 0.8
    """
    W = World(name="", deltat=5, tmax=1200, print_mode=0, save_mode=1)

    W.addNode("orig1", 0, 0)
    W.addNode("orig2", 0, 2)
    W.addNode("merge", 1, 1)
    W.addNode("dest", 2, 1)
    link1 = W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
    link2 = W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
    link3 = W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig1", "dest", 0, 1000, 0.3)
    W.adddemand("orig2", "dest", 0, 1000, 0.7)

    W.exec_simulation()

    # Outflow rates
    assert equal_tolerance(avg_outflow(link1, 200, 800), 0.3)
    assert equal_tolerance(avg_outflow(link2, 200, 800), 0.5)
    assert equal_tolerance(avg_outflow(link3, 200, 800), 0.8)
    # link1 free flow (q=D=0.3)
    
    assert equal_tolerance(link1.q(540, 550), 0.3)
    assert equal_tolerance(link1.k(540, 550), q2k_free(0.3))
    assert equal_tolerance(link1.v(540, 550), 20)
    # link3 at capacity
    assert equal_tolerance(link3.q(540, 550), 0.8)
    assert equal_tolerance(link3.k(540, 550), q2k_free(0.8))
# ============================================================
# Diverge tests
# ============================================================

def test_diverge_nocongestion():
    """Diverge, equal split, no congestion.
    Theory: beta2=beta3=0.5, D=0.6
    q1=0.6, q2=0.3, q3=0.3. All free flow.
    """
    W = World(name="", deltat=5, tmax=1200, print_mode=0, save_mode=1, auto_diverge=True)

    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1)
    W.addNode("dest1", 2, 0)
    W.addNode("dest2", 2, 2)
    link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    link2 = W.addLink("link2", "mid", "dest1", length=1000, free_flow_speed=20, jam_density=0.2)
    link3 = W.addLink("link3", "mid", "dest2", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest1", 0, 1000, 0.3)
    W.adddemand("orig", "dest2", 0, 1000, 0.3)

    W.exec_simulation()

    # Outflow rates
    assert equal_tolerance(avg_outflow(link1, 200, 800), 0.6)
    assert equal_tolerance(avg_inflow(link2, 200, 800), 0.3)
    assert equal_tolerance(avg_inflow(link3, 200, 800), 0.3)
    # Edie state: all free flow
    
    assert equal_tolerance(link1.q(540, 550), 0.6)
    assert equal_tolerance(link1.k(540, 550), q2k_free(0.6))
    assert equal_tolerance(link1.v(540, 550), 20)
    assert equal_tolerance(link2.q(540, 550), 0.3)
    assert equal_tolerance(link3.q(540, 550), 0.3)
    # Volume
    df = W.analyzer.link_to_pandas()
    assert equal_tolerance(df[df["link"]=="link1"]["traffic_volume"].values[0], 600)
    assert equal_tolerance(df[df["link"]=="link2"]["traffic_volume"].values[0], 300)
    assert equal_tolerance(df[df["link"]=="link3"]["traffic_volume"].values[0], 300)
def test_diverge_unequal_split():
    """Diverge, unequal split (0.2:0.4), no congestion.
    Theory: beta2=1/3, beta3=2/3, q1=0.6, q2=0.2, q3=0.4.
    """
    W = World(name="", deltat=5, tmax=1200, print_mode=0, save_mode=1, auto_diverge=True)

    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1)
    W.addNode("dest1", 2, 0)
    W.addNode("dest2", 2, 2)
    link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    link2 = W.addLink("link2", "mid", "dest1", length=1000, free_flow_speed=20, jam_density=0.2)
    link3 = W.addLink("link3", "mid", "dest2", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest1", 0, 1000, 0.2)
    W.adddemand("orig", "dest2", 0, 1000, 0.4)

    W.exec_simulation()

    assert equal_tolerance(avg_outflow(link1, 200, 800), 0.6)
    assert equal_tolerance(avg_inflow(link2, 200, 800), 0.2)
    assert equal_tolerance(avg_inflow(link3, 200, 800), 0.4)
    # Diverge ratio check
    assert equal_tolerance(avg_inflow(link3, 200, 800) / avg_inflow(link2, 200, 800), 2.0)
    # Flow conservation
    assert equal_tolerance(
        avg_inflow(link2, 200, 800) + avg_inflow(link3, 200, 800),
        avg_outflow(link1, 200, 800))
def test_diverge_capacity_in_congestion():
    """Diverge with capacity_in causing upstream congestion.
    Theory: demand to dest1=0.4, dest2=0.3. beta2=4/7, beta3=3/7.
    S_link2 = 0.2 (capacity_in).
    q1 = min{0.7, 0.2/(4/7), 0.8/(3/7)} = min{0.7, 0.35, 1.867} = 0.35
    q_link2 = 0.2, q_link3 = 0.15.
    """
    W = World(name="", deltat=5, tmax=1200, print_mode=0, save_mode=1, auto_diverge=True)

    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1)
    W.addNode("dest1", 2, 0)
    W.addNode("dest2", 2, 2)
    link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    link2 = W.addLink("link2", "mid", "dest1", length=1000, free_flow_speed=20, jam_density=0.2, capacity_in=0.2)
    link3 = W.addLink("link3", "mid", "dest2", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest1", 0, 1000, 0.4)
    W.adddemand("orig", "dest2", 0, 1000, 0.3)

    W.exec_simulation()

    # Outflow rates
    assert equal_tolerance(avg_outflow(link1, 200, 800), 0.35)
    assert equal_tolerance(avg_inflow(link2, 200, 800), 0.2)
    assert equal_tolerance(avg_inflow(link3, 200, 800), 0.15)
    # link2, link3: free flow
    
    assert equal_tolerance(link2.q(540, 550), 0.2)
    assert equal_tolerance(link2.v(540, 550), 20)
    assert equal_tolerance(link3.q(540, 550), 0.15)
    assert equal_tolerance(link3.v(540, 550), 20)
    # Flow conservation
    assert equal_tolerance(
        avg_inflow(link2, 200, 800) + avg_inflow(link3, 200, 800),
        avg_outflow(link1, 200, 800))
def test_diverge_capacity_in_nocongestion():
    """Diverge with capacity_in but demand within limit.
    Theory: demand to dest1=0.2, dest2=0.3. beta2=2/5, beta3=3/5.
    S_link2 = 0.2. q1=min{0.5, 0.2/0.4, 0.8/0.6}=min{0.5, 0.5, 1.33}=0.5.
    q2=0.2, q3=0.3. All free flow.
    """
    W = World(name="", deltat=5, tmax=1200, print_mode=0, save_mode=1, auto_diverge=True)

    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1)
    W.addNode("dest1", 2, 0)
    W.addNode("dest2", 2, 2)
    link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    link2 = W.addLink("link2", "mid", "dest1", length=1000, free_flow_speed=20, jam_density=0.2, capacity_in=0.2)
    link3 = W.addLink("link3", "mid", "dest2", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest1", 0, 1000, 0.2)
    W.adddemand("orig", "dest2", 0, 1000, 0.3)

    W.exec_simulation()

    assert equal_tolerance(avg_outflow(link1, 200, 800), 0.5)
    assert equal_tolerance(avg_inflow(link2, 200, 800), 0.2)
    assert equal_tolerance(avg_inflow(link3, 200, 800), 0.3)
    # All free flow
    
    assert equal_tolerance(link1.v(540, 550), 20)
    assert equal_tolerance(link2.v(540, 550), 20)
    assert equal_tolerance(link3.v(540, 550), 20)
# ============================================================
# Flow conservation tests (cumulative count verification)
# ============================================================

def test_flow_conservation_merge():
    """Verify cum_departure of inlinks = cum_arrival of outlink at every timestep."""
    W = World(name="", deltat=5, tmax=1200, print_mode=0, save_mode=1)

    W.addNode("orig1", 0, 0)
    W.addNode("orig2", 0, 2)
    W.addNode("merge", 1, 1)
    W.addNode("dest", 2, 1)
    link1 = W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
    link2 = W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=3)
    link3 = W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig1", "dest", 0, 1000, 0.6)
    W.adddemand("orig2", "dest", 0, 1000, 0.6)

    W.exec_simulation()

    for t in range(W.TSIZE + 1):
        outflow = link1.cum_departure[t] + link2.cum_departure[t]
        inflow = link3.cum_arrival[t]
        assert abs(outflow - inflow) < 1e-6, f"Flow conservation violated at t={t}"
def test_flow_conservation_diverge():
    """Verify cum_departure of inlink = sum of cum_arrival of outlinks at every timestep."""
    W = World(name="", deltat=5, tmax=1200, print_mode=0, save_mode=1, auto_diverge=True)

    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1)
    W.addNode("dest1", 2, 0)
    W.addNode("dest2", 2, 2)
    link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    link2 = W.addLink("link2", "mid", "dest1", length=1000, free_flow_speed=20, jam_density=0.2)
    link3 = W.addLink("link3", "mid", "dest2", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest1", 0, 1000, 0.5)
    W.adddemand("orig", "dest2", 0, 1000, 0.3)

    W.exec_simulation()

    for t in range(W.TSIZE + 1):
        outflow = link1.cum_departure[t]
        inflow = link2.cum_arrival[t] + link3.cum_arrival[t]
        assert abs(outflow - inflow) < 1e-6, f"Flow conservation violated at t={t}"


# ================================================================
# Manual diverge_ratio / absorption_ratio
# ================================================================

def _avg_outflow(link, t1, t2):
    dt = link.W.DELTAT
    i1 = max(0, min(int(t1/dt), len(link.cum_departure)-1))
    i2 = max(0, min(int(t2/dt), len(link.cum_departure)-1))
    if i2 <= i1:
        return 0
    return (link.cum_departure[i2] - link.cum_departure[i1]) / ((i2-i1)*dt)

def _avg_inflow(link, t1, t2):
    dt = link.W.DELTAT
    i1 = max(0, min(int(t1/dt), len(link.cum_arrival)-1))
    i2 = max(0, min(int(t2/dt), len(link.cum_arrival)-1))
    if i2 <= i1:
        return 0
    return (link.cum_arrival[i2] - link.cum_arrival[i1]) / ((i2-i1)*dt)


def test_manual_diverge_equal_split():
    """Manual diverge_ratio: 50/50 split."""
    W = World(deltat=5, tmax=1200, print_mode=0)
    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1, diverge_ratio={"link2": 0.5, "link3": 0.5})
    W.addNode("dest1", 2, 0)
    W.addNode("dest2", 2, 2)
    W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    W.addLink("link2", "mid", "dest1", length=1000, free_flow_speed=20, jam_density=0.2)
    W.addLink("link3", "mid", "dest2", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest1", 0, 1000, 0.6)
    W.exec_simulation()
    assert equal_tolerance(_avg_inflow(W.get_link("link2"), 200, 800), 0.3)
    assert equal_tolerance(_avg_inflow(W.get_link("link3"), 200, 800), 0.3)


def test_manual_diverge_unequal_split():
    """Manual diverge_ratio: 30/70 split."""
    W = World(deltat=5, tmax=1200, print_mode=0)
    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1, diverge_ratio={"link2": 0.3, "link3": 0.7})
    W.addNode("dest1", 2, 0)
    W.addNode("dest2", 2, 2)
    W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    W.addLink("link2", "mid", "dest1", length=1000, free_flow_speed=20, jam_density=0.2)
    W.addLink("link3", "mid", "dest2", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest1", 0, 1000, 0.6)
    W.exec_simulation()
    assert equal_tolerance(_avg_inflow(W.get_link("link2"), 200, 800), 0.18)
    assert equal_tolerance(_avg_inflow(W.get_link("link3"), 200, 800), 0.42)


def test_manual_diverge_supply_constrained():
    """Manual diverge with capacity_in causing congestion."""
    W = World(deltat=5, tmax=1200, print_mode=0)
    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1, diverge_ratio={"link2": 0.5, "link3": 0.5})
    W.addNode("dest1", 2, 0)
    W.addNode("dest2", 2, 2)
    W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    W.addLink("link2", "mid", "dest1", length=1000, free_flow_speed=20, jam_density=0.2, capacity_in=0.2)
    W.addLink("link3", "mid", "dest2", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest1", 0, 1000, 0.6)
    W.exec_simulation()
    assert equal_tolerance(_avg_outflow(W.get_link("link1"), 200, 800), 0.4)
    assert equal_tolerance(_avg_inflow(W.get_link("link2"), 200, 800), 0.2)


def test_manual_diverge_flow_conservation():
    """Flow conservation at diverge node with manual ratios."""
    W = World(deltat=5, tmax=1200, print_mode=0)
    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1, diverge_ratio={"link2": 0.4, "link3": 0.6})
    W.addNode("dest1", 2, 0)
    W.addNode("dest2", 2, 2)
    link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    link2 = W.addLink("link2", "mid", "dest1", length=1000, free_flow_speed=20, jam_density=0.2)
    link3 = W.addLink("link3", "mid", "dest2", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest1", 0, 1000, 0.7)
    W.exec_simulation()
    for t in range(W.TSIZE + 1):
        assert abs(link1.cum_departure[t] - link2.cum_arrival[t] - link3.cum_arrival[t]) < 1e-6


def test_absorption_full():
    """absorption_ratio=1.0: all traffic absorbed at intermediate node."""
    W = World(deltat=5, tmax=2000, print_mode=0)
    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1, absorption_ratio=1.0)
    W.addNode("dest", 2, 2)
    link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    link2 = W.addLink("link2", "mid", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "mid", 0, 500, 0.5)
    W.exec_simulation()
    assert link2.cum_arrival[-1] < 0.1
    assert equal_tolerance(link1.cum_departure[-1], 250)


def test_absorption_partial():
    """absorption_ratio=0.5: half absorbed, half passes through."""
    W = World(deltat=5, tmax=1200, print_mode=0)
    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1, absorption_ratio=0.5)
    W.addNode("dest", 2, 2)
    link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    link2 = W.addLink("link2", "mid", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest", 0, 1000, 0.4)
    W.exec_simulation()
    assert equal_tolerance(_avg_outflow(link1, 200, 800), 0.4)
    assert equal_tolerance(_avg_inflow(link2, 200, 800), 0.2)


def test_manual_vs_auto_diverge():
    """Manual diverge_ratio={0.5, 0.5} matches auto_diverge with equal demands."""
    W1 = World(deltat=5, tmax=1200, print_mode=0)
    W1.addNode("o", 0, 0)
    W1.addNode("m", 1, 1, diverge_ratio={"l2": 0.5, "l3": 0.5})
    W1.addNode("d1", 2, 0); W1.addNode("d2", 2, 2)
    W1.addLink("l1", "o", "m", length=1000, free_flow_speed=20, jam_density=0.2)
    W1.addLink("l2", "m", "d1", length=1000, free_flow_speed=20, jam_density=0.2)
    W1.addLink("l3", "m", "d2", length=1000, free_flow_speed=20, jam_density=0.2)
    W1.adddemand("o", "d1", 0, 1000, 0.3); W1.adddemand("o", "d2", 0, 1000, 0.3)
    W1.exec_simulation(); W1.analyzer.basic_analysis()

    W2 = World(deltat=5, tmax=1200, print_mode=0, auto_diverge=True)
    W2.addNode("o", 0, 0); W2.addNode("m", 1, 1)
    W2.addNode("d1", 2, 0); W2.addNode("d2", 2, 2)
    W2.addLink("l1", "o", "m", length=1000, free_flow_speed=20, jam_density=0.2)
    W2.addLink("l2", "m", "d1", length=1000, free_flow_speed=20, jam_density=0.2)
    W2.addLink("l3", "m", "d2", length=1000, free_flow_speed=20, jam_density=0.2)
    W2.adddemand("o", "d1", 0, 1000, 0.3); W2.adddemand("o", "d2", 0, 1000, 0.3)
    W2.exec_simulation(); W2.analyzer.basic_analysis()

    assert equal_tolerance(W1.analyzer.total_travel_time, W2.analyzer.total_travel_time)


# ================================================================
# INM (Incremental Node Model) - Floetteroed & Rohde (2011)
# ================================================================

def _build_flotterod(W, sw_capacity=None):
    """Build Floetteroed 3x3 intersection scenario."""
    W.addNode("O_S", 0, 0); W.addNode("O_E", 2, 0); W.addNode("O_N", 1, 2)
    W.addNode("intersection", 1, 1, turning_fractions={
        "P_S": {"S_N": 0.5, "S_W": 0.5},
        "P_E": {"S_W": 1.0},
        "P_N": {"S_W": 0.5, "S_S": 0.5},
    })
    W.addNode("D_N", 1, 3); W.addNode("D_W", -1, 1); W.addNode("D_S", 1, -1)
    W.addLink("P_S", "O_S", "intersection", length=1000, free_flow_speed=20,
              backward_wave_speed=5, jam_density=0.2, merge_priority=1.0)
    W.addLink("P_E", "O_E", "intersection", length=1000, free_flow_speed=20,
              backward_wave_speed=5, jam_density=0.2, merge_priority=0.1)
    W.addLink("P_N", "O_N", "intersection", length=1000, free_flow_speed=20,
              backward_wave_speed=5, jam_density=0.2, merge_priority=10.0)
    W.addLink("S_N", "intersection", "D_N", length=1000, free_flow_speed=20,
              backward_wave_speed=5, jam_density=0.2)
    kw = {"capacity": sw_capacity} if sw_capacity else {"jam_density": 0.2}
    W.addLink("S_W", "intersection", "D_W", length=1000, free_flow_speed=20,
              backward_wave_speed=5, **kw)
    W.addLink("S_S", "intersection", "D_S", length=1000, free_flow_speed=20,
              backward_wave_speed=5, jam_density=0.2)


def test_inm_uncongested():
    """Floetteroed Table 1: uncongested 3x3 intersection."""
    W = World(deltat=5, tmax=3000, print_mode=0)
    _build_flotterod(W)
    d_PS, d_PE, d_PN = 600/3600, 100/3600, 600/3600
    W.adddemand("O_S", "D_N", 0, 2000, d_PS)
    W.adddemand("O_E", "D_W", 0, 2000, d_PE)
    W.adddemand("O_N", "D_S", 0, 2000, d_PN)
    W.exec_simulation()
    assert equal_tolerance(_avg_outflow(W.get_link("P_S"), 200, 1500), d_PS)
    assert equal_tolerance(_avg_outflow(W.get_link("P_E"), 200, 1500), d_PE)
    assert equal_tolerance(_avg_outflow(W.get_link("P_N"), 200, 1500), d_PN)
    assert equal_tolerance(_avg_inflow(W.get_link("S_N"), 200, 1500), 300/3600)
    assert equal_tolerance(_avg_inflow(W.get_link("S_W"), 200, 1500), 700/3600)
    assert equal_tolerance(_avg_inflow(W.get_link("S_S"), 200, 1500), 300/3600)


def test_inm_congested():
    """Floetteroed Table 2: congested (S_W capacity=400/3600)."""
    W = World(deltat=5, tmax=3000, print_mode=0)
    _build_flotterod(W, sw_capacity=400/3600)
    W.adddemand("O_S", "D_N", 0, 2000, 600/3600)
    W.adddemand("O_E", "D_W", 0, 2000, 100/3600)
    W.adddemand("O_N", "D_S", 0, 2000, 600/3600)
    W.exec_simulation()
    assert equal_tolerance(_avg_outflow(W.get_link("P_N"), 500, 2000), 600/3600)
    assert equal_tolerance(_avg_outflow(W.get_link("P_S"), 500, 2000), 166.67/3600, rel_tol=0.15)
    assert equal_tolerance(_avg_inflow(W.get_link("S_W"), 500, 2000), 400/3600)
    assert equal_tolerance(_avg_inflow(W.get_link("S_S"), 500, 2000), 300/3600)


def test_inm_flow_conservation():
    """Flow conservation at INM node."""
    W = World(deltat=5, tmax=3000, print_mode=0)
    _build_flotterod(W, sw_capacity=400/3600)
    W.adddemand("O_S", "D_N", 0, 2000, 600/3600)
    W.adddemand("O_E", "D_W", 0, 2000, 100/3600)
    W.adddemand("O_N", "D_S", 0, 2000, 600/3600)
    W.exec_simulation()
    PS, PE, PN = W.get_link("P_S"), W.get_link("P_E"), W.get_link("P_N")
    SN, SW, SS = W.get_link("S_N"), W.get_link("S_W"), W.get_link("S_S")
    for t in range(W.TSIZE + 1):
        total_in = PS.cum_departure[t] + PE.cum_departure[t] + PN.cum_departure[t]
        total_out = SN.cum_arrival[t] + SW.cum_arrival[t] + SS.cum_arrival[t]
        assert abs(total_in - total_out) < 1e-6
