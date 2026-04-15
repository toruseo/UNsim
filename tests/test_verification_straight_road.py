"""
Verifies UNsim outputs for a straight road in various configurations.
"""

import pytest
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unsim import *

"""
default FD:
    u = 20
    kappa = 0.2
    tau = 1
    w = 5
    k^* = 0.04
    q^* = 0.8
"""

def test_1link():
    W = World(name="", deltat=5, tmax=2000, print_mode=1, save_mode=1)

    W.addNode("orig", 0, 0)
    W.addNode("dest", 1, 1)
    link = W.addLink("link", "orig", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest", 0, 500, 0.5)

    W.exec_simulation()
    W.analyzer.print_simple_stats()

    W.analyzer.basic_analysis()
    assert equal_tolerance(W.analyzer.trip_all, 250)
    assert equal_tolerance(W.analyzer.trip_completed, 250)
    assert equal_tolerance(W.analyzer.total_travel_time, 12500)
    assert equal_tolerance(W.analyzer.average_travel_time, 50)
    assert equal_tolerance(W.analyzer.average_delay, 0)

    # Newell state: during demand (t=300), free flow
    assert equal_tolerance(link.q(300, 550), 0.5)
    assert equal_tolerance(link.k(300, 550), 0.025)
    assert equal_tolerance(link.v(300, 550), 20)
    # After demand ends (t=900), no flow
    assert equal_tolerance(link.q(900, 550), 0)
    assert equal_tolerance(link.k(900, 550), 0)
    assert equal_tolerance(link.v(900, 550), 20)

def test_1link_demand_by_volume():
    W = World(name="", deltat=5, tmax=2000, print_mode=1, save_mode=1)

    W.addNode("orig", 0, 0)
    W.addNode("dest", 1, 1)
    link = W.addLink("link", "orig", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest", 0, 500, volume=0.5*500)

    W.exec_simulation()

    W.analyzer.basic_analysis()
    assert equal_tolerance(W.analyzer.trip_all, 250)
    assert equal_tolerance(W.analyzer.trip_completed, 250)
    assert equal_tolerance(W.analyzer.total_travel_time, 12500)
    assert equal_tolerance(W.analyzer.average_travel_time, 50)
    assert equal_tolerance(W.analyzer.average_delay, 0)

def test_1link_maxflow():
    W = World(name="", deltat=5, tmax=2000, print_mode=1, save_mode=1)

    W.addNode("orig", 0, 0)
    W.addNode("dest", 1, 1)
    link = W.addLink("link", "orig", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest", 0, 2000, 2)

    W.exec_simulation()

    assert equal_tolerance(link.q(300, 550), 0.8)
    assert equal_tolerance(link.q(900, 550), 0.8)
    assert equal_tolerance(link.k(300, 550), 0.04)
    assert equal_tolerance(link.k(900, 550), 0.04)
    assert equal_tolerance(link.v(300, 550), 20)
    assert equal_tolerance(link.v(900, 550), 20)

def test_2link():
    W = World(name="", deltat=5, tmax=2000, print_mode=1, save_mode=1)

    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1)
    W.addNode("dest", 2, 2)
    link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    link2 = W.addLink("link2", "mid", "dest", length=1000, free_flow_speed=10, jam_density=0.2)
    W.adddemand("orig", "dest", 0, 500, 0.5)

    W.exec_simulation()

    W.analyzer.basic_analysis()
    assert equal_tolerance(W.analyzer.trip_all, 250)
    assert equal_tolerance(W.analyzer.trip_completed, 250)
    assert equal_tolerance(W.analyzer.total_travel_time, 37500)
    assert equal_tolerance(W.analyzer.average_travel_time, 150)
    assert equal_tolerance(W.analyzer.average_delay, 0)

    # link1: u=20, free flow
    assert equal_tolerance(link1.q(300, 550), 0.5)
    assert equal_tolerance(link1.k(300, 550), 0.025)
    assert equal_tolerance(link1.v(300, 550), 20)
    assert equal_tolerance(link1.q(900, 550), 0)
    # link2: u=10, free flow
    assert equal_tolerance(link2.q(300, 550), 0.5)
    assert equal_tolerance(link2.k(300, 550), 0.05)
    assert equal_tolerance(link2.v(300, 550), 10)
    assert equal_tolerance(link2.q(900, 550), 0)

def test_2link_maxflow():
    W = World(name="", deltat=5, tmax=2000, print_mode=1, save_mode=1)

    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1)
    W.addNode("dest", 2, 2)
    link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    link2 = W.addLink("link2", "mid", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest", 0, 2000, 2)

    W.exec_simulation()

    for t in [300, 900]:
        assert equal_tolerance(link1.q(t, 550), 0.8)
        assert equal_tolerance(link1.k(t, 550), 0.04)
        assert equal_tolerance(link1.v(t, 550), 20)
        assert equal_tolerance(link2.q(t, 550), 0.8)
        assert equal_tolerance(link2.k(t, 550), 0.04)
        assert equal_tolerance(link2.v(t, 550), 20)

def test_2link_bottleneck_due_to_free_flow_speed():
    W = World(name="", deltat=5, tmax=2000, print_mode=1, save_mode=1)

    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1)
    W.addNode("dest", 2, 2)
    link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    link2 = W.addLink("link2", "mid", "dest", length=1000, free_flow_speed=10, jam_density=0.2)
    W.adddemand("orig", "dest", 0, 500, 0.8)
    W.adddemand("orig", "dest", 500, 1500, 0.4)

    W.exec_simulation()

    W.analyzer.basic_analysis()
    assert equal_tolerance(W.analyzer.trip_all, 800)
    assert equal_tolerance(W.analyzer.trip_completed, 800)
    assert equal_tolerance(W.analyzer.total_travel_time, 146000.0)
    assert equal_tolerance(W.analyzer.average_travel_time, 182.5)
    assert equal_tolerance(W.analyzer.average_delay, 32.5)

    # Congestion before bottleneck (t=300, link1)
    assert equal_tolerance(link1.q(300, 550), 0.6667)
    assert equal_tolerance(link1.k(300, 550), 0.06667)
    assert equal_tolerance(link1.v(300, 550), 10)
    # Free flow after demand drop (t=900, link1)
    assert equal_tolerance(link1.q(900, 550), 0.4)
    assert equal_tolerance(link1.k(900, 550), 0.02)
    assert equal_tolerance(link1.v(900, 550), 20)
    # Free flow after BN (t=300, link2)
    assert equal_tolerance(link2.q(300, 550), 0.6667)
    assert equal_tolerance(link2.k(300, 550), 0.06667)
    assert equal_tolerance(link2.v(300, 550), 10)

def test_2link_bottleneck_due_to_capacity_out():
    W = World(name="", deltat=5, tmax=2000, print_mode=1, save_mode=1)

    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1)
    W.addNode("dest", 2, 2)
    link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2, capacity_out=0.66666)
    link2 = W.addLink("link2", "mid", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest", 0, 500, 0.8)
    W.adddemand("orig", "dest", 500, 1500, 0.4)

    W.exec_simulation()

    W.analyzer.basic_analysis()
    assert equal_tolerance(W.analyzer.trip_all, 800)
    assert equal_tolerance(W.analyzer.trip_completed, 800)
    assert equal_tolerance(W.analyzer.total_travel_time, 104775)
    assert equal_tolerance(W.analyzer.average_travel_time, 131)
    assert equal_tolerance(W.analyzer.average_delay, 31)

    # Congestion before BN (t=300, link1)
    assert equal_tolerance(link1.q(300, 550), 0.6667)
    assert equal_tolerance(link1.k(300, 550), 0.06667)
    assert equal_tolerance(link1.v(300, 550), 10)
    # link2 free flow at 0.6667
    assert equal_tolerance(link2.q(300, 550), 0.6667)
    assert equal_tolerance(link2.k(300, 550), 0.03333)
    assert equal_tolerance(link2.v(300, 550), 20)

def test_2link_bottleneck_due_to_jam_density():
    W = World(name="", deltat=5, tmax=2000, print_mode=1, save_mode=1)

    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1)
    W.addNode("dest", 2, 2)
    link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    link2 = W.addLink("link2", "mid", "dest", length=1000, free_flow_speed=20, jam_density=0.1)
    W.adddemand("orig", "dest", 0, 500, 0.8)
    W.adddemand("orig", "dest", 500, 1500, 0.4)

    W.exec_simulation()

    W.analyzer.basic_analysis()
    assert equal_tolerance(W.analyzer.trip_all, 800)
    assert equal_tolerance(W.analyzer.trip_completed, 800)
    assert equal_tolerance(W.analyzer.total_travel_time, 106000)
    assert equal_tolerance(W.analyzer.average_travel_time, 132.5)
    assert equal_tolerance(W.analyzer.average_delay, 32.5)

    assert equal_tolerance(link1.q(300, 550), 0.6667)
    assert equal_tolerance(link1.k(300, 550), 0.06667)
    assert equal_tolerance(link1.v(300, 550), 10)

def test_2link_bottleneck_due_to_node_capacity():
    W = World(name="", deltat=5, tmax=2000, print_mode=1, save_mode=1)

    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1, flow_capacity=0.666)
    W.addNode("dest", 2, 2)
    link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    link2 = W.addLink("link2", "mid", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest", 0, 500, 0.8)
    W.adddemand("orig", "dest", 500, 1500, 0.4)

    W.exec_simulation()

    W.analyzer.basic_analysis()
    assert equal_tolerance(W.analyzer.trip_all, 800)
    assert equal_tolerance(W.analyzer.trip_completed, 800)
    assert equal_tolerance(W.analyzer.total_travel_time, 106000)
    assert equal_tolerance(W.analyzer.average_travel_time, 132.5)
    assert equal_tolerance(W.analyzer.average_delay, 32.5)

    assert equal_tolerance(link1.q(300, 550), 0.6667)
    assert equal_tolerance(link2.q(300, 550), 0.6667)
    assert equal_tolerance(link2.v(300, 550), 20)

def test_2link_leave_at_middle():
    W = World(name="", deltat=5, tmax=2000, print_mode=1, save_mode=1, auto_diverge=True)

    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1)
    W.addNode("dest", 2, 2)
    link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    link2 = W.addLink("link2", "mid", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "mid", 0, 500, 0.666)

    W.exec_simulation()

    W.analyzer.basic_analysis()
    assert equal_tolerance(W.analyzer.trip_all, 330)
    assert equal_tolerance(W.analyzer.trip_completed, 330)
    assert equal_tolerance(W.analyzer.total_travel_time, 17300)
    assert equal_tolerance(W.analyzer.average_travel_time, 52.4)
    assert equal_tolerance(W.analyzer.average_delay, 2.4, abs_tol=3)

    assert equal_tolerance(link1.q(300, 550), 0.6667)
    assert equal_tolerance(link1.v(300, 550), 20)
    assert equal_tolerance(link1.q(900, 550), 0.0)
    # link2: no traffic
    assert equal_tolerance(link2.q(300, 550), 0.0)
    assert equal_tolerance(link2.q(900, 550), 0.0)

def test_3link_queuespillback():
    W = World(name="", deltat=5, tmax=2000, print_mode=1, save_mode=1)

    W.addNode("orig", 0, 0)
    W.addNode("mid1", 1, 1)
    W.addNode("mid2", 2, 2)
    W.addNode("dest", 3, 3)
    link1 = W.addLink("link1", "orig", "mid1", length=1000, free_flow_speed=20, jam_density=0.2)
    link2 = W.addLink("link2", "mid1", "mid2", length=1000, free_flow_speed=20, jam_density=0.2, capacity_out=0.4)
    link3 = W.addLink("link3", "mid2", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest", 0, 400, 0.8)
    W.adddemand("orig", "dest", 400, 800, 0.4)
    W.adddemand("orig", "dest", 800, 2000, 0.1)

    W.exec_simulation()

    W.analyzer.basic_analysis()
    assert equal_tolerance(W.analyzer.trip_all, 600)
    assert equal_tolerance(W.analyzer.trip_completed, 585, rel_tol=0.2)
    assert equal_tolerance(W.analyzer.total_travel_time, 221050, rel_tol=0.2)
    assert equal_tolerance(W.analyzer.average_travel_time, 378, rel_tol=0.2)
    assert equal_tolerance(W.analyzer.average_delay, 228, rel_tol=0.2)

    # Before congestion (t=180, link1 downstream)
    assert equal_tolerance(link1.q(180, 850), 0.8, rel_tol=0.2)
    assert equal_tolerance(link1.v(180, 850), 20, rel_tol=0.2)
    # During congestion (t=660, link1 downstream)
    assert equal_tolerance(link1.q(660, 850), 0.4, rel_tol=0.2)
    assert equal_tolerance(link1.k(660, 850), 0.12, rel_tol=0.2)
    # link3 free flow during bottleneck
    assert equal_tolerance(link3.q(660, 850), 0.4, rel_tol=0.2)
    assert equal_tolerance(link3.v(660, 850), 20, rel_tol=0.2)

def test_1link_iterative_exec():
    W = World(name="", deltat=5, tmax=2000, print_mode=0, save_mode=1)

    W.addNode("orig", 0, 0)
    W.addNode("dest", 1, 1)
    link = W.addLink("link", "orig", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest", 0, 500, 0.5)

    while W.check_simulation_ongoing():
        W.exec_simulation(duration_t=random.randint(50, 200))

    W.analyzer.basic_analysis()
    assert equal_tolerance(W.analyzer.trip_all, 250)
    assert equal_tolerance(W.analyzer.trip_completed, 250)
    assert equal_tolerance(W.analyzer.total_travel_time, 12500)
    assert equal_tolerance(W.analyzer.average_travel_time, 50)
    assert equal_tolerance(W.analyzer.average_delay, 0)

def test_1link_range_query():
    """Test that q/k/v accept [lo, hi] ranges for area averaging."""
    W = World(name="", deltat=5, tmax=2000, print_mode=0, save_mode=1)

    W.addNode("orig", 0, 0)
    W.addNode("dest", 1, 1)
    link = W.addLink("link", "orig", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig", "dest", 0, 500, 0.5)

    W.exec_simulation()

    # Time range average, single x
    assert equal_tolerance(link.q([200, 400], 500), 0.5)
    # Space range average, single t
    assert equal_tolerance(link.k(300, [100, 900]), 0.025)
    # Both ranges
    assert equal_tolerance(link.q([200, 400], [100, 900]), 0.5)
    # Default x (midpoint)
    assert equal_tolerance(link.q(300), 0.5)
