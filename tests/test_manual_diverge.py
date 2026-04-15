"""
Tests for manual diverge_ratio / absorption_ratio specification (auto_diverge=False).
Verifies that addNode(diverge_ratio=..., absorption_ratio=...) works correctly.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unsim import *


def avg_outflow(link, t1, t2):
    dt = link.W.DELTAT
    i1, i2 = int(t1/dt), int(t2/dt)
    i1 = max(0, min(i1, len(link.cum_departure)-1))
    i2 = max(0, min(i2, len(link.cum_departure)-1))
    if i2 <= i1:
        return 0
    return (link.cum_departure[i2] - link.cum_departure[i1]) / ((i2-i1)*dt)

def avg_inflow(link, t1, t2):
    dt = link.W.DELTAT
    i1, i2 = int(t1/dt), int(t2/dt)
    i1 = max(0, min(i1, len(link.cum_arrival)-1))
    i2 = max(0, min(i2, len(link.cum_arrival)-1))
    if i2 <= i1:
        return 0
    return (link.cum_arrival[i2] - link.cum_arrival[i1]) / ((i2-i1)*dt)


class TestManualDiverge:
    """Test diverge_ratio specified via addNode."""

    def test_equal_split(self):
        """50/50 split specified manually."""
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

        assert equal_tolerance(avg_inflow(W.get_link("link2"), 200, 800), 0.3)
        assert equal_tolerance(avg_inflow(W.get_link("link3"), 200, 800), 0.3)

    def test_unequal_split(self):
        """30/70 split specified manually."""
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

        assert equal_tolerance(avg_inflow(W.get_link("link2"), 200, 800), 0.18)
        assert equal_tolerance(avg_inflow(W.get_link("link3"), 200, 800), 0.42)
        # Ratio check
        assert equal_tolerance(
            avg_inflow(W.get_link("link3"), 200, 800) / avg_inflow(W.get_link("link2"), 200, 800),
            7/3)

    def test_supply_constrained(self):
        """Manual diverge with capacity_in causing congestion.
        beta2=0.5, beta3=0.5, but link2 capacity_in=0.2.
        q1 = min{0.6, 0.2/0.5, 0.8/0.5} = min{0.6, 0.4, 1.6} = 0.4
        q2 = 0.2, q3 = 0.2.
        """
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

        assert equal_tolerance(avg_outflow(W.get_link("link1"), 200, 800), 0.4)
        assert equal_tolerance(avg_inflow(W.get_link("link2"), 200, 800), 0.2)
        assert equal_tolerance(avg_inflow(W.get_link("link3"), 200, 800), 0.2)

    def test_default_equal_split(self):
        """No diverge_ratio specified -> default equal split."""
        W = World(deltat=5, tmax=1200, print_mode=0)
        W.addNode("orig", 0, 0)
        W.addNode("mid", 1, 1)  # no diverge_ratio
        W.addNode("dest1", 2, 0)
        W.addNode("dest2", 2, 2)
        W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
        W.addLink("link2", "mid", "dest1", length=1000, free_flow_speed=20, jam_density=0.2)
        W.addLink("link3", "mid", "dest2", length=1000, free_flow_speed=20, jam_density=0.2)
        W.adddemand("orig", "dest1", 0, 1000, 0.6)

        W.exec_simulation()

        # Default: 50/50 split
        assert equal_tolerance(avg_inflow(W.get_link("link2"), 200, 800), 0.3)
        assert equal_tolerance(avg_inflow(W.get_link("link3"), 200, 800), 0.3)

    def test_flow_conservation_manual(self):
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
            out = link1.cum_departure[t]
            inp = link2.cum_arrival[t] + link3.cum_arrival[t]
            assert abs(out - inp) < 1e-6


class TestManualAbsorption:
    """Test absorption_ratio specified via addNode."""

    def test_full_absorption(self):
        """absorption_ratio=1.0: all traffic absorbed at intermediate node."""
        W = World(deltat=5, tmax=2000, print_mode=0)
        W.addNode("orig", 0, 0)
        W.addNode("mid", 1, 1, absorption_ratio=1.0)
        W.addNode("dest", 2, 2)
        link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
        link2 = W.addLink("link2", "mid", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
        W.adddemand("orig", "mid", 0, 500, 0.5)

        W.exec_simulation()

        # link2 should have no traffic
        assert link2.cum_arrival[-1] < 0.1
        # link1 should have carried all traffic
        assert equal_tolerance(link1.cum_departure[-1], 250)

    def test_partial_absorption(self):
        """absorption_ratio=0.5: half absorbed, half passes through."""
        W = World(deltat=5, tmax=1200, print_mode=0)
        W.addNode("orig", 0, 0)
        W.addNode("mid", 1, 1, absorption_ratio=0.5)
        W.addNode("dest", 2, 2)
        link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
        link2 = W.addLink("link2", "mid", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
        W.adddemand("orig", "dest", 0, 1000, 0.4)

        W.exec_simulation()

        # Outflow from link1 = 0.4, inflow to link2 = 0.4 * 0.5 = 0.2
        assert equal_tolerance(avg_outflow(link1, 200, 800), 0.4)
        assert equal_tolerance(avg_inflow(link2, 200, 800), 0.2)

    def test_zero_absorption(self):
        """absorption_ratio=0: all traffic passes through (default)."""
        W = World(deltat=5, tmax=1200, print_mode=0)
        W.addNode("orig", 0, 0)
        W.addNode("mid", 1, 1, absorption_ratio=0.0)
        W.addNode("dest", 2, 2)
        link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
        link2 = W.addLink("link2", "mid", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
        W.adddemand("orig", "dest", 0, 1000, 0.5)

        W.exec_simulation()

        assert equal_tolerance(avg_outflow(link1, 200, 800), 0.5)
        assert equal_tolerance(avg_inflow(link2, 200, 800), 0.5)


class TestManualVsAuto:
    """Verify manual and auto_diverge produce same results for equivalent configs."""

    def test_diverge_manual_equals_auto(self):
        """Manual diverge_ratio={0.5, 0.5} should match auto_diverge with equal demands."""
        # Manual
        W1 = World(deltat=5, tmax=1200, print_mode=0)
        W1.addNode("o", 0, 0)
        W1.addNode("m", 1, 1, diverge_ratio={"l2": 0.5, "l3": 0.5})
        W1.addNode("d1", 2, 0)
        W1.addNode("d2", 2, 2)
        W1.addLink("l1", "o", "m", length=1000, free_flow_speed=20, jam_density=0.2)
        W1.addLink("l2", "m", "d1", length=1000, free_flow_speed=20, jam_density=0.2)
        W1.addLink("l3", "m", "d2", length=1000, free_flow_speed=20, jam_density=0.2)
        W1.adddemand("o", "d1", 0, 1000, 0.3)
        W1.adddemand("o", "d2", 0, 1000, 0.3)
        W1.exec_simulation()
        W1.analyzer.basic_analysis()

        # Auto
        W2 = World(deltat=5, tmax=1200, print_mode=0, auto_diverge=True)
        W2.addNode("o", 0, 0)
        W2.addNode("m", 1, 1)
        W2.addNode("d1", 2, 0)
        W2.addNode("d2", 2, 2)
        W2.addLink("l1", "o", "m", length=1000, free_flow_speed=20, jam_density=0.2)
        W2.addLink("l2", "m", "d1", length=1000, free_flow_speed=20, jam_density=0.2)
        W2.addLink("l3", "m", "d2", length=1000, free_flow_speed=20, jam_density=0.2)
        W2.adddemand("o", "d1", 0, 1000, 0.3)
        W2.adddemand("o", "d2", 0, 1000, 0.3)
        W2.exec_simulation()
        W2.analyzer.basic_analysis()

        assert equal_tolerance(W1.analyzer.total_travel_time, W2.analyzer.total_travel_time)
        # Cumulative counts should match
        for i in range(len(W1.LINKS)):
            assert max(abs(a-b) for a, b in zip(W1.LINKS[i].cum_departure, W2.LINKS[i].cum_departure)) < 1e-6
