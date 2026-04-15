"""
Tests for automatic deltat computation and deltat sensitivity.
Verifies that:
1. deltat is auto-computed as min(d/u) when not specified.
2. Different deltat values produce consistent results (LTM is exact for any valid deltat).
3. All node types (merge, diverge, dummy, origin, destination) are insensitive to deltat.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unsim import *


# ================================================================
# Helper
# ================================================================

def _compare(factory, deltats, rel_tol=0.1):
    """Run a scenario at multiple deltat values and verify consistency.

    Parameters
    ----------
    factory : callable(deltat) -> World (executed, with basic_analysis done)
    deltats : list of float or None
    rel_tol : float
    """
    results = {}
    for dt in deltats:
        W = factory(dt)
        W.exec_simulation()
        W.analyzer.basic_analysis()
        results[dt] = {
            "trip_all": W.analyzer.trip_all,
            "trip_completed": W.analyzer.trip_completed,
            "total_travel_time": W.analyzer.total_travel_time,
            "average_travel_time": W.analyzer.average_travel_time,
        }

    ref_dt = deltats[0]
    ref = results[ref_dt]
    for dt in deltats[1:]:
        r = results[dt]
        assert equal_tolerance(r["total_travel_time"], ref["total_travel_time"], rel_tol=rel_tol), \
            f"TTT mismatch: dt={ref_dt}->{ref['total_travel_time']:.1f}, dt={dt}->{r['total_travel_time']:.1f}"
        assert equal_tolerance(r["average_travel_time"], ref["average_travel_time"], rel_tol=rel_tol), \
            f"ATT mismatch: dt={ref_dt}->{ref['average_travel_time']:.1f}, dt={dt}->{r['average_travel_time']:.1f}"
        assert equal_tolerance(r["trip_completed"], ref["trip_completed"], rel_tol=rel_tol), \
            f"trips mismatch: dt={ref_dt}->{ref['trip_completed']:.1f}, dt={dt}->{r['trip_completed']:.1f}"


# ================================================================
# Auto computation tests
# ================================================================

class TestDeltatAuto:
    def test_auto_single_link(self):
        W = World(tmax=1000, print_mode=0)
        W.addNode("o", 0, 0)
        W.addNode("d", 1, 1)
        W.addLink("l", "o", "d", length=1000, free_flow_speed=20)
        W.adddemand("o", "d", 0, 500, 0.5)
        W.exec_simulation()
        assert W.DELTAT == 1000 / 20

    def test_auto_two_links_different_speed(self):
        W = World(tmax=1000, print_mode=0)
        W.addNode("o", 0, 0)
        W.addNode("m", 1, 1)
        W.addNode("d", 2, 2)
        W.addLink("l1", "o", "m", length=1000, free_flow_speed=20)  # d/u=50
        W.addLink("l2", "m", "d", length=1000, free_flow_speed=10)  # d/u=100
        W.adddemand("o", "d", 0, 500, 0.5)
        W.exec_simulation()
        assert W.DELTAT == 50

    def test_auto_short_link(self):
        W = World(tmax=1000, print_mode=0)
        W.addNode("o", 0, 0)
        W.addNode("m", 1, 1)
        W.addNode("d", 2, 2)
        W.addLink("l1", "o", "m", length=100, free_flow_speed=20)   # d/u=5
        W.addLink("l2", "m", "d", length=1000, free_flow_speed=20)  # d/u=50
        W.adddemand("o", "d", 0, 500, 0.5)
        W.exec_simulation()
        assert W.DELTAT == 5

    def test_manual_deltat_respected(self):
        W = World(deltat=10, tmax=1000, print_mode=0)
        W.addNode("o", 0, 0)
        W.addNode("d", 1, 1)
        W.addLink("l", "o", "d", length=1000, free_flow_speed=20)
        W.adddemand("o", "d", 0, 500, 0.5)
        W.exec_simulation()
        assert W.DELTAT == 10

    def test_manual_deltat_too_large_raises(self):
        W = World(deltat=100, tmax=1000, print_mode=0)
        W.addNode("o", 0, 0)
        W.addNode("d", 1, 1)
        W.addLink("l", "o", "d", length=1000, free_flow_speed=20)
        W.adddemand("o", "d", 0, 500, 0.5)
        with pytest.raises(ValueError, match="DELTAT.*exceeds"):
            W.exec_simulation()

    def test_auto_freeflow_theory(self):
        """Auto deltat: 1-link free flow matches theory (tt=50, TTT=12500)."""
        W = World(tmax=2000, print_mode=0)
        W.addNode("o", 0, 0)
        W.addNode("d", 1, 1)
        W.addLink("l", "o", "d", length=1000, free_flow_speed=20, jam_density=0.2)
        W.adddemand("o", "d", 0, 500, 0.5)
        W.exec_simulation()
        W.analyzer.basic_analysis()
        assert equal_tolerance(W.analyzer.trip_all, 250)
        assert equal_tolerance(W.analyzer.trip_completed, 250)
        assert equal_tolerance(W.analyzer.total_travel_time, 12500)
        assert equal_tolerance(W.analyzer.average_travel_time, 50)
        assert equal_tolerance(W.analyzer.average_delay, 0)

    def test_auto_bottleneck_theory(self):
        """Auto deltat: 2-link bottleneck matches theory."""
        W = World(tmax=2000, print_mode=0)
        W.addNode("o", 0, 0)
        W.addNode("m", 1, 1)
        W.addNode("d", 2, 2)
        W.addLink("l1", "o", "m", length=1000, free_flow_speed=20, jam_density=0.2)
        W.addLink("l2", "m", "d", length=1000, free_flow_speed=10, jam_density=0.2)
        W.adddemand("o", "d", 0, 500, 0.8)
        W.adddemand("o", "d", 500, 1500, 0.4)
        W.exec_simulation()
        W.analyzer.basic_analysis()
        assert equal_tolerance(W.analyzer.trip_all, 800)
        assert equal_tolerance(W.analyzer.trip_completed, 800)
        assert equal_tolerance(W.analyzer.average_travel_time, 182.5)

    def test_auto_merge_theory(self):
        """Auto deltat: merge with priority 1:2 matches theory."""
        W = World(tmax=1200, print_mode=0)
        W.addNode("o1", 0, 0)
        W.addNode("o2", 0, 2)
        W.addNode("m", 1, 1)
        W.addNode("d", 2, 1)
        link1 = W.addLink("l1", "o1", "m", length=1000, free_flow_speed=20,
                           jam_density=0.2, merge_priority=1)
        link2 = W.addLink("l2", "o2", "m", length=1000, free_flow_speed=20,
                           jam_density=0.2, merge_priority=2)
        W.addLink("l3", "m", "d", length=1000, free_flow_speed=20, jam_density=0.2)
        W.adddemand("o1", "d", 0, 1000, 0.8)
        W.adddemand("o2", "d", 0, 1000, 0.8)
        W.exec_simulation()
        # Outflow rate at steady state should follow priority ratio 1:2
        dt = W.DELTAT
        i1, i2 = int(200/dt), int(800/dt)
        q1 = (link1.cum_departure[i2] - link1.cum_departure[i1]) / ((i2-i1)*dt)
        q2 = (link2.cum_departure[i2] - link2.cum_departure[i1]) / ((i2-i1)*dt)
        assert equal_tolerance(q2/q1, 2.0)

    def test_auto_diverge_theory(self):
        """Auto deltat: diverge split matches demand ratio."""
        W = World(tmax=1200, print_mode=0, auto_diverge=True)
        W.addNode("o", 0, 0)
        W.addNode("m", 1, 1)
        W.addNode("d1", 2, 0)
        W.addNode("d2", 2, 2)
        W.addLink("l1", "o", "m", length=1000, free_flow_speed=20, jam_density=0.2)
        link2 = W.addLink("l2", "m", "d1", length=1000, free_flow_speed=20, jam_density=0.2)
        link3 = W.addLink("l3", "m", "d2", length=1000, free_flow_speed=20, jam_density=0.2)
        W.adddemand("o", "d1", 0, 1000, 0.2)
        W.adddemand("o", "d2", 0, 1000, 0.4)
        W.exec_simulation()
        df = W.analyzer.link_to_pandas()
        vol2 = df[df["link"]=="l2"]["traffic_volume"].values[0]
        vol3 = df[df["link"]=="l3"]["traffic_volume"].values[0]
        assert equal_tolerance(vol3/vol2, 2.0)


# ================================================================
# Sensitivity tests: straight road
# ================================================================

class TestDeltatStraightRoad:
    def test_1link_freeflow(self):
        """Single link free flow: deltat=1,5,10,25,50."""
        def factory(dt):
            W = World(deltat=dt, tmax=2000, print_mode=0)
            W.addNode("o", 0, 0)
            W.addNode("d", 1, 1)
            W.addLink("l", "o", "d", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("o", "d", 0, 500, 0.5)
            return W
        _compare(factory, [None, 1, 5, 10, 25, 50])

    def test_1link_maxflow(self):
        """Single link overcapacity: deltat=1,5,50."""
        def factory(dt):
            W = World(deltat=dt, tmax=2000, print_mode=0)
            W.addNode("o", 0, 0)
            W.addNode("d", 1, 1)
            W.addLink("l", "o", "d", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("o", "d", 0, 2000, 2)
            return W
        _compare(factory, [None, 1, 5, 50])

    def test_2link_bottleneck(self):
        """2-link bottleneck: deltat=1,5,10,25,50."""
        def factory(dt):
            W = World(deltat=dt, tmax=2000, print_mode=0)
            W.addNode("o", 0, 0)
            W.addNode("m", 1, 1)
            W.addNode("d", 2, 2)
            W.addLink("l1", "o", "m", length=1000, free_flow_speed=20, jam_density=0.2)
            W.addLink("l2", "m", "d", length=1000, free_flow_speed=10, jam_density=0.2)
            W.adddemand("o", "d", 0, 500, 0.8)
            W.adddemand("o", "d", 500, 1500, 0.4)
            return W
        _compare(factory, [None, 1, 5, 10, 25, 50])

    def test_bottleneck_capacity_out(self):
        """capacity_out bottleneck: deltat=5,25,50."""
        def factory(dt):
            W = World(deltat=dt, tmax=2000, print_mode=0)
            W.addNode("o", 0, 0)
            W.addNode("m", 1, 1)
            W.addNode("d", 2, 2)
            W.addLink("l1", "o", "m", length=1000, free_flow_speed=20, jam_density=0.2,
                       capacity_out=0.66666)
            W.addLink("l2", "m", "d", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("o", "d", 0, 500, 0.8)
            W.adddemand("o", "d", 500, 1500, 0.4)
            return W
        _compare(factory, [None, 5, 25, 50])

    def test_bottleneck_node_capacity(self):
        """Node flow_capacity bottleneck: deltat=5,25,50."""
        def factory(dt):
            W = World(deltat=dt, tmax=2000, print_mode=0)
            W.addNode("o", 0, 0)
            W.addNode("m", 1, 1, flow_capacity=0.666)
            W.addNode("d", 2, 2)
            W.addLink("l1", "o", "m", length=1000, free_flow_speed=20, jam_density=0.2)
            W.addLink("l2", "m", "d", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("o", "d", 0, 500, 0.8)
            W.adddemand("o", "d", 500, 1500, 0.4)
            return W
        _compare(factory, [None, 5, 25, 50])

    def test_3link_spillback(self):
        """3-link queue spillback: deltat=5,25,50."""
        def factory(dt):
            W = World(deltat=dt, tmax=2000, print_mode=0)
            W.addNode("o", 0, 0)
            W.addNode("m1", 1, 1)
            W.addNode("m2", 2, 2)
            W.addNode("d", 3, 3)
            W.addLink("l1", "o", "m1", length=1000, free_flow_speed=20, jam_density=0.2)
            W.addLink("l2", "m1", "m2", length=1000, free_flow_speed=20, jam_density=0.2,
                       capacity_out=0.4)
            W.addLink("l3", "m2", "d", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("o", "d", 0, 400, 0.8)
            W.adddemand("o", "d", 400, 800, 0.4)
            W.adddemand("o", "d", 800, 2000, 0.1)
            return W
        _compare(factory, [None, 5, 25, 50])

    def test_leave_at_middle(self):
        """Destination at intermediate node: deltat=5,25,50."""
        def factory(dt):
            W = World(deltat=dt, tmax=2000, print_mode=0, auto_diverge=True)
            W.addNode("o", 0, 0)
            W.addNode("m", 1, 1)
            W.addNode("d", 2, 2)
            W.addLink("l1", "o", "m", length=1000, free_flow_speed=20, jam_density=0.2)
            W.addLink("l2", "m", "d", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("o", "m", 0, 500, 0.666)
            return W
        _compare(factory, [None, 5, 25, 50])


# ================================================================
# Sensitivity tests: merge nodes
# ================================================================

class TestDeltatMerge:
    def test_merge_fair_nocongestion(self):
        """Equal priority, no congestion: deltat=5,25,50."""
        def factory(dt):
            W = World(deltat=dt, tmax=1200, print_mode=0)
            W.addNode("o1", 0, 0)
            W.addNode("o2", 0, 2)
            W.addNode("m", 1, 1)
            W.addNode("d", 2, 1)
            W.addLink("l1", "o1", "m", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
            W.addLink("l2", "o2", "m", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
            W.addLink("l3", "m", "d", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("o1", "d", 0, 1000, 0.3)
            W.adddemand("o2", "d", 0, 1000, 0.3)
            return W
        _compare(factory, [None, 5, 25, 50])

    def test_merge_fair_congestion(self):
        """Equal priority, congestion: deltat=5,25,50."""
        def factory(dt):
            W = World(deltat=dt, tmax=1200, print_mode=0)
            W.addNode("o1", 0, 0)
            W.addNode("o2", 0, 2)
            W.addNode("m", 1, 1)
            W.addNode("d", 2, 1)
            W.addLink("l1", "o1", "m", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
            W.addLink("l2", "o2", "m", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
            W.addLink("l3", "m", "d", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("o1", "d", 0, 1000, 0.5)
            W.adddemand("o2", "d", 0, 1000, 0.5)
            return W
        _compare(factory, [None, 5, 25, 50])

    def test_merge_unfair(self):
        """Priority 1:1.5, congestion: deltat=5,25,50."""
        def factory(dt):
            W = World(deltat=dt, tmax=1200, print_mode=0)
            W.addNode("o1", 0, 0)
            W.addNode("o2", 0, 2)
            W.addNode("m", 1, 1)
            W.addNode("d", 2, 1)
            W.addLink("l1", "o1", "m", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
            W.addLink("l2", "o2", "m", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1.5)
            W.addLink("l3", "m", "d", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("o1", "d", 0, 1000, 0.5)
            W.adddemand("o2", "d", 0, 1000, 0.5)
            return W
        _compare(factory, [None, 5, 25, 50])

    def test_merge_veryunfair(self):
        """Priority 1:100: deltat=5,25,50."""
        def factory(dt):
            W = World(deltat=dt, tmax=1200, print_mode=0)
            W.addNode("o1", 0, 0)
            W.addNode("o2", 0, 2)
            W.addNode("m", 1, 1)
            W.addNode("d", 2, 1)
            W.addLink("l1", "o1", "m", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
            W.addLink("l2", "o2", "m", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=100)
            W.addLink("l3", "m", "d", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("o1", "d", 0, 1000, 0.5)
            W.adddemand("o2", "d", 0, 1000, 0.5)
            return W
        _compare(factory, [None, 5, 25, 50])

    def test_merge_priority_1to2(self):
        """Priority 1:2, oversaturated: deltat=5,25,50."""
        def factory(dt):
            W = World(deltat=dt, tmax=1200, print_mode=0)
            W.addNode("o1", 0, 0)
            W.addNode("o2", 0, 2)
            W.addNode("m", 1, 1)
            W.addNode("d", 2, 1)
            W.addLink("l1", "o1", "m", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
            W.addLink("l2", "o2", "m", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=2)
            W.addLink("l3", "m", "d", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("o1", "d", 0, 1000, 0.8)
            W.adddemand("o2", "d", 0, 1000, 0.8)
            return W
        _compare(factory, [None, 5, 25, 50])

    def test_merge_one_free_one_congested(self):
        """One free, one congested: deltat=5,25,50."""
        def factory(dt):
            W = World(deltat=dt, tmax=1200, print_mode=0)
            W.addNode("o1", 0, 0)
            W.addNode("o2", 0, 2)
            W.addNode("m", 1, 1)
            W.addNode("d", 2, 1)
            W.addLink("l1", "o1", "m", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
            W.addLink("l2", "o2", "m", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
            W.addLink("l3", "m", "d", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("o1", "d", 0, 1000, 0.3)
            W.adddemand("o2", "d", 0, 1000, 0.7)
            return W
        _compare(factory, [None, 5, 25, 50])


# ================================================================
# Sensitivity tests: diverge nodes
# ================================================================

class TestDeltatDiverge:
    def test_diverge_equal_split(self):
        """Equal split, no congestion: deltat=5,25,50."""
        def factory(dt):
            W = World(deltat=dt, tmax=1200, print_mode=0, auto_diverge=True)
            W.addNode("o", 0, 0)
            W.addNode("m", 1, 1)
            W.addNode("d1", 2, 0)
            W.addNode("d2", 2, 2)
            W.addLink("l1", "o", "m", length=1000, free_flow_speed=20, jam_density=0.2)
            W.addLink("l2", "m", "d1", length=1000, free_flow_speed=20, jam_density=0.2)
            W.addLink("l3", "m", "d2", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("o", "d1", 0, 1000, 0.3)
            W.adddemand("o", "d2", 0, 1000, 0.3)
            return W
        _compare(factory, [None, 5, 25, 50])

    def test_diverge_unequal_split(self):
        """Unequal split: deltat=5,25,50."""
        def factory(dt):
            W = World(deltat=dt, tmax=1200, print_mode=0, auto_diverge=True)
            W.addNode("o", 0, 0)
            W.addNode("m", 1, 1)
            W.addNode("d1", 2, 0)
            W.addNode("d2", 2, 2)
            W.addLink("l1", "o", "m", length=1000, free_flow_speed=20, jam_density=0.2)
            W.addLink("l2", "m", "d1", length=1000, free_flow_speed=20, jam_density=0.2)
            W.addLink("l3", "m", "d2", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("o", "d1", 0, 1000, 0.2)
            W.adddemand("o", "d2", 0, 1000, 0.4)
            return W
        _compare(factory, [None, 5, 25, 50])

    def test_diverge_capacity_in_congestion(self):
        """capacity_in causing congestion: deltat=5,25,50."""
        def factory(dt):
            W = World(deltat=dt, tmax=1200, print_mode=0, auto_diverge=True)
            W.addNode("o", 0, 0)
            W.addNode("m", 1, 1)
            W.addNode("d1", 2, 0)
            W.addNode("d2", 2, 2)
            W.addLink("l1", "o", "m", length=1000, free_flow_speed=20, jam_density=0.2)
            W.addLink("l2", "m", "d1", length=1000, free_flow_speed=20, jam_density=0.2, capacity_in=0.2)
            W.addLink("l3", "m", "d2", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("o", "d1", 0, 1000, 0.4)
            W.adddemand("o", "d2", 0, 1000, 0.3)
            return W
        _compare(factory, [None, 5, 25, 50])

    def test_diverge_capacity_in_nocongestion(self):
        """capacity_in within limit: deltat=5,25,50."""
        def factory(dt):
            W = World(deltat=dt, tmax=1200, print_mode=0, auto_diverge=True)
            W.addNode("o", 0, 0)
            W.addNode("m", 1, 1)
            W.addNode("d1", 2, 0)
            W.addNode("d2", 2, 2)
            W.addLink("l1", "o", "m", length=1000, free_flow_speed=20, jam_density=0.2)
            W.addLink("l2", "m", "d1", length=1000, free_flow_speed=20, jam_density=0.2, capacity_in=0.2)
            W.addLink("l3", "m", "d2", length=1000, free_flow_speed=20, jam_density=0.2)
            W.adddemand("o", "d1", 0, 1000, 0.2)
            W.adddemand("o", "d2", 0, 1000, 0.3)
            return W
        _compare(factory, [None, 5, 25, 50])
