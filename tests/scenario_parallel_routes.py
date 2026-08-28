"""
Small parallel-routes test scenario (highway vs arterial), adapted from UXsim example 26.

Two parallel routes connect the OD pair: a highway (fast, low capacity, with a bottleneck) and an arterial road (slow, high capacity).
Travelers from origin 4 choose between the onramp to the highway and the arterial at the diverge node 5, so duo_logit route choice is active.
Total demand (1.5 veh/s) exceeds the combined downstream capacity (about 1.4 veh/s), so congestion arises and tolls have an effect.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unsim import World


def create_world(route_choice="duo_logit", deltat=10, tmax=6000, logit_temperature=60.0):
    """Build the parallel-routes world.

    Parameters
    ----------
    route_choice : str, optional
        Route choice model. Default "duo_logit".
    deltat : float, optional
        Timestep (s). Default 10.
    tmax : float, optional
        Simulation duration (s). Default 6000.
    logit_temperature : float, optional
        Logit temperature (s). Default 60.

    Returns
    -------
    W : World
    demands : tuple of Demand
        (highway-origin demand, arterial-origin demand).
    """
    W = World(name="parallel_routes", deltat=deltat, tmax=tmax, print_mode=0,
              route_choice=route_choice, logit_temperature=logit_temperature)

    W.addNode("1", 0, 1)
    W.addNode("2", 1, 1)
    W.addNode("3", 5, 1)
    W.addNode("4", 0, 0)
    W.addNode("5", 1, 0)
    W.addNode("6", 5, 0)
    W.addNode("7", 6, 0.5)

    W.addLink("highway12", "1", "2", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
    W.addLink("highway23", "2", "3", length=3000, free_flow_speed=20, jam_density=0.2, merge_priority=1, capacity_out=0.6)
    W.addLink("highway37", "3", "7", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
    W.addLink("onramp", "5", "2", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=0.5)
    W.addLink("arterial45", "4", "5", length=1000, free_flow_speed=10, jam_density=0.4, merge_priority=0.5)
    W.addLink("arterial56", "5", "6", length=3000, free_flow_speed=10, jam_density=0.4, merge_priority=0.5)
    W.addLink("arterial67", "6", "7", length=1000, free_flow_speed=10, jam_density=0.4, merge_priority=0.5)

    d1 = W.adddemand("1", "7", 0, 3000, flow=0.3)
    d2 = W.adddemand("4", "7", 0, 3000, flow=1.2)

    return W, (d1, d2)
