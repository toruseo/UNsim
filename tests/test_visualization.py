"""
Tests that visualization methods run without error and produce valid output.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from unsim import *


@pytest.fixture
def bottleneck_world():
    """2-link bottleneck scenario."""
    W = World(name="test_vis", deltat=5, tmax=1000, print_mode=0)
    W.addNode("orig", 0, 0)
    W.addNode("mid", 1, 1)
    W.addNode("dest", 2, 2)
    link1 = W.addLink("link1", "orig", "mid", length=1000, free_flow_speed=20, jam_density=0.2)
    link2 = W.addLink("link2", "mid", "dest", length=1000, free_flow_speed=10, jam_density=0.2)
    W.adddemand("orig", "dest", 0, 500, 0.8)
    W.exec_simulation()
    return W, link1, link2


@pytest.fixture
def merge_world():
    """2-to-1 merge scenario."""
    W = World(name="test_merge_vis", deltat=5, tmax=800, print_mode=0)
    W.addNode("orig1", 0, 0)
    W.addNode("orig2", 0, 2)
    W.addNode("merge", 1, 1)
    W.addNode("dest", 2, 1)
    W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
    W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=2)
    W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
    W.adddemand("orig1", "dest", 0, 600, 0.5)
    W.adddemand("orig2", "dest", 0, 600, 0.5)
    W.exec_simulation()
    return W


class TestTimeSpaceDiagram:
    def test_density(self, bottleneck_world):
        W, link1, link2 = bottleneck_world
        fig = W.analyzer.time_space_diagram(link1, mode="density")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_N(self, bottleneck_world):
        W, link1, link2 = bottleneck_world
        fig = W.analyzer.time_space_diagram(link1, mode="N")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_flow(self, bottleneck_world):
        W, link1, link2 = bottleneck_world
        fig = W.analyzer.time_space_diagram(link1, mode="flow")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_speed(self, bottleneck_world):
        W, link1, link2 = bottleneck_world
        fig = W.analyzer.time_space_diagram(link1, mode="speed")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_chained_links(self, bottleneck_world):
        W, link1, link2 = bottleneck_world
        fig = W.analyzer.time_space_diagram([link1, link2], mode="density")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_links_default(self, bottleneck_world):
        W, link1, link2 = bottleneck_world
        fig = W.analyzer.time_space_diagram(mode="N")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestNetwork:
    def test_default(self, merge_world):
        fig = merge_world.analyzer.network()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_specific_time(self, merge_world):
        fig = merge_world.analyzer.network(t=300)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_right_handed(self, merge_world):
        fig = merge_world.analyzer.network(t=300, left_handed=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestNetworkAverage:
    def test_default(self, merge_world):
        fig = merge_world.analyzer.network_average()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_params(self, merge_world):
        fig = merge_world.analyzer.network_average(minwidth=1, maxwidth=8)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestNetworkAnim:
    def test_gif_creation(self, bottleneck_world, tmp_path):
        W, _, _ = bottleneck_world
        out = str(tmp_path / "test.gif")
        result = W.analyzer.network_anim(timestep_skip=40, dpi=50, file_name=out)
        assert os.path.exists(result)
        img = Image.open(result)
        assert img.format == "GIF"
        assert img.n_frames > 1
        plt.close("all")
