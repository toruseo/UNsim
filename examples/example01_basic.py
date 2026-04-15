import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unsim import *

W = World(name="", deltat=5, tmax=1200, print_mode=1, save_mode=1)

W.addNode("orig1", 0, 0)
W.addNode("orig2", 0, 2)
W.addNode("merge", 1, 1)
W.addNode("dest", 2, 1)
link1 = W.addLink("link1", "orig1", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
link2 = W.addLink("link2", "orig2", "merge", length=1000, free_flow_speed=20, jam_density=0.2, merge_priority=1)
link3 = W.addLink("link3", "merge", "dest", length=1000, free_flow_speed=20, jam_density=0.2)
W.adddemand("orig1", "dest", 0, 1000, 0.45)
W.adddemand("orig2", "dest", 400, 1000, 0.6)

W.exec_simulation()

W.analyzer.print_simple_stats()

W.analyzer.time_space_diagram(mode="k_norm", links="link1", cmap="jet", vmax=1)
W.analyzer.time_space_diagram(mode="k_norm", links="link2", cmap="jet", vmax=1)
W.analyzer.time_space_diagram(mode="k_norm", links="link3", cmap="jet", vmax=1)

fname = W.analyzer.network_anim()
#display_image_in_notebook(fname)