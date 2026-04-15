import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unsim import *

W = World(name="", deltat=10, tmax=6000, print_mode=1, auto_diverge=True)

NN = W.addNode("NN", 0, 1)
EE = W.addNode("EE", 1, 0)
SS = W.addNode("SS", 0, -1)
WW = W.addNode("WW", -1, 0)
NO = W.addNode("NO", 0, 2)
SO = W.addNode("SO", 0, -2)
EI = W.addNode("EI", 2, 0)
WI = W.addNode("WI", -2, 0)

merge_priority_inner = 1
WNl = W.addLink("WNl", WW, NN, length=1000, merge_priority=merge_priority_inner)
NEl = W.addLink("NEl", NN, EE, length=1000, merge_priority=merge_priority_inner)
ESl = W.addLink("ESl", EE, SS, length=1000, merge_priority=merge_priority_inner)
SWl = W.addLink("SWl", SS, WW, length=1000, merge_priority=merge_priority_inner)

merge_priority_outer = 2 #0.5 to prevent gridlock
IWl = W.addLink("IWl", WI, WW, length=1000, merge_priority=merge_priority_outer)
IEl = W.addLink("IEl", EI, EE, length=1000, merge_priority=merge_priority_outer)
NOl = W.addLink("NOl", NN, NO, length=1000, merge_priority=merge_priority_outer)
SOl = W.addLink("SOl", SS, SO, length=1000, merge_priority=merge_priority_outer)

W.adddemand(WI, SO, 0, 3000, 0.6)
W.adddemand(EI, NO, 500, 3000, 0.6)

W.exec_simulation()
W.analyzer.print_simple_stats()

W.analyzer.basic_analysis()

W.analyzer.time_space_diagram(mode="N",links=[WNl, NEl, ESl, SWl], cmap="jet", vmax=1)
W.analyzer.time_space_diagram(mode="k_norm",links=[WNl, NEl, ESl, SWl], cmap="jet", vmax=1)
W.analyzer.time_space_diagram(mode="q_norm", links=[WNl, NEl, ESl, SWl], cmap="jet", vmax=1)
W.analyzer.time_space_diagram(mode="v_norm", links=[WNl, NEl, ESl, SWl], cmap="jet", vmax=1)
W.analyzer.network_average()

fname = W.analyzer.network_anim()
#display_image_in_notebook(fname)