"""
Detailed comparison of Python DUO vs JAX DUO.
Outputs tables + scatter plots. Run: python tests/test_duo_comparison.py
"""

import time, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unsim import *

try:
    import jax
    import jax.numpy as jnp
    from unsim.unsim_diff import world_to_jax, simulate_duo, total_travel_time
    HAS_JAX = True
except (ImportError, RuntimeError):
    HAS_JAX = False
    print("JAX not available")
    if __name__ != "__main__":
        import pytest; pytest.skip("JAX not available", allow_module_level=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_kuwahara():
    W = World(tmax=5*3600, print_mode=0, route_choice="duo")
    W.addNode("1",-2,0); W.addNode("2",0,-1); W.addNode("3",2,-1)
    W.addNode("4",-1,1); W.addNode("5",0.5,1); W.addNode("6",1.5,1)
    for s,e,L,lw,lwp,km,fm in [("1","4",2,0.05,0.1,450,6000),("4","5",16,0.2,0.8,375,6000),
        ("5","6",8,0.1,0.4,250,4000),("6","3",2,0.05,0.1,225,3000),
        ("1","2",16,0.4,0.8,450,6000),("2","3",12,0.3,0.6,450,6000),
        ("5","2",2,0.05,0.1,450,6000)]:
        W.addLink(f"{s}_{e}",s,e,length=L*1000,free_flow_speed=(L/lw)*1000/3600,
                  backward_wave_speed=(L/lwp)*1000/3600,capacity=fm/3600)
    W.adddemand("1","2",0,3600,1000/3600); W.adddemand("1","2",3600,10800,2000/3600)
    W.adddemand("1","3",0,3600,2000/3600); W.adddemand("1","3",3600,10800,4000/3600)
    return W


def build_grid():
    GRID=6; tmax=21600
    W=World(tmax=tmax,print_mode=0,route_choice="duo")
    for i in range(GRID):
        for j in range(GRID):
            W.addNode(f"n{i}{j}",j*2,-i*2)
    for i in range(GRID):
        for j in range(GRID-1):
            u_r = 18 + 0.5*j + 0.3*i; u_l = 17 + 0.4*(GRID-j) + 0.2*i
            W.addLink(f"h{i}{j}r",f"n{i}{j}",f"n{i}{j+1}",length=2000,free_flow_speed=u_r,backward_wave_speed=5,capacity=0.6)
            W.addLink(f"h{i}{j}l",f"n{i}{j+1}",f"n{i}{j}",length=2000,free_flow_speed=u_l,backward_wave_speed=5,capacity=0.6)
    for i in range(GRID-1):
        for j in range(GRID):
            u_d = 19 + 0.3*i + 0.4*j; u_u = 18 + 0.2*(GRID-i) + 0.5*j
            W.addLink(f"v{i}{j}d",f"n{i}{j}",f"n{i+1}{j}",length=2000,free_flow_speed=u_d,backward_wave_speed=5,capacity=0.6)
            W.addLink(f"v{i}{j}u",f"n{i+1}{j}",f"n{i}{j}",length=2000,free_flow_speed=u_u,backward_wave_speed=5,capacity=0.6)
    od_data=[("o1","n00","d1","n55",0,5400,0.55),("o2","n05","d2","n50",0,5400,0.55),
             ("o3","n02","d3","n52",600,5400,0.45),("o4","n20","d4","n25",600,5400,0.45),
             ("o5","n50","d5","n05",0,5400,0.45),("o6","n55","d6","n00",0,5400,0.45)]
    for on,og,dn,dg,ts,te,fl in od_data:
        oi,oj=int(og[1]),int(og[2]); di,dj=int(dg[1]),int(dg[2])
        if on not in [n.name for n in W.NODES]:
            W.addNode(on,oj*2-1,-oi*2+1); W.addLink(f"l_{on}",on,og,length=2000,free_flow_speed=20,backward_wave_speed=5,capacity=1.0)
        if dn not in [n.name for n in W.NODES]:
            W.addNode(dn,dj*2+1,-di*2-1); W.addLink(f"l_{dg}_{dn}",dg,dn,length=2000,free_flow_speed=20,backward_wave_speed=5,capacity=1.0)
        W.adddemand(on,dn,ts,te,fl)
    return W


def compare(name, build_fn):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Python
    W = build_fn()
    t0 = time.time(); W.exec_simulation(); t_py = time.time() - t0
    W.analyzer.basic_analysis()
    dt = W.DELTAT

    py_vols = np.array([l.cum_departure[-1] for l in W.LINKS])
    py_tts = []
    for l in W.LINKS:
        vt = sum(max(l.cum_arrival[t]-l.cum_departure[t], 0)*dt for t in range(W.TSIZE))
        vol = max(l.cum_departure[-1], 1e-10)
        py_tts.append(vt / vol)
    py_tts = np.array(py_tts)
    py_ttt = W.analyzer.total_travel_time

    # JAX
    W2 = build_fn(); W2.exec_simulation()
    params, config = world_to_jax(W2)
    _ = simulate_duo(params, config)  # warmup
    t0 = time.time()
    state = simulate_duo(params, config)
    jax.block_until_ready(state.cum_arrival)
    t_jax = time.time() - t0

    jax_vols = np.array([float(state.cum_departure[i, -1]) for i in range(len(W.LINKS))])
    jax_tts = []
    for i in range(len(W.LINKS)):
        vt = float(jnp.sum(jnp.maximum(
            state.cum_arrival[i, :config.tsize] - state.cum_departure[i, :config.tsize], 0))) * dt
        vol = max(float(state.cum_departure[i, -1]), 1e-10)
        jax_tts.append(vt / vol)
    jax_tts = np.array(jax_tts)
    jax_ttt = float(total_travel_time(state, config))

    # Stats
    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - ss_res / max(ss_tot, 1e-10)

    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def mape(y_true, y_pred):
        mask = y_true > 1
        if np.sum(mask) == 0: return 0
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100

    # Filter zero-volume links for TT comparison
    active = py_vols > 1
    py_tts_a = py_tts[active]
    jax_tts_a = jax_tts[active]

    print(f"\n  Python: {t_py:.3f}s | JAX: {t_jax:.3f}s | ratio: {t_jax/max(t_py,0.001):.1f}x")
    print(f"\n  --- Total Travel Time ---")
    print(f"  Python: {py_ttt:.0f}s | JAX: {jax_ttt:.0f}s | diff: {abs(jax_ttt-py_ttt)/max(py_ttt,1)*100:.1f}%")
    print(f"\n  --- Link Volume ---")
    print(f"  R2:   {r2(py_vols, jax_vols):.4f}")
    print(f"  MAE:  {mae(py_vols, jax_vols):.1f} veh")
    print(f"  MAPE: {mape(py_vols, jax_vols):.1f}%")
    print(f"\n  --- Link Travel Time (active links: {np.sum(active)}/{len(active)}) ---")
    print(f"  R2:   {r2(py_tts_a, jax_tts_a):.4f}")
    print(f"  MAE:  {mae(py_tts_a, jax_tts_a):.1f}s")
    print(f"  MAPE: {mape(py_tts_a, jax_tts_a):.1f}%")

    return {
        "name": name, "py_vols": py_vols, "jax_vols": jax_vols,
        "py_tts": py_tts_a, "jax_tts": jax_tts_a,
        "py_ttt": py_ttt, "jax_ttt": jax_ttt,
        "t_py": t_py, "t_jax": t_jax,
    }


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / max(ss_tot, 1e-10)

results = []
results.append(compare("Kuwahara (6n/7l/2d)", build_kuwahara))
results.append(compare("Grid 6x6 (48n/132l/6d)", build_grid))

# Scatter plots
fig, axes = plt.subplots(2, 2, figsize=(12, 11))

for idx, r in enumerate(results):
    # Volume scatter
    ax = axes[idx, 0]
    ax.scatter(r["py_vols"], r["jax_vols"], s=20, alpha=0.7)
    vmax = max(r["py_vols"].max(), r["jax_vols"].max()) * 1.1
    ax.plot([0, vmax], [0, vmax], "k--", lw=0.5)
    ax.set_xlabel("Python Volume (veh)"); ax.set_ylabel("JAX Volume (veh)")
    ax.set_title(f'{r["name"]}\nLink Volume (R2={r2(r["py_vols"], r["jax_vols"]):.3f})')
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    # TT scatter
    ax = axes[idx, 1]
    ax.scatter(r["py_tts"], r["jax_tts"], s=20, alpha=0.7, color="C1")
    tmax_val = max(r["py_tts"].max(), r["jax_tts"].max()) * 1.1
    ax.plot([0, tmax_val], [0, tmax_val], "k--", lw=0.5)
    ax.set_xlabel("Python Travel Time (s)"); ax.set_ylabel("JAX Travel Time (s)")
    ax.set_title(f'{r["name"]}\nLink Travel Time (R2={r2(r["py_tts"], r["jax_tts"]):.3f})')
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

plt.tight_layout()
outdir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, "duo_py_vs_jax.png")
fig.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved to {outpath}")
