"""
Generate reference results for the parallel-routes scenario using the raw JAX core only.

The outputs gate the facade (unsim.api): the facade must reproduce these values.
Everything here uses world_to_jax / simulate_duo / manual Adam directly, without importing unsim.api.

Usage:
  python tests/generate_parallel_routes_reference.py
Output:
  tests/data/parallel_routes_reference.npz
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax
import jax.numpy as jnp

from unsim.unsim_diff import world_to_jax, simulate_duo, total_travel_time, trip_completed
from scenario_parallel_routes import create_world

STEPS = 40
LR = 1.0
REG_LAMBDA = 1e-3
B1, B2, EPS = 0.9, 0.999, 1e-8


def main():
    W, _ = create_world()
    params, config = world_to_jax(W)

    # --- Baseline forward run ---
    state0 = simulate_duo(params, config)
    ttt0 = float(total_travel_time(state0, config))
    trips0 = float(trip_completed(state0, config))
    print(f"baseline TTT = {ttt0:.1f} s, trips = {trips0:.1f}")

    # --- Toll optimization on the bottleneck link (highway23) ---
    link_names = [l.name for l in W.LINKS]
    toll_link_idx = link_names.index("highway23")
    ids = jnp.array([toll_link_idx], dtype=jnp.int32)
    n_toll = int(config.n_toll_steps)

    def loss_fn(theta):
        toll = params.toll.at[ids].set(theta.reshape(1, n_toll))
        p = params._replace(toll=toll)
        state = simulate_duo(p, config)
        return total_travel_time(state, config) + REG_LAMBDA * jnp.sum(theta ** 2)

    vgrad = jax.jit(jax.value_and_grad(loss_fn))

    theta = jnp.zeros(n_toll, dtype=jnp.float32)
    grad0 = np.asarray(vgrad(theta)[1])

    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    history = np.zeros(STEPS)
    for step in range(STEPS):
        value, g = vgrad(theta)
        history[step] = float(value)
        m = B1 * m + (1.0 - B1) * g
        v = B2 * v + (1.0 - B2) * g ** 2
        m_hat = m / (1.0 - B1 ** (step + 1))
        v_hat = v / (1.0 - B2 ** (step + 1))
        theta = theta - LR * m_hat / (jnp.sqrt(v_hat) + EPS)
        theta = jnp.maximum(theta, 0.0)
        if step % 10 == 0:
            print(f"  step {step}: loss={history[step]:.1f}")

    theta_opt = np.asarray(theta)
    ttt_opt = float(total_travel_time(simulate_duo(
        params._replace(toll=params.toll.at[ids].set(theta.reshape(1, n_toll))), config), config))
    print(f"optimized TTT = {ttt_opt:.1f} s (baseline {ttt0:.1f} s)")

    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, "parallel_routes_reference.npz"),
        ttt0=ttt0,
        trips0=trips0,
        cum_arrival=np.asarray(state0.cum_arrival),
        cum_departure=np.asarray(state0.cum_departure),
        grad0=grad0,
        theta_opt=theta_opt,
        loss_history=history,
        ttt_opt=ttt_opt,
        steps=STEPS,
        lr=LR,
        reg_lambda=REG_LAMBDA,
        toll_link_idx=toll_link_idx,
        n_toll_steps=n_toll,
    )
    print("saved tests/data/parallel_routes_reference.npz")


if __name__ == "__main__":
    main()
