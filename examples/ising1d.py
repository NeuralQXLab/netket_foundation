"""Train a foundational NQS on 1D Ising and compare against exact results."""

import os

os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import netket as nk
import netket_foundation as nkf
import numpy as np
import optax

from tqdm import tqdm

from netket_foundation._src.model.vit import ViTFNQS

# Setup: model, space, sampler, and foundational state.
key = jax.random.key(1)
hi = nk.hilbert.Spin(0.5, 10)
ps = nkf.ParameterSpace(N=1, min=0.8, max=1.2)

ma = ViTFNQS(
    num_layers=2,
    d_model=12,
    heads=4,
    L_eff=hi.size // 2,
    n_coups=ps.size,
    b=2,
    complex=False,
    disorder=False,
    transl_invariant=True,
    two_dimensional=False,
)

sa = nk.sampler.MetropolisLocal(hi, n_chains=5016)
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=8, seed=1)
vs.parameter_array = jnp.linspace(0.8, 1.2, vs.n_replicas).reshape(-1, 1)


def create_operator(params):
    assert params.shape == (1,)
    h = params[0]

    ha_x = sum(nkf.operator.sigmax(hi, i) for i in range(hi.size))
    ha_zz = sum(
        nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size)
        for i in range(hi.size)
    )
    return -h * ha_x - ha_zz


# Observables used during optimization and evaluation.
mz = sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size)) * (1 / float(hi.size))
mz2 = mz @ mz
mz2_mat = mz2.to_sparse()

ha_p = nkf.operator.ParametrizedOperator(hi, ps, create_operator)
mz_p = nkf.operator.ParametrizedOperator(
    hi,
    ps,
    lambda _: sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size))
    * (1 / float(hi.size)),
)


# Train foundational state over parameter grid.
optimizer = optax.sgd(0.005)
driver = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=1e-4)

log = nk.logging.JsonLog("2")
driver.run(
    1000,
    out=(log, nk.logging.SaveVariationalState("2", 10)),
    obs={"ham": ha_p, "mz": mz_p},
    step_size=10,
)


# Exact values for comparison across the same h-range.
exact = {
    "h": np.linspace(0.8, 1.2, 40),
    "Energy": [],
    "Mz2": [],
}
for pars in tqdm(exact["h"]):
    ha = create_operator(pars.reshape(-1))
    e0, psi0 = nk.exact.lanczos_ed(ha, k=1, compute_eigenvectors=True)
    psi0 = psi0.reshape(-1)
    exact["Energy"].append(e0.item())
    exact["Mz2"].append((psi0.T.conj() @ (mz2_mat @ psi0)).item())

exact = {
    "h": np.asarray(exact["h"]),
    "Energy": np.asarray(exact["Energy"]),
    "Mz2": np.asarray(exact["Mz2"]),
}


# VMC estimates at the trained parameter points.
vmc_vals = {"Energy": [], "Mz2": []}
for pars in tqdm(vs.parameter_array):
    ha = create_operator(pars)
    vstate = vs.get_state(pars)
    vstate.reset()
    vstate.sample()
    vstate.sample()
    vmc_vals["Energy"].append(vstate.expect(ha).Mean)
    vmc_vals["Mz2"].append(vstate.expect(mz2).Mean)

vmc_vals = {
    "h": np.asarray(vs.parameter_array),
    "Energy": np.asarray(vmc_vals["Energy"]),
    "Mz2": np.asarray(vmc_vals["Mz2"]),
}


plt.plot(exact["h"], exact["Energy"], label="Exact")
plt.plot(vmc_vals["h"], vmc_vals["Energy"], "x", label="VMC")
plt.xlabel("h")
plt.ylabel("Energy")
plt.legend()
plt.savefig("energy.pdf")
plt.clf()

plt.plot(exact["h"], exact["Mz2"], label="Exact")
plt.plot(vmc_vals["h"], vmc_vals["Mz2"], "x", label="VMC")
plt.xlabel("h")
plt.ylabel("Mz2")
plt.legend()
plt.savefig("mz2.pdf")
plt.clf()


# Convergence curves: relative error vs exact ED energy.
conv_data = []
for i, pars in tqdm(enumerate(vs.parameter_array)):
    ha = create_operator(pars)
    ed = nk.exact.lanczos_ed(ha, k=1, compute_eigenvectors=False).item()
    e0 = log.data["ham"][i].Mean
    conv_data.append(
        {
            "h": float(pars.item()),
            "e0": e0,
            "iters": log.data["ham"][i].iters,
            "err_val": e0 - ed,
        }
    )

for data in conv_data:
    plt.plot(
        data["iters"],
        np.abs(data["err_val"] / data["e0"]),
        label=f"h = {data['h']:.2f}",
    )

plt.xlabel("Iteration")
plt.ylabel("Rel Error")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("convergence.pdf")
plt.clf()
