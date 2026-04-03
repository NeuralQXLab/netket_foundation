"""Train a foundational NQS on disordered Ising instances and benchmark on test draws."""

import os

os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import netket as nk
import netket_foundation as nkf
import numpy as np
import optax
from scipy.stats import gaussian_kde
from tqdm import tqdm

from netket_foundation._src.model.vit import ViTFNQS


key = jax.random.key(1)
N = 100
h0 = 1.0
hi = nk.hilbert.Spin(0.5, 8)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=h0)


def generate_disorder_realizations(N, system_size, h0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(0.0, h0, size=(N, system_size))


ma = ViTFNQS(
    num_layers=1,
    d_model=8,
    heads=8,
    L_eff=hi.size // 2,
    n_coups=ps.size,
    b=2,
    complex=True,
    disorder=True,
    transl_invariant=False,
    two_dimensional=False,
)

sa = nk.sampler.MetropolisLocal(hi, n_chains=N * 32)

vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=N, n_samples=N * 64, seed=1)

params_list = generate_disorder_realizations(N, hi.size, h0)
print(params_list.shape)
vs.parameter_array = params_list

Mz = sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size)) * (1 / float(hi.size))


def create_operator(params):
    # params: array of shape (system_size,)
    assert params.shape == (hi.size,)

    # Transverse field term: sum_i h_i sigma^x_i
    ha_X = sum(params[i] * nkf.operator.sigmax(hi, i) for i in range(hi.size))

    # Ising interaction: sum_i sigma^z_i sigma^z_{i+1}
    ha_ZZ = sum(
        nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size)
        for i in range(hi.size)
    )

    return -ha_X - (1 / np.exp(1)) * ha_ZZ


ha_p = nkf.operator.ParametrizedOperator(hi, ps, create_operator)
mz_p = nkf.operator.ParametrizedOperator(
    hi,
    ps,
    lambda _: sum(nkf.operator.sigmaz(hi, i) for i in range(hi.size))
    * (1 / float(hi.size)),
)

learning_rate = optax.linear_schedule(
    init_value=0.03, end_value=0.005, transition_steps=300
)
optimizer = optax.sgd(learning_rate)
gs = nkf.VMC_NG(ha_p, optimizer, variational_state=vs, diag_shift=1e-4)

log = nk.logging.JsonLog("2")
gs.run(
    20,
    out=log,
    obs={"ham": ha_p, "mz": mz_p},
    step_size=10,
    callback=nk.logging.SaveVariationalState(
        "2",
        10,
    ),
)

# Convergence curves on training disorder instances.
print("Plotting convergence curves...")
conv_data = []
for i, pars in tqdm(enumerate(vs.parameter_array)):
    _ha = create_operator(pars)
    ed = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=False).item()

    err_val = log.data["ham"][i].Mean - ed
    conv_data.append(
        {
            "h": h0,
            "e0": log.data["ham"][i].Mean,
            "energy": ed,
            "iters": log.data["ham"][i].iters,
            "err_val": log.data["ham"][i].Mean - ed,
        }
    )

for _data in conv_data:
    plt.plot(
        _data["iters"],
        np.abs(_data["err_val"] / _data["e0"]),
    )

plt.xlabel("Iteration")
plt.ylabel("Rel Error")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("convergence.pdf")
plt.clf()

# Evaluate on a fresh test set of disorder realizations.
N_test = 100
params_list = generate_disorder_realizations(N_test, hi.size, h0)
print(params_list.shape)
vs.parameter_array = params_list
Mz2 = Mz @ Mz
Mz2_mat = Mz2.to_sparse()

exact = {
    "h": vs.parameter_array,
    "Energy": [],
    "Mz2": [],
}
print("computing exact values on test set...")
for pars in tqdm(exact["h"]):
    _ha = create_operator(pars.reshape(-1))
    E0, psi0 = nk.exact.lanczos_ed(_ha, k=1, compute_eigenvectors=True)
    E0 = E0.item()
    psi0 = psi0.reshape(-1)
    exact["Energy"].append(E0)
    exact["Mz2"].append((psi0.T.conj() @ (Mz2_mat @ psi0)).item())

exact = {
    "h": np.array(exact["h"]),
    "Energy": np.array(exact["Energy"]),
    "Mz2": jnp.real(np.array(exact["Mz2"])),
}

vmc_vals = {
    "Energy": [],
    "Mz2": [],
}


print("Computing the nqs predictions for the squared magnetizations on the test set...")
for pars in tqdm(vs.parameter_array):
    # Build the corresponding operator and evaluate observables with full summation.
    _ha = create_operator(pars)
    _vs = vs.get_state(pars)
    _vs.reset()

    vs_fs = nk.vqs.FullSumState(
        hilbert=hi, model=_vs.model, chunk_size=_vs.chunk_size, variables=_vs.variables
    )
    _e = vs_fs.expect(_ha)
    _o = vs_fs.expect(Mz2)
    vmc_vals["Energy"].append(_e.Mean)
    vmc_vals["Mz2"].append(_o.Mean)

vmc_vals = {
    "h": np.array(vs.parameter_array),
    "Energy": np.array(vmc_vals["Energy"]),
    "Mz2": jnp.real(np.array(vmc_vals["Mz2"])),
}

# Magnetization agreement plot: exact diagonalization vs foundational predictions.
plt.scatter(exact["Mz2"], vmc_vals["Mz2"], alpha=0.7)
plt.plot(
    np.linspace(jnp.min(exact["Mz2"]), jnp.max(exact["Mz2"]), 100),
    np.linspace(jnp.min(exact["Mz2"]), jnp.max(exact["Mz2"]), 100),
    linestyle="--",
    color="black",
    label="id",
)
plt.xlabel("exact mag squared")
plt.ylabel("vmc mag squared")
plt.legend()
plt.savefig("mag_accordance.pdf")
plt.clf()

# Distribution comparison for exact and foundational Mz2 values.
kde_exact = gaussian_kde(np.real(exact["Mz2"]))
x_exact = np.linspace(exact["Mz2"].min(), exact["Mz2"].max(), 500)
plt.plot(x_exact, kde_exact(x_exact), linestyle="--", color="black", label="Exact")

# VMC
kde_vmc = gaussian_kde(np.real(vmc_vals["Mz2"]))
x_vmc = np.linspace(vmc_vals["Mz2"].min(), vmc_vals["Mz2"].max(), 500)
plt.plot(x_vmc, kde_vmc(x_vmc), label="VMC")

plt.legend()
plt.xlabel("Mz2")
plt.ylabel("Distrib.")
plt.savefig("mz2.pdf")
plt.clf()
