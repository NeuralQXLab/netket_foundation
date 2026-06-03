import os

os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

import netket_foundation as nkf
import netket as nk
import flax.linen as nn
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import jax
import optax
from tqdm import tqdm

initializer = nn.initializers.lecun_normal()

seed = 42
key = jax.random.PRNGKey(seed)
pars_type = jnp.float64

# Hamiltonian
from netket_foundation.operator import create as fcdag
from netket_foundation.operator import destroy as fc
from netket_foundation.operator import number as fnc

L = 8
graph = nk.graph.Chain(L, pbc=True)
N = graph.n_nodes
N_fermions = 4  # Half-filling
hi = nk.hilbert.SpinOrbitalFermions(
    N, s=1 / 2, n_fermions_per_spin=(N_fermions, N_fermions)
)

n_layers = 2
d_output = hi.n_orbitals * hi.n_fermions
d_latent = 8
heads = 2
b = 2
n_patches = 4
n_coups = 1
out_activation = nn.tanh

print("Model construction...")

from netket_foundation._src.model.fermionic_model.fermi_vit.body import (
    foundation_ViT_trans_equi,
)

vit = foundation_ViT_trans_equi(
    n_layers=n_layers,
    d_model=d_latent,
    d_output=d_output,
    d_latent=d_latent,
    heads=heads,
    b=b,
    is_2d=False,
    n_patches=n_patches,
    n_coups=n_coups,
    graph=graph,
    out_activation=out_activation,
    param_dtype=pars_type,
)

from netket_foundation._src.model.fermionic_model.fermi_backflow import (
    foundation_backflow,
)

backflow = foundation_backflow(
    model=vit, hilbert=hi, graph=graph, param_dtype=pars_type
)

from netket_foundation._src.model.fermionic_model.fermi_jastrow import (
    foundation_fermi_Jastrow_MLP,
)
from netket_foundation._src.model.fermionic_model.activation import log_cosh

f_jastrow_mlp = foundation_fermi_Jastrow_MLP(
    n_layers=n_layers,
    n_coups=n_coups,
    d_model=d_latent,
    initializer=initializer,
    param_dtype=pars_type,
    out_activation=log_cosh,
)

from netket_foundation._src.model.fermionic_model.prod_module import ProductModule

J_multi = ProductModule(f_jastrow_mlp, backflow)

print("Model initialization...")

key, subkey = jax.random.split(key)
parameter_array = jnp.linspace(0.0, 4.0, 50, dtype=pars_type).reshape(-1, 1)
n_replicas = parameter_array.shape[0]
ps = nkf.ParameterSpace(N=1, min=0.0, max=4.0)

sa = nk.sampler.MetropolisFermionHop(
    hilbert=hi,
    n_chains=n_replicas,
    graph=graph,
    sweep_size=hi.size,
)

vs = nkf.FoundationalQuantumState(
    sa, J_multi, ps, n_replicas=n_replicas, seed=seed, n_samples=n_replicas * 16
)

vs.parameter_array = parameter_array

print("Model training...")

up, down = +1, -1

bonds_nn = [tuple(e) for e in graph.edges()]


def create_operator(params):
    assert params.shape == (1,)
    U = params[0]
    t = 1.0
    H_t = sum(
        (fcdag(hi, i, spin) @ fc(hi, j, spin) + fcdag(hi, j, spin) @ fc(hi, i, spin))
        for i, j in bonds_nn
        for spin in (up, down)
    )

    H_U = sum(fnc(hi, i, up) @ fnc(hi, i, down) for i in range(graph.n_nodes))

    H = -t * H_t.to_jax_operator() + U * H_U.to_jax_operator()
    return H


ha_p = nkf.operator.ParametrizedOperator(hi, ps, create_operator)

epochs = 500
lr = 5e-2
lr_factor = 0.001

lr_scheduler = optax.cosine_decay_schedule(
    init_value=lr, decay_steps=epochs, alpha=lr_factor
)
print("n_params:", vs.n_parameters)

optimizer = optax.sgd(learning_rate=lr_scheduler)
ds = optax.linear_schedule(1e-4, 1e-8, transition_steps=epochs)
gs = nkf.VMC_SR(
    ha_p, optimizer, variational_state=vs, diag_shift=ds, use_ntk=True, mode="real"
)

log = nk.logging.JsonLog("hubbard_model_log")

gs.run(
    epochs,
    out=log,
    obs={"ham": ha_p},
)

print("Plotting convergence curves...")
conv_data = []

from netket_foundation._src.model.fermionic_model.ED_calculation import (
    exact_ground_state_energy,
)

for i, pars in tqdm(enumerate(vs.parameter_array)):
    _ha = create_operator(pars)
    ed = exact_ground_state_energy(_ha).item()

    err_val = log.data["ham"][i].Mean - ed
    conv_data.append(
        {
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
plt.savefig("convergence.pdf")
plt.clf()
