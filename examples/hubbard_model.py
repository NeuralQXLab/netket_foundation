import os

os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

import netket_foundation as nkf
import netket as nk
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import jax
import optax
from tqdm import tqdm

from netket_foundation.operator import create as fcdag
from netket_foundation.operator import destroy as fc
from netket_foundation.operator import number as fnc
from netket_foundation.model import ViTFermionicFNQS

seed = 42
key = jax.random.PRNGKey(seed)
pars_type = jnp.float64

L = 8
graph = nk.graph.Chain(L, pbc=True)
N = graph.n_nodes
N_fermions = 4  # Half-filling
hi = nk.hilbert.SpinOrbitalFermions(
    N, s=1 / 2, n_fermions_per_spin=(N_fermions, N_fermions)
)

n_coups = 1
ps = nkf.ParameterSpace(N=1, min=0.0, max=4.0)

print("Model construction...")

ma = ViTFermionicFNQS(
    hilbert=hi,
    graph=graph,
    n_coups=n_coups,
    num_layers=2,
    d_model=8,
    heads=2,
    b=2,
    param_dtype=pars_type,
)

print("Model initialization...")

parameter_array = jnp.linspace(0.0, 4.0, 50, dtype=pars_type).reshape(-1, 1)
n_replicas = parameter_array.shape[0]

sa = nk.sampler.MetropolisFermionHop(
    hilbert=hi,
    n_chains=n_replicas,
    graph=graph,
    sweep_size=hi.size,
)

vs = nkf.FoundationalQuantumState(
    sa, ma, ps, n_replicas=n_replicas, seed=seed, n_samples=n_replicas * 16
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
lr_scheduler = optax.cosine_decay_schedule(
    init_value=lr, decay_steps=epochs, alpha=0.001
)
print("n_params:", vs.n_parameters)

optimizer = optax.sgd(learning_rate=lr_scheduler)
ds = optax.linear_schedule(1e-4, 1e-8, transition_steps=epochs)
gs = nkf.VMC_NG(
    ha_p, optimizer, variational_state=vs, diag_shift=ds, use_ntk=True, mode="real"
)

log = nk.logging.JsonLog("hubbard_model_log")
gs.run(epochs, out=log, obs={"ham": ha_p})

print("Plotting convergence curves...")
conv_data = []

for i, pars in tqdm(enumerate(vs.parameter_array)):
    _ha = create_operator(pars)
    ed = nk.exact.lanczos_ed(_ha, k=1)[0].item()
    conv_data.append(
        {
            "e0": log.data["ham"][i].Mean,
            "energy": ed,
            "iters": log.data["ham"][i].iters,
            "err_val": log.data["ham"][i].Mean - ed,
        }
    )

for _data in conv_data:
    plt.plot(_data["iters"], np.abs(_data["err_val"] / _data["e0"]))

plt.xlabel("Iteration")
plt.ylabel("Rel Error")
plt.xscale("log")
plt.yscale("log")
plt.savefig("convergence.pdf")
plt.clf()
