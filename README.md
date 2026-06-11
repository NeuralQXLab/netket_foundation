# NetKet Foundation

NetKet Foundation is an extension for [NetKet](https://github.com/netket/netket) to train and evaluate *foundation neural quantum states* over families of Hamiltonians parameterized by couplings or disorder realizations.

The library builds on top of NetKet's concepts (samplers, operators, logging, drivers) and introduces foundational workflows where one model is optimized across many parameter points at once.

## Installation

You can install `netket-foundation` with `pip` or `uv` using one of the two commands below. We strongly recomend against using `conda`.

```sh
uv add netket-foundation
pip install --upgrade netket-foundation
```

**With GPU support (Linux only):**
```sh
uv add netket-foundation 'netket[cuda]'
```

**Development version:**
```sh
uv add git+https://github.com/netket/netket_foundation.git
```

For detailed installation instructions of NetKet, including GPU setup, we refer to [its installation guide](https://netket.readthedocs.io/en/latest/install.html).

## Getting Started

To get started with NetKet Foundation, we recommend you give a look at our [tutorials](https://github.com/netket/netket_foundation/tree/main/docs/tutorials), by running them on your computer or on [Google Colaboratory](https://colab.research.google.com).
There are also several [example scripts](https://github.com/netket/netket_foundation/tree/main/examples) that you can download, run and edit that showcase some use-cases of NetKet Foundation, although they are not commented.

If you want to get in touch with us, feel free to open an issue or a discussion here on GitHub, or to join the MLQuantum slack group where several people involved with NetKet hang out.
The link is on [NetKet's website](https://www.netket.org).


**New concepts:**

Compared to base NetKet, this package introduces:

- `ParameterSpace`: a Hilbert-space-like class describing the space where hamiltonian parameters live.
- `FoundationalQuantumState`: a variational state that samples physical configurations together with parameter replicas.
- `ParametrizedOperator`: operators whose matrix elements are generated from per-sample parameters.

## Minimal Usage

### 1) Define a foundational state over a parameter range

```python
import jax.numpy as jnp
import netket as nk
import netket_foundation as nkf

from netket_foundation._src.model.vit import ViTFNQS

hi = nk.hilbert.Spin(0.5, 10)
ps = nkf.ParameterSpace(N=1, min=0.8, max=1.2)

model = ViTFNQS(
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

sampler = nk.sampler.MetropolisLocal(hi, n_chains=2048)
vstate = nkf.FoundationalQuantumState(sampler, model, ps, n_replicas=8, seed=1)

# Define the coupling used during training
vstate.parameter_array = jnp.linspace(0.8, 1.2, vstate.n_replicas).reshape(-1, 1)
```

### 2) Build a parameter-dependent operator

```python
import netket_foundation as nkf

def create_operator(params):
	h = params[0]
	ha_x = sum(nkf.operator.sigmax(hi, i) for i in range(hi.size))
	ha_zz = sum(
		nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i + 1) % hi.size)
		for i in range(hi.size)
	)
	return -h * ha_x - ha_zz

ham = nkf.operator.ParametrizedOperator(hi, ps, create_operator)
```

### 3) Optimize with foundational natural-gradient VMC

```python
import optax
import netket_foundation as nkf

optimizer = optax.sgd(5e-3)
driver = nkf.VMC_SR(ham, optimizer, variational_state=vstate, diag_shift=1e-4)
driver.run(100)
```
