"""Tests for ParametrizedOperator and fermionic operators."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import netket_foundation as nkf
from helpers import make_hilbert, make_parameter_space, make_ising, make_fermion_hilbert


@pytest.fixture(scope="module")
def hi():
    return make_hilbert()


@pytest.fixture(scope="module")
def ps():
    return make_parameter_space()


@pytest.fixture(scope="module")
def create_ising(hi):
    return make_ising(hi)


@pytest.fixture(scope="module")
def ham(hi, ps, create_ising):
    return nkf.operator.ParametrizedOperator(hi, ps, create_ising)


def test_hilbert_is_product(hi, ps, ham):
    """Operator's Hilbert space is the product hi * ps."""
    assert ham.hilbert == hi * ps


def test_get_conn_padded_shape(hi, ps, ham):
    """Connected states and matrix elements have compatible shapes."""
    x_phys = hi.random_state(jax.random.key(7), 8)
    x_params = ps.random_state(jax.random.key(8), 8)
    x = jnp.concatenate([x_phys, x_params], axis=-1)

    xs, mels = ham.get_conn_padded(x)

    # xs: (n_samples, max_conn, hilbert.size)  mels: (n_samples, max_conn)
    assert xs.ndim == 3
    assert mels.ndim == 2
    assert xs.shape[0] == 8
    assert mels.shape[0] == 8
    assert xs.shape[1] == mels.shape[1]
    assert xs.shape[2] == hi.size + ps.size


def test_get_conn_padded_params_preserved(hi, ps, ham):
    """Parameter portion is unchanged in all connected states."""
    x_phys = hi.random_state(jax.random.key(3), 6)
    x_params = jnp.ones((6, ps.size)) * 1.0  # fixed coupling = 1.0
    x = jnp.concatenate([x_phys, x_params], axis=-1)

    xs, _ = ham.get_conn_padded(x)

    params_in_conn = xs[..., hi.size :]
    np.testing.assert_allclose(params_in_conn, 1.0, atol=1e-6)


def test_values_match_reference_operator(hi, ps, create_ising):
    """At a fixed coupling, ParametrizedOperator agrees with the direct operator."""
    h_val = 1.0
    ref_op = create_ising(jnp.array([h_val]))

    x_phys = hi.random_state(jax.random.key(5), 10)
    x_params = jnp.full((10, ps.size), h_val)
    x_joint = jnp.concatenate([x_phys, x_params], axis=-1)

    par_op = nkf.operator.ParametrizedOperator(hi, ps, create_ising)
    xs_par, mels_par = par_op.get_conn_padded(x_joint)
    xs_ref, mels_ref = ref_op.get_conn_padded(x_phys)

    np.testing.assert_allclose(mels_par, mels_ref, atol=1e-6)
    np.testing.assert_allclose(xs_par[..., : hi.size], xs_ref, atol=1e-6)


def test_with_params_returns_operator(hi, ps, create_ising):
    """with_params(1-D array) returns a concrete operator."""
    par_op = nkf.operator.ParametrizedOperator(hi, ps, create_ising)
    op = par_op.with_params(jnp.array([1.0]))
    assert hasattr(op, "get_conn_padded")


def test_pytree_roundtrip(ham):
    """ParametrizedOperator survives JAX pytree flatten/unflatten."""
    leaves, treedef = jax.tree_util.tree_flatten(ham)
    ham2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert ham2.hilbert == ham.hilbert


# ---------------------------------------------------------------------------
# Fermionic operators
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fhi():
    return make_fermion_hilbert(n_orbitals=4)


def _to_dense(op):
    """Convert a fermionic operator to a dense numpy matrix."""
    return op.to_sparse().toarray()


class TestFermionConstructors:
    def test_create_returns_operator(self, fhi):
        op = nkf.operator.create(fhi, 0)
        assert hasattr(op, "get_conn_padded")

    def test_destroy_returns_operator(self, fhi):
        op = nkf.operator.destroy(fhi, 0)
        assert hasattr(op, "get_conn_padded")

    def test_number_returns_operator(self, fhi):
        op = nkf.operator.number(fhi, 0)
        assert hasattr(op, "get_conn_padded")

    def test_create_destroy_type(self, fhi):
        from netket_foundation._src.operator.fermion2nd.numba import FermionOperator2nd
        assert isinstance(nkf.operator.create(fhi, 0), FermionOperator2nd)
        assert isinstance(nkf.operator.destroy(fhi, 0), FermionOperator2nd)
        assert isinstance(nkf.operator.number(fhi, 0), FermionOperator2nd)


class TestAnticommutation:
    """Canonical anti-commutation relations: {c_i, c_j†} = δ_{ij}."""

    def test_same_site(self, fhi):
        # c_i c_i† + c_i† c_i = I
        for i in range(fhi.size):
            c  = nkf.operator.destroy(fhi, i)
            cd = nkf.operator.create(fhi, i)
            anticomm = _to_dense(c @ cd + cd @ c)
            np.testing.assert_allclose(anticomm, np.eye(fhi.n_states), atol=1e-12)

    def test_diff_site(self, fhi):
        # c_i c_j† + c_j† c_i = 0 for i != j
        for i in range(fhi.size):
            for j in range(fhi.size):
                if i == j:
                    continue
                ci  = nkf.operator.destroy(fhi, i)
                cdj = nkf.operator.create(fhi, j)
                anticomm = _to_dense(ci @ cdj + cdj @ ci)
                np.testing.assert_allclose(anticomm, 0.0, atol=1e-12)


class TestNumberOperator:
    def test_number_equals_cdaggerc(self, fhi):
        # n_i = c_i† c_i
        for i in range(fhi.size):
            n   = _to_dense(nkf.operator.number(fhi, i))
            cdc = _to_dense(nkf.operator.create(fhi, i) @ nkf.operator.destroy(fhi, i))
            np.testing.assert_allclose(n, cdc, atol=1e-12)

    def test_number_is_projection(self, fhi):
        # n_i^2 = n_i
        for i in range(fhi.size):
            n = _to_dense(nkf.operator.number(fhi, i))
            np.testing.assert_allclose(n @ n, n, atol=1e-12)

    def test_number_is_hermitian(self, fhi):
        for i in range(fhi.size):
            n = _to_dense(nkf.operator.number(fhi, i))
            np.testing.assert_allclose(n, n.conj().T, atol=1e-12)


class TestNumbaJaxAgreement:
    """FermionOperator2nd (Numba) and FermionOperator2ndJax produce identical matrix elements."""

    def test_hopping_mels(self, fhi):
        from netket_foundation._src.operator.fermion2nd.jax import FermionOperator2ndJax

        # H = c0† c1 + c1† c0
        hop_numba = (
            nkf.operator.create(fhi, 0) @ nkf.operator.destroy(fhi, 1)
            + nkf.operator.create(fhi, 1) @ nkf.operator.destroy(fhi, 0)
        )
        hop_jax = hop_numba.to_jax_operator()
        assert isinstance(hop_jax, FermionOperator2ndJax)

        mat_numba = _to_dense(hop_numba)
        mat_jax   = hop_jax.to_sparse().toarray()
        np.testing.assert_allclose(mat_numba, mat_jax, atol=1e-12)

    def test_number_mels(self, fhi):
        n_numba = nkf.operator.number(fhi, 0) + nkf.operator.number(fhi, 1)
        n_jax   = n_numba.to_jax_operator()
        np.testing.assert_allclose(
            _to_dense(n_numba), n_jax.to_sparse().toarray(), atol=1e-12
        )


class TestHoppingHamiltonian:
    """Simple 2-site hopping: physical sanity checks."""

    def _make_hopping(self, fhi):
        return (
            nkf.operator.create(fhi, 0) @ nkf.operator.destroy(fhi, 1)
            + nkf.operator.create(fhi, 1) @ nkf.operator.destroy(fhi, 0)
        )

    def test_is_hermitian(self, fhi):
        H = _to_dense(self._make_hopping(fhi))
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_ground_state_energy(self, fhi):
        # Single particle on 4 sites: E0 of hopping = -1 (1-particle sector)
        hi2 = nkf.operator.create(fhi, 0).__class__.__mro__  # just make a 2-site system
        import netket as nk
        hi2 = nk.hilbert.SpinOrbitalFermions(2)
        H = (
            nkf.operator.create(hi2, 0) @ nkf.operator.destroy(hi2, 1)
            + nkf.operator.create(hi2, 1) @ nkf.operator.destroy(hi2, 0)
        )
        evals = np.linalg.eigvalsh(_to_dense(H))
        # ground state in the 1-particle sector is -1
        assert np.any(np.abs(evals - (-1.0)) < 1e-10)
