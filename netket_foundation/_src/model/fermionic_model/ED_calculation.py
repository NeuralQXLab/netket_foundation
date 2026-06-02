"""Exact diagonalization helper utilities.

This module provides a small utility to compute the ground-state energy
of a given operator by constructing the full many-body Hamiltonian matrix
in the computational basis and performing a sparse eigenvalue solve.

Notes
-----
The input operator `op` must provide a Hilbert space accessible via
``op.hilbert`` with a method ``all_states()`` and must implement
``get_conn_padded(states)`` returning connected states and matrix
elements for each input basis state.
"""

import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import jax.numpy as jnp


def exact_ground_state_energy(op):
    """Compute the lowest eigenvalue (ground-state energy) of ``op``.

    The function enumerates the computational basis, builds the sparse
    Hamiltonian matrix by querying ``op.get_conn_padded`` for each basis
    state, and uses ``scipy.sparse.linalg.eigsh`` to obtain the smallest
    eigenvalue.

    Parameters
    ----------
    op : object
        Operator object providing ``op.hilbert.all_states()`` and
        ``op.get_conn_padded(states)``.

    Returns
    -------
    float
        Ground-state energy (smallest eigenvalue) of the assembled matrix.
    """

    basis_states = np.asarray(op.hilbert.all_states(), dtype=np.int8)
    basis_index = {tuple(state.tolist()): idx for idx, state in enumerate(basis_states)}

    rows = []
    cols = []
    data = []

    for col, state in enumerate(basis_states):
        conn_states, mels = op.get_conn_padded(jnp.asarray(state[None, :]))
        conn_states = np.asarray(conn_states[0])
        mels = np.asarray(mels[0])

        for conn_state, mel in zip(conn_states, mels):
            if np.abs(mel) < 1e-12:
                continue

            row = basis_index.get(tuple(np.asarray(conn_state, dtype=np.int8).tolist()))
            if row is None:
                continue

            rows.append(row)
            cols.append(col)
            data.append(mel)

    matrix = sp.coo_matrix(
        (data, (rows, cols)), shape=(len(basis_states), len(basis_states))
    ).tocsr()
    return spla.eigsh(matrix, k=1, which="SA", return_eigenvectors=False)[0]
