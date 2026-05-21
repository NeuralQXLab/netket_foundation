import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import jax.numpy as jnp

def exact_ground_state_energy(op):

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