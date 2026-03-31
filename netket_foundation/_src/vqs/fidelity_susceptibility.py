from flax import core as fcore
import jax
import netket as nk
import jax.numpy as jnp


def susceptibility_stats(dparams_logpsi):
    """
    Compute the mean and variance of the fidelity susceptibility matrix from parameter derivatives.

    Args:
        dparams_logpsi: Array of shape (n_samples, n_params) containing the derivatives
                        of log(ψ) with respect to each parameter for each sample.

    Returns:
        tuple: (chi_mean, chi_var) where:
            - chi_mean: The mean susceptibility matrix of shape (n_params, n_params),
                        computed as the covariance of centered derivatives.
            - chi_var: The unbiased variance of the sample mean, shape (n_params, n_params).
    """
    n_samples, _ = dparams_logpsi.shape

    # Center the derivatives
    centered = dparams_logpsi - jnp.mean(
        dparams_logpsi, axis=0, keepdims=True
    )  # (n_samples, n_params)
    # Compute per-sample covariance tensors (vectorized outer products)
    chi_samples = (
        centered[:, :, None] * centered[:, None, :]
    )  # (n_samples, n_params, n_params)
    # Compute the mean susceptibility matrix
    chi_mean = jnp.mean(chi_samples, axis=0)  # (n_params, n_params)
    # Unbiased variance of the sample mean:
    # Var[mean] = sum((x - mean)^2) / (n * (n - 1))
    chi_var = jnp.sum((chi_samples - chi_mean[None, :, :]) ** 2, axis=0) / (
        n_samples * (n_samples - 1)
    )

    return chi_mean, chi_var


def susceptibility(vs):
    """
    Compute the fidelity susceptibility matrix for a variational state.

    The fidelity susceptibility is defined as:
        χ_ij = <∂_i(logψ) ∂_j(logψ)> - <∂_i(logψ)><∂_j(logψ)>
    where ∂_i denotes the derivative with respect to the i-th foundational parameter.

    This quantity measures how sensitive the quantum state is to changes in the
    foundational parameters and is related to the quantum geometric tensor.

    Args:
        vs: A NetKet variational state object

    Returns:
        dict: A dictionary with keys:
            - "Mean": The mean susceptibility matrix of shape (n_params, n_params)
            - "Variance": The variance of the susceptibility estimate
    """
    # Extract model and variables
    model = vs.model
    vars_no_h, h_dict = fcore.pop(vs.variables, "foundational")

    # Flatten samples
    samples_flat = vs.samples.reshape(-1, vs.hilbert.size)

    # Define logψ function
    def logpsi(h, vars_no_h, x):
        variables_with_h = fcore.copy(vars_no_h, {"foundational": h})
        return model.apply(variables_with_h, x)

    # Compute jacobian wrt foundational parameters
    if vs.chunk_size is None:
        df = jax.jacfwd(logpsi, argnums=0)(h_dict, vars_no_h, samples_flat)
    else:
        # Chunked version to save memory
        def jac_block(h, vars_no_h, x_chunk):
            return jax.jacfwd(logpsi, argnums=0)(h, vars_no_h, x_chunk)

        jac_block_chunked = nk.jax.apply_chunked(
            jac_block, in_axes=(None, None, 0), chunk_size=vs.chunk_size
        )
        jac_block_chunked = jax.jit(jac_block_chunked)
        df = jac_block_chunked(h_dict, vars_no_h, samples_flat)

    # Focus on 'parameters'
    dpararams_logpsi = df["parameters"]  # shape (n_samples, n_params)
    chi_Mean, chi_Variance = susceptibility_stats(dpararams_logpsi)
    chi_stat = {"Mean": chi_Mean, "Variance": chi_Variance}

    return chi_stat
