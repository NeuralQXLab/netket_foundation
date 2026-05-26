"""
Importance Sampling for NetKet.

Self-normalized formula:
    <O>_target = sum_i w_i * O_loc_target(sigma_i) / sum_i w_i

with:
    w_i = |psi_target(sigma_i)|^2 / |psi_ref(sigma_i)|^2
    O_loc_target(sigma) = sum_eta <sigma|O|eta> * psi_target(eta) / psi_target(sigma)
    sigma_i ~ |psi_ref|^2
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import core as fcore

from netket.stats import Stats

from netket.utils import struct


@struct.dataclass
class ISResult(struct.Pytree):
    stats: Stats
    ess: float
    ess_fraction: float
    n_samples: int


def _compute_is_weights(log_psi_target, log_psi_ref):
    log_w = 2.0 * jnp.real(log_psi_target - log_psi_ref)
    log_w = log_w - jnp.max(log_w)
    weights = jnp.exp(log_w)
    W = jnp.sum(weights)
    ess = float(W**2 / jnp.sum(weights**2))
    return weights, ess


def _compute_local_values(
    apply_fn, target_variables, samples, operator, chunk_size=None
):
    samples_np = np.asarray(samples)
    sigma_prime, mels = operator.get_conn_padded(samples_np)
    sigma_prime = jnp.asarray(sigma_prime)
    mels = jnp.asarray(mels)

    n_samples = samples.shape[0]
    N = samples.shape[-1]
    n_conn = sigma_prime.shape[1]

    if chunk_size is not None:
        _apply = lambda s: apply_fn(target_variables, s)
        log_psi_sigma = _apply_chunked(_apply, samples, chunk_size)
    else:
        log_psi_sigma = apply_fn(target_variables, samples)

    sigma_prime_flat = sigma_prime.reshape(-1, N)
    if chunk_size is not None:
        log_psi_conn = _apply_chunked(_apply, sigma_prime_flat, chunk_size)
    else:
        log_psi_conn = apply_fn(target_variables, sigma_prime_flat)
    log_psi_conn = log_psi_conn.reshape(n_samples, n_conn)

    O_loc = jnp.sum(
        mels * jnp.exp(log_psi_conn - log_psi_sigma[:, None]),
        axis=-1,
    )

    return O_loc, log_psi_sigma


def _apply_chunked(apply_fn, samples, chunk_size):
    n = samples.shape[0]
    results = []
    for i in range(0, n, chunk_size):
        results.append(apply_fn(samples[i : i + chunk_size]))
    return jnp.concatenate(results, axis=0)


def expect_is(operator, mc_ref, target_variables, chunk_size=None):
    """
    Compute <O>_target by importance sampling from a reference MCState.

    Args:
        operator:           NetKet operator (diagonal or non-diagonal).
        mc_ref:             nk.vqs.MCState already sampled (call mc_ref.sample() first).
        target_variables:   Variables dict for the target state.
        chunk_size:         Chunk size (None = all at once).

    Returns:
        ISResult(.stats, .ess, .ess_fraction, .n_samples)
    """
    samples = mc_ref.samples
    if samples.ndim >= 3:
        samples = jax.lax.collapse(samples, 0, 2)
    n_samples = samples.shape[0]

    apply_fn = mc_ref._apply_fun

    O_loc, log_psi_target = _compute_local_values(
        apply_fn, target_variables, samples, operator, chunk_size
    )

    ref_variables = mc_ref.variables
    if chunk_size is not None:
        log_psi_ref = _apply_chunked(
            lambda s: apply_fn(ref_variables, s), samples, chunk_size
        )
    else:
        log_psi_ref = apply_fn(ref_variables, samples)

    weights, ess = _compute_is_weights(log_psi_target, log_psi_ref)

    W = jnp.sum(weights)
    mean = jnp.sum(weights * O_loc) / W
    variance = jnp.real(jnp.sum(weights * jnp.abs(O_loc - mean) ** 2) / W)
    error_of_mean = jnp.sqrt(variance / max(ess, 1.0))

    stats = Stats(mean=mean, error_of_mean=error_of_mean, variance=variance)

    return ISResult(
        stats=stats,
        ess=ess,
        ess_fraction=ess / n_samples,
        n_samples=n_samples,
    )


@struct.dataclass
class QFIISResult(struct.Pytree):
    chi: jax.Array   # (n_params, n_params)
    ess: float
    ess_fraction: float
    n_samples: int


def qfi_is(vs, parameters, mc_ref):
    """
    Compute the fidelity susceptibility chi at `parameters` via IS from mc_ref.

    Args:
        vs:         FoundationalQuantumState.
        parameters: 1D array of shape (n_params,).
        mc_ref:     MCState already sampled at a nearby reference point.

    Returns:
        QFIISResult(chi, ess, ess_fraction, n_samples)
        where chi has shape (n_params, n_params).
    """
    parameters = jnp.asarray(parameters)
    mc_tgt   = vs.get_state(parameters)
    tgt_vars = mc_tgt.variables

    samples = mc_ref.samples
    if samples.ndim >= 3:
        samples = jax.lax.collapse(samples, 0, 2)
    n_samples = samples.shape[0]

    apply_fn = mc_ref._apply_fun

    log_psi_tgt = apply_fn(tgt_vars, samples)
    log_psi_ref = apply_fn(mc_ref.variables, samples)
    weights, ess = _compute_is_weights(log_psi_tgt, log_psi_ref)
    W = jnp.sum(weights)

    apply_fn_tgt = mc_tgt._apply_fun
    vars_no_h, h_dict = fcore.pop(tgt_vars, "foundational")

    def logpsi_h_batch(h, x_batch):
        return apply_fn_tgt(fcore.copy(vars_no_h, {"foundational": h}), x_batch)

    dlog = jax.jacfwd(logpsi_h_batch)(h_dict, samples)["parameters"]
    # shape: (n_samples, n_params)

    w_n = weights / W
    mu  = jnp.einsum("i,ij->j", w_n, dlog)       # (n_params,)
    d   = dlog - mu[None, :]                       # (n_samples, n_params)
    chi = jnp.einsum("i,ij,ik->jk", w_n, d, d)   # (n_params, n_params)

    return QFIISResult(
        chi=chi,
        ess=float(ess),
        ess_fraction=float(ess / n_samples),
        n_samples=n_samples,
    )
