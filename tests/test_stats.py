"""ReplicaStats container and the per-replica combination rule."""

import jax
import jax.numpy as jnp
import numpy as np

from netket.stats import Stats

from netket_foundation.stats import ReplicaStats, combine_replica_stats


def _make_stats(mean, err, var, rhat):
    return Stats(
        mean=jnp.asarray(mean),
        error_of_mean=jnp.asarray(err),
        variance=jnp.asarray(var),
        R_hat=jnp.asarray(rhat),
    )


def test_replica_stats_is_a_list():
    stats = ReplicaStats([_make_stats(1.0, 0.1, 1.0, 1.0), _make_stats(3.0, 0.2, 2.0, 1.1)])
    assert isinstance(stats, list)
    assert len(stats) == 2
    assert float(stats[0].mean) == 1.0


def test_total_combination_rule():
    stats = ReplicaStats([_make_stats(1.0, 0.3, 1.0, 1.01), _make_stats(3.0, 0.4, 3.0, 1.20)])
    total = stats.total
    np.testing.assert_allclose(float(total.mean), 2.0)
    np.testing.assert_allclose(float(total.error_of_mean), np.sqrt(0.3**2 + 0.4**2) / 2)
    np.testing.assert_allclose(float(total.variance), 2.0)
    np.testing.assert_allclose(float(total.R_hat), 1.20)
    # combine_replica_stats also works on a plain list
    total2 = combine_replica_stats(list(stats))
    np.testing.assert_allclose(float(total2.mean), float(total.mean))


def test_replica_stats_is_a_pytree():
    stats = ReplicaStats([_make_stats(1.0, 0.1, 1.0, 1.0), _make_stats(3.0, 0.2, 2.0, 1.1)])
    doubled = jax.tree.map(lambda x: 2 * x, stats)
    assert isinstance(doubled, ReplicaStats)
    np.testing.assert_allclose(float(doubled[1].mean), 6.0)
    # survives a jit boundary
    out = jax.jit(lambda s: s)(stats)
    assert isinstance(out, ReplicaStats)
