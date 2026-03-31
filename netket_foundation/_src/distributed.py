from functools import lru_cache
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
from jax.lax import with_sharding_constraint
from jax.experimental import multihost_utils

from netket import config as nkconfig
from netket import jax as nkjax
from netket.utils import module_version

if module_version("jax") >= (0, 7, 0):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P


@lru_cache
def mode() -> str:
    """
    Returns the distributed mode used by NetKet.

    This can be one of the following: ``None``, ``"sharding"``
    """
    if nkconfig.netket_experimental_sharding:
        return "sharding"
    else:
        return None


@lru_cache
def process_count() -> int:
    """
    Returns the total number of JAX processes running NetKet.

    If you are running with experimental sharding, this is
    equivalent to ``jax.process_count()``.
    """
    return jax.process_count()


@lru_cache
def device_count() -> int:
    """
    Returns total number of devices.
    """
    return jax.device_count()


@lru_cache
def process_index() -> int:
    """
    Returns the index of this process running NetKet.

    If you are running with experimental sharding, this is
    equivalent to :func:`jax.process_index`.

    This is an integer between 0 and
    :func:`netket_foundation.distributed.process_count`.
    """

    return jax.process_index()


def is_master_process() -> bool:
    """
    Returns whether the current process is the master process.
    """
    return process_index() == 0


def broadcast_key(key: Optional[jax.Array] = None, *, root: int = 0) -> jax.Array:
    """
    Given a `jax.random.key`, distribute it among all nodes.
    """
    return nkjax.PRNGKey(key, root=root)


def broadcast(array: jax.Array, *, root: int):
    """
    Broadcasts an array from the root process to all other processes, giving a replicated
    array.

    The input array on non-root processes must be a dummy array with
    the same shape as the array on the root process.

    Args:
        array: The array to broadcast. On non-root processes, this should be a
            placeholder array with the right shape and dtype.
        root: The root process that holds the original array.
    """
    if mode() == "sharding":
        result = multihost_utils.broadcast_one_to_all(
            array, is_source=jax.process_index() == root
        )
    else:
        result = array
    return result


def shard_replicated(array, *, axis=0):
    """
    Shards a replicated array across jax processes.

    The input must be a replicated array, obtained either from
    :func:`netket_foundation.distributed.broadcast`, :func:`netket_foundation.distributed.allgather` or
    from executing the same function on all nodes.

    When running under sharding, we set the sharding constraint accordingly.

    Args:
        array: The array to shard. Must be replicated!
        axis: The axis along which to shard (Default 0).
    """

    def _shard(array):
        lenght = array.shape[axis]
        if not lenght % process_count() == 0:
            raise ValueError(
                "Sharded axis size must be a multiple of the number of processes"
            )

        if mode() == "sharding":
            # Do not use process_count() because we could have more than
            # 1 GPU per process

            sharding = sharding_along_axis(array, axis=axis)
            array = jax.lax.with_sharding_constraint(array, sharding)
        else:
            pass
        return array

    return jax.tree_util.tree_map(_shard, array)


def declare_replicated_array(x):
    """
    Declares that an array is replicated across all processes.

    This should be used when we build an array by 'hand' and we know it's the same on
    every process, but by default jax does not know that it is the same everywhere.
    So we declare explicitly that it is replcated.

    .. note::

        This only does something when sharding with 1 device per process. Does nothing
        otherwise.

    Args:
        x: The array to declare as replicated.

    Returns:
        An array with the same shape as the input, but declared as replicated.
    """
    if mode() == "sharding" and process_count() == device_count():
        par_sharding = replicate_sharding()

        return jax.make_array_from_single_device_arrays(x.shape, par_sharding, [x])
    else:
        return x


def allgather(array, *, axis: int = 0, token=None):
    """
    Gathers (unshard) a distributed (sharded) array to all processes.

    The resulting array will have the same shape as the input array except
    the first axis, which will be :ref:`netket_foundation.distributed.process_count`
    times longer.

    .. note::

        An input array of shape :math:`(M, N, ...)` will lead to a gathered
        array of shape :math:`(P \times M, N, ...)`, where :math:`P` is the
        number of processes.

    .. note::

        The resulting array will be unsharded, or fully addressable locally
        and on every process.

    Args:
        array: The array to gather.

    Returns:
        A tuple of the gathered array and the token.

    """
    if axis != 0:
        raise NotImplementedError("Only axis=0 is supported for now. Open a PR.")

    if mode() == "sharding":
        sharding = replicate_sharding()
        array = jax.lax.with_sharding_constraint(array, sharding)
    else:
        pass
    return array, token


def pad_axis_for_sharding(
    array: jax.Array, *, axis: int = 0, padding_value: float | jax.Array = 0
) -> jax.Array:
    """
    Pads an array along an axis to make it divisible by the number of processes.

    Args:
        array: The array to pad.
        axis: The axis along which to pad.
        padding_value: The value to use for padding.

    Returns:
        The padded array.
    """
    axis_size = array.shape[axis]
    n_devices = device_count()

    if axis_size % n_devices != 0:
        padded_axis_size = int(n_devices * np.ceil(axis_size / n_devices))
        padding_shape = [(0, 0) for _ in range(array.ndim)]
        padding_shape[axis] = (0, padded_axis_size - axis_size)

        array = jnp.pad(
            array,
            padding_shape,
            constant_values=padding_value,
        )
    return array


def reshard(
    array: jax.Array,
    *,
    sharded_axis: int = 0,
    out_sharded_axis: int = 1,
    token=None,
    pad: bool = False,
    pad_value: jax.Array = 0.0,
) -> jax.Array:
    """
    Reshards an array to distribute another axis among the processes.

    The input array is assumed to be sharded along axis `sharded_axis`, and the resulting
    array will be sharded along axis `out_sharded_axis`. The sharded axis will be collected
    while the output sharded axis will be distributed.

    .. note::

        If the input array has shape :math:`(x, y, z)` and the input sharded axis is `y`,
        and the output sharded axis is `x`, the resulting array will have shape :math:`(x, y*P, z/P)`.

    Args:
        array: The array to reshard / alltoall.
        sharded_axis: The axis to be collected.
        out_sharded_axis: The axis to be distributed.
        pad: Whether to pad the axis to be sharded to be a multiple of the number of processes. If this is
            set to `False`, the size of the sharded axis must be a multiple of the number of processes.
            (Default: `False`)
        pad_value: The value to use for padding. (Default: `0.0`)

    """
    assert sharded_axis != out_sharded_axis
    assert 0 <= sharded_axis < array.ndim
    assert 0 <= out_sharded_axis < array.ndim

    # Pad the number of parameters to be a multiple of the number of proceses
    # -> (#n_nodes, np_padded)
    if array.shape[out_sharded_axis] % device_count() != 0:
        if pad:
            array = pad_axis_for_sharding(
                array, axis=out_sharded_axis, padding_value=pad_value
            )
        else:
            raise ValueError(
                "Sharded axis size must be a multiple of the number of processes"
            )

    if mode() == "sharding":
        del sharded_axis  # unused

        sharding = sharding_along_axis(array, axis=out_sharded_axis)
        array = with_sharding_constraint(array, sharding)
    return array, token


def barrier(name: str):
    """
    Synchronizes all processes. This function ensures that all processes reach this point
    before continuing.

    Args:
        name: A unique string to identify the synchronization point.
    """
    if mode() == "sharding":
        multihost_utils.sync_global_devices(name)


def broadcast_string(s: str, root: int = 0) -> str:
    def _encode_string_to_uint64_array(s):
        """Encodes a string into a NumPy array of uint64."""
        byte_data = s.encode("utf-8")  # Convert to bytes
        padding_size = (
            8 - len(byte_data) % 8
        ) % 8  # Compute padding to make it multiple of 8
        byte_data += b"\x00" * padding_size  # Pad with null bytes
        uint64_array = np.frombuffer(byte_data, dtype=np.uint64)  # Interpret as uint64
        return uint64_array, padding_size

    def _decode_uint64_array_to_string(uint64_array, padding_size):
        """Decodes a NumPy uint64 array back to a string."""
        byte_data = uint64_array.tobytes()  # Convert back to bytes
        return (
            byte_data[:-padding_size].decode("utf-8")
            if padding_size
            else byte_data.decode("utf-8")
        )

    if mode() == "sharding":
        if root != 0:
            raise ValueError("Only root=0 is supported in sharding mode")

        encoded_array, pad_size = _encode_string_to_uint64_array(s)
        encoded_array = multihost_utils.broadcast_one_to_all(encoded_array)
        pad_size = multihost_utils.broadcast_one_to_all(pad_size)
        s = _decode_uint64_array_to_string(encoded_array, pad_size)

    return s


def _inspect(name: str, x: jax.Array):
    """
    Internal function to inspect the sharding of an array. To be used for debugging inside
    of :func:`jax.jit`-ted functions.

    Args:
        name: A string to identify the array, usually the name, but can contain anything else.
        x: The array
    """
    if mode() == "sharding":

        def _cb(y):
            if process_index() == 0:
                print(
                    f"{name}: shape={x.shape}, sharding_shape: {y.shape}, sharding:",
                    y,
                    flush=True,
                )

        jax.debug.inspect_array_sharding(x, callback=_cb)


# TODO: Remove this fucntion when we require jax 0.7
def replicate_sharding():
    """
    Create a replicated sharding that works with both old and new JAX versions.
    Local copy to avoid netket_foundation dependency.
    """
    # TODO: always use the NamedShardng version
    if module_version("jax") >= (0, 7, 0):
        return NamedSharding(jax.sharding.get_abstract_mesh(), P())
    else:
        from jax.sharding import PositionalSharding

        return PositionalSharding(jax.devices()).replicate()


# TODO: Remove this fucntion when we require jax 0.7
def sharding_with_shape(shape):
    """
    Create a sharding with a specific shape that works with both old and new JAX versions.

    Args:
        shape: A tuple/list where:
               - -1 indicates sharding across devices
               - 1 indicates replication
               - Any positive integer > 1 also indicates sharding (will become 'S' in NamedSharding)
               For NamedSharding, -1 or any number > 1 becomes 'S', and 1 becomes None.

    Returns:
        A sharding object with the specified shape.
    """
    if module_version("jax") >= (0, 7, 0):
        # Convert shape to PartitionSpec
        # -1 or numbers > 1 -> 'S' (sharded), 1 -> None (replicated)
        spec_parts = []
        for dim in shape:
            if dim == -1 or (isinstance(dim, int) and dim > 1):
                spec_parts.append("S")
            elif dim == 1:
                spec_parts.append(None)
            else:
                raise ValueError(
                    f"Unsupported sharding dimension: {dim}. Use -1 or positive integer for sharding, 1 for replication."
                )

        # Remove trailing None values as they can be dropped
        while spec_parts and spec_parts[-1] is None:
            spec_parts.pop()

        return NamedSharding(jax.sharding.get_abstract_mesh(), P(*spec_parts))
    else:
        from jax.sharding import PositionalSharding

        return PositionalSharding(jax.devices()).reshape(shape)


# TODO: Remove this fucntion when we require jax 0.7
def sharding_along_axis(array, *, axis):
    """
    Create a sharding that shards along a specific axis and replicates along all others.

    Args:
        array: The array to get the shape from.
        axis: The axis to shard along (0-indexed). Must be provided as keyword argument.

    Returns:
        A sharding object that shards along the specified axis.
    """
    shape = [1] * array.ndim
    shape[axis] = -1
    return sharding_with_shape(shape)
