from typing_extensions import Union

import jax.numpy as jnp
import jax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml

# ------------------------------------------------------------------------------
# Functions for parsing time series data


# from: https://github.com/google/jax/issues/3171
def time_series_idxs(past_steps: int, future_steps: int, delta_t: int, total_steps: int) -> tuple:
    """
    Get the input and output indices to split a time series into overlapping sequences of past steps and
    future steps.

    args:
        past_steps: number of historical steps to use in the model
        future_steps: number of future steps of the output
        delta_t: number of timesteps per model step, applies to past and future steps
        total_steps: total number of timesteps that we are batching

    Returns:
        tuple of jnp.arrays of input and output idxs, 1st axis num sequences, 2nd axis actual sequences
    """
    first_start = 0
    last_start = (
        total_steps - future_steps * delta_t - (past_steps - 1) * delta_t
    )  # one past step is included
    assert (
        first_start < last_start
    ), f"time_series_idxs: {total_steps}-{future_steps}*{delta_t} - ({past_steps}-1)*{delta_t}"
    in_idxs = (
        jnp.arange(first_start, last_start)[:, None]
        + jnp.arange(0, past_steps * delta_t, delta_t)[None, :]
    )

    first_start = past_steps * delta_t
    last_start = total_steps - (future_steps - 1) * delta_t
    assert (
        first_start < last_start
    ), f"time_series_idxs: {total_steps}-({future_steps}-1)*{delta_t}, {past_steps}*{delta_t}"
    out_idxs = (
        jnp.arange(first_start, last_start)[:, None]
        + jnp.arange(0, future_steps * delta_t, delta_t)[None, :]
    )
    assert len(in_idxs) == len(out_idxs)

    return in_idxs, out_idxs


def times_series_to_multi_images(
    D: int,
    dynamic_fields: dict[tuple[int, int], jax.Array],
    constant_fields: dict[tuple[int, int], jax.Array],
    is_torus: Union[bool, tuple[bool, ...]],
    past_steps: int,
    future_steps: int,
    skip_initial: int = 0,
    delta_t: int = 1,
    downsample: int = 0,
) -> tuple[geom.MultiImage, geom.MultiImage]:
    """
    Given time series fields, convert them to input and output MultiImages based on the number of past steps,
    future steps, and any subsampling/downsampling.

    args:
        D: dimension of problem
        dynamic_fields: the fields to build MultiImages, dict with keys (k,parity) and values
            of array of shape (batch,time,spatial,tensor)
        constant_fields: fields constant over time, dict with keys (k,parity) and values
            of array of shape (batch,spatial,tensor)
        is_torus: whether the images are tori
        past_steps: number of historical steps to use in the model
        future_steps: number of future steps
        skip_initial: number of initial time steps to skip
        delta_t: number of timesteps per model step
        downsample: number of times to downsample the image by average pooling, decreases by a factor
            of 2

    returns:
        tuple of MultiImages multi_image_X and multi_image_Y
    """
    assert len(dynamic_fields.values()) != 0

    dynamic_fields = {k: v[:, skip_initial:] for k, v in dynamic_fields.items()}
    t_steps = next(iter(dynamic_fields.values())).shape[1]  # time steps, also total steps
    # add a time dimension to force and fill it with copies
    constant_fields = {
        k: jnp.full((len(v), t_steps) + v.shape[1:], v[:, None]) for k, v in constant_fields.items()
    }

    input_idxs, output_idxs = time_series_idxs(past_steps, future_steps, delta_t, t_steps)
    input_dynamic_fields = {
        k: v[:, input_idxs].reshape((-1, past_steps) + v.shape[2:])
        for k, v in dynamic_fields.items()
    }
    input_constant_fields = {
        k: v[:, input_idxs].reshape((-1, past_steps) + v.shape[2:])[:, :1]
        for k, v in constant_fields.items()
    }
    output_dynamic_fields = {
        k: v[:, output_idxs].reshape((-1, future_steps) + v.shape[2:])
        for k, v in dynamic_fields.items()
    }

    multi_image_X = geom.MultiImage(input_dynamic_fields, D, is_torus).concat(
        geom.MultiImage(input_constant_fields, D, is_torus),
        axis=1,
    )
    multi_image_Y = geom.MultiImage(output_dynamic_fields, D, is_torus)

    for _ in range(downsample):
        multi_image_X = multi_image_X.average_pool(2)
        multi_image_Y = multi_image_Y.average_pool(2)

    return multi_image_X, multi_image_Y
