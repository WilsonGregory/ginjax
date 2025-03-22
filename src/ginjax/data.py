from typing_extensions import Union

import jax.numpy as jnp
import jax

import ginjax.geometric as geom

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


def batch_time_series(
    dynamic_fields: geom.MultiImage,
    constant_fields: geom.MultiImage,
    total_steps: int,
    past_steps: int,
    future_steps: int,
    skip_initial: int = 0,
    delta_t: int = 1,
    downsample: int = 0,
) -> tuple[geom.MultiImage, geom.MultiImage]:
    """
    Given time series fields batch an initial batch dimension, convert them to input and output
    MultiImages based on the number of past steps, future steps, and any subsampling/downsampling.

    args:
        dynamic_fields: the dynamic fields, shape (batch,channels*time,spatial,tensor)
        constant_fields: the constant fields, shape (batch,channels,spatial,tensor)
        total_steps: total number of timesteps we are working with
        past_steps: number of historical steps to use in the model
        future_steps: number of future steps
        skip_initial: number of initial time steps to skip
        delta_t: number of timesteps per model step
        downsample: number of times to downsample the image by average pooling, decreases by a factor
            of 2

    returns:
        tuple of MultiImages multi_image_X and multi_image_Y
    """
    vmap_f = jax.vmap(times_series_to_multi_images, in_axes=(0, 0) + (None,) * 6)
    multi_image_x, multi_image_y = vmap_f(
        dynamic_fields,
        constant_fields,
        total_steps,
        past_steps,
        future_steps,
        skip_initial,
        delta_t,
        downsample,
    )
    for key, image in multi_image_x.items():
        multi_image_x[key] = image.reshape((-1,) + image.shape[2:])

    for key, image in multi_image_y.items():
        multi_image_y[key] = image.reshape((-1,) + image.shape[2:])

    return multi_image_x, multi_image_y


def times_series_to_multi_images(
    dynamic_fields: geom.MultiImage,
    constant_fields: geom.MultiImage,
    total_steps: int,
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
        dynamic_fields: the dynamic fields, shape (channels*time,spatial,tensor)
        constant_fields: the constant fields, shape (channels,spatial,tensor)
        total_steps: total number of timesteps we are working with
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

    spatial_dims = dynamic_fields.get_spatial_dims()
    D = dynamic_fields.D
    input_idxs, output_idxs = time_series_idxs(
        past_steps, future_steps, delta_t, total_steps - skip_initial
    )

    multi_image_x = dynamic_fields.empty()
    multi_image_y = dynamic_fields.empty()
    for (k, parity), image in dynamic_fields.items():
        image = image.reshape((-1, total_steps) + spatial_dims + (D,) * k)
        image = image[:, skip_initial:]
        n_channels = len(image)

        input_image = image[:, input_idxs].reshape(
            (n_channels,) + (-1, past_steps) + spatial_dims + (D,) * k
        )
        output_image = image[:, output_idxs].reshape(
            (n_channels,) + (-1, future_steps) + spatial_dims + (D,) * k
        )

        multi_image_x.append(
            k,
            parity,
            jnp.moveaxis(input_image, 1, 0).reshape(
                (-1, n_channels * past_steps) + spatial_dims + (D,) * k
            ),
        )
        multi_image_y.append(
            k,
            parity,
            jnp.moveaxis(output_image, 1, 0).reshape(
                (-1, n_channels * future_steps) + spatial_dims + (D,) * k
            ),
        )

    batch = len(next(iter(multi_image_x.values())))
    for (k, parity), image in constant_fields.items():
        multi_image_x.append(k, parity, jnp.full((batch,) + image.shape, image), axis=1)

    for _ in range(downsample):
        multi_image_x = multi_image_x.average_pool(2)
        multi_image_y = multi_image_y.average_pool(2)

    return multi_image_x, multi_image_y
