from typing_extensions import Optional, Union
import numpy as np

import jax
import jax.numpy as jnp

import geometricconvolutions.geometric as geom


def timestep_smse_loss(
    multi_image_x: geom.BatchMultiImage,
    multi_image_y: geom.BatchMultiImage,
    n_steps: int,
    reduce: Optional[str] = "mean",
) -> jax.Array:
    """
    Returns loss for each timestep. Loss is summed over the channels, and mean over spatial dimensions
    and the batch.

    args:
        multi_image_x: predicted data
        multi_image_y: target data
        n_steps: number of timesteps, all channels should be a multiple of this
        reduce: how to reduce over the batch, one of mean or max

    returns:
        the loss array with shape (batch,n_steps) if reduce is None or (n_steps,)
    """
    assert reduce in {"mean", "max", None}
    spatial_size = np.multiply.reduce(multi_image_x.get_spatial_dims())
    batch = multi_image_x.get_L()
    loss_per_step = jnp.zeros((batch, n_steps))
    for image_a, image_b in zip(
        multi_image_x.values(), multi_image_y.values()
    ):  # loop over image types
        image_a = image_a.reshape((batch, -1, n_steps) + image_a.shape[2:])
        image_b = image_b.reshape((batch, -1, n_steps) + image_b.shape[2:])
        loss = (
            jnp.sum((image_a - image_b) ** 2, axis=(1,) + tuple(range(3, image_a.ndim)))
            / spatial_size
        )
        loss_per_step = loss_per_step + loss

    if reduce == "mean":
        return jnp.mean(loss_per_step, axis=0)
    elif reduce == "max":
        return loss_per_step[jnp.argmax(jnp.sum(loss_per_step, axis=1))]
    else:
        return loss_per_step


def smse_loss(
    multi_image_x: geom.BatchMultiImage,
    multi_image_y: geom.BatchMultiImage,
    reduce: Optional[str] = "mean",
) -> jax.Array:
    """
    Sum of mean squared error loss. The sum is over the channels, the mean is over the spatial
    dimensions. Mean is also taken over batch if reduce == 'mean', or it returns each loss if
    reduce is None.

    args:
        multi_image_x: the input BatchMultiImage
        multi_image_y: the target BatchMultiImage
        reduce: how to reduce over batch. Either "mean" or None.

    returns:
        the loss value
    """
    assert reduce in {"mean", None}
    spatial_size = np.multiply.reduce(multi_image_x.get_spatial_dims())
    loss_per_batch = jnp.sum(
        (multi_image_x.to_vector() - multi_image_y.to_vector()) ** 2 / spatial_size, axis=1
    )

    if reduce == "mean":
        return jnp.mean(loss_per_batch)
    else:
        return loss_per_batch


def normalized_smse_loss(
    multi_image_x: geom.BatchMultiImage, multi_image_y: geom.BatchMultiImage, eps: float = 1e-5
) -> jax.Array:
    """
    Pointwise normalized loss. We find the norm of each channel at each spatial point of the true value
    and divide the tensor by that norm. Then we take the l2 loss, mean over the spatial dimensions, sum
    over the channels, then mean over the batch.

    args:
        multi_image_x: input BatchMultiImage
        multi_image_y: target BatchMultiImage
        eps: ensure that we aren't dividing by 0 norm

    returns:
        the loss value
    """
    spatial_size = np.multiply.reduce(multi_image_x.get_spatial_dims())

    order_loss = jnp.zeros(multi_image_x.get_L())
    for (k, parity), img_block in multi_image_y.items():
        # (b,c,spatial, (1,)*k)
        norm = geom.norm(multi_image_y.D + 2, img_block, keepdims=True) ** 2
        normalized_l2 = ((multi_image_x[(k, parity)] - img_block) ** 2) / (norm + eps)
        # (b,)
        order_loss = order_loss + (
            jnp.sum(normalized_l2, axis=range(1, img_block.ndim)) / spatial_size
        )

    return jnp.mean(order_loss)
