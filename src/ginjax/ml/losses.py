from typing_extensions import Optional
import numpy as np

import jax
import jax.numpy as jnp

import ginjax.geometric as geom


def timestep_smse_loss(
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    n_steps: int,
    reduce: Optional[str] = "mean",
) -> jax.Array:
    """
    Returns loss for each timestep. Loss is summed over the channels, and mean over spatial dimensions
    and the batch.

    args:
        multi_image_x: predicted data, image_blocks are shape (batch,channels,spatial,tensor)
        multi_image_y: target data, image_blocks are shape (batch,channels,spatial,tensor)
        n_steps: number of timesteps, all channels should be a multiple of this
        reduce: how to reduce over the batch, one of mean or max

    returns:
        the loss array with shape (batch,n_steps) if reduce is None or (n_steps,)
    """
    reduce_options = {"mean", "max", None}
    assert (
        reduce in reduce_options
    ), f"timestep_smse_loss: reduce={reduce} must be one of {reduce_options}"
    assert (
        multi_image_x.get_n_leading() == multi_image_x.get_n_leading() == 2
    ), "timestep_smse_loss: MultiImages must have batch and channel axes"

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
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    reduce: Optional[str] = "mean",
) -> jax.Array:
    """
    Sum of mean squared error loss. The sum is over the channels, the mean is over the spatial
    dimensions. Mean is also taken over batch if reduce == 'mean', or it returns each loss if
    reduce is None.

    args:
        multi_image_x: predicted data, image_blocks are shape (batch,channels,spatial,tensor)
        multi_image_y: target data, image_blocks are shape (batch,channels,spatial,tensor)
        reduce: how to reduce over batch. Either "mean" or None.

    returns:
        the loss value
    """
    reduce_options = {"mean", None}
    assert reduce in reduce_options, f"smse_loss: reduce={reduce} must be one of {reduce_options}"
    assert (
        multi_image_x.get_n_leading() == multi_image_x.get_n_leading() == 2
    ), "smse_loss: MultiImages must have batch and channel axes"

    spatial_size = np.multiply.reduce(multi_image_x.get_spatial_dims())
    loss_per_batch = jnp.zeros(multi_image_x.get_L())
    for image_a, image_b in zip(multi_image_x.values(), multi_image_y.values()):
        loss = jnp.sum((image_a - image_b) ** 2, axis=tuple(range(1, image_a.ndim))) / spatial_size
        loss_per_batch = loss_per_batch + loss

    return jnp.mean(loss_per_batch) if reduce == "mean" else loss_per_batch


def normalized_smse_loss(
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    reduce: str | None = "mean",
    eps: float = 1e-5,
) -> jax.Array:
    """
    Pointwise normalized loss. We find the norm of each channel at each spatial point of the true value
    and divide the tensor by that norm. Then we take the l2 loss, mean over the spatial dimensions, sum
    over the channels, then mean over the batch.

    args:
        multi_image_x: predicted data, image_blocks are shape (batch,channels,spatial,tensor)
        multi_image_y: target data, image_blocks are shape (batch,channels,spatial,tensor)
        reduce: how to reduce over batch. Either "mean" or None.
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

    return jnp.mean(order_loss) if reduce == "mean" else order_loss


def nrmse_per_pixel_loss(
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    reduce: str | None = "mean",
    eps: float = 0,
) -> jax.Array:
    """
    The normalized root mean squared error. The error is relative to the second input per pixel.

    The average is taken over each pixel, and channel. If reduce is 'mean' it is also
    taken over the batch.

    args:
        multi_image_x: predicted data, image_blocks are shape (batch,channels,spatial,tensor)
        multi_image_y: target data, image_blocks are shape (batch,channels,spatial,tensor)
        reduce: how to reduce over batch. Either "mean" or None.
        eps: epsilon to add to the denominator to avoid divide by zero errors

    returns:
        average root mean squared error with respect to the second input.
    """
    reduce_options = {"mean", None}
    assert (
        reduce in reduce_options
    ), f"l1_rel_error: reduce={reduce} must be one of {reduce_options}"
    assert (
        multi_image_x.get_n_leading() == multi_image_y.get_n_leading() == 2
    ), "l1_rel_error: MultiImages must have batch and channel axes"

    batch = multi_image_x.get_L()
    D = multi_image_x.D
    error_per_batch = jnp.zeros((batch, 0))
    for image_a, image_b in zip(multi_image_x.values(), multi_image_y.values()):
        diff_norm = geom.norm(D + 2, image_a - image_b)  # (batch,channels,spatial)
        image_b_norm = geom.norm(D + 2, image_b)  # (batch,channels,spatial)
        rel_error = diff_norm / (image_b_norm + eps)
        # (batch,channels*spatial)
        error_per_batch = jnp.concatenate([error_per_batch, rel_error.reshape((batch, -1))], axis=1)

    error_per_batch = jnp.mean(error_per_batch, axis=1)  # mean over channels, spatial -> (batch,)

    return jnp.mean(error_per_batch) if reduce == "mean" else error_per_batch


def timestep_nrmse_loss(
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    reduce_batch: str | None = "mean",
    eps: float = 0,
    mask: geom.MultiImage | None = None,
    n_steps: int = 1,
    reduce_steps: str | None = "mean",
) -> jax.Array:
    """
    The normalized root mean squared error. This definition follows the standard one used in
    literature where the norm is taken over the entire difference image and reference image
    before doing the division diff / reference.

    This version can also take in a mask to only calculate the loss over the mask, and take in
    n_steps to optionally not reduce over timesteps.

    args:
        multi_image_x: predicted data, image_blocks are shape (batch,channels,spatial,tensor)
        multi_image_y: target data, image_blocks are shape (batch,channels,spatial,tensor)
        reduce: how to reduce over batch. Either "mean" or None.
        eps: epsilon to add to the denominator to avoid divide by zero errors
        mask: mask to only calculate the loss on a particular region of the image. Should be a
            multi image that is the same shape as x and y.

    returns:
        average root mean squared error with respect to the second input.
    """
    reduce_options = {"mean", None}
    assert (
        reduce_batch in reduce_options
    ), f"nrmse_loss: reduce={reduce_batch} must be one of {reduce_options}"
    assert (
        reduce_steps in reduce_options
    ), f"nrmse_loss: reduce_steps={reduce_steps} must be one of {reduce_options}"
    assert (
        multi_image_x.get_n_leading() == multi_image_y.get_n_leading() == 2
    ), "nrmse_loss: MultiImages must have batch and channel axes"

    batch = multi_image_x.get_L()

    # (batch,channels,timesteps,spatial,tensor)
    multi_image_x = multi_image_x.expand(1, n_steps)
    multi_image_y = multi_image_y.expand(1, n_steps)
    mask = None if mask is None else mask.expand(1, n_steps)

    error = jnp.zeros((batch, 0, n_steps))
    for (key_a, image_a), (key_b, image_b) in zip(multi_image_x.items(), multi_image_y.items()):
        assert key_a == key_b
        if mask is not None:
            mask_arr = mask[key_a]
            assert (
                mask_arr.shape == image_a.shape == image_b.shape
            ), f"mask:{mask_arr.shape}, image_a:{image_a.shape}, image_b:{image_b.shape}"
            image_a = jnp.where(mask_arr != 0, image_a, 0)
            image_b = jnp.where(mask_arr != 0, image_b, 0)

        # reshape to (batch,channels,timesteps,spatial*tensor)
        image_a = image_a.reshape(image_a.shape[:3] + (-1,))
        image_b = image_b.reshape(image_b.shape[:3] + (-1,))

        diff_norms = jnp.linalg.norm(image_a - image_b, axis=3)
        target_norms = jnp.linalg.norm(image_b, axis=3)
        error = jnp.concatenate([error, diff_norms / (target_norms + eps)], axis=1)

    if reduce_steps == "mean":
        error = jnp.mean(error, axis=2)  # (batch,channels)

    error = jnp.mean(error, axis=1)  # mean over channels

    if reduce_batch == "mean":
        error = jnp.mean(error, axis=0)  # mean over batch

    return error


def nrmse_loss(
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    reduce: str | None = "mean",
    eps: float = 0,
) -> jax.Array:
    """
    The normalized root mean squared error. This definition follows the standard one used in
    literature where the norm is taken over the entire difference image and reference image
    before doing the division diff / reference. We then take the mean over the channels,
    and then reduce over the batch.

    args:
        multi_image_x: predicted data, image_blocks are shape (batch,channels,spatial,tensor)
        multi_image_y: target data, image_blocks are shape (batch,channels,spatial,tensor)
        reduce: how to reduce over batch. Either "mean" or None.
        eps: epsilon to add to the denominator to avoid divide by zero errors

    returns:
        average root mean squared error with respect to the second input.
    """
    reduce_options = {"mean", None}
    assert reduce in reduce_options, f"nrmse_loss: reduce={reduce} must be one of {reduce_options}"
    assert (
        multi_image_x.get_n_leading() == multi_image_y.get_n_leading() == 2
    ), "nrmse_loss: MultiImages must have batch and channel axes"

    batch = multi_image_x.get_L()

    error_per_batch = jnp.zeros((batch, 0))
    for image_a, image_b in zip(multi_image_x.values(), multi_image_y.values()):
        # reshape to (batch,channels,spatial*tensor)
        image_a = image_a.reshape(image_a.shape[:2] + (-1,))
        image_b = image_b.reshape(image_b.shape[:2] + (-1,))
        diff_norms = jnp.linalg.norm(image_a - image_b, axis=2)
        target_norms = jnp.linalg.norm(image_b, axis=2)
        error_per_batch = jnp.concatenate(
            [error_per_batch, diff_norms / (target_norms + eps)], axis=1
        )

    error_per_batch = jnp.mean(error_per_batch, axis=1)  # (batch,channels) -> (batch,)

    return jnp.mean(error_per_batch) if reduce == "mean" else error_per_batch


def l2_rel_error(
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    reduce: str | None = "mean",
    eps: float = 0,
) -> jax.Array:
    """
    The relative error, taken as a norm over the entire difference image divided by the norm over
    the entire reference image. We then take the mean over the image types, and then reduce over
    the batch.

    args:
        multi_image_x: predicted data, image_blocks are shape (batch,channels,spatial,tensor)
        multi_image_y: target data, image_blocks are shape (batch,channels,spatial,tensor)
        reduce: how to reduce over batch. Either "mean" or None.
        eps: epsilon to add to the denominator to avoid divide by zero errors

    returns:
        average percent relative error with respect to the second input.
    """
    return nrmse_loss(multi_image_x, multi_image_y, reduce, eps) * 100  # convert to percent


def l2_per_pixel_rel_error(
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    reduce: str | None = "mean",
    eps: float = 0,
) -> jax.Array:
    """
    Average per tensor relative error as a percentage. The error is relative to the second input per pixel.

    The average is taken over each pixel, and channel. If reduce is 'mean' it is also
    taken over the batch.

    args:
        multi_image_x: predicted data, image_blocks are shape (batch,channels,spatial,tensor)
        multi_image_y: target data, image_blocks are shape (batch,channels,spatial,tensor)
        reduce: how to reduce over batch. Either "mean" or None.
        eps: epsilon to add to the denominator to avoid divide by zero errors

    returns:
        average percent relative error with respect to the second input.
    """
    return (
        nrmse_per_pixel_loss(multi_image_x, multi_image_y, reduce, eps) * 100
    )  # convert to percent
