# generate gravitational field
from __future__ import annotations
import time
from typing_extensions import Self

import jax.numpy as jnp
import jax.random as random
import jax
import equinox as eqx

import ginjax.geometric as geom
import ginjax.ml as ml
import ginjax.models as models


def get_1d_sine_data(
    N: int, t: float, n_batch: int, key: jax.Array
) -> tuple[geom.MultiImage, geom.MultiImage]:
    # function is u(x,t) = sin(x-t), du_dt(x,t) = -cos(x-t)
    D = 1
    is_torus = True
    n_cycles = 3

    initial_shift = random.uniform(key, shape=(n_batch, 1)) * (N / 3)  # (0,2pi)

    x_range = jnp.linspace(0, n_cycles * 2 * jnp.pi, num=N, endpoint=False)  # [0,6pi]
    u0 = jnp.sin(x_range[None] - initial_shift)  # (batch,spatial) currently batch is 1
    du_dt0 = -jnp.cos(x_range[None] - initial_shift)  # (batch,spatial)

    ut = jnp.sin((x_range[None] - initial_shift) - t)  # (batch,spatial)

    # make it one input channel for simplicity
    x0_img = geom.MultiImage({((), 0): u0[:, None]}, D, is_torus)
    xt_img = geom.MultiImage({((), 0): ut[:, None]}, D, is_torus)

    return x0_img, xt_img


def get_sine_data(
    D: int, N: int, t: float, n_batch: int, key: jax.Array
) -> tuple[geom.MultiImage, geom.MultiImage]:
    assert D >= 1
    spatial_dims = (N,) * D
    x0_1d, xt_1d = get_1d_sine_data(N, t, n_batch, key)
    x0_1d_data = x0_1d[((), 0)]
    xt_1d_data = xt_1d[((), 0)]

    new_shape = x0_1d_data.shape[:2] + spatial_dims
    x0 = geom.MultiImage(
        {((), 0): jnp.full(new_shape, x0_1d_data.reshape(x0_1d_data.shape + (1,) * (D - 1)))},
        D,
        True,
    )

    new_shape = xt_1d_data.shape[:2] + spatial_dims
    xt = geom.MultiImage(
        {((), 0): jnp.full(new_shape, xt_1d_data.reshape(xt_1d_data.shape + (1,) * (D - 1)))},
        D,
        True,
    )

    return x0, xt


class Mapper:
    """
    Functor for map_and_loss in train, map_loss_in_batches, etc, where arguments can be provided
    beforehand. In this case, it is useful for smse vs relative error, and whether to learn the
    residual or not.
    """

    residual: bool
    nrmse: bool
    smse: bool
    l2_rel: bool

    def __init__(
        self: Self,
        residual: bool = False,
        nrmse: bool = True,
        smse: bool = False,
        l2_rel: bool = False,
        eps: float = 0,
    ) -> None:
        """
        Docstring for __init__

        args:
            residual: Whether the network should learn the residual, defaults to False
            nrmse: Whether __call__ the normalized root mean squared error loss, defaults to True
            smse: Whether __call__ returns the smse loss, defaults to False
            l2_rel: Whether __call__ returns the l2 relative error, defaults to False
            eps: epsilon value to use for nrmse and lr_rel, avoid dividing by 0
        """
        assert nrmse or smse or l2_rel, "At least one of nrmse, smse, or l2_rel must be true."
        self.residual = residual
        self.nrmse = nrmse
        self.smse = smse
        self.l2_rel = l2_rel
        self.eps = eps

    @eqx.filter_jit
    def map(
        self: Self,
        model: models.MultiImageModule,
        multi_image_x: geom.MultiImage,
        aux_data: eqx.nn.State | None = None,
    ) -> tuple[geom.MultiImage, eqx.nn.State | None]:
        """
        The map function using the model and the input data.
        """
        out, aux_data = jax.vmap(model, in_axes=(0, None), out_axes=(0, None), axis_name="batch")(
            multi_image_x, aux_data
        )

        if self.residual:
            # add the last timestep to the residual
            pred_y = out.empty()
            for ((k, parity), img_in), img_resid in zip(multi_image_x.items(), out.values()):
                pred_y.append(k, parity, img_in[:, -1:] + img_resid)

            return pred_y, aux_data
        else:
            return out, aux_data

    @eqx.filter_jit
    def __call__(
        self: Self,
        model: models.MultiImageModule,
        multi_image_x: geom.MultiImage,
        multi_image_y: geom.MultiImage,
        aux_data: eqx.nn.State | None = None,
    ) -> tuple[jax.Array, eqx.nn.State | None]:
        """
        Equivalent of the map_and_loss function.
        """
        pred_y, aux_data = self.map(model, multi_image_x, aux_data)

        losses = []
        if self.nrmse:
            losses.append(ml.nrmse_loss(pred_y, multi_image_y, eps=self.eps))

        if self.smse:
            losses.append(ml.smse_loss(pred_y, multi_image_y))

        if self.l2_rel:
            losses.append(ml.l2_rel_error(pred_y, multi_image_y, eps=self.eps))

        return jnp.squeeze(jnp.stack(losses)), aux_data


N = 64
t = 4.3
n_batch = 8
test_D = 2
key = random.PRNGKey(time.time_ns())

M = 5
max_pixel_l1 = 2
gaussian_filters_dict = {}
for D in range(1, 4):
    group_actions = geom.make_all_operators(D)
    gaussian_filters_dict[D] = geom.get_invariant_filters(
        Ms=[M],
        ks=[0, 2],
        parities=[0],
        D=D,
        operators=group_actions,
        scale=geom.FilterScaling.NORMALIZE,
        max_pixel_l1=max_pixel_l1,
        combine_equal_l1=True,
    )

for train_D in [1, 2]:
    for test_D in [2, 3]:
        D_increase = test_D - train_D
        if D_increase == 0:
            continue

        print(f"(train_D,test_D): ({train_D}, {test_D})")

        key, subkey = random.split(key)
        data_1d_x0, data_1d_xt = get_sine_data(train_D, N, t, n_batch, subkey)
        data_testD_x0, data_testD_xt = get_sine_data(
            test_D, N, t, n_batch, subkey
        )  # same random key, should gen same data

        for i in range(N**D_increase):
            image_testD_x0 = data_testD_x0[((), 0)].reshape(
                data_testD_x0[((), 0)].shape[: 2 + train_D] + (-1,)
            )
            image_testD_xt = data_testD_xt[((), 0)].reshape(
                data_testD_xt[((), 0)].shape[: 2 + train_D] + (-1,)
            )
            assert jnp.allclose(data_1d_x0[((), 0)], image_testD_x0[..., i])
            assert jnp.allclose(data_1d_xt[((), 0)], image_testD_xt[..., i])

        print("Data embedding holds!")

        # initialize a random network
        key, subkey = random.split(key)
        model_1d = models.SimpleConvSeries(
            data_1d_x0.get_signature(),
            data_1d_xt.get_signature(),
            gaussian_filters_dict[train_D],
            width=10,
            depth=2,
            use_bias=False,
            key=subkey,
        )

        # make a converted test_D model
        key, subkey = random.split(key)
        model_testD = model_1d.convertD(
            gaussian_filters_dict[test_D], geom.Rescaling.COMPATIBILITY, subkey
        )

        out_1d, _ = Mapper().map(model_1d, data_1d_x0)
        out_testD, _ = Mapper().map(model_testD, data_testD_x0)

        print("Now testing model compatibility...")

        for i in range(N ** (test_D - 1)):
            image_testD_out = out_testD[((), 0)].reshape(
                out_testD[((), 0)].shape[: 2 + train_D] + (-1,)
            )

            first = out_1d[((), 0)]
            second = image_testD_out[..., i]
            assert jnp.allclose(first, second, rtol=1e-3, atol=1e-3), jnp.max(
                jnp.abs(first - second)
            )

        print("Model compatibility holds!")

print("Passed all tests!")
