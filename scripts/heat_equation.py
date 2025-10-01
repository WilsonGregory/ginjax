from __future__ import annotations
import argparse
import functools as ft
import math
import matplotlib.pyplot as plt
import time
from typing_extensions import Self

import jax
import jax.numpy as jnp
import jax.random as random
import equinox as eqx
import optax

import ginjax.geometric as geom
import ginjax.ml as ml
import ginjax.models as models
import ginjax.utils as utils


def heat_step(D: int, x0: jax.Array, t: float, k: float, is_torus: bool) -> jax.Array:
    """
    Given the initial temperature, time, and diffusion coefficient, calculate the new heat field.

    args:
        D: the dimension
        x0: the starting temperature array, shape (batch, spatial)
        t: the length of the timestep
        k: diffusion coefficient
        is_torus: whether the data lives on the torus

    returns:
        the heat field after time t
    """
    assert t >= 0
    if t == 0:
        return x0

    batch = len(x0)
    spatial_dims = x0.shape[1:]
    x0_flat = x0.reshape((batch, -1))
    meshgrid_dims = (jnp.arange(N) for N in spatial_dims)

    # (spatial_size,D)
    idxs = jnp.stack(jnp.meshgrid(*meshgrid_dims, indexing="ij"), axis=-1).reshape((-1, D))
    x1 = []
    for idx in idxs:
        # (spatial_size,D)
        if is_torus:
            idxs_diff = jnp.abs(idxs - idx[None])
            idxs_diff_wrapped = jnp.stack(spatial_dims).reshape((1, D)) - idxs_diff
            dist = jnp.where(idxs_diff < idxs_diff_wrapped, idxs_diff, idxs_diff_wrapped)
        else:
            dist = idxs - idx[None]

        # (1,spatial_size)
        heat_kernel = jnp.exp((jnp.linalg.norm(dist, axis=1) ** 2) / (-4 * k * t))[None]

        x1.append(jnp.sum(heat_kernel * x0_flat, axis=1) / ((4 * jnp.pi * k * t) ** (D / 2)))

    return jnp.stack(x1, axis=1).reshape(x0.shape)


def get_data(
    D: int, N: int, is_torus: bool, n_train: int, n_val: int, n_test: int, key: jax.Array
) -> tuple[
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
]:
    """
    Get an input, output data pair of heat diffusion after 1 timestep, diffusion constant 10.
    The initial data is uniform on the range 0 to 5.

    args:
        D: the dimension of the space
        N: sidelength of a cube of data
        is_torus: whether the data is on the torus
        batch: number of data points
        key: key for randomness

    returns:
        input multi image, output multi image
    """
    max_temp = 5
    k = 1
    t = 1

    key, subkey1, subkey2, subkey3 = random.split(key, num=4)
    train_x0 = random.uniform(subkey1, shape=(n_train,) + (N,) * D, maxval=max_temp)
    train_xt = heat_step(D, train_x0, t, k, is_torus)
    val_x0 = random.uniform(subkey2, shape=(n_val,) + (N,) * D, maxval=max_temp)
    val_xt = heat_step(D, val_x0, t, k, is_torus)
    test_x0 = random.uniform(subkey3, shape=(n_test,) + (N,) * D, maxval=max_temp)
    test_xt = heat_step(D, test_x0, t, k, is_torus)

    train_x0_img = geom.MultiImage({(0, 0): train_x0[:, None]}, D, is_torus)
    train_xt_img = geom.MultiImage({(0, 0): train_xt[:, None]}, D, is_torus)
    val_x0_img = geom.MultiImage({(0, 0): val_x0[:, None]}, D, is_torus)
    val_xt_img = geom.MultiImage({(0, 0): val_xt[:, None]}, D, is_torus)
    test_x0_img = geom.MultiImage({(0, 0): test_x0[:, None]}, D, is_torus)
    test_xt_img = geom.MultiImage({(0, 0): test_xt[:, None]}, D, is_torus)

    return train_x0_img, train_xt_img, val_x0_img, val_xt_img, test_x0_img, test_xt_img


class TwoLayerModel(models.MultiImageModule):
    layers: list[models.ConvBlock]

    def __init__(
        self: Self,
        input_keys: geom.Signature,
        target_keys: geom.Signature,
        conv_filters: geom.MultiImage,
        depth: int,
        key: jax.Array,
    ) -> None:
        D = conv_filters.D
        mid_keys = geom.signature_union(input_keys, target_keys, depth)

        key, subkey1, subkey2 = random.split(key, num=3)
        self.layers = [
            models.ConvBlock(
                D,
                input_keys,
                mid_keys,
                "auto",
                "gelu",
                True,
                conv_filters,
                key=subkey1,
            ),
            models.ConvBlock(
                D, mid_keys, target_keys, "auto", None, True, conv_filters, key=subkey2
            ),
        ]

    def __call__(
        self: Self, x: geom.MultiImage, aux_data: eqx.nn.State | None = None
    ) -> tuple[geom.MultiImage, eqx.nn.State | None]:
        for layer in self.layers:
            x, aux_data = layer(x, aux_data)

        return x, aux_data


@eqx.filter_jit
def map_and_loss(
    model: models.MultiImageModule,
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    aux_data: eqx.nn.State | None = None,
) -> tuple[jax.Array, eqx.nn.State | None]:
    pred_y, aux_data = jax.vmap(model, in_axes=(0, None), out_axes=(0, None))(
        multi_image_x, aux_data
    )
    return ml.smse_loss(pred_y, multi_image_y), aux_data


def train_and_eval(
    data: tuple[geom.MultiImage, ...],
    key: jax.Array,
    model_name: str,
    model: models.MultiImageModule,
    lr: float,
    batch_size: int,
    epochs: int,
    save_model: str | None,
    load_model: str | None,
    images_dir: str | None,
    has_aux: bool = False,
    verbose: int = 1,
    is_wandb: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    train_X, train_Y, val_X, val_Y, test_X, test_Y = data
    batch_stats = eqx.nn.State(model) if has_aux else None

    print(f"Model params: {models.count_params(model):,}")

    if load_model is None:
        steps_per_epoch = int(math.ceil(train_X.get_L() / batch_size))
        key, subkey = random.split(key)
        trained_model, batch_stats, train_loss, val_loss = ml.train(
            train_X,
            train_Y,
            map_and_loss,
            model,
            subkey,
            stop_condition=ml.EpochStop(epochs, verbose=verbose),
            batch_size=batch_size,
            optimizer=optax.adamw(
                optax.warmup_cosine_decay_schedule(
                    1e-8, lr, 5 * steps_per_epoch, epochs * steps_per_epoch, 1e-7
                ),
                weight_decay=1e-5,
            ),
            validation_X=val_X,
            validation_Y=val_Y,
            aux_data=batch_stats,
            is_wandb=is_wandb,
        )
        assert isinstance(train_loss, jax.Array)
        assert isinstance(val_loss, jax.Array)

        if save_model is not None:
            # TODO: need to save batch_stats as well
            ml.save(
                f"{save_model}{model_name}_L{train_X.get_L()}_e{epochs}_model.eqx", trained_model
            )
    else:
        trained_model = ml.load(
            f"{load_model}{model_name}_L{train_X.get_L()}_e{epochs}_model.eqx", model
        )

        key, subkey1, subkey2 = random.split(key, num=3)
        train_loss = ml.map_loss_in_batches(
            map_and_loss,
            trained_model,
            train_X,
            train_Y,
            batch_size,
            subkey1,
            aux_data=batch_stats,
        )
        val_loss = ml.map_loss_in_batches(
            map_and_loss,
            trained_model,
            val_X,
            val_Y,
            batch_size,
            subkey2,
            aux_data=batch_stats,
        )

    key, subkey = random.split(key)
    test_loss = ml.map_loss_in_batches(
        map_and_loss,
        trained_model,
        test_X,
        test_Y,
        batch_size,
        subkey,
        aux_data=batch_stats,
    )
    print(f"Test Loss: {test_loss}")

    if images_dir is not None:
        x0 = geom.GeometricImage(test_X[(), 0][0, 0], 0, test_X.D, test_X.is_torus)
        xt = geom.GeometricImage(test_Y[(), 0][0, 0], 0, test_Y.D, test_Y.is_torus)

        xt_pred_multi_image, _ = trained_model(test_X.get_one(keepdims=False), batch_stats)
        xt_pred = geom.GeometricImage(xt_pred_multi_image[(), 0][0], 0, test_Y.D, test_Y.is_torus)

        fig, axes = plt.subplots(1, 4, figsize=(6 * 4, 6 * 1))
        max_temp = 5
        x0.plot(axes[0], "input", vmin=0, vmax=max_temp, colorbar=True, cmap="hot")
        xt.plot(axes[1], "truth", vmin=0, vmax=max_temp, colorbar=True, cmap="hot")
        xt_pred.plot(axes[2], "model prediction", vmin=0, vmax=max_temp, colorbar=True, cmap="hot")
        (xt - xt_pred).plot(axes[3], "difference", colorbar=True)

        plt.savefig(f"../images/heat_equation/sample_model.png")

    return train_loss, val_loss, test_loss


def handleArgs() -> argparse.Namespace:
    parser = utils.get_common_parser()
    # need do to --wandb to activate, also need --wandb-entity your_wandb_name_here
    parser.add_argument(
        "--wandb-project", help="the wandb project", type=str, default="heat-equation"
    )

    return parser.parse_args()


# MAIN
args = handleArgs()
D = 2
N = 128

key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)

key, subkey = random.split(key)
print("Generating data...", end="", flush=True)
t_start = time.time()
data = get_data(D, N, True, args.n_train, args.n_val, args.n_test, subkey)
print(f"done. ({time.time() - t_start:.2f}s)", flush=True)

input_keys = data[0].get_signature()
output_keys = data[1].get_signature()

# start with basic 3x3 scalar, vector, and 2nd order tensor images
group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    Ms=[3], ks=[0], parities=[0], D=D, operators=group_actions
)
assert conv_filters is not None

train_kwargs = {
    "batch_size": args.batch,
    "epochs": args.epochs,
    "save_model": args.save_model,
    "load_model": args.load_model,
    "images_dir": args.images_dir,
    "verbose": args.verbose,
    "is_wandb": args.wandb,
}

key, subkey = random.split(key)
model_list = [
    (
        "two_layer",
        train_and_eval,
        {
            "model": TwoLayerModel(
                input_keys,
                output_keys,
                conv_filters,
                10,
                key=subkey,
            ),
            "lr": 1e-2,
            **train_kwargs,
        },
    ),
]

key, subkey = random.split(key)
# Use this for benchmarking the models with known learning rates.
results = ml.benchmark(
    lambda _: data,
    model_list,
    subkey,
    "",
    [0],
    benchmark_type=ml.BENCHMARK_NONE,
    num_trials=args.n_trials,
    num_results=3,
    is_wandb=args.wandb,
    wandb_project=args.wandb_project,
    wandb_entity=args.wandb_entity,
)

# T = 5
# fig, axes = plt.subplots(1, T, figsize=(6 * T, 6 * 1))

# for t in range(T):
#     x_t = heat_step(D, x0, t, k)
#     xt_img = geom.GeometricImage(x_t[0], 0, D, False)
#     xt_img.plot(
#         axes[t],
#         f"t={t}, mean={jnp.mean(x_t[0]):.3e}",
#         vmin=0,
#         vmax=max_temp,
#         colorbar=True,
#         cmap="hot",
#     )

# plt.savefig(f"../images/heat_equation/sample.png")
