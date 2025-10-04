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

    test_d2_x0 = random.uniform(subkey3, shape=(n_test,) + (N,) * 2, maxval=max_temp)
    test_d2_xt = heat_step(2, test_d2_x0, t, k, is_torus)
    test_d3_x0 = random.uniform(subkey3, shape=(n_test,) + (N,) * 3, maxval=max_temp)
    test_d3_xt = test_d3_x0  # its slow
    # test_p2_xt = heat_step(3, test_d3_x0, t, k, is_torus)

    train_x0_img = geom.MultiImage({(0, 0): train_x0[:, None]}, D, is_torus)
    train_xt_img = geom.MultiImage({(0, 0): train_xt[:, None]}, D, is_torus)
    val_x0_img = geom.MultiImage({(0, 0): val_x0[:, None]}, D, is_torus)
    val_xt_img = geom.MultiImage({(0, 0): val_xt[:, None]}, D, is_torus)
    test_x0_img = geom.MultiImage({(0, 0): test_x0[:, None]}, D, is_torus)
    test_xt_img = geom.MultiImage({(0, 0): test_xt[:, None]}, D, is_torus)

    test_d2_x0_img = geom.MultiImage({(0, 0): test_d2_x0[:, None]}, 2, is_torus)
    test_d2_xt_img = geom.MultiImage({(0, 0): test_d2_xt[:, None]}, 2, is_torus)
    test_d3_x0_img = geom.MultiImage({(0, 0): test_d3_x0[:, None]}, 3, is_torus)
    test_d3_xt_img = geom.MultiImage({(0, 0): test_d3_xt[:, None]}, 3, is_torus)

    return (
        train_x0_img,
        train_xt_img,
        val_x0_img,
        val_xt_img,
        test_x0_img,
        test_xt_img,
        test_d2_x0_img,
        test_d2_xt_img,
        test_d3_x0_img,
        test_d3_xt_img,
    )


class TwoLayerModel(models.MultiImageModule):
    layers: list[models.ConvBlock]

    D: int
    input_keys: geom.Signature
    target_keys: geom.Signature
    depth: int

    def __init__(
        self: Self,
        input_keys: geom.Signature,
        target_keys: geom.Signature,
        conv_filters: geom.MultiImage,
        depth: int,
        key: jax.Array,
    ) -> None:
        self.D = conv_filters.D
        self.input_keys = input_keys
        self.target_keys = target_keys
        self.depth = depth

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

    def convertD(self: Self, conv_filters: geom.MultiImage, key: jax.Array) -> Self:
        new_model = self.__class__(self.input_keys, self.target_keys, conv_filters, self.depth, key)

        # sets all the new weights equal to the old weights?
        # this also resets the conv_filters
        is_leaf = lambda n: eqx.is_array(n) or isinstance(n, geom.MultiImage)
        get_weights = lambda m: jax.tree_util.tree_leaves(m, is_leaf)
        old_weights = get_weights(self)
        # we still want to use the new conv_filters, so set those back
        # this wouldn't work if there are upsample filters as well
        old_weights = [conv_filters if isinstance(x, geom.MultiImage) else x for x in old_weights]

        new_model = eqx.tree_at(get_weights, new_model, old_weights, is_leaf=is_leaf)

        return new_model

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
    conv_filters_d2: geom.MultiImage,
    conv_filters_d3: geom.MultiImage,
    batch_size: int,
    epochs: int,
    save_model: str | None,
    load_model: str | None,
    images_dir: str | None,
    has_aux: bool = False,
    verbose: int = 1,
    is_wandb: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    (
        train_X,
        train_Y,
        val_X,
        val_Y,
        test_d1_X,
        test_d1_Y,
        test_d2_X,
        test_d2_Y,
        test_d3_X,
        test_d3_Y,
    ) = data
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
    test_loss_d1 = ml.map_loss_in_batches(
        map_and_loss,
        trained_model,
        test_d1_X,
        test_d1_Y,
        batch_size,
        subkey,
        aux_data=batch_stats,
    )
    print(f"Test Loss D=1: {test_loss_d1}")

    assert isinstance(trained_model, TwoLayerModel)
    key, subkey = random.split(key)
    trained_model_d2 = trained_model.convertD(conv_filters_d2, subkey)

    key, subkey = random.split(key)
    test_loss_d2 = ml.map_loss_in_batches(
        map_and_loss,
        trained_model_d2,
        test_d2_X,
        test_d2_Y,
        batch_size,
        subkey,
        aux_data=batch_stats,
    )
    print(f"Test Loss D=2: {test_loss_d2}")

    key, subkey = random.split(key)
    trained_model_d3 = trained_model.convertD(conv_filters_d3, subkey)

    key, subkey = random.split(key)
    test_loss_d3 = ml.map_loss_in_batches(
        map_and_loss,
        trained_model_d3,
        test_d3_X,
        test_d3_Y,
        batch_size,
        subkey,
        aux_data=batch_stats,
    )
    print(f"Test Loss D=3: {test_loss_d3}")

    if images_dir is not None:
        x0 = geom.GeometricImage(test_d2_X[(), 0][0, 0], 0, test_d2_X.D, test_d2_X.is_torus)
        xt = geom.GeometricImage(test_d2_Y[(), 0][0, 0], 0, test_d2_Y.D, test_d2_Y.is_torus)

        xt_pred_multi_image, _ = trained_model(test_d2_X.get_one(keepdims=False), batch_stats)
        xt_pred = geom.GeometricImage(
            xt_pred_multi_image[(), 0][0], 0, test_d2_Y.D, test_d2_Y.is_torus
        )

        _, axes = plt.subplots(1, 4, figsize=(6 * 4, 6 * 1))
        max_temp = 5
        x0.plot(axes[0], "input", vmin=0, vmax=max_temp, colorbar=True, cmap="hot")
        xt.plot(axes[1], "truth", vmin=0, vmax=max_temp, colorbar=True, cmap="hot")
        xt_pred.plot(axes[2], "model prediction", vmin=0, vmax=max_temp, colorbar=True, cmap="hot")
        (xt - xt_pred).plot(axes[3], "difference", colorbar=True)

        plt.savefig(f"../images/heat_equation/sample_model_D{test_d2_X.D}.png")

    return train_loss, val_loss, test_loss_d1, test_loss_d2, test_loss_d3


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
group_actions_d1 = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    Ms=[3], ks=[0], parities=[0], D=D, operators=group_actions_d1, exclude_corners=True
)

group_actions_d2 = geom.make_all_operators(2)
conv_filters_d2 = geom.get_invariant_filters(
    Ms=[3], ks=[0], parities=[0], D=2, operators=group_actions_d2, exclude_corners=True
)

group_actions_d3 = geom.make_all_operators(3)
conv_filters_d3 = geom.get_invariant_filters(
    Ms=[3], ks=[0], parities=[0], D=3, operators=group_actions_d3, exclude_corners=True
)

train_kwargs = {
    "conv_filters_d2": conv_filters_d2,
    "conv_filters_d3": conv_filters_d3,
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
    num_results=5,
    is_wandb=args.wandb,
    wandb_project=args.wandb_project,
    wandb_entity=args.wandb_entity,
)
