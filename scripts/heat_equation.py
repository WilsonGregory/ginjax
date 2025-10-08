from __future__ import annotations
import argparse
import functools as ft
import math
import matplotlib.pyplot as plt
import pathlib
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
    for i, idx in enumerate(idxs):
        print(f"{D},{batch}: {i}/{len(idxs)}")
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


def get_data_d(
    D: int,
    N: int,
    is_torus: bool,
    k: float,
    t: float,
    max_temp: float,
    batch: int,
    key: jax.Array,
    data_dir: pathlib.Path | None,
    data_name: str,
) -> tuple[geom.MultiImage, geom.MultiImage]:
    """
    Get an input, output data pair of heat diffusion after t timestep, diffusion coefficient k.
    The initial data is uniform on the range 0 to max_temp.

    args:
        D: the dimension of the space
        N: sidelength of a cube of data
        is_torus: whether the data is on the torus
        k: diffusion coefficient
        t: size of timestep
        max_temp: max temperature
        batch: number of data points
        key: key for randomness
        data_dir: directory
        data_name: name where to save the data

    returns:
        input multi image, output multi image
    """
    if data_dir is None:
        raise ValueError

    data_dir = (
        data_dir
        / f"{data_name}_N{N}_istorus{int(is_torus)}_n{batch}_k{k}_t{t}_maxtemp{max_temp}.npy"
    )

    if data_dir.is_file():
        dataset = jnp.load(data_dir, allow_pickle=True).item()
        x0 = dataset["x0"]
        xt = dataset["xt"]
    else:
        key, subkey = random.split(key)
        x0 = random.uniform(subkey, shape=(batch,) + (N,) * D, maxval=max_temp)
        xt = heat_step(D, x0, t, k, is_torus)

        print(f"saving at {data_dir}...")
        jnp.save(data_dir, {"x0": x0, "xt": xt})

    x0_img = geom.MultiImage({(0, 0): x0[:, None]}, D, is_torus)
    xt_img = geom.MultiImage({(0, 0): xt[:, None]}, D, is_torus)

    return x0_img, xt_img


def get_data(
    D: int,
    N: int,
    is_torus: bool,
    n_train: int,
    n_val: int,
    n_test: int,
    key: jax.Array,
    data_dir: str | None,
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
    k = 2
    t = 1
    data_dir_path = pathlib.Path(data_dir) if data_dir is not None else None

    key, subkey1, subkey2, subkey3, subkey4, subkey5 = random.split(key, num=6)
    train_x0, train_xt = get_data_d(
        D, N, is_torus, k, t, max_temp, n_train, subkey1, data_dir_path, f"train_D{D}"
    )
    val_x0, val_xt = get_data_d(
        D, N, is_torus, k, t, max_temp, n_val, subkey2, data_dir_path, f"val_D{D}"
    )
    test_x0, test_xt = get_data_d(
        D, N, is_torus, k, t, max_temp, n_test, subkey3, data_dir_path, f"test_D{D}"
    )
    test_d2_x0, test_d2_xt = get_data_d(
        2, N, is_torus, k, t, max_temp, n_test, subkey4, data_dir_path, "test_d2"
    )
    test_d3_x0, test_d3_xt = get_data_d(
        3, N, is_torus, k, t, max_temp, n_test, subkey5, data_dir_path, "test_d3"
    )

    return (
        train_x0,
        train_xt,
        val_x0,
        val_xt,
        test_x0,
        test_xt,
        test_d2_x0,
        test_d2_xt,
        test_d3_x0,
        test_d3_xt,
    )


class TwoLayerModel(models.MultiImageModule):
    layers: list[models.ConvBlock]

    D: int = eqx.field(static=True)
    input_keys: geom.Signature = eqx.field(static=True)
    target_keys: geom.Signature = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    use_bias: bool | str = eqx.field(static=True)

    def __init__(
        self: Self,
        input_keys: geom.Signature,
        target_keys: geom.Signature,
        conv_filters: geom.MultiImage,
        depth: int,
        use_bias: bool | str,
        key: jax.Array,
    ) -> None:
        self.D = conv_filters.D
        self.input_keys = input_keys
        self.target_keys = target_keys
        self.depth = depth
        self.use_bias = use_bias

        mid_keys = geom.signature_union(input_keys, target_keys, depth)

        key, subkey1, subkey2 = random.split(key, num=3)
        self.layers = [
            models.ConvBlock(
                D, input_keys, mid_keys, use_bias, "gelu", True, conv_filters, key=subkey1
            ),
            models.ConvBlock(
                D, mid_keys, target_keys, use_bias, None, True, conv_filters, key=subkey2
            ),
        ]

    @staticmethod
    def _scale_weights(
        weights: dict[tuple[tuple[bool, ...], int], dict[tuple[tuple[bool, ...], int], jax.Array]],
        old_filters: geom.MultiImage,
        new_filters: geom.MultiImage,
    ) -> dict[tuple[tuple[bool, ...], int], dict[tuple[tuple[bool, ...], int], jax.Array]]:
        new_weights = {}

        for (in_k, in_p), in_weights in weights.items():
            new_weights[(in_k, in_p)] = {}
            for (out_k, out_p), old_weights_block in in_weights.items():
                filter_key = ((False,) * (len(in_k) + len(out_k)), (in_p + out_p) % 2)

                weights_mul = old_weights_block.reshape(
                    old_weights_block.shape + (1,) * old_filters.D
                )
                old_weights_sum = jnp.sum(
                    old_filters[filter_key][None, None] * weights_mul,
                    axis=tuple(range(2, 3 + old_filters.D)),
                )  # (out_c,in_c)

                weights_mul = old_weights_block.reshape(
                    old_weights_block.shape + (1,) * new_filters.D
                )
                new_weights_sum = jnp.sum(
                    new_filters[filter_key][None, None] * weights_mul,
                    axis=tuple(range(2, 3 + new_filters.D)),
                )  # (out_c, in_c)

                ratios = (old_weights_sum / new_weights_sum)[..., None]

                new_weights[(in_k, in_p)][(out_k, out_p)] = old_weights_block * ratios

        return new_weights

    def convertD(
        self: Self,
        old_conv_filters: geom.MultiImage,
        conv_filters: geom.MultiImage,
        rescale: bool,
        key: jax.Array,
    ) -> Self:
        new_model = self.__class__(
            self.input_keys, self.target_keys, conv_filters, self.depth, self.use_bias, key
        )
        is_conv = lambda n: isinstance(n, ml.ConvContract)
        get_weights = lambda m: [
            x.weights for x in jax.tree_util.tree_leaves(m, is_leaf=is_conv) if is_conv(x)
        ]
        weights = get_weights(self)
        if rescale:
            new_weights = [
                TwoLayerModel._scale_weights(weight, old_conv_filters, conv_filters)
                for weight in weights
            ]
        else:
            new_weights = weights

        new_model = eqx.tree_at(get_weights, new_model, new_weights)

        get_biases = lambda m: [
            x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_conv) if is_conv(x)
        ]
        new_model = eqx.tree_at(get_biases, new_model, get_biases(self))

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
    conv_filters_d1: geom.MultiImage,
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
    trained_model_d2_unscaled = trained_model.convertD(
        conv_filters_d1, conv_filters_d2, False, subkey
    )

    key, subkey = random.split(key)
    test_loss_d2 = ml.map_loss_in_batches(
        map_and_loss,
        trained_model_d2_unscaled,
        test_d2_X,
        test_d2_Y,
        batch_size,
        subkey,
        aux_data=batch_stats,
    )
    print(f"Test Loss D=2: {test_loss_d2}")

    key, subkey = random.split(key)
    trained_model_d2 = trained_model.convertD(conv_filters_d1, conv_filters_d2, True, subkey)

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
    print(f"Test Loss D=2 (rescaled): {test_loss_d2}")

    key, subkey = random.split(key)
    trained_model_d3_unscaled = trained_model.convertD(
        conv_filters_d1, conv_filters_d3, False, subkey
    )

    key, subkey = random.split(key)
    test_loss_d3 = ml.map_loss_in_batches(
        map_and_loss,
        trained_model_d3_unscaled,
        test_d3_X,
        test_d3_Y,
        batch_size,
        subkey,
        aux_data=batch_stats,
    )
    print(f"Test Loss D=3: {test_loss_d3}")
    test_loss_d3 = jnp.ones_like(test_loss_d2)

    key, subkey = random.split(key)
    trained_model_d3 = trained_model.convertD(conv_filters_d1, conv_filters_d3, True, subkey)

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
    print(f"Test Loss D=3 (rescaled): {test_loss_d3}")
    test_loss_d3 = jnp.ones_like(test_loss_d2)

    if images_dir is not None:
        x0 = geom.GeometricImage(test_d2_X[(), 0][0, 0], 0, test_d2_X.D, test_d2_X.is_torus)
        xt = geom.GeometricImage(test_d2_Y[(), 0][0, 0], 0, test_d2_Y.D, test_d2_Y.is_torus)

        if train_X.D == 2:
            xt_pred_multi_image, _ = trained_model(test_d2_X.get_one(keepdims=False), batch_stats)
        else:
            xt_pred_multi_image, _ = trained_model_d2(
                test_d2_X.get_one(keepdims=False), batch_stats
            )
        xt_pred = geom.GeometricImage(
            xt_pred_multi_image[(), 0][0], 0, test_d2_Y.D, test_d2_Y.is_torus
        )

        _, axes = plt.subplots(1, 4, figsize=(6 * 4, 6 * 1))
        max_temp = 5
        x0.plot(axes[0], "input", vmin=0, vmax=max_temp, colorbar=True, cmap="hot")
        xt.plot(axes[1], "truth", vmin=0, vmax=max_temp, colorbar=True, cmap="hot")
        xt_pred.plot(axes[2], "model prediction", vmin=0, vmax=max_temp, colorbar=True, cmap="hot")
        diff = xt - xt_pred
        diff_max = float(jnp.max(jnp.abs(diff.data)))
        diff.plot(axes[3], "difference", vmin=-diff_max, vmax=diff_max, colorbar=True)

        plt.savefig(f"{images_dir}/trained_D{train_X.D}_converted_D{test_d2_X.D}.png")

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
D = 1
N = 128

key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)

key, subkey = random.split(key)
print("Generating data...", end="", flush=True)
t_start = time.time()
data = get_data(D, N, True, args.n_train, args.n_val, args.n_test, subkey, args.data)
print(f"done. ({time.time() - t_start:.2f}s)", flush=True)

input_keys = data[0].get_signature()
output_keys = data[1].get_signature()

# start with basic 3x3 scalar, vector, and 2nd order tensor images
group_actions_d1 = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    Ms=[3],
    ks=[0],
    parities=[0],
    D=D,
    operators=group_actions_d1,
    scale="gaussian",
    exclude_corners=True,
)

group_actions_d2 = geom.make_all_operators(2)
conv_filters_d2 = geom.get_invariant_filters(
    Ms=[3],
    ks=[0],
    parities=[0],
    D=2,
    operators=group_actions_d2,
    scale="gaussian",
    exclude_corners=True,
)

group_actions_d3 = geom.make_all_operators(3)
conv_filters_d3 = geom.get_invariant_filters(
    Ms=[3],
    ks=[0],
    parities=[0],
    D=3,
    operators=group_actions_d3,
    scale="gaussian",
    exclude_corners=True,
)

train_kwargs = {
    "conv_filters_d1": conv_filters,
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
            "model": TwoLayerModel(input_keys, output_keys, conv_filters, 10, "auto", key=subkey),
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
