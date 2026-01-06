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
        if (i % (len(idxs) // 10)) == 0:
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
        x0 = random.uniform(subkey, shape=(batch,) + (N,) * D, minval=-max_temp, maxval=max_temp)
        xt = heat_step(D, x0, t, k, is_torus)

        print(f"saving at {data_dir}...")
        jnp.save(data_dir, {"x0": x0, "xt": xt})

    x0_img = geom.MultiImage({(0, 0): x0[:, None]}, D, is_torus)
    xt_img = geom.MultiImage({(0, 0): xt[:, None]}, D, is_torus)

    return x0_img, xt_img


def get_data(
    train_D: int,
    test_D_range: tuple[int, ...],
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
    list[geom.MultiImage],
    list[geom.MultiImage],
]:
    """
    Get an input, output data pair of heat diffusion after 1 timestep, diffusion constant 10.
    The initial data is uniform on the range 0 to 5.

    args:
        D: the dimension of the space
        N: sidelength of a cube of data
        is_torus: whether the data is on the torus
        n_train: number of training data points
        n_val: number of validation data points
        ntest: number of test data points for each test dimension
        key: key for randomness
        data_dir: location to save or load the data from

    returns:
        input multi image, output multi image
    """
    max_temp = math.sqrt(3)
    k = 1
    t = 1
    data_dir_path = pathlib.Path(data_dir) if data_dir is not None else None

    key, subkey1, subkey2 = random.split(key, num=3)
    train_x0, train_xt = get_data_d(
        train_D, N, is_torus, k, t, max_temp, n_train, subkey1, data_dir_path, f"train_D{train_D}"
    )
    val_x0, val_xt = get_data_d(
        train_D, N, is_torus, k, t, max_temp, n_val, subkey2, data_dir_path, f"val_D{train_D}"
    )

    test_d_x0 = []
    test_d_xt = []
    for D in test_D_range:
        key, subkey = random.split(key)
        test_x0, test_xt = get_data_d(
            D, N, is_torus, k, t, max_temp, n_test, subkey, data_dir_path, f"test_D{D}"
        )
        x0_std = jnp.std(test_x0[((), 0)])
        xt_std = jnp.std(test_xt[((), 0)])
        xt_resid_std = jnp.std(test_xt[((), 0)] - test_x0[((), 0)])
        print(f"D={D}, x0:{x0_std:.3e}, xt:{xt_std:.3e}, xt_resid:{xt_resid_std:.3e}")
        test_d_x0.append(test_x0)
        test_d_xt.append(test_xt)

    return (train_x0, train_xt, val_x0, val_xt, test_d_x0, test_d_xt)


def plot_multi_image(
    input_multi_image: geom.MultiImage,
    actual_multi_image: geom.MultiImage,
    test_multi_image: geom.MultiImage,
    save_loc: str,
    title: str = "",
):
    """
    Plot vector x and y components of two MultiImages, and the differences between them. Each row
    is a component, and the columns are actual, test, and diff. If the image is 3D, plot the middle
    slice.

    args:
        input_multi_image_full: the input image, with past_steps number of timesteps
        actual_multi_image: the ground truth MultiImage
        test_multi_image: the predicted MultiImage
        save_loc: file location to save the image
        title: additional str to add to title, will be "test {title} {col}"
            "actual {title} {col}"
    """
    print(
        f"Printed image relative error: {ml.l2_rel_error(test_multi_image, actual_multi_image):.4f}%"
    )

    if input_multi_image.get_n_leading() == 2:
        input_multi_image = input_multi_image.get_one(keepdims=False)

    if actual_multi_image.get_n_leading() == 2:
        actual_multi_image = actual_multi_image.get_one(keepdims=False)

    if test_multi_image.get_n_leading() == 2:
        test_multi_image = test_multi_image.get_one(keepdims=False)

    # images now no longer have batch dimension

    N = input_multi_image.get_spatial_dims()[0]
    if input_multi_image.D == 3:
        img_arr = jnp.concatenate(
            [
                input_multi_image[((False,), 0)][:, :, N // 2],
                test_multi_image[((False,), 0)][:, :, N // 2],
                actual_multi_image[((False,), 0)][:, :, N // 2],
            ]
        )
    else:
        img_arr = jnp.concatenate(
            [
                input_multi_image.to_vector(),
                test_multi_image.to_vector(),
                actual_multi_image.to_vector(),
            ]
        )

    vmax = float(jnp.max(jnp.abs(img_arr)))
    vmin = -1 * vmax

    nrows = 1
    ncols = 4
    # figsize is 6 per col, 6 per row, (cols,rows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))

    input_image = input_multi_image.to_images()[0]
    if input_image.D == 3:
        input_image = geom.GeometricImage(input_image.data[N // 2], input_image.parity, 2)

    input_image.plot(axes[0], title=f"input {title}", vmin=vmin, vmax=vmax, colorbar=True)

    actual_image = actual_multi_image.to_images()[0]
    test_image = test_multi_image.to_images()[0]

    if actual_image.D == 3:
        actual_image = geom.GeometricImage(actual_image.data[N // 2], actual_image.parity, 2)
    if test_image.D == 3:
        test_image = geom.GeometricImage(test_image.data[N // 2], test_image.parity, 2)

    diff = actual_image - test_image
    diff_max = float(jnp.max(jnp.abs(diff.data)))
    diff_l2 = jnp.mean(diff.data**2)
    diff_rel_err = jnp.mean(jnp.abs(diff.data / actual_image.data)) * 100
    diff_title = f"diff (l2: {diff_l2:.3e}, rel. err: {diff_rel_err:.2f}%)"

    actual_image.plot(
        axes[1],
        title=f"output {title}",
        vmin=vmin,
        vmax=vmax,
        colorbar=True,
    )
    test_image.plot(
        axes[2],
        title=f"pred {title}",
        vmin=vmin,
        vmax=vmax,
        colorbar=True,
    )
    diff.plot(
        axes[3],
        title=diff_title,
        vmin=-diff_max,
        vmax=diff_max,
        colorbar=True,
    )

    plt.tight_layout()
    plt.savefig(save_loc)
    plt.close(fig)


class TwoLayerModel(models.AnyDimensionalModel):
    layers: list[models.ConvBlock]

    D: int = eqx.field(static=True)
    input_keys: geom.Signature = eqx.field(static=True)
    target_keys: geom.Signature = eqx.field(static=True)
    width: int = eqx.field(static=True)
    use_bias: bool | str = eqx.field(static=True)

    def __init__(
        self: Self,
        input_keys: geom.Signature,
        target_keys: geom.Signature,
        conv_filters: geom.MultiImage,
        width: int,
        use_bias: bool | str,
        key: jax.Array,
    ) -> None:
        self.D = conv_filters.D
        self.input_keys = input_keys
        self.target_keys = target_keys
        self.width = width
        self.use_bias = use_bias

        mid_keys = geom.signature_union(input_keys, target_keys, width)

        key, subkey1, subkey2 = random.split(key, num=3)
        self.layers = [
            models.ConvBlock(
                D, input_keys, mid_keys, use_bias, "gelu", True, conv_filters, key=subkey1
            ),
            models.ConvBlock(
                D, mid_keys, target_keys, use_bias, None, True, conv_filters, key=subkey2
            ),
        ]

    def convertD(
        self: Self, conv_filters: geom.MultiImage, rescale: bool, key: jax.Array, **kwargs
    ) -> Self:
        """
        Construct a new model with filters in a higher dimension.

        args:
            conv_filters: the new conv filters we are swapping to, probably in a higher dimension
            rescale: whether to force the sum of the filters in the new dimension to be equal
            key: key to initialize the weights, since they are overruled it won't matter

        returns:
            a new model with new filters but the old weights
        """
        new_model = self.__class__(
            self.input_keys,
            self.target_keys,
            conv_filters,
            self.width,
            self.use_bias,
            key,
        )

        return self.transfer_weights(new_model, rescale)

    def __call__(
        self: Self, x: geom.MultiImage, aux_data: eqx.nn.State | None = None
    ) -> tuple[geom.MultiImage, eqx.nn.State | None]:
        in_x = x
        for layer in self.layers:
            x, aux_data = layer(x, aux_data)

        return x, aux_data


@eqx.filter_jit
def map_residual(
    model: models.MultiImageModule,
    multi_image_x: geom.MultiImage,
    aux_data: eqx.nn.State | None = None,
) -> tuple[geom.MultiImage, eqx.nn.State | None]:
    residual, aux_data = jax.vmap(model, in_axes=(0, None), out_axes=(0, None), axis_name="batch")(
        multi_image_x, aux_data
    )

    # add the last timestep to the residual
    pred_y = residual.empty()
    for ((k, parity), img_in), img_resid in zip(multi_image_x.items(), residual.values()):
        pred_y.append(k, parity, img_in[:, -1:] + img_resid)

    return pred_y, aux_data


@eqx.filter_jit
def map_and_loss_rel_error(
    model: models.MultiImageModule,
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    aux_data: eqx.nn.State | None = None,
) -> tuple[jax.Array, eqx.nn.State | None]:
    """
    Calculates both the smse_loss and the rel_error.
    """
    pred_y, aux_data = map_residual(model, multi_image_x, aux_data)

    loss = ml.smse_loss(pred_y, multi_image_y)
    rel_error = ml.l2_rel_error(pred_y, multi_image_y)
    return jnp.stack([loss, rel_error]), aux_data


@eqx.filter_jit
def map_and_loss(
    model: models.MultiImageModule,
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    aux_data: eqx.nn.State | None = None,
) -> tuple[jax.Array, eqx.nn.State | None]:
    pred_y, aux_data = map_residual(model, multi_image_x, aux_data)

    return ml.smse_loss(pred_y, multi_image_y), aux_data


def train_and_eval(
    data: tuple[
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
        list[geom.MultiImage],
        list[geom.MultiImage],
    ],
    key: jax.Array,
    model_name: str,
    model: models.AnyDimensionalModel,
    lr: float,
    test_conv_filters: list[tuple[geom.MultiImage, geom.MultiImage]],
    batch_size: int,
    test_batch_size: int,
    epochs: int,
    save_model: str | None,
    load_model: str | None,
    images_dir: str | None,
    has_aux: bool = False,
    verbose: int = 1,
    is_wandb: bool = False,
) -> tuple[jax.Array | None, ...]:
    train_X, train_Y, val_X, val_Y, test_d_X, test_d_Y = data
    N = train_X.get_spatial_dims()[0]
    batch_stats = eqx.nn.State(model) if has_aux else None
    model_name_extended = f"{model_name}_L{train_X.get_L()}_N{N}_e{epochs}"

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

        if save_model is not None:
            # TODO: need to save batch_stats as well
            ml.save(f"{save_model}{model_name_extended}_model.eqx", trained_model)
    else:
        trained_model = ml.load(f"{load_model}{model_name_extended}_model.eqx", model)

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

    assert isinstance(trained_model, models.AnyDimensionalModel)
    test_losses = []
    for test_X, test_Y, (conv_filters, upsample_filters) in zip(
        test_d_X, test_d_Y, test_conv_filters
    ):
        if test_X.D == train_X.D:
            trained_model_d = trained_model
        else:
            key, subkey = random.split(key)
            # rescale can be true or false when using zero_sum, the effect is small
            trained_model_d = trained_model.convertD(
                conv_filters, True, subkey, upsample_filters=upsample_filters
            )

        key, subkey = random.split(key)
        test_loss = ml.map_loss_in_batches(
            map_and_loss_rel_error,
            trained_model_d,
            test_X,
            test_Y,
            test_batch_size,
            subkey,
            aux_data=batch_stats,
        )
        print(f"Test Loss rescale D={test_X.D}: {test_loss[0]}")
        print(f"Test Relative Error rescale D={test_X.D}: {test_loss[1]:.4f}%")

        if test_X.D == train_X.D:
            trained_model_d = trained_model
        else:
            key, subkey = random.split(key)
            # rescale can be true or false when using zero_sum, the effect is small
            trained_model_d = trained_model.convertD(
                conv_filters, False, subkey, upsample_filters=upsample_filters
            )

        key, subkey = random.split(key)
        test_loss = ml.map_loss_in_batches(
            map_and_loss_rel_error,
            trained_model_d,
            test_X,
            test_Y,
            test_batch_size,
            subkey,
            aux_data=batch_stats,
        )
        print(f"Test Loss D={test_X.D}: {test_loss[0]}")
        print(f"Test Relative Error D={test_X.D}: {test_loss[1]:.4f}%")

        test_losses.append(test_loss[0])
        test_losses.append(test_loss[1])

        if images_dir and test_X.D == 2:
            pred_y, _ = map_residual(trained_model_d, test_X.get_one(), batch_stats)

            plot_multi_image(
                test_X.get_one(),
                test_Y.get_one(),
                pred_y.get_one(),
                f"{images_dir}{model_name_extended}_D{test_X.D}_trainD{train_X.D}.png",
                "heat",
            )

    return train_loss, val_loss, *test_losses


def handleArgs() -> argparse.Namespace:
    parser = utils.get_common_parser()
    parser.add_argument(
        "--train-D", help="dimension of data to train on", choices=[1, 2, 3], default=1, type=int
    )
    parser.add_argument(
        "--max-test-D",
        help="maximum dimension of data to test on",
        choices=[1, 2, 3],
        default=3,
        type=int,
    )
    parser.add_argument("-N", help="spatial size", type=int, default=128)
    # need do to --wandb to activate, also need --wandb-entity your_wandb_name_here
    parser.add_argument(
        "--wandb-project", help="the wandb project", type=str, default="heat-equation"
    )

    return parser.parse_args()


# MAIN
args = handleArgs()
test_D_range = tuple(range(args.train_D, args.max_test_D + 1))

key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)

key, subkey = random.split(key)
print("Generating data...", end="", flush=True)
t_start = time.time()
data = get_data(
    args.train_D,
    test_D_range,
    args.N,
    True,
    args.n_train,
    args.n_val,
    args.n_test,
    subkey,
    args.data,
)
print(f"done. ({time.time() - t_start:.2f}s)", flush=True)

input_keys = data[0].get_signature()
output_keys = data[1].get_signature()

scaling = geom.FilterScaling.NORMALIZE
max_pixel_l1 = 2
M = 5
group_actions = geom.make_all_operators(args.train_D)
conv_filters = geom.get_invariant_filters(
    Ms=[M],
    ks=[0],
    parities=[0],
    D=args.train_D,
    operators=group_actions,
    scale=scaling,
    max_pixel_l1=max_pixel_l1,
    combine_equal_l1=True,
)
upsample_filters = geom.get_invariant_filters(
    Ms=[2],
    ks=[0],
    parities=[0],
    D=args.train_D,
    operators=group_actions,
    scale=scaling,
)

test_conv_filters = []
for D in test_D_range:
    group_actions_d = geom.make_all_operators(D)
    conv_filters_d = geom.get_invariant_filters(
        Ms=[M],
        ks=[0],
        parities=[0],
        D=D,
        operators=group_actions_d,
        scale=scaling,
        max_pixel_l1=max_pixel_l1,
        combine_equal_l1=True,
    )

    upsample_filters_d = geom.get_invariant_filters(
        Ms=[2],
        ks=[0],
        parities=[0],
        D=D,
        operators=group_actions_d,
        scale=scaling,
    )
    test_conv_filters.append((conv_filters_d, upsample_filters_d))


train_kwargs = {
    "test_conv_filters": test_conv_filters,
    "batch_size": args.batch,
    "test_batch_size": args.batch,
    "epochs": args.epochs,
    "save_model": args.save_model,
    "load_model": args.load_model,
    "images_dir": args.images_dir,
    "verbose": args.verbose,
    "is_wandb": args.wandb,
}

key, *subkeys = random.split(key, num=10)
model_list = [
    (
        "two_layer",
        train_and_eval,
        {
            "model": TwoLayerModel(
                input_keys, output_keys, conv_filters, 4, use_bias=False, key=subkeys[0]
            ),
            "lr": 1e-2,
            **train_kwargs,
        },
    ),
    (
        "lastStepIdentity",
        train_and_eval,
        {"model": models.LastStepIdentity(residual=True), "lr": 1, **train_kwargs},
    ),
    # comment out for now, upsample and orthoplex filters might not be working properly
    # (
    #     "unetBase_equiv48",
    #     train_and_eval,
    #     {
    #         "model": models.UNet(
    #             args.train_D,
    #             input_keys,
    #             output_keys,
    #             depth=48,
    #             num_downsamples=3 if args.N <= 64 else 4,
    #             activation_f=jax.nn.gelu,
    #             conv_filters=conv_filters,
    #             upsample_filters=upsample_filters,
    #             key=subkeys[1],
    #         ),
    #         "lr": 4e-4,  # 4e-4 to 6e-4 works, larger sometimes explodes
    #         **train_kwargs,
    #     },
    # ),
    (
        "resnet_equiv_42",
        train_and_eval,
        {
            "model": models.ResNet(
                args.train_D,
                input_keys,
                output_keys,
                depth=42,
                conv_filters=conv_filters,
                use_group_norm=True,  # want this to be true, not implemented yet
                key=subkeys[2],
            ),
            "lr": 7e-4,
            **train_kwargs,
        },
    ),
    (
        "dil_resnet_equiv20",
        train_and_eval,
        {
            "model": models.DilResNet(
                args.train_D,
                input_keys,
                output_keys,
                depth=20,
                conv_filters=conv_filters,
                key=subkeys[3],
            ),
            "lr": 1e-3,
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
