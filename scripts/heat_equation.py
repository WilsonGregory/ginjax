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
    data_dir: pathlib.Path,
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
    data_dir = (
        data_dir / f"D{D}_N{N}_istorus{int(is_torus)}_n{batch}_k{k}_t{t}_maxtemp{max_temp}.npy"
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
    train_D_range: tuple[int, ...],
    test_D_range: tuple[int, ...],
    N: int,
    is_torus: bool,
    diffusion_coef: float,
    n_train: int,
    n_val: int,
    n_test: int,
    n_tune: int,
    key: jax.Array,
    data_dir: str,
) -> tuple[
    list[geom.MultiImage],
    list[geom.MultiImage],
    list[geom.MultiImage],
    list[geom.MultiImage],
    list[geom.MultiImage],
    list[geom.MultiImage],
    list[geom.MultiImage],
    list[geom.MultiImage],
    list[geom.MultiImage],
    list[geom.MultiImage],
]:
    """
    Get an input, output data pair of heat diffusion after 1 timestep, diffusion constant 10.
    The initial data is uniform on the range 0 to 5.

    args:
        train_D_range: range of train dimensions
        test_D_range: range of test dimensions
        N: sidelength of a cube of data
        is_torus: whether the data is on the torus
        diffusion_coeff: coefficient in the diffusion equation, equivalently the timestep
        n_train: number of training data points
        n_val: number of validation data points
        n_test: number of test data points for each test dimension
        n_tune: number of tuning data points for each test dimension
        key: key for randomness
        data_dir: location to save or load the data from

    returns:
        list of training, validation, test, and tuning images for input and output
    """
    max_temp = math.sqrt(3)
    t = 1
    data_dir_path = pathlib.Path(data_dir)

    train_d_x0 = []
    train_d_xt = []
    val_d_x0 = []
    val_d_xt = []
    for D in train_D_range:
        key, subkey1, subkey2 = random.split(key, num=3)
        train_x0, train_xt = get_data_d(
            D, N, is_torus, diffusion_coef, t, max_temp, n_train, subkey1, data_dir_path / "train"
        )
        train_d_x0.append(train_x0)
        train_d_xt.append(train_xt)

        val_x0, val_xt = get_data_d(
            D, N, is_torus, diffusion_coef, t, max_temp, n_val, subkey2, data_dir_path / "val"
        )
        val_d_x0.append(val_x0)
        val_d_xt.append(val_xt)

    test_d_x0 = []
    test_d_xt = []
    tune_d_x0 = []
    tune_d_xt = []
    tune_val_d_x0 = []
    tune_val_d_xt = []
    for D in test_D_range:
        key, subkey = random.split(key)
        test_x0, test_xt = get_data_d(
            D, N, is_torus, diffusion_coef, t, max_temp, n_test, subkey, data_dir_path / "test"
        )
        x0_std = jnp.std(test_x0[((), 0)])
        xt_std = jnp.std(test_xt[((), 0)])
        xt_resid_std = jnp.std(test_xt[((), 0)] - test_x0[((), 0)])
        print(f"D={D}, x0:{x0_std:.3e}, xt:{xt_std:.3e}, xt_resid:{xt_resid_std:.3e}")
        test_d_x0.append(test_x0)
        test_d_xt.append(test_xt)

        # for the tune and tune_val datasets, reuse the train and val datasets (size depending)
        key, subkey = random.split(key)
        tune_x0, tune_xt = get_data_d(
            D, N, is_torus, diffusion_coef, t, max_temp, n_tune, subkey, data_dir_path / "train"
        )
        tune_d_x0.append(tune_x0)
        tune_d_xt.append(tune_xt)

        key, subkey = random.split(key)
        tune_val_x0, tune_val_xt = get_data_d(
            D, N, is_torus, diffusion_coef, t, max_temp, n_val, subkey, data_dir_path / "val"
        )
        tune_val_d_x0.append(tune_val_x0)
        tune_val_d_xt.append(tune_val_xt)

    return (
        train_d_x0,
        train_d_xt,
        val_d_x0,
        val_d_xt,
        test_d_x0,
        test_d_xt,
        tune_d_x0,
        tune_d_xt,
        tune_val_d_x0,
        tune_val_d_xt,
    )


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


class ConvSeriesModel(models.AnyDimensionalModel):
    """
    Simple convolution model consisting of a series of ConvBlocks, with all but the last with a
    gelu vector neuron nonlinearity.
    """

    layers: list[models.ConvBlock]

    D: int = eqx.field(static=True)
    input_keys: geom.Signature = eqx.field(static=True)
    target_keys: geom.Signature = eqx.field(static=True)
    width: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    use_bias: bool | str = eqx.field(static=True)

    def __init__(
        self: Self,
        input_keys: geom.Signature,
        target_keys: geom.Signature,
        conv_filters: geom.MultiImage,
        width: int,
        depth: int,
        use_bias: bool | str,
        key: jax.Array,
    ) -> None:
        self.D = conv_filters.D
        self.input_keys = input_keys
        self.target_keys = target_keys
        self.width = width
        self.depth = depth
        self.use_bias = use_bias

        mid_keys = geom.signature_union(input_keys, target_keys, width)

        subkey_last, *subkeys = random.split(key, num=depth)
        self.layers = []
        for subkey in subkeys:
            self.layers.append(
                models.ConvBlock(
                    D, input_keys, mid_keys, use_bias, "gelu", True, conv_filters, key=subkey
                )
            )

        self.layers.append(
            models.ConvBlock(
                D, mid_keys, target_keys, use_bias, None, True, conv_filters, key=subkey_last
            )
        )

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
            self.depth,
            self.use_bias,
            key,
        )

        return self.transfer_weights(new_model, rescale)

    def __call__(
        self: Self, x: geom.MultiImage, aux_data: eqx.nn.State | None = None
    ) -> tuple[geom.MultiImage, eqx.nn.State | None]:
        for layer in self.layers:
            x, aux_data = layer(x, aux_data)

        return x, aux_data


class HeatMapper:
    """
    Functor for map_and_loss in train, map_loss_in_batches, etc, where arguments can be provided
    beforehand. In this case, it is useful for smse vs relative error, and whether to learn the
    residual or not.
    """

    residual: bool
    smse: bool
    l2_rel: bool

    def __init__(
        self: Self, residual: bool = False, smse: bool = True, l2_rel: bool = False
    ) -> None:
        """
        Docstring for __init__

        residual: Whether the network should learn the residual, defaults to False
        smse: Whether __call__ returns the smse loss, defaults to True
        l2_rel: Whether __call__ returns the l2 relative error, defaults to False
        """
        assert smse or l2_rel, "At least one of smse or l2_rel must be true."
        self.residual = residual
        self.smse = smse
        self.l2_rel = l2_rel

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
        if self.smse:
            losses.append(ml.smse_loss(pred_y, multi_image_y))

        if self.l2_rel:
            losses.append(ml.l2_rel_error(pred_y, multi_image_y))

        return jnp.squeeze(jnp.stack(losses)), aux_data


def train_model(
    data: tuple[
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
    ],
    key: jax.Array,
    model_name: str,
    model: models.AnyDimensionalModel,
    lr: float,
    residual: bool,
    batch_size: int,
    epochs: int,
    save_model: str | None,
    load_model: str | None,
    has_aux: bool = False,
    verbose: int = 1,
    is_wandb: bool = False,
) -> models.MultiImageModule:
    train_X, train_Y, val_X, val_Y = data
    N = train_X.get_spatial_dims()[0]
    batch_stats = eqx.nn.State(model) if has_aux else None
    model_name_extended = f"{model_name}_L{train_X.get_L()}_N{N}_e{epochs}"

    print(f"{model_name} params: {models.count_params(model):,}")

    if load_model is not None:
        return ml.load(f"{load_model}{model_name_extended}_model.eqx", model)

    steps_per_epoch = int(math.ceil(train_X.get_L() / batch_size))
    key, subkey = random.split(key)
    trained_model, _, _, _ = ml.train(
        train_X,
        train_Y,
        HeatMapper(residual),
        model,
        subkey,
        stop_condition=ml.EpochStop(epochs, verbose=verbose),
        batch_size=min(train_X.get_L(), batch_size),
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

    return trained_model


def tune_and_eval(
    data: tuple[
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
    ],
    key: jax.Array,
    model_name: str,
    model: models.AnyDimensionalModel,
    lr: float,
    conv_filters_dict: dict[int, geom.MultiImage],
    residual: bool,
    batch_size: int,
    epochs: int,
    save_model: str | None,
    load_model: str | None,
    images_dir: str | None,
    has_aux: bool = False,
    verbose: int = 1,
    is_wandb: bool = False,
) -> tuple[jax.Array, jax.Array]:
    test_X, test_Y, tune_X, tune_Y, val_X, val_Y = data
    N = tune_X.get_spatial_dims()[0]
    batch_stats = eqx.nn.State(model) if has_aux else None
    model_name_extended = f"{model_name}_tuneD{tune_X.D}_L{tune_X.get_L()}_N{N}_e{epochs}"

    key, subkey = random.split(key)
    # rescale set to True, small but notable difference
    model_dprime = model.convertD(conv_filters_dict[tune_X.D], True, subkey)

    if load_model is not None:
        tuned_model_dprime = ml.load(f"{load_model}{model_name_extended}_model.eqx", model_dprime)
        tune_batch_stats = batch_stats
    else:
        # Now treat the trained_model_d as a warmstart and do some additional training
        key, subkey = random.split(key)
        steps_per_epoch = int(math.ceil(tune_X.get_L() / batch_size))
        tuned_model_dprime, tune_batch_stats, _, _ = ml.train(
            tune_X,
            tune_Y,
            HeatMapper(residual),
            model_dprime,
            subkey,
            stop_condition=ml.EpochStop(epochs, verbose=verbose),
            batch_size=min(tune_X.get_L(), batch_size),
            optimizer=optax.adamw(
                optax.warmup_cosine_decay_schedule(
                    lr * 1e-4, lr, 5 * steps_per_epoch, epochs * steps_per_epoch, lr * 1e-4
                ),
                weight_decay=1e-5,
            ),
            validation_X=val_X,
            validation_Y=val_Y,
            aux_data=batch_stats,
            is_wandb=is_wandb,
        )

        if save_model is not None:
            ml.save(f"{save_model}{model_name_extended}_model.eqx", tuned_model_dprime)

    key, subkey = random.split(key)
    tuned_loss = ml.map_loss_in_batches(
        HeatMapper(residual, True, True),
        tuned_model_dprime,
        test_X,
        test_Y,
        batch_size,
        subkey,
        aux_data=tune_batch_stats,
    )
    l2_loss = tuned_loss[0]
    rel_error = tuned_loss[1]
    print(f"Tuned Loss rescale=True, D={test_X.D}: {l2_loss:.3e} ({rel_error:.3f}%)\n")

    # assert tuned_model_dprime is not None
    # if images_dir and test_X.D == 2:
    #     pred_y, _ = HeatMapper(residual).map(tuned_model_dprime, test_X.get_one(), batch_stats)

    #     plot_multi_image(
    #         test_X.get_one(),
    #         test_Y.get_one(),
    #         pred_y.get_one(),
    #         f"{images_dir}{model_name_extended}_D{test_X.D}_trainD{tune_X.D}.png",
    #         "heat",
    #     )

    return l2_loss, rel_error


# an example of currently used script
# CUDA_VISIBLE_DEVICES=5 time python3 scripts/heat_equation.py --data /data/wgregor4/heat_equation/
# --n-test 128 --n-val 128 --n-train 128 --n-tune 4 -N 64 --train-D-range 1,2,3 --diffusion-coef 1
# --test-D-range 3 -s /data/wgregor4/runs/heat_equation/
def handleArgs() -> argparse.Namespace:
    parser = utils.get_common_parser()
    parser.add_argument(
        "--n-tune", help="the number of data points in the tuning set", default=4, type=int
    )
    parser.add_argument(
        "--train-D-range",
        help="a comma separated list of range of dims to train over, e.g. 1,2",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default="1,2",
    )
    parser.add_argument(
        "--test-D-range",
        help="a comma separated list of range of dims to test over, e.g. 1,2",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default="1,2,3",
    )
    parser.add_argument("-N", help="spatial size", type=int, default=128)
    parser.add_argument("--diffusion-coef", help="the diffusion coefficient", type=float, default=1)
    parser.add_argument(
        "--residual",
        help="learn the residual of the heat equation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--find-train-lr",
        help="benchmark trained model over the lr",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--find-tune-lr",
        help="benchmark tuned model over the lr",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save-tuned-model", help="file name to save the params", type=str, default=None
    )
    parser.add_argument(
        "--load-tuned-model", help="file name to load params from", type=str, default=None
    )
    # need do to --train-wandb or --tune-wandb to activate
    parser.add_argument(
        "--wandb-project", help="the wandb project", type=str, default="heat-equation"
    )
    parser.add_argument(
        "--train-wandb",
        help="whether to use wandb during training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--tune-wandb",
        help="whether to use wandb during tuning",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    return parser.parse_args()


# MAIN
args = handleArgs()
if args.wandb:
    print("Use --train-wandb or --test-wandb to control these individually. Exiting.")
    exit()

key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)

key, subkey = random.split(key)
print("Generating data...", end="", flush=True)
t_start = time.time()
(
    train_d_x0,
    train_d_xt,
    val_d_x0,
    val_d_xt,
    test_d_x0,
    test_d_xt,
    tune_d_x0,
    tune_d_xt,
    tune_val_d_x0,
    tune_val_d_xt,
) = get_data(
    args.train_D_range,
    args.test_D_range,
    args.N,
    True,
    args.diffusion_coef,
    args.n_train,
    args.n_val,
    args.n_test,
    args.n_tune,
    subkey,
    args.data,
)
print(f"done. ({time.time() - t_start:.2f}s)", flush=True)

max_pixel_l1 = 2
M = 5

normalize_filters_dict = {}
gaussian_filters_dict = {}
stencil_filters_dict = {}
inverse_count_filters_dict = {}

full_D_range = tuple(set(args.train_D_range).union(set(args.test_D_range)))
for D in full_D_range:
    group_actions = geom.make_all_operators(D)
    normalize_filters_dict[D] = geom.get_invariant_filters(
        Ms=[M],
        ks=[0],
        parities=[0],
        D=D,
        operators=group_actions,
        scale=geom.FilterScaling.NORMALIZE,
        max_pixel_l1=max_pixel_l1,
        combine_equal_l1=True,
    )
    gaussian_filters_dict[D] = geom.get_invariant_filters(
        Ms=[M],
        ks=[0],
        parities=[0],
        D=D,
        operators=group_actions,
        scale=geom.FilterScaling.GAUSSIAN,
        max_pixel_l1=max_pixel_l1,
        combine_equal_l1=True,
    )
    stencil_filters_dict[D] = geom.get_invariant_filters(
        Ms=[M],
        ks=[0],
        parities=[0],
        D=D,
        operators=group_actions,
        scale=geom.FilterScaling.STENCIL,
        max_pixel_l1=max_pixel_l1,
        combine_equal_l1=True,
    )
    inverse_count_filters_dict[D] = geom.get_invariant_filters(
        Ms=[M],
        ks=[0],
        parities=[0],
        D=D,
        operators=group_actions,
        scale=geom.FilterScaling.INVERSE_COUNT,
        max_pixel_l1=max_pixel_l1,
        combine_equal_l1=True,
    )

train_kwargs = {
    "residual": args.residual,
    "batch_size": args.batch,
    "epochs": args.epochs,
    "save_model": args.save_model,
    "load_model": args.load_model,
    "verbose": args.verbose,
    "is_wandb": args.train_wandb,
}

test_kwargs = {
    "residual": args.residual,
    "batch_size": args.batch,
    "epochs": args.epochs,
    "save_model": args.save_tuned_model,
    "load_model": args.load_tuned_model,
    "images_dir": args.images_dir,
    "verbose": 0,
    "is_wandb": args.tune_wandb,
}

trained_models_by_d = {}
for train_D, train_x0, train_xt, val_x0, val_xt in zip(
    args.train_D_range, train_d_x0, train_d_xt, val_d_x0, val_d_xt
):
    input_keys = train_x0.get_signature()
    output_keys = train_xt.get_signature()

    key, *subkeys = random.split(key, num=10)
    model_list = [
        # (
        #     "one_layer_normalize_scaling",
        #     train_and_eval,
        #     {
        #         "model": ConvSeriesModel(
        #             input_keys,
        #             output_keys,
        #             normalize_filters_dict[train_D],
        #             width=1,
        #             depth=1,
        #             use_bias=False,
        #             key=subkeys[0],
        #         ),
        #         "lr": 1e-2,
        #         "conv_filters_dict": normalize_filters_dict,
        #         **train_kwargs,
        #     },
        # ),
        # (
        #     "two_layer_normalize_scaling",
        #     train_and_eval,
        #     {
        #         "model": ConvSeriesModel(
        #             input_keys,
        #             output_keys,
        #             normalize_filters_dict[train_D],
        #             width=10,
        #             depth=2,
        #             use_bias=False,
        #             key=subkeys[3],
        #         ),
        #         "lr": 1e-2,
        #         "conv_filters_dict": normalize_filters_dict,
        #         **train_kwargs,
        #     },
        # ),
        # (
        #     "two_layer_gaussian_scaling",
        #     ConvSeriesModel(
        #         input_keys,
        #         output_keys,
        #         gaussian_filters_dict[train_D],
        #         width=10,
        #         depth=2,
        #         use_bias=False,
        #         key=subkeys[2],
        #     ),
        #     {
        #         "lr": 1e-2,
        #         **train_kwargs,
        #     },
        #     {
        #         "lr": 1e-2,
        #         "conv_filters_dict": gaussian_filters_dict,
        #         **test_kwargs,
        #     },
        # ),
        # (
        #     "lastStepIdentity",
        #     train_and_eval,
        #     {
        #         "model": models.LastStepIdentity(residual=args.residual),
        #         "lr": 1,
        #         "conv_filters_dict": normalize_filters_dict,
        #         **train_kwargs,
        #     },
        # ),
        # (
        #     "resnet_equiv_42_normalize_scaling",
        #     train_and_eval,
        #     {
        #         "model": models.ResNet(
        #             train_D,
        #             input_keys,
        #             output_keys,
        #             depth=42,
        #             conv_filters=normalize_filters_dict[train_D],
        #             use_group_norm=True,
        #             key=subkeys[6],
        #         ),
        #         "lr": 7e-4,
        #         "conv_filters_dict": normalize_filters_dict,
        #         **train_kwargs,
        #     },
        # ),
        (
            f"resnet_equiv_42_gaussian_scaling_D{train_D}",
            {  # train kwargs
                "model": models.ResNet(
                    train_D,
                    input_keys,
                    output_keys,
                    depth=42,
                    conv_filters=gaussian_filters_dict[train_D],
                    use_group_norm=True,
                    key=subkeys[6],
                ),
                "lr": {1: 1e-3, 2: 5e-4, 3: 5e-4},  # will be replaced before running
                **train_kwargs,
            },
            {  # tune and eval kwargs
                "lr": 1e-4,
                "conv_filters_dict": gaussian_filters_dict,
                **test_kwargs,
            },
        ),
    ]

    trained_models_by_d[train_D] = []
    for model_name, _train_kwargs, _test_kwargs in model_list:
        data = (train_x0, train_xt, val_x0, val_xt)

        if args.find_train_lr:
            key, subkey = random.split(key)

            def train_f(data, subkey, name, **kwargs):
                train_model(data, subkey, name, **kwargs)
                return 1.0  # train model returns the model, but benchmark expects a float

            ml.benchmark_lr(
                lambda _: data,
                [(model_name, train_f, _train_kwargs)],
                subkey,
                [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
                args.n_trials,
                1,
                args.train_wandb,
                args.wandb_project,
                args.wandb_entity,
                {
                    **vars(args),
                    "D": train_D,
                    "n_points": train_x0.get_L(),
                    "train_or_tune": "train",
                },
            )
        else:
            key, subkey = random.split(key)
            _train_kwargs["lr"] = _train_kwargs["lr"][train_D]
            trained_model = train_model(data, subkey, model_name, **_train_kwargs)
            _test_kwargs["model"] = trained_model
            trained_models_by_d[train_D].append((model_name, tune_and_eval, _test_kwargs))

if args.find_train_lr:
    exit()
    # if tuning baseline models, don't bother with tuning/evaluation

n_results = 2  # tuned l2, rel_error
results_d_d = []
for train_D, train_x0, train_xt, val_x0, val_xt in zip(
    args.train_D_range, train_d_x0, train_d_xt, val_d_x0, val_d_xt
):
    model_list = trained_models_by_d[train_D]

    results_d = []
    for test_D, test_x0, test_xt, tune_x0, tune_xt, tune_val_x0, tune_val_xt in zip(
        args.test_D_range, test_d_x0, test_d_xt, tune_d_x0, tune_d_xt, tune_val_d_x0, tune_val_d_xt
    ):
        # if not finding the tuning lr, range is set to [] and value in model list is used
        lr_range = (
            [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3] if args.find_tune_lr else []
        )

        if train_D == test_D and args.n_tune != 0:
            continue

        data = (test_x0, test_xt, tune_x0, tune_xt, tune_val_x0, tune_val_xt)

        key, subkey = random.split(key)
        # (n_trials, benchmark, models, n_results)
        results_d.append(
            ml.benchmark_lr(
                lambda _: data,
                model_list,
                subkey,
                lr_range,
                num_trials=args.n_trials,
                num_results=n_results,
                is_wandb=args.tune_wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                args={
                    **vars(args),
                    "tune_D1_D2": str(tuple((train_D, test_D))),
                    "n_points": tune_x0.get_L(),
                    "train_or_tune": "tune",
                },
            )
        )

    results_d_d.append(jnp.stack(results_d))

# (train_D, test_D, n_trials, benchmark, n_models, [l2, relative error])
test_results = jnp.stack(results_d_d)
print(test_results.shape)
