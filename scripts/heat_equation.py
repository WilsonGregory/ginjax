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
    key: jax.Array,
    data_dir: str,
) -> tuple[
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
        key: key for randomness
        data_dir: location to save or load the data from

    returns:
        input multi image, output multi image
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

    return (train_d_x0, train_d_xt, val_d_x0, val_d_xt, test_d_x0, test_d_xt)


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
    conv_filters_dict: dict[int, geom.MultiImage],
    residual: bool,
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
            HeatMapper(residual),
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
            HeatMapper(residual),
            trained_model,
            train_X,
            train_Y,
            batch_size,
            subkey1,
            aux_data=batch_stats,
        )
        val_loss = ml.map_loss_in_batches(
            HeatMapper(residual),
            trained_model,
            val_X,
            val_Y,
            batch_size,
            subkey2,
            aux_data=batch_stats,
        )

    assert isinstance(trained_model, models.AnyDimensionalModel)
    test_losses = []
    for test_X, test_Y in zip(test_d_X, test_d_Y):
        conv_filters = conv_filters_dict[test_X.D]

        if test_X.D == train_X.D:
            trained_model_d = trained_model
        else:
            key, subkey = random.split(key)
            # rescale can be true or false when using zero_sum, the effect is small
            trained_model_d = trained_model.convertD(conv_filters, True, subkey)

        key, subkey = random.split(key)
        test_loss_rescale = ml.map_loss_in_batches(
            HeatMapper(residual, True, True),
            trained_model_d,
            test_X,
            test_Y,
            test_batch_size,
            subkey,
            aux_data=batch_stats,
        )
        print(f"Test Loss rescale D={test_X.D}: {test_loss_rescale[0]}")
        print(f"Test Relative Error rescale D={test_X.D}: {test_loss_rescale[1]:.4f}%")

        test_losses.append(test_loss_rescale[0])
        test_losses.append(test_loss_rescale[1])

        if test_X.D == train_X.D:
            trained_model_d = trained_model
        else:
            key, subkey = random.split(key)
            # rescale can be true or false when using zero_sum, the effect is small
            trained_model_d = trained_model.convertD(conv_filters, False, subkey)

        key, subkey = random.split(key)
        test_loss = ml.map_loss_in_batches(
            HeatMapper(residual, True, True),
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
            pred_y, _ = HeatMapper(residual).map(trained_model_d, test_X.get_one(), batch_stats)

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
        "--min-train-D",
        help="the start of the train D range",
        choices=[1, 2, 3],
        default=1,
        type=int,
    )
    parser.add_argument(
        "--max-train-D", help="the end of the train D range", choices=[1, 2, 3], default=1, type=int
    )
    parser.add_argument(
        "--max-test-D",
        help="maximum dimension of data to test on",
        choices=[1, 2, 3],
        default=3,
        type=int,
    )
    parser.add_argument("-N", help="spatial size", type=int, default=128)
    parser.add_argument("--diffusion-coef", help="the diffusion coefficient", type=float, default=1)
    parser.add_argument(
        "--residual",
        help="learn the residual of the heat equation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    # need do to --wandb to activate, also need --wandb-entity your_wandb_name_here
    parser.add_argument(
        "--wandb-project", help="the wandb project", type=str, default="heat-equation"
    )

    return parser.parse_args()


# MAIN
args = handleArgs()
train_D_range = tuple(range(args.min_train_D, args.max_train_D + 1))
test_D_range = tuple(range(args.min_train_D, args.max_test_D + 1))

key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)

key, subkey = random.split(key)
print("Generating data...", end="", flush=True)
t_start = time.time()
train_d_x0, train_d_xt, val_d_x0, val_d_xt, test_d_x0, test_d_xt = get_data(
    train_D_range,
    test_D_range,
    args.N,
    True,
    args.diffusion_coef,
    args.n_train,
    args.n_val,
    args.n_test,
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

full_D_range = range(args.min_train_D, max(args.max_train_D, args.max_test_D) + 1)
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

test_results = []

for train_D, train_x0, train_xt, val_x0, val_xt in zip(
    train_D_range, train_d_x0, train_d_xt, val_d_x0, val_d_xt
):
    input_keys = train_x0.get_signature()
    output_keys = train_xt.get_signature()
    data = (train_x0, train_xt, val_x0, val_xt, test_d_x0, test_d_xt)

    train_kwargs = {
        "residual": args.residual,
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
            "one_layer_normalize_scaling",
            train_and_eval,
            {
                "model": ConvSeriesModel(
                    input_keys,
                    output_keys,
                    normalize_filters_dict[train_D],
                    width=1,
                    depth=1,
                    use_bias=False,
                    key=subkeys[0],
                ),
                "lr": 1e-2,
                "conv_filters_dict": normalize_filters_dict,
                **train_kwargs,
            },
        ),
        (
            "one_layer_gaussian_scaling",
            train_and_eval,
            {
                "model": ConvSeriesModel(
                    input_keys,
                    output_keys,
                    gaussian_filters_dict[train_D],
                    width=1,
                    depth=1,
                    use_bias=False,
                    key=subkeys[0],
                ),
                "lr": 1e-2,
                "conv_filters_dict": gaussian_filters_dict,
                **train_kwargs,
            },
        ),
        # (
        #     "one_layer_stencil_scaling",
        #     train_and_eval,
        #     {
        #         "model": ConvSeriesModel(
        #             input_keys,
        #             output_keys,
        #             stencil_filters_dict[train_D],
        #             width=1,
        #             depth=1,
        #             use_bias=False,
        #             key=subkeys[1],
        #         ),
        #         "lr": 1e-2,
        #         "conv_filters_dict": stencil_filters_dict,
        #         **train_kwargs,
        #     },
        # ),
        # (
        #     "one_layer_inverse_count_scaling",
        #     train_and_eval,
        #     {
        #         "model": ConvSeriesModel(
        #             input_keys,
        #             output_keys,
        #             inverse_count_filters_dict[train_D],
        #             width=1,
        #             depth=1,
        #             use_bias=False,
        #             key=subkeys[2],
        #         ),
        #         "lr": 1e-2,
        #         "conv_filters_dict": inverse_count_filters_dict,
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
        #     "two_layer_stencil_scaling",
        #     train_and_eval,
        #     {
        #         "model": ConvSeriesModel(
        #             input_keys,
        #             output_keys,
        #             stencil_filters_dict[train_D],
        #             width=10,
        #             depth=2,
        #             use_bias=False,
        #             key=subkeys[4],
        #         ),
        #         "lr": 1e-2,
        #         "conv_filters_dict": stencil_filters_dict,
        #         **train_kwargs,
        #     },
        # ),
        # (
        #     "two_layer_inverse_count_scaling",
        #     train_and_eval,
        #     {
        #         "model": ConvSeriesModel(
        #             input_keys,
        #             output_keys,
        #             inverse_count_filters_dict[train_D],
        #             width=10,
        #             depth=2,
        #             use_bias=False,
        #             key=subkeys[5],
        #         ),
        #         "lr": 1e-2,
        #         "conv_filters_dict": inverse_count_filters_dict,
        #         **train_kwargs,
        #     },
        # ),
        (
            "lastStepIdentity",
            train_and_eval,
            {
                "model": models.LastStepIdentity(residual=args.residual),
                "lr": 1,
                "conv_filters_dict": normalize_filters_dict,
                **train_kwargs,
            },
        ),
        # comment out for now, upsample and orthoplex filters might not be working properly
        # (
        #     "unetBase_equiv48",
        #     train_and_eval,
        #     {
        #         "model": models.UNet(
        #             train_D,
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
        #             use_group_norm=True,  # want this to be true, not implemented yet
        #             key=subkeys[6],
        #         ),
        #         "lr": 7e-4,
        #         "conv_filters_dict": normalize_filters_dict,
        #         **train_kwargs,
        #     },
        # ),
        # (
        #     "resnet_equiv_42_stencil_scaling",
        #     train_and_eval,
        #     {
        #         "model": models.ResNet(
        #             train_D,
        #             input_keys,
        #             output_keys,
        #             depth=42,
        #             conv_filters=stencil_filters_dict[train_D],
        #             use_group_norm=True,  # want this to be true, not implemented yet
        #             key=subkeys[7],
        #         ),
        #         "lr": 7e-4,
        #         "conv_filters_dict": stencil_filters_dict,
        #         **train_kwargs,
        #     },
        # ),
        # (
        #     "resnet_equiv_42_inverse_count_scaling",
        #     train_and_eval,
        #     {
        #         "model": models.ResNet(
        #             train_D,
        #             input_keys,
        #             output_keys,
        #             depth=42,
        #             conv_filters=inverse_count_filters_dict[train_D],
        #             use_group_norm=True,  # want this to be true, not implemented yet
        #             key=subkeys[8],
        #         ),
        #         "lr": 7e-4,
        #         "conv_filters_dict": inverse_count_filters_dict,
        #         **train_kwargs,
        #     },
        # ),
        # (
        #     "dil_resnet_equiv20",
        #     train_and_eval,
        #     {
        #         "model": models.DilResNet(
        #             train_D,
        #             input_keys,
        #             output_keys,
        #             depth=20,
        #             conv_filters=conv_filters,
        #             key=subkeys[3],
        #         ),
        #         "lr": 1e-3,
        #         **train_kwargs,
        #     },
        # ),
    ]

    key, subkey = random.split(key)
    # Use this for benchmarking the models with known learning rates.
    # (n_trials, benchmark, models, n_results)
    results = ml.benchmark(
        lambda _: data,
        model_list,
        subkey,
        "",
        [0],
        benchmark_type=ml.BENCHMARK_NONE,
        num_trials=args.n_trials,
        num_results=2 + len(test_D_range) * 4,  # train, val, test
        is_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    test_results.append(
        results[:, 0, :, 2:].reshape((args.n_trials, len(model_list), len(test_D_range), 2, 2))
    )

# (train_D, n_trials, n_models, test_D, [rescale,normal], [l2, relative error])
test_results = jnp.stack(test_results)

print(test_results.shape)
