import argparse
import math
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pathlib
import time

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax

import ginjax.geometric as geom
from ginjax import models
from ginjax import ml
from ginjax import utils


def unit_step(x: jax.Array) -> jax.Array:
    """
    The unit step function, returns 1 for pos, 0 for neg, and 0.5 for 0.

    args:
        x: array of inputs

    returns:
        array the same shape as x
    """
    return 0.5 * (x == 0) + 1 * (x > 0)


def wave_step(t: float, u0: jax.Array, du_dt0: jax.Array, is_torus: bool) -> jax.Array:
    """
    Take one step of the wave equation of timestep t.

    args:
        t: the timestep
        u0: the input wave displacement, shape (batch,spatial)
        du_dt0: the input wave velocity, shape (batch,spatial)
        is_torus:
    """
    assert t >= 0
    if t == 0:
        return u0

    D = u0.ndim - 1

    batch = len(u0)
    spatial_dims = u0.shape[1:]
    u0_flat = u0.reshape((batch, -1))
    du_dt0_flat = du_dt0.reshape((batch, -1))
    meshgrid_dims = (jnp.arange(N) for N in spatial_dims)

    # (spatial_size,D)
    idxs = jnp.stack(jnp.meshgrid(*meshgrid_dims, indexing="ij"), axis=-1).reshape((-1, D))
    ut = []
    for i, idx in enumerate(idxs):
        # if (i % (len(idxs) // 10)) == 0:
        #     print(f"{D},{batch}: {i}/{len(idxs)}")
        # (spatial_size,D)
        if is_torus:
            idxs_diff = jnp.abs(idxs - idx[None])
            idxs_diff_wrapped = jnp.stack(spatial_dims).reshape((1, D)) - idxs_diff
            dist = jnp.where(idxs_diff < idxs_diff_wrapped, idxs_diff, idxs_diff_wrapped)
        else:
            dist = idxs - idx[None]

        if D == 1:

            dist_norm = jnp.abs(dist[:, 0])  # (spatial_size,)

            # ut is the output field, * here is convolution
            # ut = g * du_dt0 + dg_dt * u0

            # (1,spatial_size)

            g_kernel = 0.5 * unit_step(t - dist_norm)[None]
            # derivative in t of unit step function is dirac delta
            partial_g_kernel = 0.5 * ((t - dist_norm) == 0)[None]
            # i might want to convolve with the interpolation?
            # or somehow generate the data analytically?

            print(t)
            print(dist_norm)
            print(g_kernel)
            print(partial_g_kernel)
            exit()

            # (batch,spatial_size) -> (batch,)
            ut.append(
                jnp.sum(partial_g_kernel * u0_flat, axis=1)
                + jnp.sum(g_kernel * du_dt0_flat, axis=1)
            )
        else:
            raise NotImplementedError("wave_step for D!=1 has not been implemented yet.")

    return jnp.stack(ut, axis=1).reshape(u0.shape)


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

    x0_img = geom.MultiImage({((), 0): jnp.stack([u0, du_dt0], axis=1)}, D, is_torus)
    xt_img = geom.MultiImage({((), 0): ut[:, None]}, D, is_torus)

    return x0_img, xt_img


def get_general_1d_sine_data(
    N: int, t: float, n_batch: int, key: jax.Array
) -> tuple[geom.MultiImage, geom.MultiImage]:
    """
    Construct initial 1d data from a sum of sine waves: u(x,t) = SUM_i a_i sin(k_i x - t + theta_i)
    where a_i is amplitude, k_i is scaling, and theta_i is shift.

    args:
        N: grid size
        t: timestep to take
        n_batch: batch size
        key: randomness key

    returns:
        input image with field u(x,0), field du_dt(x,0) and output image of field u(x,t)
    """
    D = 1
    is_torus = True  # this could be on the torus, but 3d image won't be so we dont
    n_cos = 3

    subkey1, subkey2, subkey3 = random.split(key, num=3)

    shift = random.uniform(subkey1, shape=(n_batch, n_cos, 1)) * N
    scale = random.normal(subkey2, shape=(n_batch, n_cos, 1))
    scale_norm = jnp.abs(scale)
    amp = random.normal(subkey3, shape=(n_batch, n_cos, 1))

    x_range = jnp.linspace(0, 2 * jnp.pi, num=N, endpoint=False)  # [0,6pi]
    scale_shift_x = x_range[None, None] * scale - shift  # (batch,n_sines,spatial)

    # (batch,spatial)
    u0 = jnp.sum(amp * jnp.cos(scale_shift_x), axis=1)
    du_dt0 = jnp.sum(amp * scale_norm * jnp.sin(scale_shift_x), axis=1)

    ut = jnp.sum(amp * jnp.cos(scale_shift_x - scale_norm * t), axis=1)  # (batch,spatial)

    x0_img = geom.MultiImage({((), 0): jnp.stack([u0, du_dt0], axis=1)}, D, is_torus)
    xt_img = geom.MultiImage({((), 0): ut[:, None]}, D, is_torus)

    return x0_img, xt_img


def get_3d_sine_data(
    N: int, t: float, n_batch: int, key: jax.Array
) -> tuple[geom.MultiImage, geom.MultiImage]:
    D = 3
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


def get_data_d(
    D: int, N: int, t: float, n_batch: int, key: jax.Array, data_dir: pathlib.Path
) -> tuple[geom.MultiImage, geom.MultiImage]:
    if D == 1:
        return get_1d_sine_data(N, t, n_batch, key)
    elif D == 3:
        return get_3d_sine_data(N, t, n_batch, key)
    else:
        raise NotImplementedError(f"get_data_d:: Only D=1,3 are implemented, not D={D}")


def get_data(
    D: int,
    N: int,
    t: float,
    n_train: int,
    n_val: int,
    n_test: int,
    key: jax.Array,
    data_dir: str,
) -> tuple[
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
]:
    """
    Generate full dataset with train, validation, and test data sets.

    args:
        D: dimension
        N: side length of images
        n_train: train dataset size
        n_val: validation dataset size
        n_test: test dataset size
        key: jax key for randomness
        data_dir: director to save/load the data

    returns:
        tuple of geometric images, (train_X, train_Y, val_X, val_Y, test_X, test_Y)
    """
    data_dir_path = pathlib.Path(data_dir)

    key, subkey1, subkey2, subkey3 = random.split(key, num=4)
    train_x0, train_xt = get_data_d(D, N, t, n_train, subkey1, data_dir_path / "train")

    val_x0, val_xt = get_data_d(D, N, t, n_val, subkey2, data_dir_path / "val")

    test_x0, test_xt = get_data_d(D, N, t, n_test, subkey3, data_dir_path / "test")

    return train_x0, train_xt, val_x0, val_xt, test_x0, test_xt


def plot_results(
    results_dict: dict[int, list[jax.Array]],
    results_labels: list[str],
    n_tune_range: tuple[int, ...],
    model_names_d: dict[int, list[str]],
    saveloc: str,
) -> None:
    """
    Plot the results of the experiments. For each test metric, create a plot with the
    number of tuning points on the x-axis and the test metric on the y-axis.

    args:
        results_dict: The results with train_D, test_D keys, then a list over n_tune.
        results_labels: e.g. 'l2', 'relative error'
        n_tune_range: number of fine-tuning points, or training points for the baseline model
        model_names_d: model names for each dimension
        saveloc: beginning of save location

    returns:
        none
    """
    # group the results by model, across all trained dimensions
    results_by_model = {}
    for train_D, results in results_dict.items():
        for i, name in enumerate(model_names_d[train_D]):
            name_trimmed = name[:-3]  # this assumes that all models end in _D2, or _D3
            display_name = f"{name} (baseline)" if train_D == 3 else name

            if name_trimmed in results_by_model:
                # (n_tune,n_trials,n_results)
                results_by_model[name_trimmed].append(
                    (display_name, jnp.stack(results)[:, :, 0, i])
                )
            else:
                results_by_model[name_trimmed] = [(display_name, jnp.stack(results)[:, :, 0, i])]

    # figsize is 8 per col, 6 per row, (cols,rows)
    nrows = 1  # D=3 is the only test dimension
    ncols = len(results_labels)
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 6 * nrows))
    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    colors = ["b", "g", "r", "c", "m", "y"]

    for error_idx, ylabel, ax in zip(range(len(results_labels)), results_labels, axes):
        assert isinstance(ax, Axes)

        # looping over 'two_layer_gaussian', 'resnet_equiv_42', ...
        for i, model_results in enumerate(results_by_model.values()):
            for j, (display_name, results_arr) in enumerate(model_results):
                ax.plot(
                    jnp.mean(results_arr, axis=1)[:, error_idx],
                    marker="o",
                    linestyle=linestyles[j],
                    label=display_name,
                    color=colors[i],
                )

        ax.legend()
        ax.set_xlabel("Number of tuning points")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_xticks(range(len(n_tune_range)), [str(x) for x in n_tune_range])
        ax.set_title(f"Test D={test_D} {ylabel}")

    plt.tight_layout()
    plt.savefig(f"{saveloc}warmstart_plot.png")
    plt.close()


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
    model_dir: pathlib.Path | None,
    overwrite_save_model: bool,
    images_dir: str | None,
    has_aux: bool = False,
    verbose: int = 1,
    is_wandb: bool = False,
) -> models.MultiImageModule:
    train_X, train_Y, val_X, val_Y = data
    N = train_X.get_spatial_dims()[0]
    batch_stats = eqx.nn.State(model) if has_aux else None
    model_name_extended = f"{model_name}_L{train_X.get_L()}_N{N}_e{epochs}"
    model_path = model_dir / f"{model_name_extended}_model.eqx" if model_dir else None

    print(f"{model_name_extended} params: {models.count_params(model):,}")

    if model_path and model_path.is_file() and not overwrite_save_model:
        trained_model = ml.load(model_path, model)
    else:
        steps_per_epoch = int(math.ceil(train_X.get_L() / batch_size))
        key, subkey = random.split(key)
        trained_model, _, _, _, _ = ml.train(
            train_X,
            train_Y,
            ml.Mapper([geom.Losses.SMSE], residual, eps=1e-5),
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
            val_map_and_loss=ml.Mapper([geom.Losses.NRMSE], residual, eps=1e-5),
            aux_data=batch_stats,
            is_wandb=is_wandb,
        )

        if model_path:
            assert not model_path.is_file() or overwrite_save_model
            # TODO: need to save batch_stats as well
            ml.save(model_path, trained_model)

    assert trained_model is not None
    # if images_dir and val_X.D == 2:
    #     pred_y, _ = ml.Mapper([geom.Losses.NRMSE], residual, eps=1e-5).map(trained_model, val_X.get_one(), batch_stats)
    #     plot_multi_image(
    #         val_X.get_one(),
    #         val_Y.get_one(),
    #         pred_y.get_one(),
    #         f"{images_dir}{model_name_extended}_D{val_X.D}.png",
    #         "burgers",
    #     )

    return trained_model


def train_all_models(
    data: tuple[
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
    ],
    key: jax.Array,
    model_list: list[tuple[str, dict, dict]],
    lr_range,
    args: argparse.Namespace,
):
    train_D = data[0].D
    n_points = data[0].get_L()

    trained_models = []
    for model_name, _train_kwargs, _test_kwargs in model_list:
        _train_kwargs["lr"] = _train_kwargs["lr"][train_D][n_points]
        key, subkey = random.split(key)
        if args.find_train_lr:

            def train_f(data, subkey, name, **kwargs):
                train_model(data, subkey, name, **kwargs)
                return 1.0  # train model returns the model, but benchmark expects a float

            ml.benchmark_lr(
                lambda _: data,
                [(model_name, train_f, _train_kwargs)],
                subkey,
                lr_range,
                args.n_trials,
                1,
                args.train_wandb,
                args.wandb_project,
                args.wandb_entity,
                {
                    **vars(args),
                    "D": train_D,
                    "n_points": n_points,
                    "train_or_tune": "train",
                },
            )
        else:
            trained_model = train_model(data, subkey, model_name, **_train_kwargs)
            _test_kwargs = {**_test_kwargs, "model": trained_model}
            trained_models.append((model_name, tune_and_eval, _test_kwargs))

    return trained_models


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
    conv_filters_dict: dict[int, geom.MultiImage] | None,
    rescale: geom.Rescaling | None,
    residual: bool,
    batch_size: int,
    epochs: int,
    model_dir: pathlib.Path | None,
    overwrite_save_model: bool,
    images_dir: str | None,
    upsample_filters_dict: dict[int, geom.MultiImage] | None = None,
    has_aux: bool = False,
    verbose: int = 1,
    is_wandb: bool = False,
) -> tuple[jax.Array, jax.Array]:
    tune_X, tune_Y, val_X, val_Y, test_X, test_Y = data
    N = tune_X.get_spatial_dims()[0]
    batch_stats = eqx.nn.State(model) if has_aux else None
    model_name_extended = f"{model_name}_tuneD{tune_X.D}_L{tune_X.get_L()}_N{N}_e{epochs}"

    key, subkey = random.split(key)
    if conv_filters_dict is not None and rescale is not None:
        model_dprime = model.convertD(
            conv_filters_dict[tune_X.D],
            rescale,
            subkey,
            upsample_filters=upsample_filters_dict[tune_X.D] if upsample_filters_dict else None,
        )
        model_name_extended += f"_rescale{rescale.name}"
    else:
        model_dprime = model

    model_path = model_dir / f"{model_name_extended}_model.eqx" if model_dir else None

    print(f"tuning: {model_name_extended}")

    if model_path and model_path.is_file() and not overwrite_save_model:
        tuned_model_dprime = ml.load(model_path, model_dprime)
        tune_batch_stats = batch_stats
    else:
        if tune_X.get_L() > 0:
            # Now treat the trained_model_d as a warmstart and do some additional training
            key, subkey = random.split(key)
            steps_per_epoch = int(math.ceil(tune_X.get_L() / batch_size))
            tuned_model_dprime, tune_batch_stats, _, _, _ = ml.train(
                tune_X,
                tune_Y,
                ml.Mapper([geom.Losses.SMSE], residual, eps=1e-5),
                model_dprime,
                subkey,
                stop_condition=ml.EpochStop(epochs, verbose=verbose),
                batch_size=min(tune_X.get_L(), batch_size),
                optimizer=optax.adamw(
                    optax.warmup_cosine_decay_schedule(
                        1e-8, lr, 5 * steps_per_epoch, epochs * steps_per_epoch, 1e-7
                    ),
                    weight_decay=1e-5,
                ),
                validation_X=val_X,
                validation_Y=val_Y,
                val_map_and_loss=ml.Mapper([geom.Losses.NRMSE], residual, eps=1e-5),
                aux_data=batch_stats,
                is_wandb=is_wandb,
            )
        else:
            tuned_model_dprime = model_dprime
            tune_batch_stats = batch_stats

        if model_path:
            assert not model_path.is_file() or overwrite_save_model
            ml.save(model_path, tuned_model_dprime)

    key, subkey = random.split(key)
    tuned_loss = ml.map_loss_in_batches(
        ml.Mapper([geom.Losses.NRMSE, geom.Losses.SMSE], residual, eps=1e-5),
        tuned_model_dprime,
        test_X,
        test_Y,
        batch_size,
        subkey,
        aux_data=tune_batch_stats,
    )
    nrmse_loss = tuned_loss[0]
    smse_loss = tuned_loss[1]
    print(f"Tuned Loss rescale=True, D={test_X.D}: nrmse={nrmse_loss:.3e} smse={smse_loss:.3e}\n")

    return nrmse_loss, smse_loss


# time python3 scripts/wave_anyd.py --n-train 128 --n-val 128 --n-test 128 --n-tune-range 0,1,4,32,128
def handleArgs() -> argparse.Namespace:
    parser = utils.get_common_parser()
    parser.add_argument(
        "--n-tune-range",
        help="the number of data points in the tuning set",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default="0,1,4,32,128",
    )
    parser.add_argument("-N", help="spatial size", type=int, default=64)
    # now doing --timestep 4.2 to differentiate complicated sine data on wandb
    parser.add_argument(
        "--timestep", help="timestep of wave output, t=1 is 1 grid point", type=float, default=4.3
    )
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
    # need do to --train-wandb or --tune-wandb to activate
    parser.add_argument("--wandb-project", help="the wandb project", type=str, default="wave")
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

if args.load_model or args.save_model:
    print("Use --model-dir and possibly --overwrite-save-model instead of --save-model")
    exit()

key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)
lr_range = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

# Only do 1D -> 3D
train_D = 1
test_D = 3
max_pixel_l1 = 2
M = 5

train_kwargs = {
    "residual": args.residual,
    "batch_size": args.batch,
    "epochs": args.epochs,
    "model_dir": pathlib.Path(args.model_dir) if args.model_dir else None,
    "overwrite_save_model": args.overwrite_save_model,
    "images_dir": None,
    "verbose": args.verbose,
    "is_wandb": args.train_wandb,
}

test_kwargs = {
    "residual": args.residual,
    "batch_size": args.batch,
    "epochs": args.epochs,
    "model_dir": pathlib.Path(args.model_dir) if args.model_dir else None,
    "overwrite_save_model": args.overwrite_save_model,
    "images_dir": None,
    "verbose": args.verbose,
    "is_wandb": args.tune_wandb,
}

normalize_filters_dict = {}
gaussian_filters_dict = {}
upsample_filters_dict = {}

full_D_range = [train_D, test_D]
for D in [train_D, test_D]:
    group_actions = geom.make_all_operators(D)
    normalize_filters_dict[D] = geom.get_invariant_filters(
        Ms=[M],
        ks=[0, 2],
        parities=[0],
        D=D,
        operators=group_actions,
        scale=geom.FilterScaling.NORMALIZE,
        max_pixel_l1=max_pixel_l1,
        combine_equal_l1=True,
    )
    gaussian_filters_dict[D] = geom.get_invariant_filters(
        Ms=[M],
        ks=[0, 2],
        parities=[0],
        D=D,
        operators=group_actions,
        scale=geom.FilterScaling.GAUSSIAN,
        max_pixel_l1=max_pixel_l1,
        combine_equal_l1=True,
    )
    upsample_filters_dict[D] = geom.get_invariant_filters(
        Ms=[2],
        ks=[0, 2],
        parities=[0],
        D=D,
        operators=group_actions,
        scale=geom.FilterScaling.NORMALIZE,  # for N=2, all pixels are equidistant
    )

print("Define the models!")
model_list_d = {}
for D in full_D_range:
    key, subkey = random.split(key)
    train_x0, train_xt, _, _, _, _ = get_data(
        D,
        args.N,
        args.timestep,
        args.n_train,
        0,
        0,
        subkey,
        args.data,
    )
    input_keys = train_x0.get_signature()
    output_keys = train_xt.get_signature()

    key, *subkeys = random.split(key, num=10)
    model_list = [
        # (
        #     f"two_layer_gaussian_scaling_D{D}",
        #     {
        #         "model": models.SimpleConvSeries(
        #             input_keys,
        #             output_keys,
        #             gaussian_filters_dict[D],
        #             width=10,
        #             depth=2,
        #             use_bias=False,
        #             key=subkeys[0],
        #         ),
        #         "lr": {1: {128: 1e-2}, 3: {0: 5e-2, 1: 5e-2, 4: 5e-2, 32: 1e-2, 128: 1e-2}},
        #         **train_kwargs,
        #     },
        #     {  # test_kwargs
        #         "lr": {3: {0: 1e-2, 1: 1e-2, 4: 1e-2, 32: 5e-3, 128: 1e-2}},
        #         "rescale": geom.Rescaling.NO_SCALING,
        #         "conv_filters_dict": gaussian_filters_dict,
        #         **test_kwargs,
        #     },
        # ),
        # (
        #     f"two_layer_gaussian_scaling_D{D}",
        #     {
        #         "model": models.SimpleConvSeries(
        #             input_keys,
        #             output_keys,
        #             gaussian_filters_dict[D],
        #             width=10,
        #             depth=2,
        #             use_bias=False,
        #             key=subkeys[0],
        #         ),
        #         "lr": {1: {128: 1e-2}, 3: {0: 5e-2, 1: 5e-2, 4: 5e-2, 32: 1e-2, 128: 1e-2}},
        #         **train_kwargs,
        #     },
        #     {  # test_kwargs
        #         "lr": {3: {0: 1e-2, 1: 1e-2, 4: 1e-2, 32: 5e-3, 128: 1e-2}},
        #         "rescale": geom.Rescaling.VOLUME,
        #         "conv_filters_dict": gaussian_filters_dict,
        #         **test_kwargs,
        #     },
        # ),
        (
            f"two_layer_gaussian_scaling_D{D}",  # rescale is added to name in tune
            {
                "model": models.SimpleConvSeries(
                    input_keys,
                    output_keys,
                    gaussian_filters_dict[D],
                    width=10,
                    depth=2,
                    use_bias=False,
                    key=subkeys[0],
                ),
                "lr": {1: {128: 1e-2}, 3: {0: 5e-2, 1: 5e-2, 4: 5e-2, 32: 1e-2, 128: 1e-2}},
                **train_kwargs,
            },
            {  # test_kwargs
                "lr": {3: {0: 1e-4, 1: 1e-4, 4: 1e-4, 32: 5e-2, 128: 5e-2}},
                "rescale": geom.Rescaling.COMPATIBILITY,
                "conv_filters_dict": gaussian_filters_dict,
                **test_kwargs,
            },
        ),
        (
            f"unetBase_equiv48_gaussian_scaling_D{D}",
            {  # train_kwargs
                "model": models.UNet(
                    D,
                    input_keys,
                    output_keys,
                    depth=48,
                    activation_f=jax.nn.gelu,
                    conv_filters=gaussian_filters_dict[D],
                    upsample_filters=upsample_filters_dict[D],
                    key=subkeys[1],
                ),
                "lr": {1: {128: 5e-4}, 3: {0: 1e-3, 1: 1e-3, 4: 1e-3, 32: 5e-4, 128: 1e-3}},
                **train_kwargs,
            },
            {  # tune and eval kwargs
                "lr": {3: {0: 5e-4, 1: 5e-4, 4: 5e-4, 32: 5e-4, 128: 1e-3}},
                "rescale": geom.Rescaling.VOLUME,
                "conv_filters_dict": gaussian_filters_dict,
                "upsample_filters_dict": upsample_filters_dict,
                **test_kwargs,
            },
        ),
    ]
    model_list_d[D] = model_list

# train the models, i.e. the warmstart lower dimensional models
print("Train the models (warmstart)!")
key, subkey = random.split(key)
train_x0, train_xt, val_x0, val_xt, _, _ = get_data(
    train_D,
    args.N,
    args.timestep,
    args.n_train,
    args.n_val,
    0,
    subkey,
    args.data,
)

train_data = (train_x0, train_xt, val_x0, val_xt)
key, subkey = random.split(key)
trained_model_list = train_all_models(train_data, subkey, model_list_d[train_D], lr_range, args)

if args.find_train_lr:
    exit()

# evaluate the models
print("Tune and evaluate the models!")
results_dict = {k: [] for k in (test_D, train_D)}
for n_tune in args.n_tune_range:
    print(f"D={test_D}, n_tune={n_tune}.\n")
    # the data is saved, so this is still reasonably efficient
    key, subkey = random.split(key)
    tune_data = get_data(
        test_D,
        args.N,
        args.timestep,
        n_tune,
        args.n_val,
        args.n_test,
        subkey,
        args.data,
    )

    # need to train the baseline model on tune_x0, etc. aka models without the warmstart
    baseline_model_list = [
        (
            name,
            tune_and_eval,
            {
                **_train_kwargs,
                "lr": _train_kwargs["lr"][test_D][n_tune],
                "conv_filters_dict": None,
                "rescale": None,
                "is_wandb": args.tune_wandb,
            },
        )
        for name, _train_kwargs, _ in model_list_d[test_D]
    ]

    key, subkey = random.split(key)
    # although this uses train_kwargs and lr, it is in the tune section
    # (n_trials, benchmark, models, n_results)
    baseline_results = ml.benchmark_lr(
        lambda _: tune_data,
        baseline_model_list,
        subkey,
        lr_range if args.find_tune_lr else [],
        num_trials=args.n_trials,
        num_results=2,  # l2, rel_error
        is_wandb=args.tune_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        args={
            **vars(args),
            "tune_D1_D3": f"(-,{test_D})",
            "n_points": n_tune,
            "train_or_tune": "tune",
        },
    )
    results_dict[test_D].append(baseline_results)

    trained_model_list_input = [
        (name, model_f, {**_test_kwargs, "lr": _test_kwargs["lr"][test_D][n_tune]})
        for name, model_f, _test_kwargs in trained_model_list
    ]

    key, subkey = random.split(key)
    # (n_trials, benchmark, models, n_results)
    tune_results = ml.benchmark_lr(
        lambda _: tune_data,
        trained_model_list_input,
        subkey,
        lr_range if args.find_tune_lr else [],
        num_trials=args.n_trials,
        num_results=2,  # l2, rel_error
        is_wandb=args.tune_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        args={
            **vars(args),
            "tune_D1_D3": str(tuple((train_D, test_D))),
            "n_points": n_tune,
            "train_or_tune": "tune",
        },
    )

    results_dict[train_D].append(tune_results)

if args.images_dir is not None:
    model_names_d = {
        D: [f'{x[2]["rescale"].name}_{x[0]}' for x in model_list]
        for D, model_list in model_list_d.items()
    }
    plot_results(
        results_dict,
        ["nrmse_loss", "smse_loss"],
        args.n_tune_range,
        model_names_d,
        args.images_dir,
    )
