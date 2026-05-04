import argparse
import math
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pathlib
import time
from typing_extensions import Self

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
import apebench

import ginjax.geometric as geom
from ginjax import models
from ginjax import ml
import ginjax.data as gc_data
from ginjax import utils


def get_data_d(
    D: int,
    N: int,
    diffusion_coef: float,
    convection_coef: float,
    subsample: int,  # new
    n_batch: int,
    key: jax.Array,
    data_dir: pathlib.Path,
) -> tuple[geom.MultiImage, geom.MultiImage]:
    """
    Generate data for a particular dimension using the apebench code.

    args:
        D: dimension
        N: side length of images
        diffusion_coef: parameter for burgers
        convection_coef: parameter for burgers
        subsample: additional steps to generate, which are then subsampled for the output
        n_batch: batch size, or total dataset size in this case
        key: jax key for randomness
        data_dir: director to save/load the data

    returns:
        a tuple of the input and output geometric images
    """
    is_torus = True
    n_timesteps = 50  # 50?
    n_timesteps_int = n_timesteps * subsample  # integrator time steps
    n_warmup_steps = 0  # in 3D, seems like there is an initial problem
    scenario = "diff_burgers"  # diff setting guaranteed to avoid NaNs, so prefer it over norm, phy

    # we multiply the coefs by D to remove the dimension normalizing effect from diff_burgers
    scaled_diff_coef = diffusion_coef * D
    scaled_conv_coef = convection_coef * D

    # information about relationship:
    # https://github.com/Ceyron/exponax/blob/main/exponax/stepper/generic/_convection.py
    # default values are given (diffusion_gamma, convection_delta)
    # apebench.scenarios.physical.Burgers()  # (0.0003,-0.125)
    # apebench.scenarios.normalized.Burgers()  # (0.00003,-0.0125)
    # apebench.scenarios.difficulty.Burgers()  # (1.5,-1.5)

    train_name = f"D{D}_{scenario}_N{N}_n{n_batch}_diffusion{scaled_diff_coef}_convection{scaled_conv_coef}_t{n_timesteps_int}"
    train_path = pathlib.Path(f"{data_dir}") / f"{train_name}_train.npy"
    if not train_path.is_file():
        print(f"Generating data:", train_path)
        key, subkey = random.split(key)
        train_seed, test_seed = random.randint(subkey, shape=(2,), minval=0, maxval=10000)

        apebench.scraper.scrape_data_and_metadata(
            data_dir,  # warning, but it works
            scenario=scenario,
            name=train_name,
            num_spatial_dims=D,
            num_points=N,
            num_warmup_steps=n_warmup_steps,
            num_train_samples=n_batch,
            num_test_samples=0,
            train_seed=int(train_seed),
            test_seed=int(test_seed),
            train_temporal_horizon=n_timesteps_int - 1,
            test_temporal_horizon=n_timesteps_int - 1,
            # diffusion_coef=diffusion_coef,  # for phy_burgers
            # convection_coef=convection_delta,  # for phy_burgers
            # diffusion_alpha=diffusion_coef,  # for norm_burgers
            # convection_beta=convection_coef,  # for norm_burgers
            diffusion_gamma=scaled_diff_coef,  # for diff_burgers
            convection_delta=scaled_conv_coef,  # for diff_burgers
        )

    cpu = jax.devices("cpu")[0]
    # (batch,timesteps,tensor,spatial) -> (batch,timesteps,spatial,tensor)
    train_data = jnp.moveaxis(jax.device_put(jnp.load(train_path)[:, ::subsample], cpu), 2, -1)
    # subsample here for memory efficiency

    # Plot the training data
    # print(train_data.shape)
    # vmax = float(jnp.max(jnp.abs(train_data)))
    # vmin = -1 * vmax

    # print(vmax)

    # timesteps = 10

    # nrows = D
    # ncols = timesteps
    # # figsize is 6 per col, 6 per row, (cols,rows)
    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))
    # for component in range(D):
    #     comp_name = ["x", "y", "z"][component]

    #     print(int(n_timesteps / timesteps))
    #     timestep_images = train_data[0, :: int(n_timesteps / timesteps), ..., component]
    #     print(timestep_images.shape)
    #     for i, timestep_image in enumerate(timestep_images):
    #         geom.GeometricImage(timestep_image, 0, D).plot(
    #             axes[component, i],
    #             title=f"time {i} {comp_name}",
    #             vmin=vmin,
    #             vmax=vmax,
    #             colorbar=True,
    #         )

    # plt.tight_layout()
    # plt.savefig("/data/wgregor4/images/apebench/burgers/spaced_10_steps.png")
    # plt.close(fig)

    # exit()

    constant_fields = geom.MultiImage({}, D, is_torus)
    x0, xt = gc_data.batch_time_series(
        geom.MultiImage({(1, 0): train_data}, D, is_torus), constant_fields, n_timesteps, 1, 1
    )

    return x0, xt


def get_data(
    D: int,
    N: int,
    diffusion_coef: float,
    convection_coef: float,
    subsample: int,  # new
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
        diffusion_coef: parameter for burgers
        convection_coef: parameter for burgers
        subsample: additional steps to generate, which are then subsampled for the output
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
    train_x0, train_xt = get_data_d(
        D, N, diffusion_coef, convection_coef, subsample, n_train, subkey1, data_dir_path / "train"
    )

    val_x0, val_xt = get_data_d(
        D, N, diffusion_coef, convection_coef, subsample, n_val, subkey2, data_dir_path / "val"
    )

    test_x0, test_xt = get_data_d(
        D, N, diffusion_coef, convection_coef, subsample, n_test, subkey3, data_dir_path / "test"
    )

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

    timesteps = len(input_multi_image[((False,), 0)])

    nrows = test_multi_image.D
    ncols = timesteps + 3
    # figsize is 6 per col, 6 per row, (cols,rows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))
    for component in range(test_multi_image.D):
        comp_name = ["x", "y", "z"][component]

        input_multi_image_comp = input_multi_image.get_component(component, timesteps)
        for i, input_image in enumerate(input_multi_image_comp.to_images()):
            if input_image.D == 3:
                input_image = geom.GeometricImage(input_image.data[N // 2], input_image.parity, 2)

            input_image.plot(
                axes[component, i],
                title=f"input {(i+1)-timesteps} {title} {comp_name}",
                vmin=vmin,
                vmax=vmax,
                colorbar=True,
            )

        actual_multi_image_comp = actual_multi_image.get_component(component)
        test_multi_image_comp = test_multi_image.get_component(component)

        actual_image = actual_multi_image_comp.to_images()[0]
        test_image = test_multi_image_comp.to_images()[0]

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
            axes[component, timesteps],
            title=f"output {title} {comp_name}",
            vmin=vmin,
            vmax=vmax,
            colorbar=True,
        )
        test_image.plot(
            axes[component, timesteps + 1],
            title=f"pred {title} {comp_name}",
            vmin=vmin,
            vmax=vmax,
            colorbar=True,
        )
        diff.plot(
            axes[component, timesteps + 2],
            title=diff_title,
            vmin=-diff_max,
            vmax=diff_max,
            colorbar=True,
        )

    plt.tight_layout()
    plt.savefig(save_loc)
    plt.close(fig)


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
        trained_model, _, _, _ = ml.train(
            train_X,
            train_Y,
            ml.Mapper([geom.Losses.NRMSE], residual),
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

        if model_path:
            assert not model_path.is_file() or overwrite_save_model
            # TODO: need to save batch_stats as well
            ml.save(model_path, trained_model)

    assert trained_model is not None
    if images_dir and val_X.D == 2:
        pred_y, _ = ml.Mapper([geom.Losses.NRMSE], residual).map(
            trained_model, val_X.get_one(), batch_stats
        )
        plot_multi_image(
            val_X.get_one(),
            val_Y.get_one(),
            pred_y.get_one(),
            f"{images_dir}{model_name_extended}_D{val_X.D}.png",
            "burgers",
        )

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
    model_path = model_dir / f"{model_name_extended}_model.eqx" if model_dir else None

    key, subkey = random.split(key)
    # rescale set to True, small but notable difference
    if conv_filters_dict is not None and rescale is not None:
        model_dprime = model.convertD(
            conv_filters_dict[tune_X.D],
            rescale,
            subkey,
            upsample_filters=upsample_filters_dict[tune_X.D] if upsample_filters_dict else None,
        )
    else:
        model_dprime = model

    if model_path and model_path.is_file() and not overwrite_save_model:
        tuned_model_dprime = ml.load(model_path, model_dprime)
        tune_batch_stats = batch_stats
    else:
        if tune_X.get_L() > 0:
            # Now treat the trained_model_d as a warmstart and do some additional training
            key, subkey = random.split(key)
            steps_per_epoch = int(math.ceil(tune_X.get_L() / batch_size))
            tuned_model_dprime, tune_batch_stats, _, _ = ml.train(
                tune_X,
                tune_Y,
                ml.Mapper([geom.Losses.NRMSE], residual),
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
        ml.Mapper([geom.Losses.L2_REL, geom.Losses.SMSE], residual),
        tuned_model_dprime,
        test_X,
        test_Y,
        batch_size,
        subkey,
        aux_data=tune_batch_stats,
    )
    l2_rel = tuned_loss[0]
    smse_loss = tuned_loss[1]
    print(f"Tuned Loss rescale=True, D={test_X.D}: lr_rel={l2_rel:.3f}% smse={smse_loss:.3e}\n")

    return l2_rel, smse_loss


def handleArgs() -> argparse.Namespace:
    parser = utils.get_common_parser()
    parser.add_argument(
        "--n-tune-range",
        help="the number of data points in the tuning set",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default="0,1,4,32,128",
    )
    parser.add_argument(
        "--subsample", help="how much to subsample the trajectories", type=int, default=1
    )
    parser.add_argument("-N", help="spatial size", type=int, default=64)
    # defaults for diff_burgers are 1.5 and -1.5, we scale them by D, so for these defaults we
    # unscale by 2 so that D=2 uses the defaults
    parser.add_argument(
        "--diffusion-coef", help="the diffusion coefficient", type=float, default=1.5 / 2
    )
    parser.add_argument(
        "--convection-coef", help="the convection coefficient", type=float, default=-1.5 / 2
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
    parser.add_argument("--wandb-project", help="the wandb project", type=str, default="burgers")
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
# lr_range = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
lr_range = [1e-5, 5e-5, 1e-4, 5e-4]

# D=1 doesn't make sense for a vector field, so we restrict the problem to only this case
train_D = 2
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
free_filters_dict = {}

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
        scale=geom.FilterScaling.ONE,  # for N=2, all pixels are equidistant
    )
    free_filters_dict[D] = geom.get_invariant_filters(
        Ms=[3],
        ks=[0, 2],
        parities=[0],
        D=D,
        operators=group_actions,
        scale=geom.FilterScaling.ONE,
    )

print("Define the models!")
model_list_d = {}
for D in full_D_range:
    key, subkey = random.split(key)
    train_x0, train_xt, _, _, _, _ = get_data(
        D,
        args.N,
        args.diffusion_coef,
        args.convection_coef,
        args.subsample,
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
        #         "lr": 5e-2,  # (D=2,5e-2) (D=2,5e-2) could also be 1e-2
        #         **train_kwargs,
        #     },
        #     {
        #         "lr": 1e-2,
        #         "conv_filters_dict": gaussian_filters_dict,
        #         **test_kwargs,
        #     },
        # ),
        # (
        #     f"lastStepIdentity_D{D}",
        #     {  # train kwargs
        #         "model": models.LastStepIdentity(residual=args.residual),
        #         "lr": 1,
        #         **train_kwargs,
        #     },
        #     {  # tune and eval kwargs
        #         "lr": 1,
        #         **test_kwargs,
        #     },
        # ),
        # (
        #     f"resnet_equiv_42_gaussian_scaling_D{D}",
        #     {  # train kwargs
        #         "model": models.ResNet(
        #             D,
        #             input_keys,
        #             output_keys,
        #             depth=42,
        #             conv_filters=gaussian_filters_dict[D],
        #             use_group_norm=True,
        #             key=subkeys[1],
        #         ),
        #         "lr": {2: 5e-4, 3: 1e-3}[D],  # (D=2,5e-4) (D=2,1e-3) very close
        #         **train_kwargs,
        #     },
        #     {  # tune and eval kwargs
        #         "lr": 5e-5,
        #         "conv_filters_dict": gaussian_filters_dict,
        #         **test_kwargs,
        #     },
        # ),
        # (
        #     f"unetBase_equiv48_gaussian_scaling_D{D}",
        #     {  # train_kwargs
        #         "model": models.UNet(
        #             D,
        #             input_keys,
        #             output_keys,
        #             depth=48,
        #             activation_f=jax.nn.gelu,
        #             conv_filters=gaussian_filters_dict[D],
        #             upsample_filters=upsample_filters_dict[D],
        #             key=subkeys[2],
        #         ),
        #         "lr": {2: 1e-3, 3: 5e-3}[D],  # (D=2,1e-3) (D=3,5e-3)
        #         **train_kwargs,
        #     },
        #     {  # tune and eval kwargs
        #         "lr": 1e-3,
        #         "rescale": geom.Rescaling.VOLUME,
        #         "conv_filters_dict": gaussian_filters_dict,
        #         "upsample_filters_dict": upsample_filters_dict,
        #         **test_kwargs,
        #     },
        # ),
        (
            f"unetBase_equiv48_free_filters_D{D}",
            {  # train_kwargs
                "model": models.UNet(
                    D,
                    input_keys,
                    output_keys,
                    depth=48,
                    activation_f=jax.nn.gelu,
                    conv_filters=free_filters_dict[D],
                    upsample_filters=upsample_filters_dict[D],
                    key=subkeys[2],
                ),
                "lr": {2: 1e-4, 3: 5e-3}[D],  # (D=2,1e-3) (D=3,5e-3)
                # D=3, (n=1, 1e-4 or 5e-4) (n=4,1e-4)
                **train_kwargs,
            },
            {  # tune and eval kwargs
                # 4: could also be 5e-5 also
                # 1: 1e-4, 4: 1e-4 (or 5e-5), 8:
                "lr": 1e-4,
                "rescale": geom.Rescaling.COMPAT_FLEX,
                "conv_filters_dict": free_filters_dict,
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
    args.diffusion_coef,
    args.convection_coef,
    args.subsample,
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
        args.diffusion_coef,
        args.convection_coef,
        args.subsample,
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
            "tune_D1_D2": f"(-,{test_D})",
            "n_points": n_tune,
            "train_or_tune": "tune",
        },
    )
    results_dict[test_D].append(baseline_results)

    key, subkey = random.split(key)
    # (n_trials, benchmark, models, n_results)
    tune_results = ml.benchmark_lr(
        lambda _: tune_data,
        trained_model_list,
        subkey,
        lr_range if args.find_tune_lr else [],
        num_trials=args.n_trials,
        num_results=2,  # l2, rel_error
        is_wandb=args.tune_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        args={
            **vars(args),
            "tune_D1_D2": str(tuple((train_D, test_D))),
            "n_points": n_tune,
            "train_or_tune": "tune",
        },
    )

    results_dict[train_D].append(tune_results)

if args.images_dir is not None:
    model_names_d = {D: [x[0] for x in model_list] for D, model_list in model_list_d.items()}
    plot_results(
        results_dict,
        ["nrmse_loss", "smse_loss"],
        args.n_tune_range,
        model_names_d,
        args.images_dir,
    )
