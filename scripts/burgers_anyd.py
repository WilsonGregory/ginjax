import argparse
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pathlib
import time

import jax
import jax.numpy as jnp
from jax import random
import apebench

import ginjax.geometric as geom
from ginjax import models
from ginjax import ml
import ginjax.data as gc_data
from ginjax import utils
from anyd_helpers import train_all_models, tune_and_eval


def get_data_d(
    D: int,
    N: int,
    diffusion_coef: float,
    convection_coef: float,
    n_timesteps: int,
    subsample: int,  # new
    n_batch: int,
    key: jax.Array,
    data_dir: pathlib.Path,
) -> tuple[geom.MultiImage, geom.MultiImage, float]:
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
        start_time = time.time()
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

        data_generation_time = time.time() - start_time
        print(f"Finished: {data_generation_time} seconds.")
    else:
        # dictionary by D, then n_batch
        # Since these values aren't saved with the data, hardcode them from some previous run.
        data_generation_times = {
            64: {
                2: {
                    0: jnp.array([1.0652430057525635, 0.40758848190307617]),
                    8: jnp.array([7.801406621932983, 3.1244280338287354]),
                },
                3: {
                    0: jnp.array([0.6283245086669922, 2.0740256309509277, 0.9826910495758057]),
                    1: jnp.array([5.805211067199707]),
                    4: jnp.array([7.013747453689575]),
                    8: jnp.array([8.912535429000854, 5.779284477233887, 7.701824903488159]),
                },
            },
            96: {  # 96 was also done with n_timesteps=25, rather than 50
                2: {
                    0: jnp.array([0.6557881832122803, 0.3366715908050537]),
                    8: jnp.array([4.890353679656982, 2.4403231143951416]),
                },
                3: {
                    0: jnp.array([0.3930032253265381, 0.38234496116638184]),
                    1: jnp.array([0]),
                    4: jnp.array([0]),
                    8: jnp.array([15.993224382400513]),
                },
            },
        }
        data_generation_time = float(jnp.mean(data_generation_times[N][D][n_batch]))

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

    return x0, xt, data_generation_time


def get_data(
    D: int,
    N: int,
    diffusion_coef: float,
    convection_coef: float,
    n_timesteps: int,
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
    float,
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
    train_x0, train_xt, train_data_time = get_data_d(
        D,
        N,
        diffusion_coef,
        convection_coef,
        n_timesteps,
        subsample,
        n_train,
        subkey1,
        data_dir_path / "train",
    )

    val_x0, val_xt, _ = get_data_d(
        D,
        N,
        diffusion_coef,
        convection_coef,
        n_timesteps,
        subsample,
        n_val,
        subkey2,
        data_dir_path / "val",
    )

    test_x0, test_xt, _ = get_data_d(
        D,
        N,
        diffusion_coef,
        convection_coef,
        n_timesteps,
        subsample,
        n_test,
        subkey3,
        data_dir_path / "test",
    )

    return train_x0, train_xt, val_x0, val_xt, test_x0, test_xt, train_data_time


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
        results_dict: The results dict of test_D, then a list over n_tune, array n_results
            in this case n_results is smse_mean, smse_std, rel_mean, rel_std
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
            display_name = "UNet Baseline" if train_D == test_D else "UNet Pretrained"

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
    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    colors = ["b", "g", "r", "c", "m", "y"]

    for error_idx, ylabel in zip(range(len(results_labels)), results_labels):
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(8 * 1, 6 * 1))

        assert isinstance(ax, Axes)

        # looping over 'two_layer_gaussian', 'resnet_equiv_42', ...
        for i, model_results in enumerate(results_by_model.values()):
            for j, (display_name, results_arr) in enumerate(model_results):

                mean_result = jnp.mean(results_arr, axis=1)[:, error_idx * 2]
                stdev = jnp.mean(results_arr, axis=1)[:, error_idx * 2 + 1]
                ax.plot(
                    mean_result,
                    marker="o",
                    linestyle=linestyles[j],
                    label=display_name,
                    color=colors[j],  # was i, currently only model
                )
                ax.fill_between(
                    range(len(n_tune_range)),
                    mean_result - stdev,
                    mean_result + stdev,
                    color=colors[j],
                    alpha=0.2,
                )

        ax.legend(fontsize=24)
        ax.set_xlabel("Number of tuning points", fontsize=28)
        ax.set_ylabel(ylabel, fontsize=28)
        ax.set_yscale("log")
        ax.set_xticks(range(len(n_tune_range)), [str(x) for x in n_tune_range])
        ax.set_title(f"Burgers' 2D->3D, by tuning points", fontsize=28)

        plt.tight_layout()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(
            f"{saveloc}burgers_warmstart_plot_{test_D}D_{''.join(ylabel.split()).lower()}.png"
        )
        plt.close()


def plot_time_results(
    results_dict: dict[int, list[jax.Array]],
    results_labels: list[str],
    model_names_d: dict[int, list[str]],
    saveloc: str,
) -> None:
    """
    Plot the results of each model versus the time it took to get those results, including the time
    to generate the training data.

    args:
        results_dict: The results dict of test_D, then a list over n_tune, array n_results
            in this case n_results is smse_mean, smse_std, rel_mean, rel_std, time
        results_labels: e.g. 'l2', 'relative error'
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
            display_name = "UNet Baseline" if train_D == test_D else "UNet Pretrained"

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
    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    colors = ["b", "g", "r", "c", "m", "y"]

    for error_idx, ylabel in zip(range(len(results_labels)), results_labels):
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(8 * 1, 6 * 1))
        assert isinstance(ax, Axes)

        # looping over 'two_layer_gaussian', 'resnet_equiv_42', ...
        for i, model_results in enumerate(results_by_model.values()):
            for j, (display_name, results_arr) in enumerate(model_results):

                # mean is over trials
                times = jnp.mean(results_arr, axis=1)[:, -1] / 60
                mean_result = jnp.mean(results_arr, axis=1)[:, error_idx * 2]
                stdev = jnp.mean(results_arr, axis=1)[:, error_idx * 2 + 1]
                ax.plot(
                    times,
                    mean_result,
                    marker="o",
                    linestyle=linestyles[j],
                    label=display_name,
                    color=colors[j],  # was i, currently only model
                )
                ax.fill_between(
                    times,
                    mean_result - stdev,
                    mean_result + stdev,
                    color=colors[j],
                    alpha=0.2,
                )

        ax.legend(fontsize=24)
        ax.set_xlabel("Total time (minutes)", fontsize=28)
        ax.set_ylabel(ylabel, fontsize=28)
        ax.set_yscale("log")
        ax.set_title(f"Burgers' 2D->3D, by time", fontsize=28)

        plt.tight_layout()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(
            f"{saveloc}burgers_warmstart_time_plot_{test_D}D_{''.join(ylabel.split()).lower()}.png"
        )
        plt.close()


# Something like
# CUDA_VISIBLE_DEVICES=0,1 time python3 scripts/burgers_anyd.py --data /data/wgregor4/apebench/burgers/
# --n-train 8 --n-val 8 --n-test 8 --batch 2 --model-dir /data/wgregor4/runs/burgers_anyd/
def handleArgs() -> argparse.Namespace:
    parser = utils.get_common_parser()
    parser.add_argument(
        "--n-tune-range",
        help="the number of data points in the tuning set",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default="0,1,4,8",
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
        "--n-timesteps", help="the number of timesteps in each trajectory", type=int, default=50
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
n_results = 5

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
    train_x0, train_xt, _, _, _, _, _ = get_data(
        D,
        args.N,
        args.diffusion_coef,
        args.convection_coef,
        args.n_timesteps,
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
                "lr": 1e-4,
                # D=3, for all of them (and tuning) its just 1e-4
                **train_kwargs,
            },
            {  # tune and eval kwargs
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
train_x0, train_xt, val_x0, val_xt, _, _, pretrain_data_time = get_data(
    train_D,
    args.N,
    args.diffusion_coef,
    args.convection_coef,
    args.n_timesteps,
    args.subsample,
    args.n_train,
    args.n_val,
    0,
    subkey,
    args.data,
)

train_data = (train_x0, train_xt, val_x0, val_xt)
key, subkey = random.split(key)
trained_model_list, train_times = train_all_models(
    train_data, subkey, model_list_d[train_D], lr_range, args
)
train_times = np.array(train_times)[:, None]  # (models,1)
# will want to re-run, but for now this is 2gpus, batch=8 1587.40561104 1gpu batch=8 1026.90299463
train_times = np.array([1026.90299463]).reshape((1, 1))  # (models,1)
# (models,n_results)
train_times = np.concat([np.zeros((len(train_times), n_results - 1)), train_times], axis=1)

if args.find_train_lr:
    exit()

# evaluate the models
print("Tune and evaluate the models!")
results_dict = {k: [] for k in (test_D, train_D)}
for n_tune in args.n_tune_range:
    print(f"D={test_D}, n_tune={n_tune}.\n")
    # the data is saved, so this is still reasonably efficient
    key, subkey = random.split(key)
    *tune_data, tune_data_time = get_data(
        test_D,
        args.N,
        args.diffusion_coef,
        args.convection_coef,
        args.n_timesteps,
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
        num_results=n_results,  # smse_mean, smse_std, rel_mean, rel_std, train_time
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
    baseline_results[..., -1] += tune_data_time
    results_dict[test_D].append(baseline_results)

    key, subkey = random.split(key)
    # (n_trials, benchmark, models, n_results)
    tune_results = ml.benchmark_lr(
        lambda _: tune_data,
        trained_model_list,
        subkey,
        lr_range if args.find_tune_lr else [],
        num_trials=args.n_trials,
        num_results=n_results,  # smse_mean, smse_std, rel_mean, rel_std, train_time
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
    tune_results += train_times[None, None]
    tune_results[..., -1] += pretrain_data_time + tune_data_time

    results_dict[train_D].append(tune_results)

if args.images_dir is not None:
    model_names_d = {D: [x[0] for x in model_list] for D, model_list in model_list_d.items()}
    plot_results(
        results_dict,
        ["L2 error", "Relative error"],
        args.n_tune_range,
        model_names_d,
        args.images_dir,
    )

    plot_time_results(results_dict, ["L2 error", "Relative error"], model_names_d, args.images_dir)
