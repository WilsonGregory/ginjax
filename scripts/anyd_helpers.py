import argparse
import math
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pathlib
import time
from dataclasses import dataclass
from typing import Callable, Self, Sequence

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, PRNGKeyArray
import equinox as eqx
import optax
from torch.utils.data import DataLoader

import ginjax.geometric as geom
from ginjax import models
from ginjax import ml


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


def plot_results(
    results_test_d: dict[int, dict[int, list[dict[str, Float[Array, "n_trials n_results"]]]]],
    results_labels: list[str],
    n_tune_range: tuple[int, ...],
    model_names_d: dict[int, list[str]],
    saveloc: str,
    title: str,
) -> None:
    """
    Plot the results of the heat_equation experiments. For each test_D, create a plot with the
    number of tuning points on the x-axis and the error (either l2 or relative) on the y-axis.

    args:
        results_test_d: The results dict of test_D, train_D, then list (over n_tune_range) of dict
            of key:model_name, val: array of n_trials, n_results.
            The results are l2 error, relative error, train time, train flops
        results_labels: e.g. 'l2', 'relative error'
        n_tune_range: number of fine-tuning points, or training points for the baseline model
        model_names_d: model names for each dimension
        saveloc: beginning of save location
        title: the title of the plot

    returns:
        none
    """
    # group the results by model, across all trained dimensions
    for test_D, results_train_d in results_test_d.items():
        grouped_results = {}
        for results_n_tune_ls in results_train_d.values():
            for results_name in results_n_tune_ls:
                for name, results in results_name.items():
                    print(name, results.shape, name in grouped_results)
                    # might want to swap to display name
                    if name in grouped_results:
                        grouped_results[name].append(results)
                    else:
                        grouped_results[name] = [results]

        linestyles = ["solid", "dotted", "dashed", "dashdot"]
        colors = ["b", "g", "r", "c", "m", "y"]

        for error_idx, ylabel in zip(range(len(results_labels)), results_labels):

            # do them individually
            # figsize is 8 per col, 6 per row, (cols,rows)
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=(8 * 1, 6 * 1))
            assert isinstance(ax, Axes)

            # looping over 'two_layer_gaussian', 'resnet_equiv_42', ...
            for i, (display_name, results) in enumerate(grouped_results.items()):
                results_arr = jnp.stack(results)  # (n_tune_range,n_trials,n_results)
                print(display_name, results_arr.shape)

                # take the mean over trials
                mean_result = jnp.mean(results_arr, axis=1)[:, error_idx]
                ax.plot(
                    mean_result,
                    marker="o",
                    linestyle=linestyles[i % len(linestyles)],
                    label=display_name,
                    color=colors[i % len(colors)],  # was i, currently only model
                )
                # plot stdev range over n_trials
                stdev = jnp.std(results_arr, axis=1)[:, error_idx]
                ax.fill_between(
                    range(len(n_tune_range)),
                    mean_result - stdev,
                    mean_result + stdev,
                    color=colors[i % len(colors)],
                    alpha=0.2,
                )

            ax.legend(fontsize=24)
            ax.set_xlabel("Number of tuning points", fontsize=28)
            ax.set_ylabel(ylabel, fontsize=28)
            ax.set_yscale("log")
            ax.set_xticks(range(len(n_tune_range)), [str(x) for x in n_tune_range])
            ax.set_title(title, fontsize=28)

            plt.tight_layout()
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.savefig(f"{saveloc}warmstart_plot_{test_D}D_{''.join(ylabel.split()).lower()}.png")
            plt.close()


def plot_time_results(
    results_dict: dict[int, dict[int, list[jax.Array]]],
    results_labels: list[str],
    model_names_d: dict[int, list[str]],
    saveloc: str,
    title: str,
) -> None:
    """
    Plot the results of each model versus the time it took to get those results, including the time
    to generate the training data.

    args:
        results_dict: The results dict of test_D, train_D, then a list over n_tune, array n_results
            in this case n_results is smse_mean, smse_std, rel_mean, rel_std, time
        results_labels: e.g. 'l2', 'relative error'
        model_names_d: model names for each dimension
        saveloc: beginning of save location
        title: the title of the plot

    returns:
        none
    """
    # group the results by model, across all trained dimensions
    grouped_results = {}
    for test_D, results_train_d in results_dict.items():
        grouped_results[test_D] = {}
        for train_D, results in results_train_d.items():
            for i, name in enumerate(model_names_d[train_D]):
                name_trimmed = name[:-3]  # this assumes that all models end in _D1, _D2, or _D3
                display_name = "UNet Baseline" if train_D == test_D else "UNet Pretrained"

                if name_trimmed in grouped_results[test_D]:
                    # (n_tune,n_trials,n_results)
                    grouped_results[test_D][name_trimmed].append(
                        (display_name, jnp.stack(results)[:, :, 0, i])
                    )
                else:
                    grouped_results[test_D][name_trimmed] = [
                        (display_name, jnp.stack(results)[:, :, 0, i])
                    ]

    # figsize is 8 per col, 6 per row, (cols,rows)
    nrows = len(list(grouped_results.keys()))
    ncols = len(results_labels)

    for test_D, results_by_model in grouped_results.items():
        for error_idx, ylabel in zip(range(len(results_labels)), results_labels):

            _, ax = plt.subplots(nrows=1, ncols=1, figsize=(8 * 1, 6 * 1))
            linestyles = ["solid", "dotted", "dashed", "dashdot"]
            colors = ["b", "g", "r", "c", "m", "y"]

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
                        linestyle=linestyles[j % len(linestyles)],
                        label=display_name,
                        color=colors[j % len(colors)],  # was i, currently only model
                    )
                    ax.fill_between(
                        times,
                        mean_result - stdev,
                        mean_result + stdev,
                        color=colors[j % len(colors)],
                        alpha=0.2,
                    )

            ax.legend(fontsize=24)
            ax.set_xlabel("Total time (minutes)", fontsize=28)
            ax.set_ylabel(ylabel, fontsize=28)
            ax.set_yscale("log")
            ax.set_title(title, fontsize=28)

            plt.tight_layout()
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.savefig(
                f"{saveloc}warmstart_time_plot_{test_D}D_{''.join(ylabel.split()).lower()}.png"
            )
            plt.close()


def generate_filters(
    D_range: Sequence[int], ks: Sequence[int]
) -> tuple[dict[int, geom.MultiImage], dict[int, geom.MultiImage]]:
    """
    For the range of dimensions, generate the free filters (M=3) and upsample filters (M=3) of
    scaling geom.FilterScaling.ONE.

    args:
        D_range: range of integers to generate filters for

    returns:
        dictionary mapping dimension D to filters of that dimension.
    """
    upsample_filters_dict = {}
    free_filters_dict = {}

    for D in D_range:
        group_actions = geom.make_all_operators(D)
        upsample_filters_dict[D] = geom.get_invariant_filters(
            Ms=[2],
            ks=ks,
            parities=[0],
            D=D,
            operators=group_actions,
            scale=geom.FilterScaling.ONE,  # for N=2, all pixels are equidistant
        )
        free_filters_dict[D] = geom.get_invariant_filters(
            Ms=[3],
            ks=ks,
            parities=[0],
            D=D,
            operators=group_actions,
            scale=geom.FilterScaling.ONE,
        )

    return free_filters_dict, upsample_filters_dict


def trim_trial_name(model_name: str) -> str:
    """
    Remove the trial from the model name.
    """
    start = model_name.find("_trial")
    stop = model_name.find("_", start + len("_trial"))
    return model_name[:start] + model_name[stop:]


def train_model(
    data: tuple[DataLoader[ml.MultiImageDataset], DataLoader[ml.MultiImageDataset]],
    model_name: str,
    model: models.AnyDimensionalModel,
    lr: float,
    train_loss_f: geom.Losses,
    batch_size: int,
    epochs: int,
    model_dir: pathlib.Path | None,
    overwrite_save_model: bool,
    images_dir: str | None,
    has_aux: bool = False,
    verbose: int = 1,
    is_wandb: bool = False,
    wandb_project: str = "",
    wandb_entity: str = "",
    args: dict = {},
) -> tuple[models.MultiImageModule, float, int]:
    """
    Train the model.

    args:
        data: train and val dataloaders as a tuple
        model_name: name of the model we are training
        model: the any dimensional model to train
        lr: learning rate
        train_loss_f: the loss function to use for tuning
        batch_size: the batch size
        epochs: numbers of epochs (full passes through the tune data) to tune for
        model_dir: path to save or load the model from
        overwrite_save_model: whether to retrain and save a new tuned model even if there is
            already one saved
        images_dir: if provided and D==2, save one data point image
        has_aux: whether the model has auxilliary data like batch stats
        verbose: verbosity level for training
        is_wandb: whether to turn on wandb tracking

    returns:
        tuple of the trained model, the train wall clock time, and the train flops estimate
    """
    if has_aux:
        raise NotImplementedError(f"train_model: got has_aux={has_aux} which is not implemented.")

    train_dataloader, val_dataloader = data
    assert isinstance(train_dataloader.dataset, ml.MultiImageDataset)
    assert isinstance(val_dataloader.dataset, ml.MultiImageDataset)

    D = train_dataloader.dataset.D
    L = len(train_dataloader.dataset)
    batch_stats = eqx.nn.State(model) if has_aux else None
    model_path = model_dir / f"{model_name}.eqx" if model_dir else None

    print(f"{model_name} params: {models.count_params(model):,}")

    if model_path and model_path.is_file() and not overwrite_save_model:
        trained_model, further_args = ml.load_plus(model_path, model)
        train_time = further_args["train_time"]
    else:
        if L > 0:
            steps_per_epoch = int(math.ceil(L / batch_size))
            trained_model, _, _, _, train_time = ml.train_dl_wandb(
                train_dataloader,
                ml.Mapper([train_loss_f]),
                model,
                stop_condition=ml.EpochStop(epochs, verbose=verbose),
                optimizer=optax.adamw(
                    optax.warmup_cosine_decay_schedule(
                        1e-8, lr, 5 * steps_per_epoch, epochs * steps_per_epoch, 1e-7
                    ),
                    weight_decay=1e-5,
                ),
                val_dataloader=val_dataloader,
                val_map_and_loss=ml.Mapper([geom.Losses.NRMSE], eps=1e-5),
                aux_data=batch_stats,
                is_wandb=is_wandb,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                model_name=trim_trial_name(model_name),
                args=args,
            )
        else:
            trained_model = model
            train_time = 0.0

        if model_path:
            assert not model_path.is_file() or overwrite_save_model
            # TODO: need to save batch_stats as well
            ml.save_plus(model_path, trained_model, {"train_time": train_time})

    assert trained_model is not None
    if images_dir and D == 2:
        val_x_one, val_y_one = next(iter(val_dataloader))
        pred_y, _ = ml.Mapper([geom.Losses.NRMSE]).map(trained_model, val_x_one, batch_stats)
        plot_multi_image(
            val_x_one,
            val_y_one,
            pred_y.get_one(),
            f"{images_dir}{model_name}.png",
            "burgers",
        )

    spatial_dims = train_dataloader.dataset.get_spatial_dims()
    # flops/sample * number of samples * epochs
    flops = trained_model.get_flops(**{"spatial_dims": spatial_dims})
    total_train_flops = flops * L * epochs
    print(
        f"L={L}, epochs={epochs}: {flops / 1_000_000_000:,.3f} gflops/sample, "
        f"{total_train_flops / 1_000_000_000:,.3f} total gflops"
    )

    return trained_model, train_time, total_train_flops


def train_all_models(
    data: tuple[DataLoader[ml.MultiImageDataset], DataLoader[ml.MultiImageDataset]],
    model_list: list[tuple[str, models.AnyDimensionalModel, dict, dict, dict]],
    lr_range: list[float] | None,
    kwargs_idx: int,
    n_points: int,
    data_time: float,
    args: argparse.Namespace,
) -> list[tuple[str, models.AnyDimensionalModel, dict, dict, dict, float, int]]:
    """
    Train all the models in the model list.

    args:
        data: input and output multi image pairs for train and val
        key: key for randomness
        model_list: list of tuples of (model_name, train_kwargs, test_kwargs)
        lr_range: list of lr values to test if args.find_train_lr is true

    returns:
        list of tuples of (model_name, tune_and_eval, test_kwargs, train_time)
    """
    train_dataloader, _ = data
    assert isinstance(train_dataloader.dataset, ml.MultiImageDataset)
    train_D = train_dataloader.dataset.D
    L = len(train_dataloader.dataset)

    trained_models = []
    for model_name, model, train_kwargs, baseline_kwargs, test_kwargs in model_list:
        kwargs = [train_kwargs, baseline_kwargs, test_kwargs][kwargs_idx]

        model_name += f"_D{train_D}_L{L}"

        if lr_range is None:
            lr_range = [kwargs["lr"][train_D][n_points]]

        for lr in lr_range:
            _kwargs = {
                **kwargs,
                "lr": lr,
                "args": {
                    **vars(args),
                    "model_name": model_name,
                    "lr": lr,
                    "D": train_D,
                    "n_points": n_points,
                    "train_or_tune": ["train", "baseline", "tune"][kwargs_idx],
                },
            }  # copy kwargs and overwrite lr
            trained_model, train_time, train_flops = train_model(data, model_name, model, **_kwargs)

            trained_models.append(
                (
                    model_name,
                    trained_model,
                    train_kwargs,
                    baseline_kwargs,
                    test_kwargs,
                    train_time + data_time,
                    train_flops,
                )
            )

    return trained_models


def convert_and_tune(
    data: tuple[DataLoader[ml.MultiImageDataset], DataLoader[ml.MultiImageDataset]],
    key: jax.Array,
    model_list: list[tuple[str, models.AnyDimensionalModel, dict, dict, dict, float, int]],
    lr_range: list[float] | None,
    rescale_list: list[geom.Rescaling],
    n_points: int,
    data_time: float,
    args: argparse.Namespace,
) -> list[tuple[str, models.AnyDimensionalModel, dict, dict, dict, float, int]]:
    """
    For a given model, convert lower dimensional model to a higher one, then tune (train) the model
    on the tuning data.

    args:
        data: input and output multi images for tune, val, and test data sets
        key: key for randomness
        model_list: models to convert and tune, tuple of
            (model_name,model,train_kwargs,baseline_kwargs,test_kwargs,pretrain_time)
        lr_range: when hyperparameter tuning, list of learning rates to try
        rescale_list: list of rescalings to perform
        n_points:
        data_time:
        args:

    returns:
        tuple of smse_mean, smse_std, relative error mean, relative error std, tuning time
    """
    tune_dataloader, _ = data
    assert isinstance(tune_dataloader.dataset, ml.MultiImageDataset)

    tune_D = tune_dataloader.dataset.D

    # Convert the models that need converting
    converted_model_list = []
    convert_times = []
    pretrain_flops = []
    for (
        model_name,
        model,
        train_kwargs,
        baseline_kwargs,
        test_kwargs,
        pretrain_time,
        pretrain_flop,
    ) in model_list:
        train_D = model.D

        for rescale in rescale_list:
            start_time = time.time()

            conv_filters_d = test_kwargs["conv_filters_dict"]
            upsample_filters_d = (
                test_kwargs["upsample_filters_dict"]
                if "upsample_filters_dict" in test_kwargs
                else None
            )
            key, subkey = random.split(key)
            model_dprime = model.convertD(
                conv_filters_d[tune_D],
                rescale,
                subkey,
                upsample_filters=upsample_filters_d[tune_D] if upsample_filters_d else None,
            )
            convert_times.append(time.time() - start_time + pretrain_time)  # this should be tiny
            # convert flops is equal to about the number of params of the lower dimensional model
            pretrain_flops.append(pretrain_flop + models.count_params(model))
            _model_name = f"{model_name}_rescale{rescale.name}"

            # get the correct learning rate, remove rescale, conv_filters_dict, upsample_filters_dict
            _test_kwargs = {**test_kwargs, "lr": test_kwargs["lr"][train_D]}
            _test_kwargs.pop("conv_filters_dict", None)
            _test_kwargs.pop("upsample_filters_dict", None)

            converted_model_list.append(
                (_model_name, model_dprime, train_kwargs, baseline_kwargs, _test_kwargs)
            )

    # Fine tune the converted models
    tuned_models = train_all_models(
        data, converted_model_list, lr_range, 2, n_points, data_time, args
    )
    merged_data = []
    for (a, b, c, d, e, train_time, train_flops), convert_time, pretrain_flop in zip(
        tuned_models, convert_times, pretrain_flops
    ):
        merged_data.append((a, b, c, d, e, train_time + convert_time, train_flops + pretrain_flop))

    return merged_data


def eval(
    test_dataloader: DataLoader[ml.MultiImageDataset],
    model_list: list[tuple[str, models.AnyDimensionalModel, dict, dict, dict, float, int]],
) -> dict[str, Float[Array, "n_trials n_results"]]:
    models_dict = {}
    for model_name, model, _, _, _, train_time, train_flops in model_list:
        # (losses,)
        rel_mean, smse_mean = ml.map_loss_in_batches_dl(
            ml.Mapper([geom.Losses.L2_REL, geom.Losses.SMSE], eps=1e-9), model, test_dataloader
        )
        print(f"Eval {model_name}: {smse_mean:.3e} ({rel_mean:.3f}%)\n")

        tuned_loss = jnp.array([rel_mean, smse_mean, train_time, train_flops])

        model_name_trim = trim_trial_name(model_name)
        if model_name_trim in models_dict:
            models_dict[model_name_trim].append(tuned_loss)
        else:
            models_dict[model_name_trim] = [tuned_loss]

    return {k: jnp.stack(v) for k, v in models_dict.items()}


def run_anyd(
    train_D_range: tuple[int],
    test_D_range: tuple[int],
    n_tune_range: tuple[int],
    args: argparse.Namespace,
    key: jax.Array,
    get_data: Callable[
        [int, int, int, int, int, PRNGKeyArray],
        tuple[
            DataLoader[ml.MultiImageDataset],
            DataLoader[ml.MultiImageDataset],
            DataLoader[ml.MultiImageDataset],
            float,
        ],
    ],
    model_list_d: dict[int, list[tuple[str, models.AnyDimensionalModel, dict, dict, dict]]],
    pretrain_lr_range: list[float] | None,
    finetune_lr_range: list[float] | None,
    rescale_list: list[geom.Rescaling],
    batch_size_d: dict[int, int],
):
    """
    Run the full battery of the any-dimensional test. Train the models, convert them to higher
    dimensions and then tune/train a baseline model in that higher dimension.
    args:
        train_D_range: initial dimensions to train on
        test_D_range: test dimensions to tune and evaluate on
        n_tune_range: range of tuning dataset sizes
        args: all the command line args
        key: jax key for randomness
        get_data: a function that takes dimension, n_train, n_val, n_test, and random key, returns
            input and output multi images for train, val, and test, as well data generation time for train
    """
    pretrain_lr = pretrain_lr_range is not None
    n_results = 3
    # train the models, i.e. the warmstart lower dimensional models
    print("Train the models (warmstart)!")
    pretrain_model_list_d = {}
    for train_D in train_D_range:
        key, subkey = random.split(key)
        train_dataloader, val_dataloader, _, pretrain_data_time = get_data(
            train_D, args.n_train, args.n_val, 0, batch_size_d[train_D], subkey
        )

        train_data = (train_dataloader, val_dataloader)
        key, subkey = random.split(key)
        pretrain_model_list_d[train_D] = train_all_models(
            train_data,
            model_list_d[train_D],
            pretrain_lr_range,
            0,
            args.n_train,
            pretrain_data_time,
            args,
        )

    if pretrain_lr:
        return

    # evaluate the models
    print("Tune and evaluate the models!")
    results_dict = {}
    for test_D in test_D_range:
        results_dict[test_D] = {k: [] for k in ((test_D,) + train_D_range)}
        for n_tune in n_tune_range:
            print(f"D={test_D}, n_tune={n_tune}.\n")
            # the data is saved, so this is still reasonably efficient
            key, subkey = random.split(key)
            # TODO: as n_tune grows, it may later include data points that were used for val/test
            tune_dl, tune_val_dl, tune_test_dl, tune_data_time = get_data(
                test_D, n_tune, args.n_val, args.n_test, batch_size_d[test_D], subkey
            )

            # need to train the baseline model on tune_x0, etc. aka models without the warmstart
            baseline_trained_models = train_all_models(
                (tune_dl, tune_val_dl),
                model_list_d[test_D],
                finetune_lr_range,
                1,
                n_tune,
                tune_data_time,
                args,
            )
            # do the eval
            results_dict[test_D][test_D].append(eval(tune_test_dl, baseline_trained_models))

            for train_D in train_D_range:
                if train_D == test_D:
                    continue

                key, subkey = random.split(key)
                finetuned_models = convert_and_tune(
                    (tune_dl, tune_val_dl),
                    subkey,
                    pretrain_model_list_d[train_D],
                    finetune_lr_range,
                    rescale_list,
                    n_tune,
                    tune_data_time,
                    args,
                )

                key, subkey = random.split(key)
                results_dict[test_D][train_D].append(eval(tune_test_dl, finetuned_models))

    # if args.images_dir is not None:
    #     model_names_d = {D: [x[0] for x in model_list] for D, model_list in model_list_d.items()}
    #     plot_results(
    #         results_dict,
    #         ["L2 error", "Relative error"],
    #         n_tune_range,
    #         model_names_d,
    #         args.images_dir,
    #         f"By tuning points",  # TODO: want to say what the dimensions are
    #     )

    #     plot_time_results(
    #         results_dict,
    #         ["L2 error", "Relative error"],
    #         model_names_d,
    #         args.images_dir,
    #         f"By time",  # TODO: want to say what the dimensions are
    #     )
