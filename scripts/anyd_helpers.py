import argparse
import math
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pathlib
import time
from typing import Callable

import jax
import jax.numpy as jnp
from jax import random
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
    results_dict: dict[int, dict[int, list[jax.Array]]],
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
        results_dict: The results dict of test_D, train_D, then a list over n_tune, array n_results
            in this case n_results is smse_mean, smse_std, rel_mean, rel_std
        results_labels: e.g. 'l2', 'relative error'
        n_tune_range: number of fine-tuning points, or training points for the baseline model
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
    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    colors = ["b", "g", "r", "c", "m", "y"]

    for test_D, results_by_model in grouped_results.items():
        for error_idx, ylabel in zip(range(len(results_labels)), results_labels):

            # do them individually
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
            ax.set_title(title, fontsize=28)

            plt.tight_layout()
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.savefig(
                f"{saveloc}warmstart_time_plot_{test_D}D_{''.join(ylabel.split()).lower()}.png"
            )
            plt.close()


def train_model(
    data: tuple[DataLoader[ml.MultiImageDataset], DataLoader[ml.MultiImageDataset]],
    model_name: str,
    model: models.AnyDimensionalModel,
    lr: float,
    train_loss_f: geom.Losses,
    residual: bool,
    batch_size: int,
    epochs: int,
    model_dir: pathlib.Path | None,
    overwrite_save_model: bool,
    images_dir: str | None,
    has_aux: bool = False,
    verbose: int = 1,
    is_wandb: bool = False,
) -> tuple[models.MultiImageModule, float]:
    """
    Train the model.

    args:
        data: train and val dataloaders as a tuple
        model_name: name of the model we are training
        model: the any dimensional model to train
        lr: learning rate
        train_loss_f: the loss function to use for tuning
        residual: whether the model is or should learn the residual between input/output
        batch_size: the batch size
        epochs: numbers of epochs (full passes through the tune data) to tune for
        model_dir: path to save or load the model from
        overwrite_save_model: whether to retrain and save a new tuned model even if there is
            already one saved
        images_dir: if provided and D==2, save one data point image
        has_aux: whether the model has auxilliary data like batch stats
        verbose: verbosity level for training
        is_wandb: whether to turn on wandb tracking
    """
    train_dataloader, val_dataloader = data
    assert isinstance(train_dataloader.dataset, ml.MultiImageDataset)

    D = train_dataloader.dataset.D
    L = len(train_dataloader.dataset)
    N = train_dataloader.dataset.get_N()
    batch_stats = eqx.nn.State(model) if has_aux else None
    model_name_extended = f"{model_name}_L{L}_N{N}_e{epochs}"
    model_path = model_dir / f"{model_name_extended}_model.eqx" if model_dir else None

    print(f"{model_name_extended} params: {models.count_params(model):,}")

    if model_path and model_path.is_file() and not overwrite_save_model:
        trained_model, further_args = ml.load_plus(model_path, model)
        train_time = further_args["train_time"]
    else:
        steps_per_epoch = int(math.ceil(L / batch_size))
        trained_model, _, _, _, train_time = ml.train_dl(
            train_dataloader,
            ml.Mapper([train_loss_f], residual),
            model,
            stop_condition=ml.EpochStop(epochs, verbose=verbose),
            optimizer=optax.adamw(
                optax.warmup_cosine_decay_schedule(
                    1e-8, lr, 5 * steps_per_epoch, epochs * steps_per_epoch, 1e-7
                ),
                weight_decay=1e-5,
            ),
            val_dataloader=val_dataloader,
            val_map_and_loss=ml.Mapper([geom.Losses.NRMSE], residual, eps=1e-5),
            aux_data=batch_stats,
            is_wandb=is_wandb,
        )

        if model_path:
            assert not model_path.is_file() or overwrite_save_model
            # TODO: need to save batch_stats as well
            ml.save_plus(model_path, trained_model, {"train_time": train_time})

    assert trained_model is not None
    if images_dir and D == 2:
        val_x_one, val_y_one = next(iter(val_dataloader))
        pred_y, _ = ml.Mapper([geom.Losses.NRMSE], residual).map(
            trained_model, val_x_one, batch_stats
        )
        plot_multi_image(
            val_x_one,
            val_y_one,
            pred_y.get_one(),
            f"{images_dir}{model_name_extended}_D{D}.png",
            "burgers",
        )

    return trained_model, train_time


def train_all_models(
    data: tuple[DataLoader[ml.MultiImageDataset], DataLoader[ml.MultiImageDataset]],
    key: jax.Array,
    model_list: list[tuple[str, dict, dict]],
    lr_range: list[float],
    args: argparse.Namespace,
) -> list[tuple[str, Callable, dict, float]]:
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

    trained_models = []
    for model_name, _train_kwargs, _test_kwargs in model_list:
        _train_kwargs["lr"] = _train_kwargs["lr"][train_D][args.n_train]
        key, subkey = random.split(key)
        if args.find_train_lr:

            def train_f(data, subkey, name, **kwargs):
                train_model(data, name, **kwargs)
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
                    "n_points": args.n_train,
                    "train_or_tune": "train",
                },
            )
        else:
            trained_model, train_time = train_model(data, model_name, **_train_kwargs)
            _test_kwargs = {**_test_kwargs, "model": trained_model}
            trained_models.append((model_name, tune_and_eval, _test_kwargs, train_time))

    return trained_models


def tune_and_eval(
    data: tuple[
        DataLoader[ml.MultiImageDataset],
        DataLoader[ml.MultiImageDataset],
        DataLoader[ml.MultiImageDataset],
    ],
    key: jax.Array,
    model_name: str,
    model: models.AnyDimensionalModel,
    lr: float,
    conv_filters_dict: dict[int, geom.MultiImage] | None,
    rescale: geom.Rescaling | None,
    train_loss_f: geom.Losses,
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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, float]:
    """
    For a given model, convert lower dimensional model to a higher one, tune (train) the model on
    the tuning data, then evaluate it on the test data.

    args:
        data: input and output multi images for tune, val, and test data sets
        key: key for randomness
        model_name: name of the model we are training
        model: the any dimensional model
        lr: learning rate
        conv_filters_dict: dictionary of conv_filters by dimension
        rescale: how to rescale the convolution filter coefficients
        train_loss_f: the loss function to use for tuning
        residual: whether the model is or should learn the residual between input/output
        batch_size: the batch size
        epochs: numbers of epochs (full passes through the tune data) to tune for
        model_dir: path to save or load the model from
        overwrite_save_model: whether to retrain and save a new tuned model even if there is
            already one saved
        images_dir: currently unused
        upsample_filters_dict: dictionary by dimension of the upsample filters
        has_aux: whether the model has auxilliary data like batch stats
        verbose: verbosity level for training
        is_wandb: whether to turn on wandb tracking

    returns:
        tuple of smse_mean, smse_std, relative error mean, relative error std, tuning time
    """
    tune_dataloader, val_dataloader, test_dataloader = data
    assert isinstance(tune_dataloader.dataset, ml.MultiImageDataset)

    tune_D = tune_dataloader.dataset.D
    L = len(tune_dataloader.dataset)
    N = tune_dataloader.dataset.get_N()
    batch_stats = eqx.nn.State(model) if has_aux else None
    model_name_extended = f"{model_name}_tuneD{tune_D}_L{L}_N{N}_e{epochs}"

    key, subkey = random.split(key)
    # rescale set to True, small but notable difference
    if conv_filters_dict is not None and rescale is not None:
        start_time = time.time()
        model_dprime = model.convertD(
            conv_filters_dict[tune_D],
            rescale,
            subkey,
            upsample_filters=upsample_filters_dict[tune_D] if upsample_filters_dict else None,
        )
        convert_time = time.time() - start_time  # this should be tiny
        model_name_extended += f"_rescale{rescale.name}"
    else:
        model_dprime = model
        convert_time = 0.0

    model_path = model_dir / f"{model_name_extended}_model.eqx" if model_dir else None
    print(f"tuning: {model_name_extended}")

    if model_path and model_path.is_file() and not overwrite_save_model:
        tuned_model_dprime, further_args = ml.load_plus(model_path, model_dprime)
        tune_time = further_args["train_time"]
        tune_batch_stats = batch_stats
    else:
        if L > 0:
            # Now treat the trained_model_d as a warmstart and do some additional training
            key, subkey = random.split(key)
            steps_per_epoch = int(math.ceil(L / batch_size))
            tuned_model_dprime, tune_batch_stats, _, _, tune_time = ml.train_dl(
                tune_dataloader,
                ml.Mapper([train_loss_f], residual, eps=1e-9),
                model_dprime,
                stop_condition=ml.EpochStop(epochs, verbose=verbose),
                optimizer=optax.adamw(
                    optax.warmup_cosine_decay_schedule(
                        1e-8, lr, 5 * steps_per_epoch, epochs * steps_per_epoch, 1e-7
                    ),
                    weight_decay=1e-5,
                ),
                val_dataloader=val_dataloader,
                val_map_and_loss=ml.Mapper([geom.Losses.NRMSE], residual, eps=1e-9),
                aux_data=batch_stats,
                is_wandb=is_wandb,
            )
        else:
            tuned_model_dprime = model_dprime
            tune_time = 0
            tune_batch_stats = batch_stats

        if model_path:
            assert not model_path.is_file() or overwrite_save_model
            ml.save_plus(model_path, tuned_model_dprime, {"train_time": tune_time})

    key, subkey = random.split(key)
    # (batch,losses)
    tuned_loss = ml.map_loss_in_batches_dl(
        ml.Mapper([geom.Losses.L2_REL, geom.Losses.SMSE], residual, reduce=None, eps=1e-9),
        tuned_model_dprime,
        test_dataloader,
        aux_data=tune_batch_stats,
        reduce=None,
    )
    rel_mean = jnp.mean(tuned_loss[:, 0])
    rel_std = jnp.std(tuned_loss[:, 0])
    smse_mean = jnp.mean(tuned_loss[:, 1])
    smse_std = jnp.std(tuned_loss[:, 1])
    print(
        f"Tuned Loss rescale=True, D={tune_D}: {smse_mean:.3e} +-{smse_std:.3e} ({rel_mean:.3f}% +-{rel_std:.3f}%)\n"
    )

    return smse_mean, smse_std, rel_mean, rel_std, convert_time + tune_time


def run_anyd(
    train_D_range: tuple[int],
    test_D_range: tuple[int],
    n_tune_range: tuple[int],
    args: argparse.Namespace,
    key: jax.Array,
    get_data: Callable[
        [int, int, int, int, jax.Array],
        tuple[
            DataLoader[ml.MultiImageDataset],
            DataLoader[ml.MultiImageDataset],
            DataLoader[ml.MultiImageDataset],
            float,
        ],
    ],
    model_list_d: dict[int, list[tuple[str, dict, dict]]],
    lr_range: list[float],
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
    n_results = 5
    # train the models, i.e. the warmstart lower dimensional models
    print("Train the models (warmstart)!")
    trained_model_list_d = {}
    pretrain_data_time_d = {}
    for train_D in train_D_range:
        key, subkey = random.split(key)
        train_dataloader, val_dataloader, _, pretrain_data_time = get_data(
            train_D, args.n_train, args.n_val, 0, subkey
        )

        train_data = (train_dataloader, val_dataloader)
        key, subkey = random.split(key)
        trained_model_list_d[train_D] = train_all_models(
            train_data, subkey, model_list_d[train_D], lr_range, args
        )
        pretrain_data_time_d[train_D] = pretrain_data_time

    if args.find_train_lr:
        exit()

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
            *tune_data, tune_data_time = get_data(test_D, n_tune, args.n_val, args.n_test, subkey)

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
                        "is_wandb": _test_kwargs["is_wandb"],
                        "batch_size": _test_kwargs["batch_size"],
                        "model_dir": None,  # tmp
                    },
                )
                for name, _train_kwargs, _test_kwargs in model_list_d[test_D]
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
                num_results=n_results,  # smse_mean, smse_std, rel_mean, rel_std, tune_time
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
            results_dict[test_D][test_D].append(baseline_results)

            for train_D in train_D_range:
                if train_D == test_D:
                    continue

                # select the correct learning rate for the tuning based on train and test dimensions
                trained_model_list = [
                    (
                        name,
                        func,
                        {**_test_kwargs, "lr": _test_kwargs["lr"][(train_D, test_D)][n_tune]},
                    )
                    for name, func, _test_kwargs, _ in trained_model_list_d[train_D]
                ]
                train_model_times = np.stack(
                    [train_time for _, _, _, train_time in trained_model_list_d[train_D]]
                )
                train_model_times = np.concat(
                    [np.zeros((len(train_model_times), n_results - 1)), train_model_times[:, None]],
                    axis=1,
                )

                key, subkey = random.split(key)
                # (n_trials, benchmark, models, n_results)
                tune_results = ml.benchmark_lr(
                    lambda _: tune_data,
                    trained_model_list,
                    subkey,
                    lr_range if args.find_tune_lr else [],
                    num_trials=args.n_trials,
                    num_results=n_results,  # smse_mean, smse_std, rel_mean, rel_std, tune_time
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

                tune_results += train_model_times[None, None]
                tune_results[..., -1] += pretrain_data_time_d[train_D] + tune_data_time

                results_dict[test_D][train_D].append(tune_results)

    if args.images_dir is not None:
        model_names_d = {D: [x[0] for x in model_list] for D, model_list in model_list_d.items()}
        plot_results(
            results_dict,
            ["L2 error", "Relative error"],
            n_tune_range,
            model_names_d,
            args.images_dir,
            f"By tuning points",  # TODO: want to say what the dimensions are
        )

        plot_time_results(
            results_dict,
            ["L2 error", "Relative error"],
            model_names_d,
            args.images_dir,
            f"By time",  # TODO: want to say what the dimensions are
        )
