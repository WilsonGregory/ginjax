import argparse
import math
import matplotlib.pyplot as plt
import pathlib
import time
from typing import Callable

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax

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
) -> tuple[models.MultiImageModule, float]:
    train_X, train_Y, val_X, val_Y = data
    N = train_X.get_spatial_dims()[0]
    batch_stats = eqx.nn.State(model) if has_aux else None
    model_name_extended = f"{model_name}_L{train_X.get_L()}_N{N}_e{epochs}"
    model_path = model_dir / f"{model_name_extended}_model.eqx" if model_dir else None

    print(f"{model_name_extended} params: {models.count_params(model):,}")

    if model_path and model_path.is_file() and not overwrite_save_model:
        trained_model, further_args = ml.load_plus(model_path, model)
        train_time = further_args["train_time"]
    else:
        steps_per_epoch = int(math.ceil(train_X.get_L() / batch_size))
        key, subkey = random.split(key)
        trained_model, _, _, _, train_time = ml.train(
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
            ml.save_plus(model_path, trained_model, {"train_time": train_time})

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

    return trained_model, train_time


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
) -> tuple[list[tuple[str, Callable, dict]], list[float]]:
    train_D = data[0].D
    n_points = data[0].get_L()

    trained_models = []
    train_times = []
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
            trained_model, train_time = train_model(data, subkey, model_name, **_train_kwargs)
            _test_kwargs = {**_test_kwargs, "model": trained_model}
            trained_models.append((model_name, tune_and_eval, _test_kwargs))
            train_times.append(train_time)

    return trained_models, train_times


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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, float]:
    tune_X, tune_Y, val_X, val_Y, test_X, test_Y = data
    N = tune_X.get_spatial_dims()[0]
    batch_stats = eqx.nn.State(model) if has_aux else None
    model_name_extended = f"{model_name}_tuneD{tune_X.D}_L{tune_X.get_L()}_N{N}_e{epochs}"
    model_path = model_dir / f"{model_name_extended}_model.eqx" if model_dir else None

    key, subkey = random.split(key)
    # rescale set to True, small but notable difference
    if conv_filters_dict is not None and rescale is not None:
        start_time = time.time()
        model_dprime = model.convertD(
            conv_filters_dict[tune_X.D],
            rescale,
            subkey,
            upsample_filters=upsample_filters_dict[tune_X.D] if upsample_filters_dict else None,
        )
        convert_time = time.time() - start_time
    else:
        model_dprime = model
        convert_time = 0

    if model_path and model_path.is_file() and not overwrite_save_model:
        tuned_model_dprime, further_args = ml.load_plus(model_path, model_dprime)
        tune_time = further_args["train_time"]
        tune_batch_stats = batch_stats
    else:
        if tune_X.get_L() > 0:
            # Now treat the trained_model_d as a warmstart and do some additional training
            key, subkey = random.split(key)
            steps_per_epoch = int(math.ceil(tune_X.get_L() / batch_size))
            tuned_model_dprime, tune_batch_stats, _, _, tune_time = ml.train(
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
            tune_time = 0
            tune_batch_stats = batch_stats

        if model_path:
            assert not model_path.is_file() or overwrite_save_model
            ml.save_plus(model_path, tuned_model_dprime, {"train_time": tune_time})

    key, subkey = random.split(key)
    # (batch,losses)
    tuned_loss = ml.map_loss_in_batches(
        ml.Mapper([geom.Losses.L2_REL, geom.Losses.SMSE], residual, reduce=None),
        tuned_model_dprime,
        test_X,
        test_Y,
        batch_size,
        subkey,
        aux_data=tune_batch_stats,
        reduce=None,
    )
    rel_mean = jnp.mean(tuned_loss[:, 0])
    rel_std = jnp.std(tuned_loss[:, 0])
    smse_mean = jnp.mean(tuned_loss[:, 1])
    smse_std = jnp.std(tuned_loss[:, 1])
    print(
        f"Tuned Loss rescale=True, D={test_X.D}: {smse_mean:.3e} +-{smse_std:.3e} ({rel_mean:.3f}% +-{rel_std:.3f}%)\n"
    )

    return smse_mean, smse_std, rel_mean, rel_std, convert_time + tune_time
