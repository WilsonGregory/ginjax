import argparse
import math
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pathlib
import time
from dataclasses import dataclass
from typing import Callable, Literal, Self, Sequence

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


@dataclass(frozen=True, repr=False)
class ModelLabel:

    name: str
    trial: int | None = None
    pretrain_D: int | None = None
    pretrain_L: int | None = None
    rescale: geom.Rescaling | None = None
    tune_D: int | None = None
    tune_L: int | None = None

    def __str__(self: Self) -> str:
        return self.get()

    def get(self: Self, trial: bool = True, tune_L: bool = True) -> str:
        trial_str = f"_trial{self.trial}" if trial else ""
        pretrain_str = (
            f"_D{self.pretrain_D}_L{self.pretrain_L}" if self.pretrain_D is not None else ""
        )
        rescale_str = f"_rescale{self.rescale.name}" if self.rescale is not None else ""
        tune_str = f"_D{self.tune_D}_L{self.tune_L}" if self.tune_D is not None else ""
        return f"{self.name}{trial_str}{pretrain_str}{rescale_str}{tune_str}"

    def display_name(self: Self) -> str:
        if self.rescale is not None:
            return self.rescale.name.title()
        else:
            assert self.pretrain_D is None
            return "Baseline"


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
    test_D: int,
    grouped_results: dict[ModelLabel, Float[Array, "n_trials n_tune_range n_results"]],
    results_labels: list[str],
    n_tune_range: tuple[int, ...],
    x_axis_type: Literal["ntune", "gflops"],
    saveloc: str,
    title: str,
    middle_metric: Literal["mean", "median"] = "mean",
    include_points: bool = False,
) -> None:
    """
    Plot the results of the heat_equation experiments. For each test_D, create a plot with the
    number of tuning points on the x-axis and the error (either l2 or relative) on the y-axis.

    args:
        test_D: the dimension of the space
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
    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    colors = ["b", "g", "r", "c", "m", "y"]
    for error_idx, ylabel in enumerate(results_labels):

        # do them individually
        _, ax = plt.subplots(figsize=(8, 6))  # figsize is (cols,rows)
        assert isinstance(ax, Axes)
        for i, (model_label, results_arr) in enumerate(grouped_results.items()):
            if x_axis_type == "ntune":
                x_axis = range(len(n_tune_range))
            elif x_axis_type == "gflops":
                x_axis = jnp.mean(results_arr, axis=0)[:, -1] / 1_000_000

            results_arr = results_arr[..., error_idx]

            if middle_metric == "mean":
                # take the mean over trials
                middle_result = jnp.mean(results_arr, axis=0)
                stdev = jnp.std(results_arr, axis=0)
                lower_bound = middle_result - stdev
                upper_bound = middle_result + stdev
            elif middle_metric == "median":
                middle_result = jnp.median(results_arr, axis=0)
                lower_bound = jnp.quantile(results_arr, 0.25, axis=0)
                upper_bound = jnp.quantile(results_arr, 0.75, axis=0)

            ax.plot(
                x_axis,
                middle_result,
                marker="x",
                linestyle=linestyles[i % len(linestyles)],
                label=model_label.display_name(),
                color=colors[i % len(colors)],  # was i, currently only model
            )
            # plot bounds range
            ax.fill_between(
                x_axis,
                lower_bound,
                upper_bound,
                color=colors[i % len(colors)],
                alpha=0.2,
            )

            if include_points:
                for results_trial in results_arr:
                    ax.scatter(
                        x_axis,
                        results_trial,
                        marker=".",
                        color=colors[i % len(colors)],
                    )

        ax.legend(fontsize=12)

        ax.set_ylabel(ylabel, fontsize=24)
        ax.set_yscale("log")
        if x_axis_type == "ntune":
            ax.set_xticks(range(len(n_tune_range)), [str(x) for x in n_tune_range])
            ax.set_xlabel("Number of tuning points", fontsize=24)
        elif x_axis_type == "gflops":
            ax.set_xlabel("Number of gigaflops", fontsize=24)

        ax.set_title(title, fontsize=24)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(
            f"{saveloc}warmstart_plot_{x_axis_type}_{middle_metric}_{test_D}D_{''.join(ylabel.split()).lower()}.png"
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


def train_model(
    data: tuple[DataLoader[ml.MultiImageDataset], DataLoader[ml.MultiImageDataset]],
    model_label: ModelLabel,
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

    D = train_dataloader.dataset.D
    L = len(train_dataloader.dataset)
    batch_stats = eqx.nn.State(model) if has_aux else None
    model_path = model_dir / f"{model_label}.eqx" if model_dir else None

    print(f"{model_label} params: {models.count_params(model):,}")

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
                model_name=model_label.get(trial=False),
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
            f"{images_dir}{model_label}.png",
            "burgers",
        )

    spatial_dims = train_dataloader.dataset.get_spatial_dims()
    # flops/sample * number of samples * epochs
    flops = trained_model.get_flops(**{"spatial_dims": spatial_dims})
    total_train_flops = flops * L * epochs

    return trained_model, train_time, total_train_flops


def train_all_models(
    data: tuple[DataLoader[ml.MultiImageDataset], DataLoader[ml.MultiImageDataset]],
    model_list: list[tuple[ModelLabel, models.AnyDimensionalModel, dict, dict, dict]],
    lr_range: list[float] | None,
    kwargs_idx: int,
    n_points: int,
    data_time: float,
    args: argparse.Namespace,
) -> list[tuple[ModelLabel, models.AnyDimensionalModel, dict, dict, dict, float, int]]:
    """
    Train all the models in the model list.

    args:
        data: train and val dataloaders
        model_list: list of tuples (model_label, model, train_kwargs, baseline_kwargs, test_kwargs)
        lr_range: list of lr values to test if args.find_train_lr is true

    returns:
        list of tuples of (model_name, tune_and_eval, test_kwargs, train_time)
    """
    train_dataloader, _ = data
    assert isinstance(train_dataloader.dataset, ml.MultiImageDataset)
    train_D = train_dataloader.dataset.D
    L = len(train_dataloader.dataset)

    trained_models = []
    for model_label, model, train_kwargs, baseline_kwargs, test_kwargs in model_list:
        kwargs = [train_kwargs, baseline_kwargs, test_kwargs][kwargs_idx]

        # update model label with train information
        model_label = ModelLabel(
            model_label.name,
            model_label.trial,
            pretrain_D=train_D if kwargs_idx == 0 else model_label.pretrain_D,
            pretrain_L=L if kwargs_idx == 0 else model_label.pretrain_L,
            rescale=model_label.rescale,
            tune_D=train_D if kwargs_idx != 0 else model_label.tune_D,
            tune_L=L if kwargs_idx != 0 else model_label.tune_L,
        )

        if lr_range is None:
            lr_range = [kwargs["lr"][train_D][n_points]]

        for lr in lr_range:
            _kwargs = {
                **kwargs,
                "lr": lr,
                "args": {
                    **vars(args),
                    "model_name": str(model_label),
                    "lr": lr,
                    "D": train_D,
                    "n_points": n_points,
                    "train_or_tune": ["train", "baseline", "tune"][kwargs_idx],
                },
            }  # copy kwargs and overwrite lr
            trained_model, train_time, train_flops = train_model(
                data, model_label, model, **_kwargs
            )

            trained_models.append(
                (
                    model_label,
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
    model_list: list[tuple[ModelLabel, models.AnyDimensionalModel, dict, dict, dict, float, int]],
    lr_range: list[float] | None,
    rescale_list: list[geom.Rescaling],
    n_points: int,
    data_time: float,
    args: argparse.Namespace,
) -> list[tuple[ModelLabel, models.AnyDimensionalModel, dict, dict, dict, float, int]]:
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
        model_label,
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

            _model_label = ModelLabel(
                model_label.name,
                model_label.trial,
                model_label.pretrain_D,
                model_label.pretrain_L,
                rescale,
                model_label.tune_D,
                model_label.tune_L,
            )

            # get the correct learning rate, remove rescale, conv_filters_dict, upsample_filters_dict
            _test_kwargs = {**test_kwargs, "lr": test_kwargs["lr"][train_D]}
            _test_kwargs.pop("conv_filters_dict", None)
            _test_kwargs.pop("upsample_filters_dict", None)

            converted_model_list.append(
                (_model_label, model_dprime, train_kwargs, baseline_kwargs, _test_kwargs)
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
    model_list: list[tuple[ModelLabel, models.AnyDimensionalModel, dict, dict, dict, float, int]],
) -> dict[ModelLabel, Float[Array, "n_trials n_results"]]:
    """
    Evaluate the models against the test dataset, then stack each model by trials.
    """
    models_dict = {}
    for model_label, model, _, _, _, train_time, train_flops in model_list:
        # (losses,)
        rel_mean, smse_mean = ml.map_loss_in_batches_dl(
            ml.Mapper([geom.Losses.L2_REL, geom.Losses.SMSE], eps=1e-9), model, test_dataloader
        )
        print(f"Eval {model_label}: {smse_mean:.3e} ({rel_mean:.3f}%)\n")
        tuned_loss = jnp.array([rel_mean, smse_mean, train_time, train_flops])

        trialless_label = ModelLabel(
            model_label.name,
            None,
            model_label.pretrain_D,
            model_label.pretrain_L,
            model_label.rescale,
            model_label.tune_D,
            model_label.tune_L,
        )
        if trialless_label in models_dict:
            models_dict[trialless_label].append(tuned_loss)
        else:
            models_dict[trialless_label] = [tuned_loss]

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
    model_list_d: dict[int, list[tuple[ModelLabel, models.AnyDimensionalModel, dict, dict, dict]]],
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
    n_results = 4
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
    for test_D in test_D_range:
        results_dict = {}
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
            results_dict = {**results_dict, **eval(tune_test_dl, baseline_trained_models)}

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
                results_dict = {**results_dict, **eval(tune_test_dl, finetuned_models)}

        # stack over n_tune_range
        grouped_results = {}
        for model_label, results in results_dict.items():
            assert model_label.trial is None  # should be already stacked over trials
            model_label = ModelLabel(
                model_label.name,
                model_label.trial,
                model_label.pretrain_D,
                model_label.pretrain_L,
                model_label.rescale,
                model_label.tune_D,
                None,
            )
            if model_label in grouped_results:
                grouped_results[model_label].append(results)
            else:
                grouped_results[model_label] = [results]

        # resulting shape (n_trials,n_tune_range,n_results)
        grouped_results = {k: jnp.stack(v, axis=1) for k, v in grouped_results.items()}

        if args.images_dir is not None:
            plot_results(
                test_D,
                grouped_results,
                ["Relative error", "L2 error"],
                n_tune_range,
                "ntune",
                args.images_dir,
                f"By tuning points",
                middle_metric="median",
                include_points=True,
            )

            plot_results(
                test_D,
                grouped_results,
                ["Relative error", "L2 error"],
                n_tune_range,
                "gflops",
                args.images_dir,
                f"By flops",
                middle_metric="median",
                include_points=True,
            )
