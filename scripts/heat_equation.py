from __future__ import annotations
import argparse
import functools as ft
import math
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pathlib
import time
from typing import Callable
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

    if batch == 0:
        x0 = jnp.zeros((batch,) + (N,) * D)
        xt = jnp.zeros((batch,) + (N,) * D)
    elif data_dir.is_file():
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
    D: int,
    N: int,
    is_torus: bool,
    diffusion_coef: float,
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
    Get an input, output data pair of heat diffusion after 1 timestep, specified diffusion constant.
    The initial data is uniform on the range -sqrt(3) to sqrt(3), so it has unit variance.

    args:
        D: data dimension
        N: sidelength of a cube of data
        is_torus: whether the data is on the torus
        diffusion_coeff: coefficient in the diffusion equation, equivalently the timestep
        n_train: number of training data points
        n_val: number of validation data points
        n_test: number of test data points for each test dimension
        key: key for randomness
        data_dir: location to save or load the data from

    returns:
        training, validation, and test images for input and output
    """
    max_temp = math.sqrt(3)
    t = 1
    data_dir_path = pathlib.Path(data_dir)

    key, subkey1, subkey2, subkey3 = random.split(key, num=4)
    train_x0, train_xt = get_data_d(
        D, N, is_torus, diffusion_coef, t, max_temp, n_train, subkey1, data_dir_path / "train"
    )

    val_x0, val_xt = get_data_d(
        D, N, is_torus, diffusion_coef, t, max_temp, n_val, subkey2, data_dir_path / "val"
    )

    test_x0, test_xt = get_data_d(
        D, N, is_torus, diffusion_coef, t, max_temp, n_test, subkey3, data_dir_path / "test"
    )
    if n_test > 0:
        x0_std = jnp.std(test_x0[((), 0)])
        xt_std = jnp.std(test_xt[((), 0)])
        xt_resid_std = jnp.std(test_xt[((), 0)] - test_x0[((), 0)])
        print(f"D={D}, x0:{x0_std:.3e}, xt:{xt_std:.3e}, xt_resid:{xt_resid_std:.3e}")

    return train_x0, train_xt, val_x0, val_xt, test_x0, test_xt


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


def plot_results(
    results_dict: dict[int, dict[int, list[jax.Array]]],
    results_labels: list[str],
    n_tune_range: tuple[int, ...],
    model_names_d: dict[int, list[str]],
    saveloc: str,
) -> None:
    """
    Plot the results of the heat_equation experiments. For each test_D, create a plot with the
    number of tuning points on the x-axis and the error (either l2 or relative) on the y-axis.

    args:
        results_dict: The results with test_D, then 'baseline' or train_D, then a list over n_tune.
        results_labels: e.g. 'l2', 'relative error'
        n_tune_range: number of fine-tuning points, or training points for the baseline model
        model_names_d: model names for each dimension
        saveloc: beginning of save location

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
                display_name = f"{name} (baseline)" if train_D == test_D else name

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
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 6 * nrows))
    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    colors = ["b", "g", "r", "c", "m", "y"]

    for (test_D, results_by_model), ax_row in zip(grouped_results.items(), axes):
        for error_idx, ylabel, ax in zip(range(len(results_labels)), results_labels, ax_row):
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
        return ml.load(model_path, model)

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

    if model_path:
        assert not model_path.is_file() or overwrite_save_model
        # TODO: need to save batch_stats as well
        ml.save(model_path, trained_model)

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
    if conv_filters_dict is not None:
        model_dprime = model.convertD(
            conv_filters_dict[tune_X.D],
            True,
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
                HeatMapper(residual),
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
# CUDA_VISIBLE_DEVICES=6 time python3 scripts/heat_equation.py --data /data/wgregor4/heat_equation/
# --n-test 128 --n-val 128 --n-train 128 -N 64 --train-D-range 1,2 --diffusion-coef 1
# --test-D-range 2,3 -s /data/wgregor4/runs/heat_equation/
def handleArgs() -> argparse.Namespace:
    parser = utils.get_common_parser()
    parser.add_argument(
        "--n-tune-range",
        help="the number of data points in the tuning set",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default="0,1,4,32,128",
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
        default="2,3",
    )
    parser.add_argument("-N", help="spatial size", type=int, default=64)
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

if args.load_model or args.save_model:
    print("Use --model-dir and possibly --overwrite-save-model instead of --save-model")
    exit()

key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)
lr_range = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]

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
    "images_dir": args.images_dir,  # currently ignored
    "verbose": args.verbose,
    "is_wandb": args.tune_wandb,
}

normalize_filters_dict = {}
gaussian_filters_dict = {}
upsample_filters_dict = {}

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
    upsample_filters_dict[D] = geom.get_invariant_filters(
        Ms=[2],
        ks=[0],
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
        D, args.N, True, args.diffusion_coef, args.n_train, 0, 0, subkey, args.data
    )
    input_keys = train_x0.get_signature()
    output_keys = train_xt.get_signature()

    key, *subkeys = random.split(key, num=10)
    model_list = [
        (
            f"two_layer_gaussian_scaling_D{D}",
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
                "lr": 5e-2,  # best for all dimensions, all n_tune
                **train_kwargs,
            },
            {
                "lr": {(1, 2): 1e-2, (1, 3): 5e-2, (2, 3): 1e-2},
                # n_tune has only a small effect on the error difference, simplicity use n=1 value
                # it is typically lower so this strategy is conservative
                # (1,2): (n=1,1e-2) (n=4,1e-2) (n=32,1e-2) (n=128,1e-2)
                # (1,3): (n=1,5e-2) (n=4,5e-2) (n=32,5e-2) (n=128,5e-2) (pretty large gap for n=1,4)
                # (2,3): (n=1,1e-2) (n=4,1e-2) (n=32,5e-2) (n=128,5e-2) can do 1e-2
                "conv_filters_dict": gaussian_filters_dict,
                **test_kwargs,
            },
        ),
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
        (
            f"resnet_equiv_42_gaussian_scaling_D{D}",
            {  # train kwargs
                "model": models.ResNet(
                    D,
                    input_keys,
                    output_keys,
                    depth=42,
                    conv_filters=gaussian_filters_dict[D],
                    use_group_norm=True,
                    key=subkeys[1],
                ),
                "lr": {1: 5e-3, 2: 5e-4, 3: 1e-3}[D],
                # (D=1,n=128,5e-3) (D=2,n=1,5e-4) (D=2,n=4,5e-4) (D=2,n=32,5e-4) (D=2,n=128,5e-4)
                # (D=3,n=1,1e-3) (D=3,n=4,1e-3) (D=3,n=32,5e-4) (D=3,n=128,1e-3)
                **train_kwargs,
            },
            {  # tune and eval kwargs
                "lr": {(1, 2): 1e-4, (1, 3): 5e-5, (2, 3): 5e-5},
                # (1,2): (n=1,1e-4) (n=4,1e-4) (n=32,1e-3) (n=128,1e-3)
                # (1,3): (n=1,5e-5) (n=4,5e-5) (n=32,5e-4) (n=128,5e-3)
                # (2,3): (n=1,5e-5) (n=4,5e-5) (n=32,1e-4) (n=128,1e-4)
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
                    key=subkeys[2],
                ),
                "lr": 1e-3,
                # D=1 (n=128,1e-3)
                # D=2 (n=1,1e-3) (n=4,1e-3) (n=32,1e-3) (n=128,5e-4 although 1e-3 is close?)
                # D=3 (n=1,1e-3) (n=4,1e-3) (n=32,1e-3) (n=128,1e-3)
                **train_kwargs,
            },
            {  # tune and eval kwargs
                "lr": {(1, 2): 1e-3, (1, 3): 1e-3, (2, 3): 1e-3},
                # (1,2) (n=1,1e-3) (n=4,1e-3) (n=32,5e-3) (n=128,5e-4 or 1e-3)
                # (1,3) (n=1,1e-3) (n=4,1e-3) (n=32,1e-3) (n=128,5e-3,1e-3,5e-4)
                # (2,3) (n=1,1e-3) (n=4,1e-3) (n=32,1e-3) (n=128,1e-3 works)
                "conv_filters_dict": gaussian_filters_dict,
                "upsample_filters_dict": upsample_filters_dict,
                **test_kwargs,
            },
        ),
    ]
    model_list_d[D] = model_list

# train the models, i.e. the warmstart lower dimensional models
print("Train the models (warmstart)!")
trained_model_list_d = {}
for train_D in args.train_D_range:
    key, subkey = random.split(key)
    train_x0, train_xt, val_x0, val_xt, _, _ = get_data(
        train_D, args.N, True, args.diffusion_coef, args.n_train, args.n_val, 0, subkey, args.data
    )

    train_data = (train_x0, train_xt, val_x0, val_xt)
    key, subkey = random.split(key)
    trained_model_list_d[train_D] = train_all_models(
        train_data, subkey, model_list_d[train_D], lr_range, args
    )

if args.find_train_lr:
    exit()

# evaluate the models
print("Tune and evaluate the models!")
results_dict = {}
for test_D in args.test_D_range:
    results_dict[test_D] = {k: [] for k in ((test_D,) + tuple(args.train_D_range))}
    for n_tune in args.n_tune_range:
        print(f"D={test_D}, n_tune={n_tune}.\n")
        # the data is saved, so this is still reasonably efficient
        key, subkey = random.split(key)
        tune_data = get_data(
            test_D,
            args.N,
            True,
            args.diffusion_coef,
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
                {**_train_kwargs, "conv_filters_dict": None, "is_wandb": args.tune_wandb},
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
        results_dict[test_D][test_D].append(baseline_results)

        for train_D in args.train_D_range:
            if train_D == test_D:
                continue

            # select the correct learning rate for the tuning based on train and test dimensions
            trained_model_list = [
                (name, func, {**_test_kwargs, "lr": _test_kwargs["lr"][(train_D, test_D)]})
                for name, func, _test_kwargs in trained_model_list_d[train_D]
            ]

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

            results_dict[test_D][train_D].append(tune_results)


if args.images_dir is not None:
    model_names_d = {D: [x[0] for x in model_list] for D, model_list in model_list_d.items()}
    plot_results(
        results_dict,
        ["l2_error", "relative_error"],
        args.n_tune_range,
        model_names_d,
        args.images_dir,
    )
