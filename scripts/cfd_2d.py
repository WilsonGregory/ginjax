import sys
import time
import argparse
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import h5py
from typing import Optional

import jax.numpy as jnp
import jax
import jax.random as random
import jax.experimental.mesh_utils as mesh_utils
from jaxtyping import ArrayLike
import optax
import equinox as eqx

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml
import geometricconvolutions.ml_eqx as ml_eqx
import geometricconvolutions.utils as utils
import geometricconvolutions.data as gc_data
import geometricconvolutions.models_eqx as models


def read_one_h5(filename: str, num_trajectories: int) -> tuple:
    """
    Given a filename and a type of data (train, test, or validation), read the data and return as jax arrays.
    args:
        filename (str): the full file path
        data_class (str): either 'train', 'test', or 'valid'
    returns: u, vxy as jax arrays
    """
    data_dict = h5py.File(filename)

    # all of these are shape (num_trajectories, t, x, y) = (10K, 21, 128, 128)
    density = jax.device_put(
        jnp.array(data_dict["density"][:num_trajectories][()]), jax.devices("cpu")[0]
    )
    pressure = jax.device_put(
        jnp.array(data_dict["pressure"][:num_trajectories][()]), jax.devices("cpu")[0]
    )
    vx = jax.device_put(jnp.array(data_dict["Vx"][:num_trajectories][()]), jax.devices("cpu")[0])
    vy = jax.device_put(jnp.array(data_dict["Vy"][:num_trajectories][()]), jax.devices("cpu")[0])
    vxy = jnp.stack([vx, vy], axis=-1)

    data_dict.close()

    return density, pressure, vxy


def get_data(
    D: int,
    filename: str,
    n_train: int,
    n_val: int,
    n_test: int,
    past_steps: int,
    rollout_steps: int,
    normalize: bool = True,
) -> tuple[geom.BatchLayer]:
    density, pressure, velocity = read_one_h5(filename, n_train + n_val + n_test)

    if normalize:
        density = (density - jnp.mean(density[: (n_train + n_val)])) / jnp.std(
            density[: (n_train + n_val)]
        )
        pressure = (pressure - jnp.mean(pressure[: (n_train + n_val)])) / jnp.std(
            pressure[: (n_train + n_val)]
        )
        velocity = velocity / jnp.std(
            velocity[: (n_train + n_val)]
        )  # this one I am not so sure about

    start = 0
    stop = n_train
    train_X, train_Y = gc_data.times_series_to_layers(
        D,
        {(0, 0): density[start:stop], (1, 0): velocity[start:stop]},
        {},
        True,
        past_steps,
        1,
    )
    train_pressure_X, train_pressure_Y = gc_data.times_series_to_layers(
        D,
        {(0, 0): pressure[start:stop]},
        {},
        True,
        past_steps,
        1,
    )
    train_X = train_X.concat(train_pressure_X, axis=1)
    train_Y = train_Y.concat(train_pressure_Y, axis=1)

    start = start + n_train
    stop = start + n_val
    val_X, val_Y = gc_data.times_series_to_layers(
        D,
        {(0, 0): density[start:stop], (1, 0): velocity[start:stop]},
        {},
        True,
        past_steps,
        1,
    )
    val_pressure_X, val_pressure_Y = gc_data.times_series_to_layers(
        D,
        {(0, 0): pressure[start:stop]},
        {},
        True,
        past_steps,
        1,
    )
    val_X = val_X.concat(val_pressure_X, axis=1)
    val_Y = val_Y.concat(val_pressure_Y, axis=1)

    start = start + n_val
    stop = start + n_test
    test_X, test_Y = gc_data.times_series_to_layers(
        D,
        {(0, 0): density[start:stop], (1, 0): velocity[start:stop]},
        {},
        True,
        past_steps,
        1,
    )
    test_pressure_X, test_pressure_Y = gc_data.times_series_to_layers(
        D,
        {(0, 0): pressure[start:stop]},
        {},
        True,
        past_steps,
        1,
    )
    test_X = test_X.concat(test_pressure_X, axis=1)
    test_Y = test_Y.concat(test_pressure_Y, axis=1)

    test_rollout_X, test_rollout_Y = gc_data.times_series_to_layers(
        D,
        {(0, 0): density[start:stop], (1, 0): velocity[start:stop]},
        {},
        True,
        past_steps,
        rollout_steps,
    )
    test_rollout_pressure_X, test_rollout_pressure_Y = gc_data.times_series_to_layers(
        D,
        {(0, 0): pressure[start:stop]},
        {},
        True,
        past_steps,
        rollout_steps,
    )
    test_rollout_X = test_rollout_X.concat(test_rollout_pressure_X, axis=1)
    test_rollout_Y = test_rollout_Y.concat(test_rollout_pressure_Y, axis=1)

    return (
        train_X,
        train_Y,
        val_X,
        val_Y,
        test_X,
        test_Y,
        test_rollout_X,
        test_rollout_Y,
    )


def plot_layer(
    test_layer: geom.BatchLayer,
    actual_layer: geom.BatchLayer,
    save_loc: str,
    future_steps: int,
    component: int = 0,
    show_power: bool = False,
    title: str = "",
    minimal: bool = False,
):
    """
    Plot all timesteps of a particular component of two layers, and the differences between them.
    args:
        test_layer (BatchLayer): the predicted layer
        actual_layer (BatchLayer): the ground truth layer
        save_loc (str): file location to save the image
        future_steps (int): the number future time steps in the layer
        component (int): index of the component to plot, default to 0
        show_power (bool): whether to also plot the power spectrum, default to False
        title (str): additional str to add to title, will be "test {title} {col}"
            "actual {title} {col}"
        minimal (bool): if minimal, no titles, colorbars, or axes labels, defaults to False
    """
    test_layer_comp = test_layer.get_component(component, future_steps).get_one_layer()
    actual_layer_comp = actual_layer.get_component(component, future_steps).get_one_layer()

    test_images = test_layer_comp.to_images()
    actual_images = actual_layer_comp.to_images()

    img_arr = jnp.concatenate([test_layer_comp[(0, 0)], actual_layer_comp[(0, 0)]])
    vmax = jnp.max(jnp.abs(img_arr))
    vmin = -1 * vmax

    nrows = 4 if show_power else 3

    # figsize is 6 per col, 6 per row, (cols,rows)
    fig, axes = plt.subplots(nrows=nrows, ncols=future_steps, figsize=(6 * future_steps, 6 * nrows))
    for col, (test_image, actual_image) in enumerate(zip(test_images, actual_images)):
        diff = (actual_image - test_image).norm()
        if minimal:
            test_title = ""
            actual_title = ""
            diff_title = ""
            colorbar = False
            hide_ticks = True
            xlabel = ""
            ylabel = ""
        else:
            test_title = f"test {title} {col}"
            actual_title = f"actual {title} {col}"
            diff_title = f"diff {title} {col} (mse: {jnp.mean(diff.data)})"
            colorbar = True
            hide_ticks = False
            xlabel = "unnormalized wavenumber"
            ylabel = "unnormalized power"

        test_image.plot(axes[0, col], title=test_title, vmin=vmin, vmax=vmax, colorbar=colorbar)
        actual_image.plot(axes[1, col], title=actual_title, vmin=vmin, vmax=vmax, colorbar=colorbar)
        diff.plot(axes[2, col], title=diff_title, vmin=vmin, vmax=vmax, colorbar=colorbar)

        if show_power:
            utils.plot_power(
                [test_image.data[None, None], actual_image.data[None, None]],
                ["test", "actual"] if col == 0 else None,
                axes[3, col],
                xlabel=xlabel,
                ylabel=ylabel,
                hide_ticks=hide_ticks,
            )

    plt.tight_layout()
    plt.savefig(save_loc)
    plt.close(fig)


def plot_timestep_power(
    layers: list[geom.BatchLayer],
    labels: list[str],
    save_loc: str,
    future_steps: int,
    component: int = 0,
    title: str = "",
):
    fig, axes = plt.subplots(nrows=1, ncols=future_steps, figsize=(8 * future_steps, 6 * 1))
    for i, ax in enumerate(axes):
        utils.plot_power(
            [
                layer.get_component(component, future_steps, as_layer=False)[:, i : i + 1]
                for layer in layers
            ],
            labels,
            ax,
            title=f"{title} {i}",
        )

    plt.savefig(save_loc)
    plt.close(fig)


def map_and_loss(
    model: eqx.Module,
    layer_x: geom.BatchLayer,
    layer_y: geom.BatchLayer,
    aux_data: Optional[dict] = None,
    has_aux: bool = False,
    future_steps: int = 1,
    return_map: bool = False,
):
    batch_model = jax.vmap(model)
    result = ml_eqx.autoregressive_map(
        batch_model,
        layer_x,
        aux_data,
        layer_x[(1, 0)].shape[1],  # past_steps
        future_steps,
        has_aux=has_aux,
    )
    if has_aux:
        out_layer, aux_data = result
    else:
        out_layer = result

    loss = ml.timestep_smse_loss(out_layer, layer_y, future_steps)
    loss = loss[0] if future_steps == 1 else loss

    if has_aux or return_map:
        output = (loss,)
        if has_aux:
            output = output + (aux_data,)

        if return_map:
            output = output + (out_layer,)

        return output
    else:
        return loss


def train_and_eval(
    data: tuple[geom.BatchLayer],
    key: ArrayLike,
    model_name: str,
    model: eqx.Module,
    lr: float,
    batch_size: int,
    epochs: int,
    rollout_steps: int,
    save_model: Optional[str],
    load_model: Optional[str],
    images_dir: Optional[str],
    has_aux: bool = False,
    verbose: int = 1,
    plot_component: int = 0,
) -> tuple[float]:
    (
        train_X,
        train_Y,
        val_X,
        val_Y,
        test_single_X,
        test_single_Y,
        test_rollout_X,
        test_rollout_Y,
    ) = data

    print(f"Model params: {models.count_params(model):,}")

    if load_model is None:
        steps_per_epoch = int(np.ceil(train_X.get_L() / batch_size))
        key, subkey = random.split(key)
        results = ml_eqx.train(
            train_X,
            train_Y,
            map_and_loss,
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
            has_aux=has_aux,
        )

        if has_aux:
            model, batch_stats, train_loss, val_loss = results
        else:
            model, train_loss, val_loss = results
            batch_stats = None

        if save_model is not None:
            # TODO: need to save batch_stats as well
            ml_eqx.save(f"{save_model}{model_name}_L{train_X.L}_e{epochs}_model.eqx", model)
    else:
        model = ml_eqx.load(f"{save_model}{model_name}_L{train_X.L}_e{epochs}_model.eqx", model)

        key, subkey1, subkey2 = random.split(key)
        train_loss = ml_eqx.map_loss_in_batches(
            map_and_loss,
            model,
            train_X,
            train_Y,
            batch_size,
            subkey1,
            # map_and_loss kwargs
            has_aux=has_aux,
            aux_data=batch_stats,
        )
        val_loss = ml_eqx.map_loss_in_batches(
            map_and_loss,
            model,
            val_X,
            val_Y,
            batch_size,
            subkey2,
            # map_and_loss kwargs
            has_aux=has_aux,
            aux_data=batch_stats,
        )

    key, subkey = random.split(key)
    test_loss = ml_eqx.map_loss_in_batches(
        map_and_loss,
        model,
        test_single_X,
        test_single_Y,
        batch_size,
        subkey,
        # map_and_loss kwargs
        has_aux=has_aux,
        aux_data=batch_stats,
    )
    print(f"Test Loss: {test_loss}")

    key, subkey = random.split(key)
    test_rollout_loss, rollout_layer = ml_eqx.map_loss_in_batches(
        map_and_loss,
        model,
        test_rollout_X,
        test_rollout_Y,
        batch_size,
        subkey,
        # map_and_loss kwargs
        future_steps=rollout_steps,
        has_aux=has_aux,
        aux_data=batch_stats,
        return_map=True,
    )
    print(f"Test Rollout Loss: {test_rollout_loss}, Sum: {jnp.sum(test_rollout_loss)}")

    if images_dir is not None:
        components = ["density", "pressure", "velocity_x", "velocity_y"]
        plot_layer(
            rollout_layer.get_one(),
            test_rollout_Y.get_one(),
            f"{images_dir}{model_name}_L{train_X.L}_e{epochs}_rollout.png",
            future_steps=rollout_steps,
            component=plot_component,
            show_power=True,
            title=f"{components[plot_component]}",
        )
        plot_timestep_power(
            [rollout_layer, test_rollout_Y],
            ["test", "actual"],
            f"{images_dir}{model_name}_L{train_X.L}_e{epochs}_{components[plot_component]}_power_spectrum.png",
            future_steps=rollout_steps,
            component=plot_component,
            title=f"{components[plot_component]}",
        )

    return train_loss, val_loss, test_loss, *test_rollout_loss


def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="the data .hdf5 file", type=str)
    parser.add_argument("-e", "--epochs", help="number of epochs to run", type=int, default=50)
    parser.add_argument("-batch", help="batch size", type=int, default=8)
    parser.add_argument("-n_train", help="number of training trajectories", type=int, default=100)
    parser.add_argument(
        "-n_val",
        help="number of validation trajectories, defaults to batch",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-n_test",
        help="number of testing trajectories, defaults to batch",
        type=int,
        default=None,
    )
    parser.add_argument("-t", "--n_trials", help="number of trials to run", type=int, default=1)
    parser.add_argument("-seed", help="the random number seed", type=int, default=None)
    parser.add_argument(
        "-s", "--save_model", help="file name to save the params", type=str, default=None
    )
    parser.add_argument(
        "-l", "--load_model", help="file name to load params from", type=str, default=None
    )
    parser.add_argument(
        "-images_dir",
        help="directory to save images, or None to not save",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="verbose argument passed to trainer",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--normalize",
        help="normalize input data, equivariantly",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--plot_component",
        help="which component to plot, one of 0-3",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
    )
    parser.add_argument(
        "--rollout_steps",
        help="number of steps to rollout in test",
        type=int,
        default=5,
    )

    return parser.parse_args()


# Main
args = handleArgs(sys.argv)

D = 2
N = 128
output_keys = (((0, 0), 2), ((1, 0), 1))

past_steps = 4  # how many steps to look back to predict the next step
key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)

# an attempt to reduce recompilation, but I don't think it actually is working
n_test = args.batch if args.n_test is None else args.n_test
n_val = args.batch if args.n_val is None else args.n_val

data = get_data(
    D,
    args.data,
    args.n_train,
    n_val,
    n_test,
    past_steps,
    args.rollout_steps,
    args.normalize,
)
input_keys = data[0].get_signature()

group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    Ms=[3], ks=[0, 1, 2], parities=[0, 1], D=D, operators=group_actions
)
upsample_filters = geom.get_invariant_filters(
    Ms=[2], ks=[0, 1, 2], parities=[0, 1], D=D, operators=group_actions
)

train_and_eval = partial(
    train_and_eval,
    batch_size=args.batch,
    epochs=args.epochs,
    rollout_steps=args.rollout_steps,
    save_model=args.save_model,
    load_model=args.load_model,
    images_dir=args.images_dir,
    verbose=args.verbose,
    plot_component=args.plot_component,
)

key, subkey1, subkey2 = random.split(key, num=3)
model_list = [
    (
        "unetBase",
        partial(
            train_and_eval,
            model=models.UNet(
                D,
                input_keys,
                output_keys,
                depth=64,
                use_bias=True,
                activation_f=jax.nn.gelu,
                equivariant=False,
                kernel_size=3,
                use_group_norm=True,
                key=subkey1,
            ),
            lr=8e-4,
        ),
    ),
    (
        "unetBase_equiv48",
        partial(
            train_and_eval,
            model=models.UNet(
                D,
                input_keys,
                output_keys,
                depth=48,
                activation_f=jax.nn.gelu,
                conv_filters=conv_filters,
                upsample_filters=upsample_filters,
                key=subkey2,
            ),
            lr=4e-4,  # 4e-4 to 6e-4 works, larger sometimes explodes
        ),
    ),
]

key, subkey = random.split(key)

# # Use this for benchmarking over different learning rates
# results = ml.benchmark(
#     lambda _: data,
#     model_list,
#     subkey,
#     "lr",
#     [2e-4, 4e-4, 6e-4],
#     benchmark_type=ml.BENCHMARK_MODEL,
#     num_trials=args.n_trials,
#     num_results=3 + args.rollout_steps,
# )

# Use this for benchmarking the models with known learning rates.
results = ml.benchmark(
    lambda _: data,
    model_list,
    subkey,
    "",
    [0],
    benchmark_type=ml.BENCHMARK_NONE,
    num_trials=args.n_trials,
    num_results=3 + args.rollout_steps,
)

rollout_res = results[..., 3:]
non_rollout_res = jnp.concatenate(
    [results[..., :3], jnp.sum(rollout_res, axis=-1, keepdims=True)], axis=-1
)
print(non_rollout_res)
mean_results = jnp.mean(
    non_rollout_res, axis=0
)  # includes the sum of rollout. (benchmark_vals,models,outputs)
std_results = jnp.std(non_rollout_res, axis=0)
print("Mean", mean_results, sep="\n")

plot_mapping = {
    "dil_resnet64": ("DilResNet", "blue", "o", "dashed"),
    "dil_resnet_equiv48": ("DilResNet Equiv", "blue", "o", "solid"),
    "resnet": ("ResNet", "red", "s", "dashed"),
    "resnet_equiv_groupnorm_100": ("ResNet Equiv", "red", "s", "solid"),
    "unetBase": ("UNet LayerNorm", "green", "P", "dashed"),
    "unetBase_equiv48": ("UNet LayerNorm Equiv", "green", "P", "solid"),
    "unet2015": ("UNet", "orange", "*", "dashed"),
    "unet2015_equiv48": ("Unet Equiv", "orange", "*", "solid"),
}

# print table
output_types = ["train", "val", "test", f"rollout ({args.rollout_steps} steps)"]
print("model ", end="")
for output_type in output_types:
    print(f"& {output_type} ", end="")

print("\\\\")
print("\\hline")

for i in range(len(model_list) // 2):
    for l in range(2):  # models come in a baseline and equiv pair
        idx = 2 * i + l
        print(f"{plot_mapping[model_list[idx][0]][0]} ", end="")

        for j, result_type in enumerate(output_types):
            if jnp.trunc(std_results[0, idx, j] * 1000) / 1000 > 0:
                stdev = f"$\\pm$ {std_results[0,idx,j]:.3f}"
            else:
                stdev = ""

            if jnp.allclose(
                mean_results[0, idx, j],
                min(mean_results[0, 2 * i, j], mean_results[0, 2 * i + 1, j]),
            ):
                print(f'& \\textbf{"{"}{mean_results[0,idx,j]:.3f} {stdev}{"}"}', end="")
            else:
                print(f"& {mean_results[0,idx,j]:.3f} {stdev} ", end="")

        print("\\\\")

    print("\\hline")

print("\n")

if args.images_dir:
    for i, (model_name, _) in enumerate(model_list):
        label, color, marker, linestyle = plot_mapping[model_name]
        plt.plot(
            jnp.arange(1, 1 + args.rollout_steps),
            jnp.mean(rollout_res, axis=0)[0, i],
            label=label,
            marker=marker,
            linestyle=linestyle,
            color=color,
        )

    plt.legend()
    plt.title(f"MSE vs. Rollout Step, Mean of {args.n_trials} Trials")
    plt.xlabel("Rollout Step")
    plt.ylabel("SMSE")
    plt.yscale("log")
    plt.savefig(f"{args.images_dir}/rollout_loss_plot.png")
    plt.close()
