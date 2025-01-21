import sys
import os
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


def read_one_h5(filename: str, input_step: int, output_step: int) -> tuple:
    """
    Given a filename and a type of data (train, test, or validation), read the data and return as jax arrays.
    Data information
    data keys: ['boundary_conditions', 'dimensions', 'scalars', 't0_fields', 't1_fields', 't2_fields']
    'boundary_conditions': ['x_open', 'y_open', 'z_open']
    'dimensions': ['time', 'x', 'y', 'z']
        ['time'] evenly spaced 0 to 0.2, 59 steps
        ['x'], ['x'], ['z'] evenly spaced -1 to 1, length 64
    'scalars': ['Msun', 'T0', 'Z', 'rho0'] 4 different scalars
    't0_fields': ['density', 'pressure', 'temperature'],
        all (16,59,64,64,64) presumably (batch,timesteps,spatial)
    't1_fields': ['velocity'], (16,59,64,64,64,3) presumably (batch,timesteps,spatial,tensor)
    't2_fields': empty

    args:
        filename (str): the full file path
        data_class (str): either 'train', 'test', or 'valid'
    returns: u, vxy as jax arrays
    """
    data_dict = h5py.File(filename)

    scalar_X_ls = []
    scalar_y_ls = []
    for scalar_field in ["density", "pressure", "temperature"]:
        scalar_X_ls.append(
            jax.device_put(
                jnp.array(data_dict["t0_fields"][scalar_field][:, input_step][()]),
                jax.devices("cpu")[0],
            )
        )
        scalar_y_ls.append(
            jax.device_put(
                jnp.array(data_dict["t0_fields"][scalar_field][:, output_step][()]),
                jax.devices("cpu")[0],
            )
        )

    scalar_X = jnp.stack(scalar_X_ls, axis=1)
    scalar_y = jnp.stack(scalar_y_ls, axis=1)

    # reinsert channel dimension even though its just 1
    velocity_X = jax.device_put(
        jnp.array(data_dict["t1_fields"]["velocity"][:, input_step : input_step + 1][()]),
        jax.devices("cpu")[0],
    )
    velocity_y = jax.device_put(
        jnp.array(data_dict["t1_fields"]["velocity"][:, output_step : output_step + 1][()]),
        jax.devices("cpu")[0],
    )

    data_dict.close()

    return scalar_X, scalar_y, velocity_X, velocity_y


def merge_data(D: int, N: int, dir: str, n_traj: int, input_step: int, output_step: int):
    all_files = filter(lambda file: f"Msun_1" in file, os.listdir(dir))

    all_scalar_X = jnp.zeros((0, 3) + (N,) * D)  # 3 scalar channels, density, pressure, temp
    all_scalar_y = jnp.zeros((0, 3) + (N,) * D)
    all_velocity_X = jnp.zeros((0, 1) + (N,) * D + (D,))
    all_velocity_y = jnp.zeros((0, 1) + (N,) * D + (D,))
    for filename in all_files:
        scalar_X, scalar_y, velocity_X, velocity_y = read_one_h5(
            f"{dir}/{filename}", input_step, output_step
        )

        all_scalar_X = jnp.concatenate([all_scalar_X, scalar_X])
        all_scalar_y = jnp.concatenate([all_scalar_y, scalar_y])
        all_velocity_X = jnp.concatenate([all_velocity_X, velocity_X])
        all_velocity_y = jnp.concatenate([all_velocity_y, velocity_y])

        if len(all_scalar_X) >= n_traj:
            break

    # (n_traj, 3, spatial) and (n_traj, 1, spatial, tensor)
    all_scalar_X = all_scalar_X[:n_traj]
    all_scalar_y = all_scalar_y[:n_traj]
    all_velocity_X = all_velocity_X[:n_traj]
    all_velocity_y = all_velocity_y[:n_traj]

    # boundary conditions are open for this dataset
    layer_X = geom.BatchLayer({(0, 0): all_scalar_X, (1, 0): all_velocity_X}, D, False)
    layer_y = geom.BatchLayer({(0, 0): all_scalar_y, (1, 0): all_velocity_y}, D, False)

    return layer_X, layer_y


def get_data(
    D: int,
    N: int,
    dir: str,
    n_train: int,
    n_val: int,
    n_test: int,
    input_step: int = 0,
    output_step: int = 29,
    normalize: bool = True,
) -> tuple[geom.BatchLayer]:
    train_X, train_y = merge_data(D, N, dir + "train/", n_train, input_step, output_step)
    val_X, val_y = merge_data(D, N, dir + "valid/", n_val, input_step, output_step)
    test_X, test_y = merge_data(D, N, dir + "test/", n_test, input_step, output_step)

    if normalize:
        # log10 normalization
        # for layer in [train_X, train_y, val_X, val_y, test_X, test_y]:
        #     layer[(0, 0)] = jnp.log10(layer[(0, 0)])
        #     vec_data = layer[(1, 0)]  # (batch,channels,spatial,D)
        #     layer[(1, 0)] = jnp.concatenate(
        #         [
        #             jnp.where(vec_data > 0, jnp.log10(vec_data), jnp.zeros_like(vec_data)),
        #             jnp.where(vec_data < 0, jnp.log10(-vec_data), jnp.zeros_like(vec_data)),
        #         ],
        #         axis=1,
        #     )

        # mean, var normalization
        # do we normalize both the inputs and the outputs?
        # axes = (0,) + tuple(range(2, train_X[(0, 0)].ndim))
        # all_scalars = jnp.concatenate(
        #     [train_X[(0, 0)], train_y[(0, 0)], val_X[(0, 0)], val_y[(0, 0)]]
        # )
        # scalar_mean = jnp.mean(all_scalars, axis=axes, keepdims=True)  # (1,channels,(1,)*D)
        # scalar_std = jnp.std(all_scalars, axis=axes, keepdims=True)

        # axes = (0,) + tuple(range(2, train_X[(1, 0)].ndim))
        # all_vectors = jnp.concatenate(
        #     [train_X[(1, 0)], train_y[(1, 0)], val_X[(1, 0)], val_y[(1, 0)]]
        # )
        # vector_std = jnp.std(all_vectors, axis=axes, keepdims=True)  # (1,channels,(1,)*D,1)

        for data_group in [[train_X, val_X, test_X], [train_y, val_y, test_y]]:

            scalar_mean = jnp.mean(
                data_group[0][(0, 0)],
                axis=(0,) + tuple(range(2, data_group[0][(0, 0)].ndim)),
                keepdims=True,
            )
            scalar_std = jnp.std(
                data_group[0][(0, 0)],
                axis=(0,) + tuple(range(2, data_group[0][(0, 0)].ndim)),
                keepdims=True,
            )
            vector_std = jnp.std(
                data_group[0][(1, 0)],
                axis=(0,) + tuple(range(2, data_group[0][(1, 0)].ndim)),
                keepdims=True,
            )

            for layer in data_group:
                layer[(0, 0)] = (layer[(0, 0)] - scalar_mean) / scalar_std
                layer[(1, 0)] = layer[(1, 0)] / vector_std

    return (
        train_X,
        train_y,
        val_X,
        val_y,
        test_X,
        test_y,
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


@eqx.filter_jit
def map_and_loss(
    model: eqx.Module,
    layer_x: geom.BatchLayer,
    layer_y: geom.BatchLayer,
    aux_data: Optional[eqx.nn.State] = None,
    return_map: bool = False,
):
    out_layer = model(layer_x, aux_data)
    if isinstance(out_layer, tuple):
        out_layer, aux_data = out_layer

    loss = ml.smse_loss(out_layer, layer_y)

    return (loss, aux_data, out_layer) if return_map else (loss, aux_data)


def train_and_eval(
    data: tuple[geom.BatchLayer],
    key: ArrayLike,
    model_name: str,
    model: eqx.Module,
    lr: float,
    batch_size: int,
    epochs: int,
    save_model: Optional[str],
    load_model: Optional[str],
    has_aux: bool = False,
    verbose: int = 1,
) -> tuple[float]:
    train_X, train_Y, val_X, val_Y, test_single_X, test_single_Y = data
    batch_stats = eqx.nn.State(model) if has_aux else None

    print(f"Model params: {models.count_params(model):,}")

    if load_model is None:
        key, subkey = random.split(key)
        steps_per_epoch = int(np.ceil(train_X.get_L() / batch_size))
        model, batch_stats, train_loss, val_loss = ml_eqx.train(
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
            aux_data=batch_stats,
        )

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
            aux_data=batch_stats,
        )
        val_loss = ml_eqx.map_loss_in_batches(
            map_and_loss,
            model,
            val_X,
            val_Y,
            batch_size,
            subkey2,
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
        aux_data=batch_stats,
    )
    print(f"Test Loss: {test_loss}")

    return (train_loss, val_loss, test_loss)


def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="the data .hdf5 file", type=str)
    parser.add_argument("-e", "--epochs", help="number of epochs to run", type=int, default=50)
    parser.add_argument("-batch", help="batch size", type=int, default=8)
    parser.add_argument("-n_train", help="number of training trajectories", type=int, default=1)
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

    return parser.parse_args()


# Main
args = handleArgs(sys.argv)

D = 3
N = 64

key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)

# an attempt to reduce recompilation, but I don't think it actually is working
n_test = args.batch if args.n_test is None else args.n_test
n_val = args.batch if args.n_val is None else args.n_val

data = get_data(
    D,
    N,
    args.data,
    args.n_train,
    n_val,
    n_test,
    normalize=args.normalize,
)
input_keys = data[0].get_signature()
output_keys = input_keys

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
    save_model=args.save_model,
    load_model=args.load_model,
    verbose=args.verbose,
)

key, *subkeys = random.split(key, num=4)
model_list = [
    (
        "unet",
        partial(
            train_and_eval,
            model=models.UNet(
                D,
                input_keys,
                output_keys,
                depth=8,
                use_bias=True,
                activation_f=jax.nn.relu,
                equivariant=False,
                kernel_size=3,  # paper describes a patch size of 1?
                use_group_norm=False,
                key=subkeys[0],
            ),
            lr=3e-4,
        ),
    ),
    (
        "unet_equiv",
        partial(
            train_and_eval,
            model=models.UNet(
                D,
                input_keys,
                output_keys,
                depth=8,
                activation_f=jax.nn.gelu,
                conv_filters=conv_filters,
                upsample_filters=upsample_filters,
                key=subkeys[1],
            ),
            lr=3e-4,
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
#     [5e-5, 1e-4, 3e-4],
#     benchmark_type=ml.BENCHMARK_MODEL,
#     num_trials=args.n_trials,
#     num_results=3,
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
    num_results=3,
)

print(results)
mean_results = jnp.mean(results, axis=0)  # (benchmark_vals,models,outputs)
std_results = jnp.std(results, axis=0)
print("Mean", mean_results, sep="\n")
