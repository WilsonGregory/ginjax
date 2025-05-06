import sys
import os
import time
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import Optional, Union

import jax.numpy as jnp
import jax
import jax.random as random
from jaxtyping import ArrayLike
import optax
import equinox as eqx

import ginjax.geometric as geom
import ginjax.ml as ml
import ginjax.utils as utils
import ginjax.models as models


def plot_fields(
    images_dir: str, density_data: jax.Array, temperature_data: jax.Array, velocity_data: jax.Array
) -> None:
    density_data = density_data[0, :, 32]  # (59, spatial_2D)
    temperature_data = temperature_data[0, :, 32]
    velocity_data = velocity_data[0, :, 32]
    nrows = 4
    ncols = 5

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    for i, row in enumerate([0, 1, 2, 29]):
        geom.GeometricImage(jnp.log10(density_data[row]), 0, 2, False).plot(
            axes[i, 0], f"density step {row}", colorbar=True
        )
        geom.GeometricImage(jnp.log10(temperature_data[row]), 0, 2, False).plot(
            axes[i, 1], f"temperature step {row}", colorbar=True
        )
        vec_norm = geom.norm(2 + 1, velocity_data, keepdims=True)
        scaled_velocity_data = (jnp.log10(vec_norm) / vec_norm) * velocity_data
        geom.GeometricImage(scaled_velocity_data[row, ..., 0], 0, 2, False).plot(
            axes[i, 2], f"velocity_x step {row}", colorbar=True
        )
        geom.GeometricImage(scaled_velocity_data[row, ..., 1], 0, 2, False).plot(
            axes[i, 3], f"velocity_y step {row}", colorbar=True
        )
        geom.GeometricImage(scaled_velocity_data[row, ..., 2], 0, 2, False).plot(
            axes[i, 4], f"velocity_z step {row}", colorbar=True
        )

    plt.tight_layout()
    plt.savefig(f"{images_dir}supernova_steps.png")
    plt.close(fig)


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

    Note that the batch size of 16 is for the training files, valid and test only have 2. Thus:
    There are 400 Msun_1 total training trajectories, 50 total valid traj, and 50 total test traj

    args:
        filename (str): the full file path
        data_class (str): either 'train', 'test', or 'valid'
    returns: u, vxy as jax arrays
    """
    data_dict = h5py.File(filename)

    scalar_X_ls = []
    scalar_y_ls = []
    # only use density and temperature omitting pressure as in Keiya's paper
    for scalar_field in ["density", "temperature"]:
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

    all_scalar_X = jnp.zeros((0, 2) + (N,) * D)  # 2 scalar channels, density, temp
    all_scalar_y = jnp.zeros((0, 2) + (N,) * D)
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
    multi_image_x = geom.MultiImage({(0, 0): all_scalar_X, (1, 0): all_velocity_X}, D, False)
    multi_image_y = geom.MultiImage({(0, 0): all_scalar_y, (1, 0): all_velocity_y}, D, False)

    return multi_image_x, multi_image_y


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
    include_center: bool = False,
) -> tuple[
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
]:
    """
    Get the data.

    args:
        D: dimension of the space, should be 3
        N: sidelength of the images, should be 64
        dir: directory containing train, valid, and test folders
        n_train: number of training trajectories
        n_val: number of validation trajectories
        n_test: number of test trajectories
        input_step: timestep of the input data, defaults to 0
        output_step: timestep of the output data, defaults to 29 which is 0.1 Myr
        normalize: whether to normalize the data with log10 normalization
        include_center: break translation symmetry where energy is injected

    returns:
        returns 6 multi images corresponding to train_x, train_y, val_x, val_y, test_x, test_y
    """
    train_X, train_y = merge_data(D, N, dir + "train/", n_train, input_step, output_step)
    val_X, val_y = merge_data(D, N, dir + "valid/", n_val, input_step, output_step)
    test_X, test_y = merge_data(D, N, dir + "test/", n_test, input_step, output_step)

    if normalize:
        # log10 normalization
        for multi_image in [train_X, train_y, val_X, val_y, test_X, test_y]:
            multi_image[((), 0)] = jnp.log10(multi_image[((), 0)])
            vec_data = multi_image[((False,), 0)]  # (batch,channels,spatial,D)
            # original transformation is not equivariant, so scale the norm of the vectors by log10
            vec_norm = geom.norm(D + 2, vec_data, keepdims=True)  # (batch,channels,spatial,1)
            multi_image[((False,), 0)] = (jnp.log10(vec_norm) / vec_norm) * vec_data

        # mean and var scaling
        for data_group in [[train_X, val_X, test_X], [train_y, val_y, test_y]]:

            # (b,c,spatial) -> (1,c,1...)
            train_scalar = data_group[0][((), 0)]
            scalar_mean = jnp.mean(
                train_scalar, axis=(0,) + tuple(range(2, train_scalar.ndim)), keepdims=True
            )
            scalar_std = jnp.std(
                train_scalar, axis=(0,) + tuple(range(2, train_scalar.ndim)), keepdims=True
            )
            # (b,c,spatial,tensor) -> (1,c,1...,1)
            train_vec = data_group[0][((False,), 0)]
            vec_norm = jnp.linalg.norm(train_vec, axis=-1, keepdims=True)
            vector_std = jnp.std(
                vec_norm, axis=(0,) + tuple(range(2, vec_norm.ndim)), keepdims=True
            )

            for multi_image in data_group:
                multi_image[((), 0)] = (multi_image[((), 0)] - scalar_mean) / scalar_std
                multi_image[((False,), 0)] = multi_image[((False,), 0)] / vector_std

    if include_center:
        for data_X in [train_X, val_X, test_X]:
            # the fact that the boundary conditions are open already breaks this symmetry somewhat
            center = np.zeros((len(data_X[((), 0)]), 1) + (N,) * D)
            assert N % 2 == 0  # N will be 64 which is even
            center[(slice(None), slice(None)) + (slice(N // 2 - 1, N // 2 + 1),) * D] = 1
            data_X[((), 0)] = jnp.concatenate([data_X[((), 0)], center], axis=1)  # converts to jnp

    return (
        train_X,
        train_y,
        val_X,
        val_y,
        test_X,
        test_y,
    )


@eqx.filter_jit
def map_and_loss(
    model: models.MultiImageModule,
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    aux_data: Optional[eqx.nn.State] = None,
    return_map: bool = False,
) -> Union[
    tuple[jax.Array, Optional[eqx.nn.State], geom.MultiImage],
    tuple[jax.Array, Optional[eqx.nn.State]],
]:
    vmap_model = jax.vmap(model, in_axes=(0, None), out_axes=(0, None))
    out_image, aux_data = vmap_model(multi_image_x, aux_data)
    loss = ml.smse_loss(out_image, multi_image_y)

    return (loss, aux_data, out_image) if return_map else (loss, aux_data)


def train_and_eval(
    data: tuple[geom.MultiImage, ...],
    key: ArrayLike,
    model_name: str,
    model: models.MultiImageModule,
    lr: float,
    batch_size: int,
    epochs: int,
    save_model: Optional[str],
    load_model: Optional[str],
    images_dir: Optional[str],
    has_aux: bool = False,
    verbose: int = 1,
    is_wandb: bool = False,
) -> tuple[Optional[ArrayLike], ...]:
    train_X, train_Y, val_X, val_Y, test_single_X, test_single_Y = data
    batch_stats = eqx.nn.State(model) if has_aux else None

    print(f"Model params: {models.count_params(model):,}")

    if load_model is None:
        key, subkey = random.split(key)
        steps_per_epoch = int(np.ceil(train_X.get_L() / batch_size))
        model, batch_stats, train_loss, val_loss = ml.train(
            train_X,
            train_Y,
            map_and_loss,
            model,
            subkey,
            stop_condition=ml.AnyStop([ml.ValLoss(10), ml.EpochStop(epochs, verbose=verbose)]),
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
            is_wandb=is_wandb,
        )

        if save_model is not None:
            # TODO: need to save batch_stats as well
            ml.save(f"{save_model}{model_name}_L{train_X.get_L()}_e{epochs}_model.eqx", model)
    else:
        model = ml.load(f"{load_model}{model_name}_L{train_X.get_L()}_e{epochs}_model.eqx", model)

        key, subkey1, subkey2 = random.split(key, num=3)
        train_loss = ml.map_loss_in_batches(
            map_and_loss,
            model,
            train_X,
            train_Y,
            batch_size,
            subkey1,
            aux_data=batch_stats,
        )
        val_loss = ml.map_loss_in_batches(
            map_and_loss,
            model,
            val_X,
            val_Y,
            batch_size,
            subkey2,
            aux_data=batch_stats,
        )

    key, subkey = random.split(key)
    test_loss = ml.map_loss_in_batches(
        map_and_loss,
        model,
        test_single_X,
        test_single_Y,
        batch_size,
        subkey,
        aux_data=batch_stats,
    )
    print(f"Test Loss: {test_loss}")

    if images_dir is not None:
        # plot a single center slice of the 3d image, rather than trying to plot a 3d image
        components = ["density", "temperature", "velocity_x", "velocity_y", "velocity_z"]
        pred_y = model(test_single_X.get_one(keepdims=False))[0]
        pred_y_slice = pred_y.to_scalar_multi_image()[((), 0)][:, 32]  # (c,y,z)
        target_y = test_single_Y.get_one(keepdims=False)
        target_y_slice = target_y.to_scalar_multi_image()[((), 0)][:, 32]  # (c,y,z)
        ncols = len(pred_y_slice)
        nrows = 3

        vec_vmax = float(jnp.max(jnp.abs(jnp.stack([pred_y_slice[2:], target_y_slice[2:]]))))

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
        for i, pred_data, target_data, name in zip(
            range(ncols), pred_y_slice, target_y_slice, components
        ):
            pred_image_2d = geom.GeometricImage(pred_data, 0, 2, pred_y.is_torus[1:])
            target_image_2d = geom.GeometricImage(target_data, 0, 2, test_single_Y.is_torus[1:])
            diff_2d = (target_image_2d - pred_image_2d).norm()

            if i < 2:
                vmax = float(jnp.max(jnp.abs(jnp.stack([pred_data, target_data]))))
            else:
                vmax = vec_vmax

            pred_image_2d.plot(axes[0, i], f"test {name}", vmin=-vmax, vmax=vmax, colorbar=True)
            target_image_2d.plot(axes[1, i], f"actual {name}", vmin=-vmax, vmax=vmax, colorbar=True)
            diff_2d.plot(axes[2, i], f"diff {name}", vmin=-vmax, vmax=vmax, colorbar=True)

        plt.tight_layout()
        plt.savefig(f"{images_dir}{model_name}_L{train_X.get_L()}_e{epochs}.png")
        plt.close(fig)

    return (train_loss, val_loss, test_loss)


def handleArgs(argv):
    # n_train <=400, n_val <= 50, n_test <= 50
    # https://arxiv.org/abs/2410.23346 uses n_train=300
    parser = utils.get_common_parser()
    parser.add_argument(
        "--include-center",
        help="identify the center pixels where the supernova starts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--wandb-project", help="the wandb project", type=str, default="supernova")
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
    include_center=args.include_center,
)
input_keys = data[0].get_signature()
output_keys = data[1].get_signature()

group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    Ms=[3], ks=[0, 1, 2], parities=[0, 1], D=D, operators=group_actions
)
upsample_filters = geom.get_invariant_filters(
    Ms=[2], ks=[0, 1, 2], parities=[0, 1], D=D, operators=group_actions
)

train_kwargs = {
    "batch_size": args.batch,
    "epochs": args.epochs,
    "save_model": args.save_model,
    "load_model": args.load_model,
    "images_dir": args.images_dir,
    "verbose": args.verbose,
    "is_wandb": args.wandb,
}

key, *subkeys = random.split(key, num=4)
model_list = [
    # (
    #     "unet_keiya",
    #     train_and_eval,
    #     {
    #         "model": models.UNet(
    #             D,
    #             input_keys,
    #             output_keys,
    #             depth=8,
    #             use_bias=True,
    #             activation_f=jax.nn.relu,
    #             equivariant=False,
    #             kernel_size=1,  # paper describes a patch size of 1?
    #             use_group_norm=False,
    #             key=subkeys[0],
    #         ),
    #         "lr": 1e-3,
    #         **train_kwargs,
    #     },
    # ),
    (
        "unet",
        train_and_eval,
        {
            "model": models.UNet(
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
            "lr": 1e-3,
            **train_kwargs,
        },
    ),
    (
        "unet_equiv",
        train_and_eval,
        {
            "model": models.UNet(
                D,
                input_keys,
                output_keys,
                depth=8,
                activation_f=jax.nn.gelu,
                conv_filters=conv_filters,
                upsample_filters=upsample_filters,
                key=subkeys[1],
            ),
            "lr": 5e-4,
            **train_kwargs,
        },
    ),
]

key, subkey = random.split(key)

# Use this for benchmarking the models with known learning rates.
results = ml.benchmark(
    lambda _: data,
    model_list,
    subkey,
    "",
    [0],
    benchmark_type=ml.BENCHMARK_NONE,
    # "lr",
    # [1e-4, 3e-4, 5e-4, 1e-3],
    # benchmark_type=ml.BENCHMARK_MODEL,
    num_trials=args.n_trials,
    num_results=3,
    is_wandb=args.wandb,
    wandb_project=args.wandb_project,
    wandb_entity=args.wandb_entity,
)

print(results)
mean_results = jnp.mean(results, axis=0)  # (benchmark_vals,models,outputs)
std_results = jnp.std(results, axis=0)
print("Mean", mean_results, sep="\n")
