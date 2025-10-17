import time
import argparse
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import pathlib
import h5py
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
import ginjax.data as gc_data
import ginjax.models as models


def read_one_h5(D: int, filename: pathlib.Path, num_trajectories: int) -> tuple:
    """
    Given a dimension and filename, read the data and return as jax arrays.

    args:
        D: the dimension of the space
        filename: the full file path
        num_trajectories: number of trajectories to read

    returns:
        density, pressure, and velocity fields
    """
    data_dict = h5py.File(filename)

    # all of these are shape (num_trajectories, timesteps, spatial, tensor)
    # 1D: (10K,101,1024,tensor)
    # 2D: (1K,21,512,512,tensor)
    # 3D: (100,21,128,128,128,tensor)
    density = jax.device_put(
        jnp.array(data_dict["density"][:num_trajectories][()]), jax.devices("cpu")[0]
    )
    pressure = jax.device_put(
        jnp.array(data_dict["pressure"][:num_trajectories][()]), jax.devices("cpu")[0]
    )

    velocities = []
    for vkey in ["Vx", "Vy", "Vz"][:D]:
        velocities.append(
            jax.device_put(jnp.array(data_dict[vkey][:num_trajectories][()]), jax.devices("cpu")[0])
        )

    velocity = jnp.stack(velocities, axis=-1)

    data_dict.close()

    return density, pressure, velocity


def get_data(
    train_D: int,
    data_dir: str,
    n_train: int,
    n_val: int,
    n_test: int,
    past_steps: int,
    normalize: bool = True,
) -> tuple[geom.MultiImage, ...]:
    data_dir_path = pathlib.Path(data_dir)
    is_torus = True
    density_mean, density_std = 0, 1
    pressure_mean, pressure_std = 0, 1
    velocity_norm_std = 1

    densities = []
    pressures = []
    velocities = []
    for D, fname in zip(
        [1, 2, 3],  # ignore D=1 for now
        [
            "1D_CFD_Rand_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5",
            "2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5",
            "3D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_Train.hdf5",
        ],
    ):
        fpath = data_dir_path / f"cfd_{D}d" / fname
        n_traj = n_train + n_val + n_test if D == train_D else n_test

        density, pressure, velocity = read_one_h5(D, fpath, n_traj)

        if normalize and train_D == D:
            density_mean = jnp.mean(density[: (n_train + n_val)])
            density_std = jnp.std(density[: (n_train + n_val)])

            pressure_mean = jnp.mean(pressure[: (n_train + n_val)])
            pressure_std = jnp.std(pressure[: (n_train + n_val)])

            velocity_norm_std = jnp.std(jnp.linalg.norm(velocity[: n_train + n_val], axis=-1))

        densities.append(density)
        pressures.append(pressure)
        velocities.append(velocity)

    test_Xs = []
    test_Ys = []
    train_X, train_Y, val_X, val_Y = None, None, None, None
    for D, total_steps, density, pressure, velocity in zip(
        [1, 2, 3], [101, 21, 21], densities, pressures, velocities
    ):
        constant_fields = geom.MultiImage({}, D, is_torus)

        if normalize:
            density = (density - density_mean) / density_std
            pressure = (pressure - pressure_mean) / pressure_std
            velocity = velocity / velocity_norm_std

        # (batch,2,timesteps,spatial)
        density_pressure = jnp.concatenate([density[:, None], pressure[:, None]], axis=1)
        # (batch,2*timesteps,spatial)
        density_pressure = density_pressure.reshape(
            (len(density_pressure), -1) + density_pressure.shape[3:]
        )

        if train_D == D:
            start = 0
            stop = n_train
            train_X, train_Y = gc_data.batch_time_series(
                geom.MultiImage(
                    {(0, 0): density_pressure[start:stop], (1, 0): velocity[start:stop]},
                    D,
                    is_torus,
                ),
                constant_fields,
                total_steps,
                past_steps,
                1,
            )

            start = start + n_train
            stop = start + n_val
            val_X, val_Y = gc_data.batch_time_series(
                geom.MultiImage(
                    {(0, 0): density_pressure[start:stop], (1, 0): velocity[start:stop]},
                    D,
                    is_torus,
                ),
                constant_fields,
                total_steps,
                past_steps,
                1,
            )

            start = start + n_val
            stop = start + n_test
        else:
            start = 0
            stop = n_test

        test_X, test_Y = gc_data.batch_time_series(
            geom.MultiImage(
                {(0, 0): density_pressure[start:stop], (1, 0): velocity[start:stop]}, D, is_torus
            ),
            constant_fields,
            total_steps,
            past_steps,
            1,
        )
        test_Xs.append(test_X)
        test_Ys.append(test_Y)

    test_d1_X, test_d2_X, test_d3_X = test_Xs
    test_d1_Y, test_d2_Y, test_d3_Y = test_Ys

    assert (train_X is not None) and (train_Y is not None)
    assert (val_X is not None) and (val_Y is not None)

    return (
        train_X,
        train_Y,
        val_X,
        val_Y,
        test_d1_X,
        test_d1_Y,
        test_d2_X,
        test_d2_Y,
        test_d3_X,
        test_d3_Y,
    )


def plot_multi_image(
    test_multi_image: geom.MultiImage,
    actual_multi_image: geom.MultiImage,
    save_loc: str,
    future_steps: int,
    component: int = 0,
    show_power: bool = False,
    title: str = "",
    minimal: bool = False,
):
    """
    Plot all timesteps of a particular component of two MultiImages, and the differences between them.
    args:
        test_multi_image: the predicted MultiImage
        actual_multi_image: the ground truth MultiImage
        save_loc: file location to save the image
        future_steps: the number future time steps in the MultiImage
        component: index of the component to plot, default to 0
        show_power: whether to also plot the power spectrum
        title: additional str to add to title, will be "test {title} {col}"
            "actual {title} {col}"
        minimal: if minimal, no titles, colorbars, or axes labels
    """
    if test_multi_image.get_n_leading() == 2:
        test_multi_image = test_multi_image.get_one(keepdims=False)

    if actual_multi_image.get_n_leading() == 2:
        actual_multi_image = actual_multi_image.get_one(keepdims=False)

    test_multi_image_comp = test_multi_image.get_component(component, future_steps)
    actual_multi_image_comp = actual_multi_image.get_component(component, future_steps)

    test_images = test_multi_image_comp.to_images()
    actual_images = actual_multi_image_comp.to_images()

    img_arr = jnp.concatenate([test_multi_image_comp[((), 0)], actual_multi_image_comp[((), 0)]])
    vmax = float(jnp.max(jnp.abs(img_arr)))
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
    multi_images: list[geom.MultiImage],
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
                multi_image.batch_get_component(component, future_steps)[(0, 0)][:, i : i + 1]
                for multi_image in multi_images
            ],
            labels,
            ax,
            title=f"{title} {i}",
        )

    plt.savefig(save_loc)
    plt.close(fig)


@eqx.filter_jit
def map_and_loss(
    model: models.MultiImageModule,
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    aux_data: Optional[eqx.nn.State] = None,
    future_steps: int = 1,
    return_map: bool = False,
) -> Union[
    tuple[jax.Array, Optional[eqx.nn.State], geom.MultiImage],
    tuple[jax.Array, Optional[eqx.nn.State]],
]:
    vmap_autoregressive = jax.vmap(
        ml.autoregressive_map,
        in_axes=(None, 0, None, None, None),
        out_axes=(0, None),
        axis_name="batch",
    )
    out, aux_data = vmap_autoregressive(
        model,
        multi_image_x,
        aux_data,
        multi_image_x[((False,), 0)].shape[1],  # past_steps
        future_steps,
    )

    loss = ml.timestep_smse_loss(out, multi_image_y, future_steps)
    loss = loss[0] if future_steps == 1 else loss

    return (loss, aux_data, out) if return_map else (loss, aux_data)


def train_and_eval(
    data: tuple[geom.MultiImage, ...],
    key: ArrayLike,
    model_name: str,
    model: models.AnyDimensionalModule,
    lr: float,
    conv_filters_d2: geom.MultiImage,
    upsample_filters_d2: geom.MultiImage,
    conv_filters_d3: geom.MultiImage,
    upsample_filters_d3: geom.MultiImage,
    batch_size: int,
    epochs: int,
    rollout_steps: int,
    save_model: Optional[str],
    load_model: Optional[str],
    images_dir: Optional[str],
    has_aux: bool = False,
    verbose: int = 1,
    plot_component: int = 0,
    is_wandb: bool = False,
) -> tuple[Optional[ArrayLike], ...]:
    (
        train_X,
        train_Y,
        val_X,
        val_Y,
        test_d1_X,
        test_d1_Y,
        test_d2_X,
        test_d2_Y,
        test_d3_X,
        test_d3_Y,
    ) = data
    batch_stats = eqx.nn.State(model) if has_aux else None

    print(f"Model params: {models.count_params(model):,}")

    if load_model is None:
        steps_per_epoch = int(np.ceil(train_X.get_L() / batch_size))
        key, subkey = random.split(key)
        trained_model, batch_stats, train_loss, val_loss = ml.train(
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
            is_wandb=is_wandb,
        )

        if save_model is not None:
            # TODO: need to save batch_stats as well
            ml.save(f"{save_model}{model_name}_L{train_X.get_L()}_e{epochs}_model.eqx", model)
    else:
        trained_model = ml.load(
            f"{load_model}{model_name}_L{train_X.get_L()}_e{epochs}_model.eqx", model
        )

        key, subkey1, subkey2 = random.split(key, num=3)
        train_loss = ml.map_loss_in_batches(
            map_and_loss,
            trained_model,
            train_X,
            train_Y,
            batch_size,
            subkey1,
            aux_data=batch_stats,
        )
        val_loss = ml.map_loss_in_batches(
            map_and_loss,
            trained_model,
            val_X,
            val_Y,
            batch_size,
            subkey2,
            aux_data=batch_stats,
        )

    assert isinstance(trained_model, models.AnyDimensionalModule)
    test_losses = []
    for test_X, test_Y, conv_filters, upsample_filters in [
        (test_d1_X, test_d1_Y, None, None),
        (test_d2_X, test_d2_Y, conv_filters_d2, upsample_filters_d2),
        (test_d3_X, test_d3_Y, conv_filters_d3, upsample_filters_d3),
    ]:
        if test_X.D < train_X.D:
            continue
        elif test_X.D == train_X.D:
            trained_model_d = trained_model
        else:
            key, subkey = random.split(key)
            trained_model_d = trained_model.convertD(
                conv_filters, True, subkey, upsample_filters=upsample_filters
            )

        key, subkey = random.split(key)
        test_loss = ml.map_loss_in_batches(
            map_and_loss,
            trained_model_d,
            test_X,
            test_Y,
            batch_size,
            subkey,
            aux_data=batch_stats,
        )
        print(f"Test Loss D={test_X.D}: {test_loss}")
        test_losses.append(test_loss)

    if images_dir is not None:
        components = ["density", "pressure", "velocity_x", "velocity_y"]
        plot_multi_image(
            rollout_multi_image.get_one(),
            test_rollout_Y.get_one(),
            f"{images_dir}{model_name}_L{train_X.get_L()}_e{epochs}_rollout.png",
            future_steps=rollout_steps,
            component=plot_component,
            show_power=True,
            title=f"{components[plot_component]}",
        )
        plot_timestep_power(
            [rollout_multi_image, test_rollout_Y],
            ["test", "actual"],
            f"{images_dir}{model_name}_L{train_X.get_L()}_e{epochs}_{components[plot_component]}_power_spectrum.png",
            future_steps=rollout_steps,
            component=plot_component,
            title=f"{components[plot_component]}",
        )

    return train_loss, val_loss, *test_losses


def handleArgs() -> argparse.Namespace:
    parser = utils.get_common_parser()
    parser.add_argument(
        "--plot-component",
        help="which component to plot, one of 0-3",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
    )
    parser.add_argument(
        "--rollout-steps",
        help="number of steps to rollout in test",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--past-steps", help="number of past steps to use as input", type=int, default=4
    )
    # need do to --wandb to activate, also need --wandb-entity your_wandb_name_here
    parser.add_argument("--wandb-project", help="the wandb project", type=str, default="cfd-anyd")

    return parser.parse_args()


# Main
args = handleArgs()

D = 2  # dimension of the training data

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
    args.past_steps,
    args.normalize,
)
input_keys = data[0].get_signature()
output_keys = data[1].get_signature()  # (((0, 0), 2), ((1, 0), 1))

group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    Ms=[3], ks=[0, 1, 2], parities=[0, 1], D=D, operators=group_actions
)
upsample_filters = geom.get_invariant_filters(
    Ms=[2], ks=[0, 1, 2], parities=[0, 1], D=D, operators=group_actions
)

group_actions_d2 = geom.make_all_operators(2)
conv_filters_d2 = geom.get_invariant_filters(
    Ms=[3], ks=[0, 1, 2], parities=[0, 1], D=2, operators=group_actions_d2
)
upsample_filters_d2 = geom.get_invariant_filters(
    Ms=[2], ks=[0, 1, 2], parities=[0, 1], D=2, operators=group_actions_d2
)

group_actions_d3 = geom.make_all_operators(3)
conv_filters_d3 = geom.get_invariant_filters(
    Ms=[3], ks=[0, 1, 2], parities=[0, 1], D=3, operators=group_actions_d3
)
upsample_filters_d3 = geom.get_invariant_filters(
    Ms=[2], ks=[0, 1, 2], parities=[0, 1], D=3, operators=group_actions_d3
)

train_kwargs = {
    "conv_filters_d2": conv_filters_d2,
    "upsample_filters_d2": upsample_filters_d2,
    "conv_filters_d3": conv_filters_d3,
    "upsample_filters_d3": upsample_filters_d3,
    "batch_size": args.batch,
    "epochs": args.epochs,
    "rollout_steps": args.rollout_steps,
    "save_model": args.save_model,
    "load_model": args.load_model,
    "images_dir": args.images_dir,
    "verbose": args.verbose,
    "plot_component": args.plot_component,
    "is_wandb": args.wandb,
}

padding_mode = "CIRCULAR" if data[0].is_torus == (True,) * D else "ZEROS"
key, *subkeys = random.split(key, num=13)
model_list = [
    (
        "unetBase_equiv48",
        train_and_eval,
        {
            "model": models.UNet(
                D,
                input_keys,
                output_keys,
                depth=48,
                activation_f=jax.nn.gelu,
                conv_filters=conv_filters,
                upsample_filters=upsample_filters,
                key=subkeys[8],
            ),
            "lr": 4e-4,  # 4e-4 to 6e-4 works, larger sometimes explodes
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
    num_trials=args.n_trials,
    num_results=3 + args.rollout_steps,
    is_wandb=args.wandb,
    wandb_project=args.wandb_project,
    wandb_entity=args.wandb_entity,
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
