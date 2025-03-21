import time
import argparse
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import h5py
from typing import Optional, Union

import jax.numpy as jnp
import jax
import jax.random as random
from jaxtyping import ArrayLike
import optax
import equinox as eqx

import ginet.geometric as geom
import ginet.ml as ml
import ginet.utils as utils
import ginet.data as gc_data
import ginet.models as models


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
) -> tuple[geom.MultiImage, ...]:
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

    # (batch,2,timesteps,spatial)
    density_pressure = jnp.concatenate([density[:, None], pressure[:, None]], axis=1)
    # (batch,2*timesteps,spatial)
    density_pressure = density_pressure.reshape(
        (len(density_pressure), -1) + density_pressure.shape[3:]
    )

    is_torus = True
    total_steps = 21
    constant_fields = geom.MultiImage({}, D, is_torus)

    start = 0
    stop = n_train
    train_X, train_Y = gc_data.batch_time_series(
        geom.MultiImage(
            {(0, 0): density_pressure[start:stop], (1, 0): velocity[start:stop]}, D, is_torus
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
            {(0, 0): density_pressure[start:stop], (1, 0): velocity[start:stop]}, D, is_torus
        ),
        constant_fields,
        total_steps,
        past_steps,
        1,
    )

    start = start + n_val
    stop = start + n_test
    test_X, test_Y = gc_data.batch_time_series(
        geom.MultiImage(
            {(0, 0): density_pressure[start:stop], (1, 0): velocity[start:stop]}, D, is_torus
        ),
        constant_fields,
        total_steps,
        past_steps,
        1,
    )

    test_rollout_X, test_rollout_Y = gc_data.batch_time_series(
        geom.MultiImage(
            {(0, 0): density_pressure[start:stop], (1, 0): velocity[start:stop]}, D, is_torus
        ),
        constant_fields,
        total_steps,
        past_steps,
        rollout_steps,
    )

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

    img_arr = jnp.concatenate([test_multi_image_comp[(0, 0)], actual_multi_image_comp[(0, 0)]])
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
        multi_image_x[(1, 0)].shape[1],  # past_steps
        future_steps,
    )

    loss = ml.timestep_smse_loss(out, multi_image_y, future_steps)
    loss = loss[0] if future_steps == 1 else loss

    return (loss, aux_data, out) if return_map else (loss, aux_data)


def train_and_eval(
    data: tuple[geom.MultiImage, ...],
    key: ArrayLike,
    model_name: str,
    model: models.MultiImageModule,
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
    is_wandb: bool = False,
) -> tuple[Optional[ArrayLike], ...]:
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
    batch_stats = eqx.nn.State(model) if has_aux else None

    print(f"Model params: {models.count_params(model):,}")

    if load_model is None:
        steps_per_epoch = int(np.ceil(train_X.get_L() / batch_size))
        key, subkey = random.split(key)
        model, batch_stats, train_loss, val_loss = ml.train(
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

    key, subkey = random.split(key)
    test_rollout_loss, rollout_multi_image = ml.map_plus_loss_in_batches(
        partial(map_and_loss, future_steps=rollout_steps, return_map=True),
        model,
        test_rollout_X,
        test_rollout_Y,
        batch_size,
        subkey,
        aux_data=batch_stats,
    )
    print(f"Test Rollout Loss: {test_rollout_loss}, Sum: {jnp.sum(test_rollout_loss)}")

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

    return train_loss, val_loss, test_loss, *test_rollout_loss


def handleArgs() -> argparse.Namespace:
    parser = utils.get_common_parser()
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
    # need do to --wandb to activate, also need --wandb-entity your_wandb_name_here
    parser.add_argument("--wandb-project", help="the wandb project", type=str, default="cfd-2d")

    return parser.parse_args()


# Main
args = handleArgs()

D = 2
N = 128

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
output_keys = data[1].get_signature()  # (((0, 0), 2), ((1, 0), 1))

group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    Ms=[3], ks=[0, 1, 2], parities=[0, 1], D=D, operators=group_actions
)
upsample_filters = geom.get_invariant_filters(
    Ms=[2], ks=[0, 1, 2], parities=[0, 1], D=D, operators=group_actions
)
assert conv_filters is not None
assert upsample_filters is not None

train_kwargs = {
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

key, *subkeys = random.split(key, num=13)
model_list = [
    (
        "dil_resnet64",
        train_and_eval,
        {
            "model": models.DilResNet(
                D,
                input_keys,
                output_keys,
                depth=64,
                equivariant=False,
                kernel_size=3,
                key=subkeys[0],
            ),
            "lr": 2e-3,
            **train_kwargs,
        },
    ),
    (
        "dil_resnet_equiv20",
        train_and_eval,
        {
            "model": models.DilResNet(
                D,
                input_keys,
                output_keys,
                depth=20,
                conv_filters=conv_filters,
                key=subkeys[1],
            ),
            "lr": 1e-3,
            **train_kwargs,
        },
    ),
    (
        "dil_resnet_equiv48",
        train_and_eval,
        {
            "model": models.DilResNet(
                D,
                input_keys,
                output_keys,
                depth=48,
                conv_filters=conv_filters,
                key=subkeys[2],
            ),
            "lr": 1e-3,
            **train_kwargs,
        },
    ),
    (
        "resnet",
        train_and_eval,
        {
            "model": models.ResNet(
                D,
                input_keys,
                output_keys,
                depth=128,
                equivariant=False,
                kernel_size=3,
                key=subkeys[3],
            ),
            "lr": 1e-3,
            **train_kwargs,
        },
    ),
    (
        "resnet_equiv_groupnorm_42",
        train_and_eval,
        {
            "model": models.ResNet(
                D,
                input_keys,
                output_keys,
                depth=42,
                conv_filters=conv_filters,
                key=subkeys[4],
            ),
            "lr": 7e-4,
            **train_kwargs,
        },
    ),
    (
        "resnet_equiv_groupnorm_100",
        train_and_eval,
        {
            "model": models.ResNet(
                D,
                input_keys,
                output_keys,
                depth=100,  # very slow at 100
                conv_filters=conv_filters,
                key=subkeys[5],
            ),
            "lr": 7e-4,
            **train_kwargs,
        },
    ),
    (
        "unetBase",
        train_and_eval,
        {
            "model": models.UNet(
                D,
                input_keys,
                output_keys,
                depth=64,
                use_bias=True,
                activation_f=jax.nn.gelu,
                equivariant=False,
                kernel_size=3,
                use_group_norm=True,
                key=subkeys[6],
            ),
            "lr": 8e-4,
            **train_kwargs,
        },
    ),
    (
        "unetBase_equiv20",
        train_and_eval,
        {
            "model": models.UNet(
                D,
                input_keys,
                output_keys,
                depth=20,
                conv_filters=conv_filters,
                upsample_filters=upsample_filters,
                key=subkeys[7],
            ),
            "lr": 6e-4,  # 4e-4 to 6e-4 works, larger sometimes explodes
            **train_kwargs,
        },
    ),
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
    (
        "unet2015",
        train_and_eval,
        {
            "model": models.UNet(
                D,
                input_keys,
                output_keys,
                depth=64,
                use_bias=False,
                equivariant=False,
                kernel_size=3,
                use_batch_norm=True,
                key=subkeys[9],
            ),
            "lr": 8e-4,
            "has_aux": True,
            **train_kwargs,
        },
    ),
    (
        "unet2015_equiv20",
        train_and_eval,
        {
            "model": models.UNet(
                D,
                input_keys,
                output_keys,
                depth=20,
                use_bias=False,
                conv_filters=conv_filters,
                upsample_filters=upsample_filters,
                key=subkeys[10],
            ),
            "lr": 7e-4,  # sometimes explodes for larger values
            **train_kwargs,
        },
    ),
    (
        "unet2015_equiv48",
        train_and_eval,
        {
            "model": models.UNet(
                D,
                input_keys,
                output_keys,
                depth=48,
                use_bias=False,
                conv_filters=conv_filters,
                upsample_filters=upsample_filters,
                key=subkeys[11],
            ),
            "lr": 3e-4,
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

plot_mapping = {
    "dil_resnet64": ("DilResNet64", "blue", "o", "dashed"),
    "dil_resnet_equiv20": ("DilResNet20 (E)", "blue", "o", "dotted"),
    "dil_resnet_equiv48": ("DilResNet48 (E)", "blue", "o", "solid"),
    "resnet": ("ResNet128", "red", "s", "dashed"),
    "resnet_equiv_groupnorm_42": ("ResNet42 (E)", "red", "s", "dotted"),
    "resnet_equiv_groupnorm_100": ("ResNet100 (E)", "red", "s", "solid"),
    "unetBase": ("UNet64 Norm", "green", "P", "dashed"),
    "unetBase_equiv20": ("UNet20 Norm (E)", "green", "P", "dotted"),
    "unetBase_equiv48": ("UNet48 Norm (E)", "green", "P", "solid"),
    "unet2015": ("UNet64", "orange", "*", "dashed"),
    "unet2015_equiv20": ("Unet20 (E)", "orange", "*", "dotted"),
    "unet2015_equiv48": ("Unet48 (E)", "orange", "*", "solid"),
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
                min(float(mean_results[0, 2 * i, j]), float(mean_results[0, 2 * i + 1, j])),
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
