import argparse
import math
import numpy as np
import os
import pathlib
import time
from typing import Literal
from PIL import Image

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray
import matplotlib.pyplot as plt
import optax
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

import ginjax.geometric as geom
import ginjax.ml as ml
import ginjax.models as models
import ginjax.utils as utils
from ginjax.data import batch_time_series


def plot_multi_image(
    test_multi_image: geom.MultiImage,
    actual_multi_image: geom.MultiImage,
    save_loc: pathlib.Path,
    col_titles: list[str],
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

    # (channels,spatial)
    test_components = test_multi_image.to_scalar_multi_image()[((), 0)]
    actual_components = actual_multi_image.to_scalar_multi_image()[((), 0)]

    nrows = 3
    ncols = len(test_components)

    # max for field component
    zyxin_max = jnp.max(jnp.abs(jnp.stack([test_components[0], actual_components[0]])))
    actin_max = jnp.max(jnp.abs(jnp.stack([test_components[1], actual_components[1]])))
    force_max = jnp.max(
        jnp.abs(
            jnp.stack(
                [test_components[2], actual_components[2], test_components[3], actual_components[3]]
            )
        )
    )

    max_vals = [zyxin_max, actin_max, force_max, force_max]
    fig, axs = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows), dpi=144)
    for i, (test_field, actual_field, title, max_val) in enumerate(
        zip(test_components, actual_components, col_titles, max_vals)
    ):
        print(f"Plotting component {i}:{title}")
        geom.GeometricImage(test_field, 0, D).plot(
            axs[0][i],
            f"predicted {title}",
            vmin=-float(max_val),
            vmax=float(max_val),
            colorbar=True,
        )
        geom.GeometricImage(actual_field, 0, D).plot(
            axs[1][i], f"target {title}", vmin=-float(max_val), vmax=float(max_val), colorbar=True
        )
        geom.GeometricImage(test_field - actual_field, 0, D).plot(
            axs[2][i], f"diff {title}", vmin=-float(max_val), vmax=float(max_val), colorbar=True
        )

    plt.tight_layout()
    plt.savefig(save_loc)
    plt.close(fig)


def plot_gif(
    fields: list[Float[Array, "time spatial tensor"]], titles: list[str], save_loc: str
) -> None:
    """
    Plot a gif of timesteps, by component for vectors.

    args:
        fields: list of fields to plot
    """
    D = 2

    # -> (time, spatial, all_tensor_flat)
    flat_fields = jnp.concat(
        [field.reshape(field.shape[: D + 1] + (-1,)) for field in fields], axis=-1
    )
    flat_fields = jnp.moveaxis(flat_fields, -1, 1)  # -> (time, all_tensor_flat, spatial)

    nrows = 1
    ncols = flat_fields.shape[1]

    # max for field component
    max_vals = jnp.max(jnp.abs(flat_fields), axis=(0,) + tuple(range(2, 2 + D)))
    for frame, flat_fields_frame in enumerate(flat_fields):
        _, axs = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows), dpi=144)
        for ax, field, title, max_val in zip(axs, flat_fields_frame, titles, max_vals):
            geom.GeometricImage(field, 0, D).plot(
                ax, title, vmin=-float(max_val), vmax=float(max_val), colorbar=True
            )

        print(f"Saving at: {save_loc}_frame{frame}.png")
        plt.savefig(f"{save_loc}_frame{frame}.png")
        plt.close()

    images = []
    frame_images = [x for x in os.listdir(save_loc) if "frame" in x]
    for file_name in sorted(
        frame_images, key=lambda x: int(x[x.rfind("frame") + len("frame") : x.rfind(".png")])
    ):
        file_path = os.path.join(save_loc, file_name)
        images.append(Image.open(file_path))

    # 2. Save as an animated GIF
    images[0].save(
        f"{save_loc}_animation.gif",
        save_all=True,  # Ensures all frames are included, not just the first one
        append_images=images[1:],  # Appends the rest of the frames
        optimize=False,
        duration=200,  # Duration of each frame in milliseconds (e.g., 200ms = 5 FPS)
        loop=0,  # 0 means infinite loop; omit or change for specific iterations
    )


def plot_hist(image: Float[Array, " ..."], save_loc: pathlib.Path) -> None:
    plt.hist(image.ravel(), bins=50, log=True)
    plt.savefig(save_loc)
    plt.close()


def read_one(
    fname: pathlib.Path,
) -> tuple[
    Float[Array, " spatial"],
    Float[Array, " spatial"],
    Float[Array, "spatial D"],
    Bool[Array, " spatial"],
]:
    # shape (channels,spatial)
    data = jnp.array(np.load(fname), device=jax.devices("cpu")[0])
    zyxin = data[6]  # shape (spatial,)
    actin = data[7]

    fx = data[2]
    fy = data[3]
    force = jnp.stack([fx, fy], axis=-1)  # (spatial,tensor)

    # mask is (spatial,) of 0 for outside cell, 255 for inside cell.
    # Convert to bool, true for inside the cell, false for outside
    mask = data[4] != 0

    return zyxin, actin, force, mask


def read_cell(
    cell_dir: pathlib.Path,
) -> tuple[
    Float[Array, "steps spatial"],
    Float[Array, "steps spatial"],
    Float[Array, "steps spatial D"],
    Float[Array, "steps spatial"],
]:
    frame_files = os.listdir(cell_dir)
    # sort by the frame number, files are "yadayada_<frame>.npy".
    sorted_frames = sorted(frame_files, key=lambda s: int(s[s.rfind("_") + 1 : s.rfind(".npy")]))
    zyxin_ls = []
    actin_ls = []
    force_ls = []
    mask_ls = []
    for frame_file in sorted_frames:  # these need to be sorted properly
        zyxin, actin, force, mask = read_one(cell_dir / frame_file)
        zyxin_ls.append(zyxin)
        actin_ls.append(actin)
        force_ls.append(force)
        mask_ls.append(mask)

    zyxin = jnp.stack(zyxin_ls)
    actin = jnp.stack(actin_ls)
    force = jnp.stack(force_ls)
    mask = jnp.stack(mask_ls)

    return zyxin, actin, force, mask


def read_cells(
    D: int,
    cell_dirs: list[pathlib.Path],
    normalize: Literal["previous", "mean_std", "log1p"],
    images_dir: pathlib.Path | None,
    plot_histograms: bool,
) -> geom.MultiImage:
    """
    Read the cells and create a multi image out of them.

    args:
        D: the dimension of the space
        cell_dirs: a list of the cell directories where all the frames live

    returns:
        a multi image with shape (batch,channel*time,spatial,tensor)
    """
    zyxin_ls = []
    actin_ls = []
    force_ls = []
    for cell_dir in cell_dirs:  # do they all have the same number of timesteps?
        zyxin, actin, force, mask = read_cell(cell_dir)

        if plot_histograms:
            assert images_dir is not None
            for field, name in [(zyxin, "zyxin"), (actin, "actin"), (force, "force")]:
                plot_hist(field, images_dir / f"{cell_dir.name}_{name}_hist.png")
                plot_hist(field[mask], images_dir / f"{cell_dir.name}_{name}_masked_hist.png")

        if normalize == "previous":
            # follow the normalization done in the previous paper
            zyxin = (zyxin - jnp.mean(zyxin[mask])) / (
                jnp.mean(zyxin[mask]) - jnp.mean(zyxin[mask])
            )
            actin = (actin - jnp.mean(actin[mask == 0])) / (
                jnp.mean(actin[mask]) - jnp.mean(actin[mask])
            )
            force_norm = jnp.linalg.norm(force, axis=-1)  # (steps,spatial)
            # TODO: do I need to use the mask here?
            force = force / jnp.mean(force_norm)
        elif normalize == "mean_std":
            zyxin = (zyxin - jnp.mean(zyxin)) / jnp.std(zyxin)
            actin = (actin - jnp.mean(actin)) / jnp.std(actin)
            force = force / jnp.std(jnp.linalg.norm(force, axis=-1))
        elif normalize == "log1p":
            zyxin = jnp.log1p(zyxin)
            actin = jnp.log1p(actin)
            force_norm = jnp.linalg.norm(force, axis=-1, keepdims=True)  # (steps,spatial,1)
            force = (jnp.log1p(force_norm) / force_norm) * force

        if plot_histograms:
            assert images_dir is not None
            for field, name in [(zyxin, "zyxin"), (actin, "actin"), (force, "force")]:
                plot_hist(field, images_dir / f"{cell_dir.name}_{normalize}_{name}_hist.png")
                plot_hist(
                    field[mask], images_dir / f"{cell_dir.name}_{normalize}_{name}_masked_hist.png"
                )

        zyxin_ls.append(zyxin)
        actin_ls.append(actin)
        force_ls.append(force)

    # (batch,time,spatial,tensor)
    zyxins = jnp.stack(zyxin_ls)
    actins = jnp.stack(actin_ls)
    forces = jnp.stack(force_ls)

    # (batch,channel,time,spatial,tensor) -> (batch,channel*time,spatial,tensor)
    scalars = jnp.stack([zyxins, actins], axis=1).reshape(len(cell_dirs), -1, *zyxins.shape[2:])
    return geom.MultiImage({((), 0): scalars, ((False,), 0): forces}, D, is_torus=False)


def get_data(
    D: int,
    data_dir: pathlib.Path,
    n_train: int,
    n_val: int,
    n_test: int,
    past_steps: int,
    future_steps: int,
    batch_size: int,
    normalize: Literal["previous", "mean_std", "log1p"],
    images_dir: pathlib.Path | None,
) -> tuple[
    DataLoader[ml.MultiImageDataset],
    DataLoader[ml.MultiImageDataset],
    DataLoader[ml.MultiImageDataset],
    geom.Signature,
    geom.Signature,
]:
    """
    Load the data and put it into multi image datasets.
    """

    # use cell_1 as the test, as in the repo?

    cell_dirs = [
        data_dir / x
        for x in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, x)) and ("cell" in x)
    ]
    test_cells = cell_dirs[-1:]  # [cell_1]
    cells = cell_dirs[:-1]  # [cell_0, cell_2, cell_3]

    train_val = read_cells(D, cells, normalize, images_dir, False)
    test = read_cells(D, test_cells, normalize, images_dir, False)
    total_timesteps = train_val[((False,), 0)].shape[1]

    train_val_x, train_val_y = batch_time_series(
        train_val, geom.MultiImage({}, D, False), total_timesteps, past_steps, future_steps
    )

    train_x = train_val_x.get_subset(jnp.arange(n_train))
    train_y = train_val_y.get_subset(jnp.arange(n_train))
    val_x = train_val_x.get_subset(jnp.arange(n_train, n_train + n_val))
    val_y = train_val_y.get_subset(jnp.arange(n_train, n_train + n_val))

    test_x, test_y = batch_time_series(
        test, geom.MultiImage({}, D, False), total_timesteps, past_steps, future_steps
    )

    train_dataset = ml.MultiImageDataset(train_x, train_y)
    val_dataset = ml.MultiImageDataset(val_x, val_y)
    test_dataset = ml.MultiImageDataset(test_x, test_y)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=BatchSampler(
            RandomSampler(train_dataset), batch_size, drop_last=n_train > batch_size
        ),
        collate_fn=lambda x: x[0],
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=BatchSampler(
            SequentialSampler(val_dataset), batch_size, drop_last=n_val > batch_size
        ),
        collate_fn=lambda x: x[0],
    )
    test_dataloader = DataLoader(
        test_dataset,
        sampler=BatchSampler(
            SequentialSampler(test_dataset), batch_size, drop_last=n_test > batch_size
        ),
        collate_fn=lambda x: x[0],
    )

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        train_x.get_signature(),
        train_y.get_signature(),
    )


def train_and_eval(
    data: tuple[
        DataLoader[ml.MultiImageDataset],
        DataLoader[ml.MultiImageDataset],
        DataLoader[ml.MultiImageDataset],
    ],
    key: PRNGKeyArray,
    model_name: str,
    model: models.MultiImageModule,
    lr: float,
    batch_size: int,
    epochs: int,
    rollout_steps: int,
    model_dir: pathlib.Path | None,
    overwrite_save_model: bool,
    images_dir: pathlib.Path | None,
    has_aux: bool = False,
    verbose: int = 1,
    is_wandb: bool = False,
) -> tuple[Float[Array, ""], ...]:
    train_dl, val_dl, test_dl = data
    assert isinstance(train_dl.dataset, ml.MultiImageDataset)
    batch_stats = eqx.nn.State(model) if has_aux else None

    print(f"Model params: {models.count_params(model):,}")

    mapper = ml.Mapper([geom.Losses.NRMSE], eps=1e-5)

    model_path = model_dir / f"{model_name}.eqx" if model_dir else None
    if model_path and model_path.is_file() and not overwrite_save_model:
        trained_model, _ = ml.load_plus(model_path, model)
    else:
        steps_per_epoch = int(math.ceil(len(train_dl.dataset) / batch_size))
        trained_model, _, _, _, train_time = ml.train_dl(
            train_dl,
            mapper,
            model,
            stop_condition=ml.EpochStop(epochs, verbose=verbose),
            optimizer=optax.adamw(
                optax.warmup_cosine_decay_schedule(
                    1e-8, lr, 5 * steps_per_epoch, epochs * steps_per_epoch, 1e-7
                ),
                weight_decay=1e-5,
            ),
            val_dataloader=val_dl,
            val_map_and_loss=ml.Mapper([geom.Losses.NRMSE], eps=1e-5),
            aux_data=batch_stats,
            is_wandb=is_wandb,
        )

        if model_path:
            assert not model_path.is_file() or overwrite_save_model
            # TODO: need to save batch_stats as well
            ml.save_plus(model_path, trained_model, {"train_time": train_time})

    train_loss = ml.map_loss_in_batches_dl(mapper, trained_model, train_dl)
    val_loss = ml.map_loss_in_batches_dl(mapper, trained_model, val_dl)
    test_loss = ml.map_loss_in_batches_dl(mapper, trained_model, test_dl)

    print(f"Train Loss: {train_loss}")
    print(f"Val Loss: {val_loss}")
    print(f"Test Loss: {test_loss}")
    # rollout_mapper = ml.Mapper()
    # TODO: rollout loss

    if images_dir is not None:
        val_x_one, val_y_one = next(iter(val_dl))
        assert isinstance(val_x_one, geom.MultiImage)
        assert isinstance(val_y_one, geom.MultiImage)
        val_x_one = val_x_one.get_one(keepdims=False).get_one()
        val_y_one = val_y_one.get_one(keepdims=False).get_one()
        pred_y, _ = mapper.map(trained_model, val_x_one, batch_stats)
        one_loss, _ = mapper(trained_model, val_x_one, val_y_one)
        print(f"One Loss: {one_loss}")
        components = ["zyxin", "actin", "force_x", "force_y"]
        plot_multi_image(pred_y, val_y_one, images_dir / f"{model_name}_e{epochs}.png", components)

    return train_loss, val_loss, test_loss


def handleArgs() -> argparse.Namespace:
    """
    CUDA_VISIBLE_DEVICES=2 time python3 -m scripts.contractility \
    --data /data/wgregor4/contractility/ZyxAct_16kPa_small/ \
    --n-train 256 --n-val 32 --n-test 8 -b 2 -e 50 \
    --model-dir /data/wgregor4/runs/contractility/ \
    --images-dir /data/wgregor4/images/contractility/ \
    --normalize-type log1p

    Can do --n-val 128, but for speed do this
    """
    parser = utils.get_common_parser()
    parser.add_argument(
        "--past-steps", help="the number of past steps for the input", type=int, default=4
    )
    parser.add_argument(
        "--future-steps", help="the number of future steps to output", type=int, default=1
    )
    parser.add_argument(
        "--rollout-steps",
        help="number of steps to rollout in test",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--normalize-type",
        help="type of normalization, `previous` of the method from the earlier paper",
        choices=["previous", "mean_std", "log1p"],
        default="previous",
    )
    # need do to --wandb to activate, also need --wandb-entity your_wandb_name_here
    parser.add_argument(
        "--wandb-project", help="the wandb project", type=str, default="contractility"
    )

    return parser.parse_args()


# MAIN
args = handleArgs()

# Since we only have 4 trajectories, n_train and n_val refer to the number of data points after
# reshaping timesteps into batches.
D = 2
data_dir = pathlib.Path(args.data)
images_dir = pathlib.Path(args.images_dir) if args.images_dir else None

train_dl, val_dl, test_dl, input_keys, output_keys = get_data(
    D,
    data_dir,
    args.n_train,
    args.n_val,
    args.n_test,
    args.past_steps,
    args.future_steps,
    args.batch,
    args.normalize_type,
    images_dir,
)

if args.load_model or args.save_model:
    print("Use --model-dir and possibly --overwrite-save-model instead of --save-model")
    exit()


key = jax.random.PRNGKey(time.time_ns()) if (args.seed is None) else jax.random.PRNGKey(args.seed)

group_actions = geom.make_all_operators(D)
upsample_filters = geom.get_invariant_filters(
    Ms=[2], ks=[0, 1, 2], parities=[0], D=D, operators=group_actions
)
conv_filters = geom.get_invariant_filters(
    Ms=[3], ks=[0, 1, 2], parities=[0], D=D, operators=group_actions
)

train_kwargs = {
    "batch_size": args.batch,
    "epochs": args.epochs,
    "rollout_steps": args.rollout_steps,
    "model_dir": pathlib.Path(args.model_dir) if args.model_dir else None,
    "overwrite_save_model": args.overwrite_save_model,
    "images_dir": images_dir,
    "verbose": args.verbose,
    "is_wandb": args.wandb,
}

key, *subkeys = jax.random.split(key, num=13)
model_list = [
    (
        # batch=2 works, batch=4 fails
        "unetBase_equiv20",
        train_and_eval,
        {
            "model": models.UNet(
                D,
                input_keys,
                output_keys,
                depth=20,
                activation_f=jax.nn.gelu,
                conv_filters=conv_filters,
                upsample_filters=upsample_filters,
                key=subkeys[8],
            ),
            "lr": 4e-4,  # 4e-4 to 6e-4 works, larger sometimes explodes
            **train_kwargs,
        },
    ),
    # (
    #     "unetBase",
    #     train_and_eval,
    #     {
    #         "model": models.UNet(
    #             D,
    #             input_keys,
    #             output_keys,
    #             depth=64,
    #             use_bias=True,
    #             activation_f=jax.nn.gelu,
    #             equivariant=False,
    #             kernel_size=3,
    #             use_group_norm=False,
    #             padding_mode="ZEROS",
    #             key=subkeys[6],
    #         ),
    #         "lr": 8e-4,
    #         **train_kwargs,
    #     },
    # ),
]

key, subkey = jax.random.split(key)

# Use this for benchmarking the models with known learning rates.
results = ml.benchmark_lr(
    lambda _: (train_dl, val_dl, test_dl),
    model_list,
    subkey,
    [],  # lr_range
    num_trials=args.n_trials,
    num_results=3 + args.rollout_steps,
    is_wandb=args.wandb,
    wandb_project=args.wandb_project,
    wandb_entity=args.wandb_entity,
    # args=args, # needs to be a dictionary
)
