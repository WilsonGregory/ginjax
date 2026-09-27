import argparse
import math
import numpy as np
import os
import pathlib
import time
from typing import Literal, Self
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

PreprocessChoices = Literal[
    "identity",
    "percell_nonmask_mean",
    "percell_nonmask_scale",
    "mean",
    "std",
    "log1p_proteins",
    "log1p_force",
]


def plot_input_output(
    input: list[Float[Array, "timesteps spatial tensor"]],
    output: list[Float[Array, "timesteps spatial tensor"]],
    col_titles: list[str],
    save_loc: str,
) -> None:
    """
    Plot fields as rows and timesteps as columns, two images one for input and one for output.
    """
    for fields, name in [(input, "input"), (output, "output")]:
        flat_fields = [field.reshape(field.shape[:3] + (-1,)) for field in fields]
        max_vals = jnp.concat(
            [
                jnp.max(jnp.abs(flat_field)) * jnp.ones(flat_field.shape[-1])
                for flat_field in flat_fields
            ]
        )
        flat_fields = jnp.concat(flat_fields, axis=-1)  # (timesteps,spatial,all_flat_tensor)
        nrows = flat_fields.shape[-1]  # total channel components
        ncols = flat_fields.shape[0]  # number of timesteps

        fig, axs = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows), dpi=144)
        for row, (title, max_val) in enumerate(zip(col_titles, max_vals)):
            if title == "zyxin" or title == "actin":
                max_val = jnp.log1p(max_val)

            for col, flat_field in enumerate(flat_fields):
                if title == "zyxin" or title == "actin":
                    flat_field = jnp.log1p(flat_field)  # hacky, oh well

                print(f"Plotting {title} {col}")
                geom.GeometricImage(flat_field[..., row], 0, D).plot(
                    axs[row][col] if ncols > 1 else axs[row],
                    f"{title} {col}",
                    vmin=-float(max_val),
                    vmax=float(max_val),
                )

        plt.tight_layout()
        plt.savefig(f"{save_loc}_{name}.png")
        plt.close(fig)


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
    mask_max = jnp.max(jnp.abs(jnp.stack([test_components[2], actual_components[2]])))
    force_max = jnp.max(
        jnp.abs(
            jnp.stack(
                [test_components[3], actual_components[3], test_components[4], actual_components[4]]
            )
        )
    )

    max_vals = [zyxin_max, actin_max, mask_max, force_max, force_max]
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

    # images = []
    # frame_images = [x for x in os.listdir(save_loc) if "frame" in x]
    # for file_name in sorted(
    #     frame_images, key=lambda x: int(x[x.rfind("frame") + len("frame") : x.rfind(".png")])
    # ):
    #     file_path = os.path.join(save_loc, file_name)
    #     images.append(Image.open(file_path))

    # # 2. Save as an animated GIF
    # images[0].save(
    #     f"{save_loc}_animation.gif",
    #     save_all=True,  # Ensures all frames are included, not just the first one
    #     append_images=images[1:],  # Appends the rest of the frames
    #     optimize=False,
    #     duration=200,  # Duration of each frame in milliseconds (e.g., 200ms = 5 FPS)
    #     loop=0,  # 0 means infinite loop; omit or change for specific iterations
    # )


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
    # Convert to 1 for inside the cell, 0 for outside
    mask = (data[4] != 0).astype(int)
    force_mask = (data[5] != 0).astype(int)

    # plot_gif(
    #     [zyxin[None], actin[None], fx[None], fy[None], mask[None], force_mask[None]],
    #     ["zyxin", "actin", "force_x", "force_y", "mask", "force_mask"],
    #     "/data/wgregor4/images/contractility/example",
    # )

    # force_norm = jnp.linalg.norm(force, axis=-1)
    # plot_gif(
    #     [
    #         jnp.log1p(zyxin[None]),
    #         jnp.log1p(actin[None]),
    #         (jnp.log1p(force_norm) / force_norm) * fx[None],
    #         (jnp.log1p(force_norm) / force_norm) * fy[None],
    #         mask[None],
    #         force_mask[None],
    #     ],
    #     ["zyxin", "actin", "force_x", "force_y", "mask", "force_mask"],
    #     "/data/wgregor4/images/contractility/example_log1p",
    # )
    # exit()

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
    for frame_file in sorted_frames:
        zyxin, actin, force, mask = read_one(cell_dir / frame_file)
        zyxin_ls.append(zyxin)
        actin_ls.append(actin)
        force_ls.append(force)
        mask_ls.append(mask)

    zyxin = jnp.stack(zyxin_ls)
    actin = jnp.stack(actin_ls)
    force = jnp.stack(force_ls)
    mask = jnp.stack(mask_ls)

    # plot_input_output(
    #     [zyxin[:4], actin[:4], mask[:4], force[:4]],
    #     [zyxin[4:5], actin[4:5], mask[4:5], force[4:5]],
    #     ["zyxin", "actin", "mask", "force_x", "force_y"],
    #     "/data/wgregor4/images/contractility/setup",
    # )

    return zyxin, actin, force, mask


class Preprocessor:

    preprocess_steps: list[PreprocessChoices]
    mean: geom.MultiImage | None  # in shape (batch,channels,timesteps,spatial,tensor)
    std: geom.MultiImage | None  # in shape (batch,channels,timesteps,spatial,tensor)

    def __init__(self: Self, preprocess_steps: list[PreprocessChoices]) -> None:
        self.preprocess_steps = preprocess_steps
        self.mean = None
        self.std = None

    def __call__(
        self: Self,
        x: geom.MultiImage,
        mask: Float[Array, "batch timesteps spatial"],
        timesteps: int,
    ) -> geom.MultiImage:
        """
        Input shape (batch,channels*steps,spatial,tensor)
        """
        x = x.expand(1, timesteps)  # now (batch,channels,steps,spatial,tensor)

        for preprocess_step in self.preprocess_steps:
            out = x.empty()
            if preprocess_step == "identity":
                out = x.copy()
            elif preprocess_step == "percell_nonmask_mean":
                # follow the normalization done in the previous paper
                # vmap over batch for both, then channels for field
                def nonmask_mean(
                    field: Float[Array, "timesteps spatial"],
                    mask: Float[Array, "timesteps spatial"],
                ) -> Float[Array, "timesteps spatial"]:
                    return field - jnp.sum(jnp.where(mask == 0, field, 0.0)) / jnp.sum(mask == 0)

                vmap_nonmask_mean = jax.vmap(jax.vmap(nonmask_mean, in_axes=(0, None)))
                out = x.copy()
                # set negative values to 0 with the relu
                out[((), 0)] = jax.nn.relu(vmap_nonmask_mean(out[((), 0)], mask))
            elif preprocess_step == "percell_nonmask_scale":
                # follow the normalization done in the previous paper
                # vmap over batch for both, then channels for field
                def nonmask_scale(
                    field: Float[Array, "timesteps spatial"],
                    mask: Float[Array, "timesteps spatial"],
                ) -> Float[Array, "timesteps spatial"]:
                    return field / (jnp.mean(field[mask != 0]) - jnp.mean(field[mask == 0]))

                vmap_nonmask_scale = jax.vmap(jax.vmap(nonmask_scale, in_axes=(0, None)))
                out = x.copy()
                out[((), 0)] = vmap_nonmask_scale(out[((), 0)], mask)

                # (batch,steps,spatial,1)
                force_norm = jnp.linalg.norm(out[((False,), 0)], axis=-1, keepdims=True)
                # TODO: do I need to use the mask here?
                out[((False,), 0)] = out[((False,), 0)] / jnp.mean(
                    force_norm, axis=tuple(range(2, force_norm.ndim)), keepdims=True
                )
            elif preprocess_step == "mean":
                if self.mean is None:
                    self.mean = x.empty()
                    for (k, p), image_block in x.items():
                        if len(k) == 0:
                            mean = jnp.mean(
                                image_block,
                                axis=(0,) + tuple(range(2, image_block.ndim)),
                                keepdims=True,
                            )
                            # construct mean with batch=1 so it broadcasts with any batch
                            self.mean[k, p] = jnp.full((1,) + image_block.shape[1:], mean)
                        else:
                            self.mean[k, p] = jnp.zeros((1,) + image_block.shape[1:])

                out = x - self.mean
            elif preprocess_step == "std":
                if self.std is None:
                    self.std = x.empty()
                    for (k, p), image_block in x.items():
                        # std will broadcast with images, but axes all 1 except for channels
                        if len(k) == 0:
                            # (batch,channels,spatial) = (1,channels,1,1,1)
                            std = jnp.std(
                                image_block,
                                axis=(0,) + tuple(range(2, image_block.ndim)),
                                keepdims=True,
                            )
                            self.std[k, p] = jnp.full((1,) + image_block.shape[1:], std)
                        else:
                            norm_image_block = jnp.linalg.norm(
                                image_block,
                                axis=tuple(range(image_block.ndim - len(k), image_block.ndim)),
                                keepdims=True,
                            )
                            std = jnp.std(
                                norm_image_block,
                                axis=(0,) + tuple(range(2, image_block.ndim)),
                                keepdims=True,
                            )
                            self.std[k, p] = jnp.full((1,) + image_block.shape[1:], std)

                out = x / self.std
            elif preprocess_step == "log1p_proteins":
                # assumes that values are non-negative
                out = x.copy()
                out[((), 0)] = jnp.log1p(x[((), 0)])
            elif preprocess_step == "log1p_force":
                # assumes that values are non-negative
                out = x.copy()
                # (steps,spatial,1)
                force_norm = jnp.linalg.norm(x[((False,), 0)], axis=-1, keepdims=True)
                out[((False,), 0)] = (jnp.log1p(force_norm) / force_norm) * x[((False,), 0)]
            else:
                raise ValueError(f"{preprocess_step} not in {PreprocessChoices}")

            x = out

        return x.combine_axes([1, 2])

    def reverse(self: Self, x: geom.MultiImage, timesteps: int) -> geom.MultiImage:
        x = x.expand(1, timesteps)  # now (batch,channels,steps,spatial,tensor)

        for preprocess_step in reversed(self.preprocess_steps):
            out = x.empty()
            if preprocess_step in ["identity", "percell_nonmask_mean", "percell_nonmask_scale"]:
                # percell processing would require us to track which cell it comes from to reverse
                out = x.copy()
            elif preprocess_step == "mean":
                assert self.mean is not None
                out = x + self.mean

            elif preprocess_step == "std":
                assert self.std is not None
                out = x * self.std

            elif preprocess_step == "log1p_proteins":
                out = x.copy()
                out[((), 0)] = jnp.expm1(x[((), 0)])
            elif preprocess_step == "log1p_force":
                out = x.copy()
                # (steps,spatial,1)
                force_norm = jnp.linalg.norm(x[((False,), 0)], axis=-1, keepdims=True)
                out[((False,), 0)] = (jnp.expm1(force_norm) / force_norm) * x[((False,), 0)]
            else:
                raise ValueError(f"{preprocess_step} not in {PreprocessChoices}")

            x = out

        return x.combine_axes([1, 2])

    def __str__(self: Self) -> str:
        return "_".join(self.preprocess_steps)


def read_cells(
    D: int, cell_dirs: list[pathlib.Path], images_dir: pathlib.Path | None, plot_histograms: bool
) -> tuple[geom.MultiImage, Float[Array, "batch timesteps spatial"]]:
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
    mask_ls = []
    for cell_dir in cell_dirs:  # requires they have equal number of timesteps, currently do
        zyxin, actin, force, mask = read_cell(cell_dir)

        if plot_histograms:
            assert images_dir is not None
            for field, name in [(zyxin, "zyxin"), (actin, "actin"), (force, "force")]:
                plot_hist(field, images_dir / f"{cell_dir.name}_{name}_hist.png")
                plot_hist(field[mask != 0], images_dir / f"{cell_dir.name}_{name}_masked_hist.png")

        zyxin_ls.append(zyxin)
        actin_ls.append(actin)
        force_ls.append(force)
        mask_ls.append(mask)

    # (batch,time,spatial,tensor)
    zyxins = jnp.stack(zyxin_ls)
    actins = jnp.stack(actin_ls)
    forces = jnp.stack(force_ls)
    masks = jnp.stack(mask_ls)

    # (batch,channel,time,spatial,tensor) -> (batch,channel*time,spatial,tensor)
    scalars = jnp.stack([zyxins, actins], axis=1).reshape(len(cell_dirs), -1, *zyxins.shape[2:])

    return geom.MultiImage({((), 0): scalars, ((False,), 0): forces}, D, is_torus=False), masks


def get_data(
    D: int,
    data_dir: pathlib.Path,
    n_train: int,
    n_val: int,
    n_test: int,
    past_steps: int,
    future_steps: int,
    batch_size: int,
    preprocess: list[PreprocessChoices],
    images_dir: pathlib.Path | None,
    plot_histograms: bool,
) -> tuple[
    DataLoader[ml.MultiImageDataset],
    DataLoader[ml.MultiImageDataset],
    DataLoader[ml.MultiImageDataset],
    geom.Signature,
    geom.Signature,
    Preprocessor,
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

    train_val, train_val_mask = read_cells(D, cells, images_dir, plot_histograms)
    test, test_mask = read_cells(D, test_cells, images_dir, plot_histograms)
    total_timesteps = train_val[((False,), 0)].shape[1]

    preprocessor = Preprocessor(preprocess)
    train_val = preprocessor(train_val, train_val_mask, total_timesteps)  # sets mean,std if used
    test = preprocessor(test, test_mask, total_timesteps)

    # use only for getting the signature without the mask
    train_val_x, train_val_y = batch_time_series(
        train_val, geom.MultiImage({}, D, False), total_timesteps, past_steps, future_steps
    )
    x_sig = train_val_x.get_signature()
    y_sig = train_val_y.get_signature()

    # Add the mask after preprocessing
    train_val = (
        train_val.expand(1, total_timesteps)
        .append((), 0, train_val_mask[:, None], axis=1)
        .combine_axes([1, 2])
    )
    test = (
        test.expand(1, total_timesteps)
        .append((), 0, test_mask[:, None], axis=1)
        .combine_axes([1, 2])
    )

    # if plot_histograms:
    #     assert images_dir is not None
    #     for field, name in [(zyxins, "zyxin"), (actins, "actin"), (forces, "force")]:
    #         plot_hist(field, images_dir / f"{preprocessor}_{name}_hist.png")
    #         plot_hist(field[masks != 0], images_dir / f"{preprocessor}_{name}_masked_hist.png")

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
        x_sig,
        y_sig,
        preprocessor,
    )


class ContractilityMapper(ml.Mapper):
    """
    A mapper for the contractility data
    """

    reverse_preprocess: list[bool]
    preprocessor: Preprocessor
    past_steps: int
    future_steps: int
    has_mask: bool

    def __init__(
        self: Self,
        losses: list[geom.Losses],
        reverse_preprocess: list[bool],
        preprocessor: Preprocessor,
        past_steps: int,
        future_steps: int,
        has_mask: bool = True,
        residual: bool = False,
        reduce: str | None = "mean",
        eps: float = 0,
    ) -> None:
        """
        args:
            losses: the list of losses
            reverse_preprocess: for each loss, whether to reverse the preprocessing prior to
                computing the loss
            preprocessor: the preprocessor used to reverse the preprocessing as necessary
            past_steps: the number of historical steps in the input
            has_mask: whether the last channel of the scalars input and output is a mask
            residual: whether to calculate the residual loss, and hence the model will learn
            reduce: how to reduce over the batch, the default is mean
            eps: epsilon used for normalized losses
        """
        super().__init__(losses, residual, reduce, eps)
        self.reverse_preprocess = reverse_preprocess
        self.preprocessor = preprocessor
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.has_mask = has_mask

    @eqx.filter_jit
    def map_plus_loss(
        self: Self,
        model: models.MultiImageModule,
        multi_image_x: geom.MultiImage,
        multi_image_y: geom.MultiImage,
        aux_data: eqx.nn.State | None = None,
    ) -> tuple[geom.MultiImage, jax.Array, eqx.nn.State | None]:
        """
        Like the __call__, but also returned the mapped pred_y. Its also unreduced.
        """
        if self.has_mask:
            x_expanded = multi_image_x.expand(1, self.past_steps)
            multi_image_x = x_expanded.empty()
            input_mask = x_expanded[((), 0)][:, -1:]  # (batch,1,past_steps,spatial)
            for (k, p), image_block in x_expanded.items():
                if len(k) == 0:
                    multi_image_x.append(k, p, image_block[:, :-1])
                else:
                    multi_image_x.append(k, p, image_block)

            multi_image_x = multi_image_x.combine_axes([1, 2])

            y_expanded = multi_image_y.expand(1, self.future_steps)
            multi_image_y = y_expanded.empty()
            output_mask = y_expanded[((), 0)][:, -1:]  # (batch,1,future_steps,spatial)
            for (k, p), image_block in y_expanded.items():
                if len(k) == 0:
                    multi_image_y.append(k, p, image_block[:, :-1])
                else:
                    multi_image_y.append(k, p, image_block)

            multi_image_y = multi_image_y.combine_axes([1, 2])

        pred_y, aux_data = self.map(model, multi_image_x, aux_data)
        # TODO: its getting broadcast the wrong way during the reverse
        pred_y_reversed = self.preprocessor.reverse(pred_y, self.future_steps)
        multi_image_y_reversed = self.preprocessor.reverse(multi_image_y, self.future_steps)

        loss_outputs = []
        for loss, reverse in zip(self.losses, self.reverse_preprocess):  # the order is important

            pred_y_ = pred_y_reversed if reverse else pred_y
            multi_image_y_ = multi_image_y_reversed if reverse else multi_image_y

            if loss is geom.Losses.SMSE:
                loss_outputs.append(ml.smse_loss(pred_y_, multi_image_y_, self.reduce))
            elif loss is geom.Losses.NRMSE:
                loss_outputs.append(
                    ml.nrmse_loss(pred_y_, multi_image_y_, self.reduce, eps=self.eps)
                )
            elif loss is geom.Losses.NRMSE_PER_PIXEL:
                loss_outputs.append(
                    ml.nrmse_per_pixel_loss(pred_y_, multi_image_y_, self.reduce, eps=self.eps)
                )
            elif loss is geom.Losses.L2_REL:
                loss_outputs.append(
                    ml.l2_rel_error(pred_y_, multi_image_y_, self.reduce, eps=self.eps)
                )
            elif loss is geom.Losses.L2_REL_PER_PIXEL:
                loss_outputs.append(
                    ml.l2_per_pixel_rel_error(pred_y_, multi_image_y_, self.reduce, eps=self.eps)
                )

        # if we aren't reducing the batch dimension, we don't want to squeeze it out
        loss_outputs = jnp.stack(loss_outputs, axis=-1)
        if len(self.losses) == 1:
            squeeze_outputs = jnp.squeeze(loss_outputs, axis=1 if self.reduce is None else 0)
        else:
            squeeze_outputs = loss_outputs

        return pred_y, squeeze_outputs, aux_data

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
        _, squeeze_outputs, aux_data = self.map_plus_loss(
            model, multi_image_x, multi_image_y, aux_data
        )
        return squeeze_outputs, aux_data


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
    past_steps: int,
    future_steps: int,
    rollout_steps: int,
    preprocessor: Preprocessor,
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

    train_mapper = ContractilityMapper(
        [geom.Losses.NRMSE], [False], preprocessor, past_steps, future_steps, eps=1e-5
    )
    val_mapper = ContractilityMapper(
        [geom.Losses.NRMSE], [True], preprocessor, past_steps, future_steps, eps=1e-5
    )

    model_name_extended = f"{model_name}_{preprocessor}_L{len(train_dl.dataset)}_e{epochs}"
    model_path = model_dir / f"{model_name_extended}.eqx" if model_dir else None
    if model_path and model_path.is_file() and not overwrite_save_model:
        trained_model, _ = ml.load_plus(model_path, model)
    else:
        steps_per_epoch = int(math.ceil(len(train_dl.dataset) / batch_size))
        trained_model, _, _, _, train_time = ml.train_dl(
            train_dl,
            train_mapper,
            model,
            stop_condition=ml.EpochStop(epochs, verbose=verbose),
            optimizer=optax.adamw(
                optax.warmup_cosine_decay_schedule(
                    1e-8, lr, 5 * steps_per_epoch, epochs * steps_per_epoch, 1e-7
                ),
                weight_decay=1e-5,
            ),
            val_dataloader=val_dl,
            val_map_and_loss=val_mapper,
            aux_data=batch_stats,
            is_wandb=is_wandb,
        )

        if model_path:
            assert not model_path.is_file() or overwrite_save_model
            # TODO: need to save batch_stats as well
            ml.save_plus(model_path, trained_model, {"train_time": train_time})

    eval_mapper = ContractilityMapper(
        [geom.Losses.NRMSE, geom.Losses.NRMSE, geom.Losses.SMSE, geom.Losses.SMSE],
        [False, True, False, True],
        preprocessor,
        past_steps,
        future_steps,
        eps=1e-5,
    )
    train_loss = ml.map_loss_in_batches_dl(eval_mapper, trained_model, train_dl)
    val_loss = ml.map_loss_in_batches_dl(eval_mapper, trained_model, val_dl)
    test_loss = ml.map_loss_in_batches_dl(eval_mapper, trained_model, test_dl)

    print(f"Train Loss: {train_loss}")
    print(f"Val Loss: {val_loss}")
    print(f"Test Loss: {test_loss}")
    # TODO: rollout loss

    if images_dir is not None:
        val_x_one, val_y_one = next(iter(val_dl))
        assert isinstance(val_x_one, geom.MultiImage)
        assert isinstance(val_y_one, geom.MultiImage)
        # (1,channels*future_steps,spatial,tensor)
        val_x_one = val_x_one.get_one(keepdims=False).get_one()
        val_y_one = val_y_one.get_one(keepdims=False).get_one()
        pred_y, one_loss, _ = train_mapper.map_plus_loss(
            trained_model, val_x_one, val_y_one, batch_stats
        )
        pred_y.append((), 0, val_y_one[(), 0][:, -1:], axis=1)  # add the mask back to pred_y
        print(f"One Loss: {one_loss}")
        components = ["zyxin", "actin", "mask", "force_x", "force_y"]
        # plot_multi_image(pred_y, val_y_one, images_dir / f"{model_name_extended}.png", components)

    return train_loss[0], val_loss[0], test_loss[0]


def handleArgs() -> argparse.Namespace:
    """
    CUDA_VISIBLE_DEVICES=3 time python3 -m scripts.contractility \
    --data /data/wgregor4/contractility/ZyxAct_16kPa_small/ \
    --n-train 232 --n-val 112 --n-test 8 -b 2 -e 10 \
    --model-dir /data/wgregor4/runs/contractility/ \
    --images-dir /data/wgregor4/images/contractility/ \
    --preprocess percell_nonmask_mean,log1p_proteins,std

    Can do --n-val 128, but for speed do this
    """
    parser = utils.get_common_parser()
    parser.add_argument(
        "--past-steps", help="the number of past steps for the input", type=int, default=4
    )
    parser.add_argument(
        "--future-steps", help="number of output future steps during training", type=int, default=1
    )
    parser.add_argument(
        "--rollout-steps",
        help="number of output future steps to evaluate with",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--preprocess",
        help=f"the preprocessing steps in order, choices {PreprocessChoices}",
        type=lambda s: s.split(","),
        default="log1p,std",
    )
    parser.add_argument(
        "--plot-histograms",
        help="whether to plot the histograms of the input data",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    # need do to --wandb to activate, also need --wandb-entity your_wandb_name_here
    parser.add_argument(
        "--wandb-project", help="the wandb project", type=str, default="contractility"
    )

    return parser.parse_args()


# MAIN
args = handleArgs()

if args.load_model or args.save_model:
    print("Use --model-dir and possibly --overwrite-save-model instead of --save-model")
    exit()

# Since we only have 4 trajectories, n_train and n_val refer to the number of data points after
# reshaping timesteps into batches.
D = 2
data_dir = pathlib.Path(args.data)
images_dir = pathlib.Path(args.images_dir) if args.images_dir else None

train_dl, val_dl, test_dl, input_keys, output_keys, preprocessor = get_data(
    D,
    data_dir,
    args.n_train,
    args.n_val,
    args.n_test,
    args.past_steps,
    args.future_steps,
    args.batch,
    args.preprocess,
    images_dir,
    args.plot_histograms,
)

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
    "past_steps": args.past_steps,
    "future_steps": args.future_steps,
    "rollout_steps": args.rollout_steps,
    "preprocessor": preprocessor,
    "model_dir": pathlib.Path(args.model_dir) if args.model_dir else None,
    "overwrite_save_model": args.overwrite_save_model,
    "images_dir": images_dir,
    "verbose": args.verbose,
    "is_wandb": args.wandb,
}

key, *subkeys = jax.random.split(key, num=13)
model_list = [
    # (
    #     # batch=2 works, batch=4 fails
    #     "unetBase_equiv20",
    #     train_and_eval,
    #     {
    #         "model": models.UNet(
    #             D,
    #             input_keys,
    #             output_keys,
    #             depth=20,
    #             activation_f=jax.nn.gelu,
    #             conv_filters=conv_filters,
    #             upsample_filters=upsample_filters,
    #             key=subkeys[8],
    #         ),
    #         "lr": 4e-4,  # 4e-4 to 6e-4 works, larger sometimes explodes
    #         **train_kwargs,
    #     },
    # ),
    (
        "lastStepIdentity",
        train_and_eval,
        {
            "model": models.LastStepIdentity(D, args.past_steps),
            "lr": 3e-4,  # unused
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
