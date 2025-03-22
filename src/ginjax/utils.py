import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.colors
import matplotlib.figure
from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge
import argparse
from typing_extensions import Any, Optional, Union

import jax
import jax.numpy as jnp


# Visualize the filters.

FIGSIZE = (4, 3)
XOFF, YOFF = 0.15, -0.1
TINY = 1.0e-5


def setup_plot(figsize: tuple[int, int] = (8, 6)) -> matplotlib.axes.Axes:
    """
    Create a figure of figsize and return the axes.

    args:
        figsize: the specified figure size, (width,height)

    returns:
        the new axes
    """
    return plt.figure(figsize=figsize).gca()


def nobox(ax: matplotlib.axes.Axes) -> None:
    """
    Turn axes and xticks,yticks off.

    args:
        ax: the axis to edit
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")


def finish_plot(
    ax: matplotlib.axes.Axes, title: str, xs: np.ndarray, ys: np.ndarray, D: int
) -> None:
    """
    Set title, axis limits, equal aspect ratio, and no box.

    args:
        ax: the axis to edit
        title: title of the plot
        xs: x dimensions
        ys: y dimensions
        D: dimension of the space
    """
    ax.set_title(title)
    if D == 2:
        ax.set_xlim(np.min(xs) - 0.55, np.max(xs) + 0.55)
        ax.set_ylim(np.min(ys) - 0.55, np.max(ys) + 0.55)
    if D == 3:
        ax.set_xlim(np.min(xs) - 0.75, np.max(xs) + 0.75)
        ax.set_ylim(np.min(ys) - 0.75, np.max(ys) + 0.75)
    ax.set_aspect("equal")
    nobox(ax)


def plot_boxes(ax: matplotlib.axes.Axes, xs: np.ndarray, ys: np.ndarray) -> None:
    """
    Plot the boxes for a geometric image.

    args:
        ax: the axis to plot on
        xs: pixels in the x direction
        ys: pixels in the y direction
    """
    ax.plot(
        xs[None] + np.array([-0.5, -0.5, 0.5, 0.5, -0.5]).reshape((5, 1)),
        ys[None] + np.array([-0.5, 0.5, 0.5, -0.5, -0.5]).reshape((5, 1)),
        "k-",
        lw=0.5,
        zorder=10,
    )


def fill_boxes(
    ax: matplotlib.axes.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    ws: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: Union[matplotlib.colors.Colormap, str],
    zorder: int = -100,
    colorbar: bool = False,
    alpha: float = 1.0,
) -> None:
    """
    Fill boxes with color according to the ws values.

    args:
        ax: the axis we are plotting on
        xs: the x coordinates of the pixels
        ys: the y coordinates of the pixels
        ws: the values to determine the fill color
        vmin: min value used for color
        vmax: max value used for color
        cmap: the color map
        zorder: whether to put the color in front or behind other elements
        colorbar: whether to include the colorbar
        alpha: the opacity of the box fill
    """
    plotted_img = ax.imshow(
        ws.reshape((np.max(xs) + 1, np.max(ys) + 1)).T,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        alpha=alpha,
        zorder=zorder,
    )
    if colorbar:
        plt.colorbar(plotted_img, ax=ax)


def plot_scalars(
    ax: matplotlib.axes.Axes,
    spatial_dims: tuple[int, ...],
    xs: np.ndarray,
    ys: np.ndarray,
    ws: np.ndarray,
    boxes: bool = True,
    fill: bool = True,
    symbols: bool = True,
    vmin: float = -2.0,
    vmax: float = 2.0,
    cmap: Union[matplotlib.colors.Colormap, str] = "BrBG",
    colorbar: bool = False,
) -> None:
    """
    Plot scalars from a scalar image.

    args:
        ax: the axis we are plotting on
        spatial_dims: the spatial dimensions of the image
        xs: the x coordinates of the pixels
        ys: the y coordinates of the pixels
        ws: the scalar pixel values
        boxes: whether to plot the surrounding boxes
        fill: whether to plot the color fill
        symbols: whether to plot symbol representation of the scalar values
        vmin: min value used for color
        vmax: max value used for color
        cmap: the color map
        colorbar: whether to include the colorbar
    """
    if boxes:
        plot_boxes(ax, xs, ys)
    if fill:
        fill_boxes(ax, xs, ys, ws, vmin, vmax, cmap, colorbar=colorbar)
    if symbols:
        height = ax.get_window_extent().height
        ss = (5 * height / spatial_dims[0]) * np.abs(ws)
        ax.scatter(xs[ws > TINY], ys[ws > TINY], marker="+", c="k", s=ss[ws > TINY], zorder=100)
        ax.scatter(
            xs[ws < -TINY],
            ys[ws < -TINY],
            marker="_",
            c="k",
            s=ss[ws < -TINY],
            zorder=100,
        )


def plot_vectors(
    ax: matplotlib.axes.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    ws: np.ndarray,
    boxes: bool = True,
    fill: bool = True,
    vmin: float = 0.0,
    vmax: float = 2.0,
    cmap: Union[matplotlib.colors.Colormap, str] = "PuRd",
    scaling: float = 0.33,
) -> None:
    """
    Plot vectors from a vector image.

    args:
        ax: the axis we are plotting on
        xs: the x coordinates of the pixels
        ys: the y coordinates of the pixels
        ws: the scalar pixel values
        boxes: whether to plot the surrounding boxes
        fill: whether to plot the color fill
        vmin: min value used for color
        vmax: max value used for color
        cmap: the color map
        scaling: how much to scale the vectors
    """
    if boxes:
        plot_boxes(ax, xs, ys)
    if fill:
        fill_boxes(ax, xs, ys, np.linalg.norm(ws, axis=-1), vmin, vmax, cmap, alpha=0.25)

    normws = np.linalg.norm(ws, axis=1)

    xs = xs[normws > TINY]
    ys = ys[normws > TINY]
    ws = ws[normws > TINY]

    for x, y, w, normw in zip(xs, ys, ws, normws[normws > TINY]):
        ax.arrow(
            x - scaling * w[0],
            y - scaling * w[1],
            2 * scaling * w[0],
            2 * scaling * w[1],
            length_includes_head=True,
            head_width=0.24 * scaling * normw,
            head_length=0.72 * scaling * normw,
            color="k",
            zorder=100,
        )


def plot_one_tensor(
    ax: matplotlib.axes.Axes,
    x: float,
    y: float,
    T: np.ndarray,
    zorder: int = 0,
    scaling: float = 0.33,
) -> None:
    """
    Plot a tensor a particular coordinate on the axis.

    args:
        ax: the axis we are plotting on
        x: the pixel x coordinate
        y: the pixel y coordinate
        T: the tensor
        zorder: whether to put this plot in front or behind other plots
        scaling: how much to scale the tensor
    """
    if np.abs(T[0, 0]) > TINY:
        # plot a double-headed arrow
        ax.arrow(
            x - scaling,
            y,
            2 * scaling * np.abs(T[0, 0]),
            0,
            length_includes_head=True,
            head_width=0.24 * scaling,
            head_length=0.72 * scaling,
            color="g" if T[0, 0] > TINY else "k",
            zorder=zorder,
        )
        ax.arrow(
            x + scaling,
            y,
            -2 * scaling * np.abs(T[0, 0]),
            0,
            length_includes_head=True,
            head_width=0.24 * scaling,
            head_length=0.72 * scaling,
            color="g" if T[0, 0] > TINY else "k",
            zorder=zorder,
        )
    if np.abs(T[1, 1]) > TINY:
        # plot a double-headed arrow
        ax.arrow(
            x,
            y - scaling,
            0,
            2 * scaling * np.abs(T[1, 1]),
            length_includes_head=True,
            head_width=0.24 * scaling,
            head_length=0.72 * scaling,
            color="g" if T[1, 1] > TINY else "k",
            zorder=zorder,
        )
        ax.arrow(
            x,
            y + scaling,
            0,
            -2 * scaling * np.abs(T[1, 1]),
            length_includes_head=True,
            head_width=0.24 * scaling,
            head_length=0.72 * scaling,
            color="g" if T[1, 1] > TINY else "k",
            zorder=zorder,
        )

    patches = []
    # plot the petals
    if T[0, 1] > TINY:
        patches.append(
            Wedge(
                (x - 0.25, y - 0.25),
                0.25 * np.abs(T[0, 1]),
                45,
                225,
                color="b",
                zorder=zorder,
                alpha=0.25,
            )
        )
        patches.append(
            Wedge(
                (x + 0.25, y + 0.25),
                0.25 * np.abs(T[0, 1]),
                -135,
                45,
                color="b",
                zorder=zorder,
                alpha=0.25,
            )
        )
    if T[0, 1] < -TINY:
        patches.append(
            Wedge(
                (x - 0.25, y + 0.25),
                0.25 * np.abs(T[0, 1]),
                135,
                315,
                color="b",
                zorder=zorder,
                alpha=0.25,
            )
        )
        patches.append(
            Wedge(
                (x + 0.25, y - 0.25),
                0.25 * np.abs(T[0, 1]),
                -45,
                135,
                color="b",
                zorder=zorder,
                alpha=0.25,
            )
        )
    if T[1, 0] > TINY:
        patches.append(
            Wedge(
                (x - 0.25, y - 0.25),
                0.25 * np.abs(T[1, 0]),
                -135,
                45,
                color="b",
                zorder=zorder,
                alpha=0.25,
            )
        )
        patches.append(
            Wedge(
                (x + 0.25, y + 0.25),
                0.25 * np.abs(T[1, 0]),
                45,
                225,
                color="b",
                zorder=zorder,
                alpha=0.25,
            )
        )
    if T[1, 0] < -TINY:
        patches.append(
            Wedge(
                (x - 0.25, y + 0.25),
                0.25 * np.abs(T[1, 0]),
                -45,
                135,
                color="b",
                zorder=zorder,
                alpha=0.25,
            )
        )
        patches.append(
            Wedge(
                (x + 0.25, y - 0.25),
                0.25 * np.abs(T[1, 0]),
                135,
                315,
                color="b",
                zorder=zorder,
                alpha=0.25,
            )
        )

    p = PatchCollection(patches, alpha=0.4)
    ax.add_collection(p)


def plot_tensors(
    ax: matplotlib.axes.Axes, xs: np.ndarray, ys: np.ndarray, ws: np.ndarray, boxes: bool = True
) -> None:
    """
    Plot a tensor image.

    args:
        ax: the axis to plot on
        xs: the x coordinates of the pixels
        ys: the y coordinates of the pixels
        ws: the tensor values at the pixels
        boxes: whether to also plot the boxes
    """
    if boxes:
        plot_boxes(ax, xs, ys)
    for x, y, w in zip(xs, ys, ws):
        normw = np.linalg.norm(w)
        if normw > TINY:
            plot_one_tensor(ax, x, y, w, zorder=100)


def plot_nothing(ax: matplotlib.axes.Axes) -> None:
    """
    Set the title to empty string and print no boxes.

    args:
        ax: the axis to plot nothing on
    """
    ax.set_title(" ")
    nobox(ax)


# avoid circular import by not specifying what images is a list of
def plot_grid(
    images: list, names: list[str], n_cols: int, **kwargs: bool
) -> matplotlib.figure.Figure:
    """
    Plot a grid of GeometricImages. The grid will have columns equal to n_cols, and the number of
    rows are calculated automatically. If there are extra spots in the grid, plot_nothing is called
    on those spaces.

    args:
        images: images to plot in the grid
        names: names of each image to plot as the title
        n_cols: number of columns in the gride
        kwargs: keyword arguments passed along to plot

    returns:
        the figure plotted
    """
    n_rows = max(1, np.ceil(len(images) / n_cols).astype(int))
    assert len(images) <= n_cols * n_rows
    bar = 8.0  # figure width in inches?
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(bar, 1.15 * bar * n_rows / n_cols),  # magic
        squeeze=False,
    )
    axes = axes.flatten()
    plt.subplots_adjust(
        left=0.001 / n_cols,
        right=1 - 0.001 / n_cols,
        wspace=0.2 / n_cols,
        bottom=0.001 / n_rows,
        top=1 - 0.001 / n_rows - 0.1 / n_rows,
        hspace=0.2 / n_rows,
    )

    for img, name, axis in zip(images, names, axes):
        img.plot(ax=axis, title=name, **kwargs)

    for axis in axes[len(images) :]:
        plot_nothing(axis)

    return fig


def power(img: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Compute the power of image
    From: https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/

    args:
        img: scalar image data, shape (batch,channel,spatial)

    returns:
        tuple of k, P, N
    """
    # images are assumed to be scalar images
    spatial_dims = img.shape[2:]

    # some assumptions about the image
    assert len(spatial_dims) == 2  # assert the D=2
    assert spatial_dims[0] == spatial_dims[1]  # the image is square

    kmax = min(s for s in spatial_dims) // 2
    even = spatial_dims[0] % 2 == 0  # are the image sides even?

    img = jnp.fft.fftn(img, s=spatial_dims)  # fourier transform
    P = img.real**2 + img.imag**2
    P = jnp.sum(
        jnp.mean(P, axis=0), axis=0
    )  # mean over batch, then sum over channels. Shape (spatial,)

    kfreq = jnp.fft.fftfreq(spatial_dims[0]) * spatial_dims[0]
    kfreq2D = jnp.meshgrid(kfreq, kfreq)
    k = jnp.linalg.norm(jnp.stack(kfreq2D, axis=0), axis=0)

    N = np.full_like(P, 2, dtype=jnp.int32)
    N[..., 0] = 1
    if even:
        N[..., -1] = 1

    k = k.flatten()
    P = P.flatten()
    N = N.flatten()

    kbin = jnp.ceil(k).astype(jnp.int32)
    k = jnp.bincount(kbin, weights=k * N)
    P = jnp.bincount(kbin, weights=P * N)
    N = jnp.bincount(kbin, weights=N).round().astype(jnp.int32)

    # drop k=0 mode and cut at kmax (smallest Nyquist)
    k = k[1 : 1 + kmax]
    P = P[1 : 1 + kmax]
    N = N[1 : 1 + kmax]

    k /= N
    P /= N

    return k, P, N


def plot_power(
    fields: list[jax.Array],
    labels: Optional[list[str]],
    ax: matplotlib.axes.Axes,
    title: str = "",
    xlabel: str = "unnormalized wavenumber",
    ylabel: str = "unnormalized power",
    hide_ticks: bool = False,
) -> None:
    """
    Plot the power spectrum of each image onto the same plot.

    args:
        fields: list of fields to plot the power spectrum, shape (batch,channel,spatial)
        labels: label for each field
        ax: the axis to plot on
        title: title of the plot
        xlabel: the x axis label
        ylabel: the y axis label
        hide_ticks: whether to hide the ticks
    """

    ks, Ps = [], []
    for field in fields:
        k, P, _ = power(field)
        ks.append(k)
        Ps.append(P)

    used_labels = labels if labels else [""] * len(ks)
    for k, P, l in zip(ks, Ps, used_labels):
        ax.loglog(k, P, label=l, alpha=0.7)

    if labels:
        ax.legend(fontsize=36)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if hide_ticks:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )


def get_common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument("--data", help="the data .hdf5 file", type=str, default=None)
    parser.add_argument("--n_train", help="number of training trajectories", type=int, default=1)
    parser.add_argument("--n_val", help="number of validation trajectories", type=int, default=None)
    parser.add_argument("--n_test", help="number of testing trajectories", type=int, default=None)
    parser.add_argument(
        "--normalize",
        help="normalize input data",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # training arguments
    parser.add_argument("-e", "--epochs", help="number of epochs to run", type=int, default=50)
    parser.add_argument("-b", "--batch", help="batch size", type=int, default=8)
    parser.add_argument("-t", "--n_trials", help="number of trials to run", type=int, default=1)
    parser.add_argument("--seed", help="the random number seed", type=int, default=None)
    parser.add_argument(
        "-v", "--verbose", help="verbose argument passed to trainer", type=int, default=1
    )
    parser.add_argument(
        "-s", "--save_model", help="file name to save the params", type=str, default=None
    )
    parser.add_argument(
        "-l", "--load_model", help="file name to load params from", type=str, default=None
    )
    parser.add_argument(
        "--images_dir", help="directory to save images, or None to not save", type=str, default=None
    )

    # wandb arguments
    parser.add_argument(
        "--wandb",
        help="whether to use wandb on this run",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--wandb-entity", help="the wandb user", type=str)

    return parser
