# generate gravitational field
from __future__ import annotations
import sys
import argparse
import time
import matplotlib.pyplot as plt
from typing_extensions import Self

import jax.numpy as jnp
import jax.random as random
import jax
from jax.typing import ArrayLike
import optax
import equinox as eqx

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml_eqx as ml_eqx
import geometricconvolutions.ml as ml
from geometricconvolutions.data import get_gravity_data as get_data


def plot_results(
    model: Model,
    layer_x: geom.Layer,
    layer_y: geom.Layer,
    axs: list[plt.Axes],
    titles: list[str],
):
    assert len(axs) == len(titles)
    learned_x = model(layer_x).to_images()[0]
    x = layer_x.to_images()[0]
    y = layer_y.to_images()[0]
    images = [x, y, learned_x, y - learned_x]
    for i, image, ax, title in zip(range(len(images)), images, axs, titles):
        if i == 0:
            vmin = 0.0
            vmax = 2.0
        else:
            vmin = None
            vmax = None

        image.plot(ax, title, vmin=vmin, vmax=vmax)


class Model(eqx.Module):
    embedding: ml_eqx.ConvContract
    first_layers: list[ml_eqx.ConvContract]
    second_layers: list[ml_eqx.ConvContract]
    last_layer: ml_eqx.ConvContract

    def __init__(
        self: Self,
        spatial_dims: tuple[int],
        input_keys: tuple[tuple[ml.LayerKey, int]],
        conv_filters: geom.Layer,
        depth: int,
        key: ArrayLike,
    ) -> Self:
        D = conv_filters.D
        mid_keys = (((0, 0), depth), ((1, 0), depth))
        target_keys = (((1, 0), 1),)

        key, subkey = random.split(key)
        self.embedding = ml_eqx.ConvContract(input_keys, mid_keys, conv_filters, key=subkey)

        self.first_layers = []
        for dilation in range(1, spatial_dims[0]):  # dilations in parallel
            key, subkey = random.split(key)
            self.first_layers.append(
                ml_eqx.ConvContract(
                    mid_keys, mid_keys, conv_filters, rhs_dilation=(dilation,) * D, key=subkey
                )
            )

        self.second_layers = []
        for dilation in range(1, int(spatial_dims[0] / 2)):  # dilations in parallel
            key, subkey = random.split(key)
            self.first_layers.append(
                ml_eqx.ConvContract(
                    mid_keys, mid_keys, conv_filters, rhs_dilation=(dilation,) * D, key=subkey
                )
            )

        key, subkey = random.split(key)
        self.last_layer = ml_eqx.ConvContract(mid_keys, target_keys, conv_filters, key=subkey)

    def __call__(self: Self, x: geom.Layer) -> geom.Layer:
        x = self.embedding(x)

        out_x = None
        for layer in self.first_layers:
            out_x = layer(x) if out_x is None else out_x + layer(x)

        x = out_x
        out_x = None
        for layer in self.second_layers:
            out_x = layer(x) if out_x is None else out_x + layer(x)

        return self.last_layer(x)


def map_and_loss(model: Model, x: geom.BatchLayer, y: geom.BatchLayer) -> float:
    return ml.smse_loss(jax.vmap(model)(x), y)


def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", help="where to save the image", type=str, default=None)
    parser.add_argument("-lr", help="learning rate", type=float, default=0.01)
    parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=50)
    parser.add_argument("-batch", help="batch size", type=int, default=1)
    parser.add_argument("-seed", help="the random number seed", type=int, default=None)
    parser.add_argument(
        "-s", "--save_model", help="folder location to save the model", type=str, default=None
    )
    parser.add_argument(
        "-l", "--load_model", help="folder location to load the model", type=str, default=None
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="levels of print statements during training",
        type=int,
        default=1,
    )
    return parser.parse_args()


# Main
args = handleArgs(sys.argv)

N = 16
D = 2
num_points = 5
num_train_images = 6
num_test_images = 10
num_val_images = 6

key = random.PRNGKey(args.seed if args.seed else time.time_ns())

key, subkey = random.split(key)
validation_X, validation_Y = get_data(N, D, num_points, subkey, num_val_images)

key, subkey = random.split(key)
test_X, test_Y = get_data(N, D, num_points, subkey, num_test_images)

key, subkey = random.split(key)
train_X, train_Y = get_data(N, D, num_points, subkey, num_train_images)

# start with basic 3x3 scalar, vector, and 2nd order tensor images
group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    Ms=[3], ks=[0, 1, 2], parities=[0], D=D, operators=group_actions
)

key, subkey = random.split(key)
model = Model(train_X.get_spatial_dims(), (((0, 0), 1),), conv_filters, 10, key=subkey)
print(f"Num params: {sum([x.size for x in jax.tree_util.tree_leaves(model)]):,}")

if args.load_model:
    model = ml_eqx.load(f"{args.load_model}params.eqx", model)
else:
    optimizer = optax.adam(
        optax.exponential_decay(
            args.lr,
            transition_steps=int(train_X.get_L() / args.batch),
            decay_rate=0.995,
        )
    )
    key, subkey = random.split(key)
    model, train_loss, val_loss = ml_eqx.train(
        train_X,
        train_Y,
        map_and_loss,
        model,
        subkey,
        ml.EpochStop(epochs=args.epochs, verbose=args.verbose),
        batch_size=args.batch,
        optimizer=optimizer,
        validation_X=validation_X,
        validation_Y=validation_Y,
        save_model=f"{args.save_model}params.eqx" if args.save_model else None,
    )
    if args.save_model:
        ml_eqx.save(f"{args.save_model}params.eqx", model)

key, subkey = random.split(key)
test_loss = ml_eqx.map_loss_in_batches(map_and_loss, model, test_X, test_Y, args.batch, subkey)
print("Full Test loss:", test_loss)
print(f"One Test loss:", map_and_loss(model, test_X.get_one(), test_Y.get_one()))

if args.images_dir is not None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "STIXGeneral"
    plt.tight_layout()

    titles = ["Input", "Ground Truth", "Prediction", "Difference"]
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(24, 12))
    plot_results(
        model,
        train_X.get_one_layer(),
        train_Y.get_one_layer(),
        axs[0],
        titles,
    )
    plot_results(
        model,
        test_X.get_one_layer(),
        test_Y.get_one_layer(),
        axs[1],
        titles,
    )
    plt.savefig(f"{args.images_dir}gravity_field.png")
