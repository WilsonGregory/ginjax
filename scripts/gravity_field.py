# generate gravitational field
from __future__ import annotations
import sys
import argparse
import time
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from typing_extensions import Optional, Self

import jax.numpy as jnp
import jax.random as random
import jax
from jax.typing import ArrayLike
import optax
import equinox as eqx

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml
import geometricconvolutions.models as models

# Generate data for the gravity problem


def get_gravity_vector(position1, position2, mass):
    r_vec = position1 - position2
    r_squared = np.linalg.norm(r_vec) ** 3
    return (mass / r_squared) * r_vec


def get_gravity_field_image(N, D, point_position, point_mass):
    field = np.zeros((N,) * D + (D,))

    # this could all be vectorized
    for position in it.product(range(N), repeat=D):
        position = np.array(position)
        if np.all(position == point_position):
            continue

        field[tuple(position)] = get_gravity_vector(point_position, position, point_mass)

    return geom.GeometricImage(jnp.array(field), 0, D, is_torus=False)


def get_data(
    N, D, num_points, rand_key, num_images=1
) -> tuple[geom.BatchMultiImage, geom.BatchMultiImage]:
    rand_key, subkey = random.split(rand_key)
    planets = random.uniform(subkey, shape=(num_points,))
    planets = planets / jnp.max(planets)

    masses = []
    gravity_fields = []
    for _ in range(num_images):
        point_mass = np.zeros((N, N))
        gravity_field = geom.GeometricImage.zeros(N=N, k=1, parity=0, D=D, is_torus=False)

        # Sample uniformly the cells
        rand_key, subkey = random.split(rand_key)
        possible_locations = np.array(list(it.product(range(N), repeat=D)))
        location_choices = random.choice(
            subkey, possible_locations, shape=(num_points,), replace=False, axis=0
        )
        for (x, y), mass in zip(location_choices, planets):
            point_mass[x, y] = mass
            gravity_field = gravity_field + get_gravity_field_image(N, D, np.array([x, y]), mass)

        masses.append(geom.GeometricImage(jnp.array(point_mass), 0, D, is_torus=False))
        gravity_fields.append(gravity_field)

    masses_images = geom.BatchMultiImage.from_images(masses)
    gravity_field_images = geom.BatchMultiImage.from_images(gravity_fields)
    assert masses_images is not None
    assert gravity_field_images is not None

    return masses_images, gravity_field_images


def plot_results(
    model: models.MultiImageModule,
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    axs: list,
    titles: list[str],
):
    assert len(axs) == len(titles)
    learned_x = model(multi_image_x)[0].to_images()[0]
    x = multi_image_x.to_images()[0]
    y = multi_image_y.to_images()[0]
    images = [x, y, learned_x, y - learned_x]
    for i, image, ax, title in zip(range(len(images)), images, axs, titles):
        if i == 0:
            vmin = 0.0
            vmax = 2.0
        else:
            vmin = None
            vmax = None

        image.plot(ax, title, vmin=vmin, vmax=vmax)


class Model(models.MultiImageModule):
    embedding: ml.ConvContract
    first_layers: list[ml.ConvContract]
    second_layers: list[ml.ConvContract]
    last_layer: ml.ConvContract

    def __init__(
        self: Self,
        spatial_dims: tuple[int, ...],
        input_keys: geom.Signature,
        conv_filters: geom.MultiImage,
        depth: int,
        key: ArrayLike,
    ) -> None:
        D = conv_filters.D
        mid_keys = geom.Signature((((0, 0), depth), ((1, 0), depth)))
        target_keys = geom.Signature((((1, 0), 1),))

        key, subkey = random.split(key)
        self.embedding = ml.ConvContract(input_keys, mid_keys, conv_filters, key=subkey)

        self.first_layers = []
        for dilation in range(1, spatial_dims[0]):  # dilations in parallel
            key, subkey = random.split(key)
            self.first_layers.append(
                ml.ConvContract(
                    mid_keys, mid_keys, conv_filters, rhs_dilation=(dilation,) * D, key=subkey
                )
            )

        self.second_layers = []
        for dilation in range(1, int(spatial_dims[0] / 2)):  # dilations in parallel
            key, subkey = random.split(key)
            self.first_layers.append(
                ml.ConvContract(
                    mid_keys, mid_keys, conv_filters, rhs_dilation=(dilation,) * D, key=subkey
                )
            )

        key, subkey = random.split(key)
        self.last_layer = ml.ConvContract(mid_keys, target_keys, conv_filters, key=subkey)

    def __call__(
        self: Self, x: geom.MultiImage, aux_data: Optional[eqx.nn.State] = None
    ) -> tuple[geom.MultiImage, Optional[eqx.nn.State]]:
        x = self.embedding(x)

        out_x = None
        for layer in self.first_layers:
            out_x = layer(x) if out_x is None else out_x + layer(x)

        assert out_x is not None
        x = out_x
        out_x = None
        for layer in self.second_layers:
            out_x = layer(x) if out_x is None else out_x + layer(x)

        return self.last_layer(x), aux_data


def map_and_loss(
    model: models.MultiImageModule,
    x: geom.BatchMultiImage,
    y: geom.BatchMultiImage,
    aux_data: Optional[eqx.nn.State] = None,
) -> tuple[jax.Array, Optional[eqx.nn.State]]:
    pred_y, aux_data = jax.vmap(model, in_axes=(0, None), out_axes=(0, None))(x, aux_data)
    assert isinstance(pred_y, geom.BatchMultiImage)
    return ml.smse_loss(pred_y, y), aux_data


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
assert conv_filters is not None

key, subkey = random.split(key)
model = Model(train_X.get_spatial_dims(), train_X.get_signature(), conv_filters, 10, key=subkey)
print(f"Num params: {sum([x.size for x in jax.tree_util.tree_leaves(model)]):,}")

if args.load_model:
    model = ml.load(f"{args.load_model}params.eqx", model)
else:
    optimizer = optax.adam(
        optax.exponential_decay(
            args.lr,
            transition_steps=int(train_X.get_L() / args.batch),
            decay_rate=0.995,
        )
    )
    key, subkey = random.split(key)
    model, _, train_loss, val_loss = ml.train(
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
        ml.save(f"{args.save_model}params.eqx", model)

key, subkey = random.split(key)
test_loss = ml.map_loss_in_batches(map_and_loss, model, test_X, test_Y, args.batch, subkey)
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
        train_X.get_one(),
        train_Y.get_one(),
        axs[0],
        titles,
    )
    plot_results(
        model,
        test_X.get_one(),
        test_Y.get_one(),
        axs[1],
        titles,
    )
    plt.savefig(f"{args.images_dir}gravity_field.png")
