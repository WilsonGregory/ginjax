# generate gravitational field
from __future__ import annotations
import sys
import argparse
import time
import matplotlib.pyplot as plt
from typing_extensions import Optional, Self

import jax.numpy as jnp
import jax.random as random
import jax
from jax.typing import ArrayLike
import optax
import equinox as eqx

import ginet.geometric as geom
import ginet.ml as ml
import ginet.models as models

# Generate data for the gravity problem


def get_gravity_field(
    N: int, D: int, point_position: jax.Array, point_mass: jax.Array
) -> jax.Array:
    # (N,)*D + (D,)
    positions = jnp.stack(jnp.meshgrid(*(jnp.arange(N),) * D, indexing="ij"), axis=-1)
    r_vec = point_position.reshape((1,) * D + (D,)) - positions
    r_squared = jnp.linalg.norm(r_vec, axis=-1, keepdims=True) ** 3
    # at the pixel where the planet is, set it to 0 rather than dividing by 0
    gravity_field = jnp.where(
        r_squared == 0, jnp.zeros_like(r_squared), (point_mass / r_squared) * r_vec
    )
    assert isinstance(gravity_field, jax.Array)
    return gravity_field


def get_mass_gravity_fields(
    N: int, D: int, planets: jax.Array, rand_key: ArrayLike
) -> tuple[geom.MultiImage, geom.MultiImage]:
    # Sample uniformly the cells
    indices = jnp.stack(jnp.meshgrid(*(jnp.arange(N),) * D, indexing="ij"), axis=-1)
    location_choices = random.choice(
        rand_key, indices.reshape((-1, D)), shape=(num_points,), replace=False, axis=0
    )

    vmap_gravity_field = jax.vmap(get_gravity_field, in_axes=(None, None, 0, 0))
    gravity_field = jnp.sum(vmap_gravity_field(N, D, location_choices, planets), axis=0)

    # likely a nice jax way of doing this, but this works
    point_mass = jnp.zeros((N,) * D)
    for (x, y), mass in zip(location_choices, planets):
        point_mass = point_mass.at[x, y].set(mass)

    masses_images = geom.MultiImage({(0, 0): point_mass[None]}, D, is_torus=False)
    gravity_field_images = geom.MultiImage({(1, 0): gravity_field[None]}, D, is_torus=False)
    return masses_images, gravity_field_images


def get_data(
    N: int, D: int, num_points: int, rand_key: ArrayLike, num_images: int = 1
) -> tuple[geom.MultiImage, geom.MultiImage]:
    rand_key, subkey = random.split(rand_key)
    planets = random.uniform(subkey, shape=(num_images, num_points))
    planets = planets / jnp.max(planets, axis=1, keepdims=True)

    return jax.vmap(get_mass_gravity_fields, in_axes=(None, None, 0, 0))(
        N, D, planets, random.split(rand_key, num=num_images)
    )


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
    x: geom.MultiImage,
    y: geom.MultiImage,
    aux_data: Optional[eqx.nn.State] = None,
) -> tuple[jax.Array, Optional[eqx.nn.State]]:
    pred_y, aux_data = jax.vmap(model, in_axes=(0, None), out_axes=(0, None))(x, aux_data)
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
            transition_steps=int(num_train_images / args.batch),
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
