import time
import optax
from typing_extensions import Optional, Self

import jax
from jax import random
from jaxtyping import ArrayLike
import equinox as eqx

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml


class SimpleModel(eqx.Module):
    D: int
    net: list[ml.ConvContract]

    def __init__(
        self: Self,
        D: int,
        input_keys: tuple[tuple[ml.LayerKey, int]],
        output_keys: tuple[tuple[ml.LayerKey, int]],
        conv_filters: geom.Layer,
        key: ArrayLike,
    ):
        self.D = D
        key, subkey1, subkey2 = random.split(key, num=3)
        self.net = [
            ml.ConvContract(input_keys, output_keys, conv_filters, False, key=subkey1),
            ml.ConvContract(output_keys, output_keys, conv_filters, False, key=subkey2),
        ]

    def __call__(self: Self, x: geom.BatchLayer):
        for layer in self.net:
            x = layer(x)

        return x


def map_and_loss(
    model: eqx.Module,
    layer_x: geom.BatchLayer,
    layer_y: geom.BatchLayer,
    aux_data: Optional[eqx.nn.State] = None,
) -> float:
    """
    Given an input batch layer x and a target batch layer y, apply the neural network to the input
    layer and then calculate the mean squared error loss with the target batch layer y. The first
    5 arguments are what is expected in the ml.train function, but we don't use key and train.

    args:
        model: The model to apply on our input data
        layer_x: the input data to the network
        layer_y: the target data for the network
        aux_data: The train function expects to pass an aux_data. This is used for models with
            batch norm. In this case it will be ignored.

    Returns:
        The loss value
    """
    return ml.smse_loss(layer_y, model(layer_x)), aux_data


def target_function(
    layer: geom.BatchLayer, conv_filter_a: jax.Array, conv_filter_b: jax.Array
) -> geom.BatchLayer:
    """
    Target function that applies two convolutions in sequence

    args:
        image: layer input
        conv_filter_a: first convolution filter
        conv_filter_b: second convolution filter

    Returns:
        The layer after convolving twice
    """
    convolved_data = geom.convolve(
        layer.D,
        geom.convolve(layer.D, layer[(0, 0)], conv_filter_a[None, None], layer.is_torus),
        conv_filter_b[None, None],
        layer.is_torus,
    )
    return geom.BatchLayer({(0, 0): convolved_data}, layer.D, layer.is_torus)


# Main
key = random.PRNGKey(time.time_ns())

D = 2
N = 64  # image size
M = 3  # filter image size
num_images = 10

group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    Ms=[M], ks=[0], parities=[0], D=D, operators=group_actions
)

key, subkey = random.split(key)
layer_X = geom.BatchLayer({(0, 0): random.normal(subkey, shape=(num_images, 1) + (N,) * D)}, D)
layer_y = target_function(layer_X, conv_filters[(0, 0)][1], conv_filters[(0, 0)][2])

key, subkey = random.split(key)
model = SimpleModel(D, layer_X.get_signature(), layer_y.get_signature(), conv_filters, subkey)

key, subkey = random.split(key)
trained_model, _, _, _ = ml.train(
    layer_X,
    layer_y,
    map_and_loss,
    model,
    subkey,
    ml.EpochStop(500, verbose=1),
    num_images,
    optimizer=optax.adam(optax.exponential_decay(0.1, transition_steps=1, decay_rate=0.99)),
)

print(trained_model.net[0].weights)
print(trained_model.net[1].weights)
