# generate gravitational field
from __future__ import annotations
import sys
import argparse
import time
from typing import Callable, Optional
from typing_extensions import Self

import jax.numpy as jnp
import jax.random as random
import jax
from jax.typing import ArrayLike
import equinox as eqx

import ginet.geometric as geom
import ginet.ml as ml
import ginet.models as models


class MLP(eqx.Module):
    layers: list[eqx.Module]

    def __init__(
        self: Self,
        depth: int,
        num_layers: int = 1,
        activation_f: Optional[Callable] = None,
        key: Optional[ArrayLike] = None,
    ) -> Self:
        D = conv_filters.D

        self.layers = []
        for i in range(num_layers):
            key, subkey = random.split(key)
            self.layers.append(eqx.nn.Linear(depth, depth, use_bias=False, key=subkey))
            if activation_f and (i < (num_layers - 1)):
                self.layers.append(activation_f)

    def __call__(self: Self, x: ArrayLike) -> jax.Array:
        print("in:", jnp.var(x), jnp.mean(x))
        for i, layer in enumerate(self.layers):
            x = layer(x)
            print(f"{i}: ", jnp.var(x), jnp.mean(x))

        return x


class VanillaModel(eqx.Module):
    layers: list[eqx.Module]

    def __init__(
        self: Self,
        depth: int,
        kernel_size: int = 3,
        num_layers: int = 1,
        activation_f: Optional[Callable] = None,
        key: Optional[ArrayLike] = None,
    ) -> Self:
        D = conv_filters.D

        self.layers = []
        for i in range(num_layers):
            key, subkey = random.split(key)
            self.layers.append(
                ("conv", eqx.nn.Conv2d(depth, depth, kernel_size, use_bias=False, key=subkey))
            )
            if activation_f and (i < (num_layers - 1)):
                self.layers.append(("activation", activation_f))

    def __call__(self: Self, x: ArrayLike) -> jax.Array:
        print("in:", jnp.var(x), jnp.mean(x))
        print(jnp.cov(x[:, :3, :3].reshape((-1, 9)), rowvar=False).reshape((9, 3, 3)))
        for name, layer in self.layers:
            x = layer(x)
            print(f"{name}: ", jnp.var(x), jnp.mean(x))
            print(jnp.cov(x[:, :3, :3].reshape((-1, 9)), rowvar=False).reshape((9, 3, 3)))

        return x


class Model(eqx.Module):
    layers: list[ml.ConvContract]
    D: int

    def __init__(
        self: Self,
        input_keys: geom.Signature,
        target_keys: geom.Signature,
        conv_filters: geom.MultiImage,
        num_layers: int = 1,
        activation_f: Optional[Callable] = None,
        key: Optional[ArrayLike] = None,
    ) -> Self:
        self.D = conv_filters.D

        self.layers = []
        for i in range(num_layers):
            if i == 0:
                input_keys = target_keys

            key, subkey1, subkey2 = random.split(key, num=3)
            self.layers.append(
                (
                    "conv_contract",
                    ml.ConvContract(
                        input_keys, target_keys, conv_filters, use_bias=False, key=subkey1
                    ),
                )
            )
            if activation_f:
                self.layers.append(
                    (
                        "activation",
                        ml.VectorNeuronNonlinear(target_keys, self.D, activation_f, key=subkey2),
                    )
                )

    def __call__(self: Self, x: geom.MultiImage) -> geom.MultiImage:
        print("in:", jnp.var(x[(0, 0)]), jnp.mean(x[(0, 0)]))
        # print(jnp.cov(x[(0, 0)][:, :3, :3].reshape((-1, 9)), rowvar=False).reshape((9, 3, 3)))
        for name, layer in self.layers:
            x = layer(x)
            mean, var = jnp.mean(x[(0, 0)]), jnp.var(x[(0, 0)])
            print(f"{name}: ", var, mean)
            # print(jnp.cov(x[(0, 0)][:, :3, :3].reshape((-1, 9)), rowvar=False).reshape((9, 3, 3)))

        return x


def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="the random number seed", type=int, default=None)
    return parser.parse_args()


# Main
args = handleArgs(sys.argv)

N = 128
D = 2
in_c = 100
n_steps = 5
activation_f = jax.nn.relu

key = random.PRNGKey(args.seed if args.seed else time.time_ns())
key, subkey = random.split(key)

data = random.normal(subkey, shape=(in_c,) + (N,) * D)

input_x = geom.MultiImage({(0, 0): data}, D, is_torus=False)
target_keys = (((0, 0), in_c),)

# start with basic 3x3 scalar, vector, and 2nd order tensor images
group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    Ms=[3], ks=[0, 1, 2], parities=[0], D=D, operators=group_actions
)
print(conv_filters[(0, 0)][0])
print(conv_filters[(0, 0)][1])
print(conv_filters[(0, 0)][2])

key, subkey = random.split(key)
model = Model(
    input_x.get_signature(),
    target_keys,
    conv_filters,
    num_layers=10,
    activation_f=activation_f,
    key=subkey,
)
print(models.count_params(model))

output_x = model(input_x)
exit()

kernel_size = 3
key, subkey1, subkey2 = random.split(key, num=3)
data = random.normal(key=subkey1, shape=(in_c,) + (N,) * D)
data = data - jnp.mean(data)
data = data / jnp.std(data)
vanilla_model = VanillaModel(
    in_c, kernel_size=kernel_size, num_layers=5, activation_f=activation_f, key=subkey2
)

is_conv = lambda l: isinstance(l, eqx.nn.Conv2d)
get_weights = lambda m: [
    x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_conv) if is_conv(x)
]
weights = get_weights(vanilla_model)
bound = jnp.sqrt((3) / (in_c * (kernel_size**2)))

new_weights = [
    random.uniform(subkey, shape=weight.shape, minval=-bound, maxval=bound)
    for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
]
vanilla_model = eqx.tree_at(get_weights, vanilla_model, new_weights)

# output_x = vanilla_model(data)


key, subkey1, subkey2 = random.split(key, num=3)
data = random.normal(key=subkey1, shape=(in_c,))
data = data - jnp.mean(data)
data = data / jnp.std(data)
mlp = MLP(in_c, num_layers=5, key=subkey2)

is_linear = lambda l: isinstance(l, eqx.nn.Linear)
get_weights = lambda m: [
    x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)
]
weights = get_weights(mlp)
bound = jnp.sqrt((3) / in_c)
new_weights = [
    random.uniform(subkey, shape=weight.shape, minval=-bound, maxval=bound)
    for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
]
vanilla_model = eqx.tree_at(get_weights, mlp, new_weights)

# output_x = mlp(data)
