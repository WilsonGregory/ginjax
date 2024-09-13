import sys
from functools import partial
import argparse
import time
import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax.random as random
import jax
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml
import geometricconvolutions.utils as utils


def gen_data(N, num_imgs, key):
    key, subkey = random.split(key)
    coeffs = random.normal(subkey, shape=(num_imgs, 10, 1, 1)) * 0.4
    scalar_imgs, grad_imgs = jax.vmap(gen_scalar_gradient_imgs, in_axes=(None, 0))(N, coeffs)

    return (
        geom.BatchLayer({(0, 0): jnp.expand_dims(scalar_imgs, axis=1)}, 2, is_torus=False),
        geom.BatchLayer({(1, 0): jnp.expand_dims(grad_imgs, axis=1)}, 2, is_torus=False),
    )


def gen_scalar_gradient_imgs(N, coeffs):
    """
    Generate scalar and gradient images from the provided coefficients for a 3rd degree polynomial basis
    args:
        N (int): side length of the images
        coeffs (int): length 10 vector of the coefficients for the 3rd degree polynomial basis
    """
    # Dimension D=2
    mesh = jnp.meshgrid(jnp.linspace(-1.2, 1.2, N), jnp.linspace(-1.2, 1.2, N), indexing="ij")
    x = mesh[0]
    y = mesh[1]
    ones = jnp.ones(x.shape)
    x2 = x**2
    xy = x * y
    y2 = y**2
    x3 = x**3
    x2y = (x**2) * y
    xy2 = x * (y**2)
    y3 = y**3

    library = jnp.stack([ones, x, y, x2, xy, y2, x3, x2y, xy2, y3])

    scalar_img = jnp.sum((coeffs * library), axis=0)
    assert scalar_img.shape == (N, N)

    # dx is 0, 1, 0, 2x, y, 0, 3x2, 2xy, y2, 0
    # dy is 0, 0, 1, 0, x, 2y, 0, x2, x 2y, 3y2
    zeros = jnp.zeros(x.shape)
    dx_library = jnp.stack([zeros, ones, zeros, 2 * x, y, zeros, 3 * x2, 2 * xy, y2, zeros])
    dy_library = jnp.stack([zeros, zeros, ones, zeros, x, 2 * y, zeros, x2, 2 * xy, 3 * y2])

    dx_sum = jnp.sum((coeffs * dx_library), axis=0)
    dy_sum = jnp.sum((coeffs * dy_library), axis=0)
    gradient_img = jnp.stack([dx_sum, dy_sum], axis=2)
    assert gradient_img.shape == (N, N, 2)

    return scalar_img, gradient_img


@partial(jax.jit, static_argnums=[3, 5])
def div_net(params, layer, key, train, conv_filters, return_params=False):
    target_key = (1, 0)
    num_conv_layers = 2

    for i in range(num_conv_layers):
        layer, params = ml.batch_conv_layer(
            params,
            layer,
            {"type": "fixed", "filters": conv_filters},
            depth=1,
            target_key=target_key if (i + 1 == num_conv_layers) else None,
            mold_params=return_params,
        )
        layer = ml.batch_relu_layer(layer)

    layer = ml.batch_all_contractions(target_key[0], layer)
    layer, params = ml.batch_channel_collapse(params, layer, mold_params=return_params)

    return (layer, params) if return_params else layer


def map_and_loss(params, layer_x, layer_y, key, train, net_f):
    recon = net_f(params, layer_x, key, train)

    # We only calculate the loss on the inner section because gradient equations are defined outside
    # the grid, but the convolution can only use information inside the grid, so there are more errors
    # around the edge.
    y_img = layer_y[(1, 0)][:, :, 2:-2, 2:-2]
    recon_img = recon[(1, 0)][:, :, 2:-2, 2:-2]
    assert y_img.shape == recon_img.shape
    return jnp.mean(
        jax.vmap(ml.l2_loss)(recon_img, y_img)
    )  # calculate loss, then average over batch


def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", help="where to save the image", type=str)
    parser.add_argument("-e", "--epochs", help="number of epochs", type=float, default=10)
    parser.add_argument("-lr", help="learning rate", type=float, default=1e-2)
    parser.add_argument("-batch", help="batch size", type=int, default=10)
    parser.add_argument("-seed", help="the random number seed", type=int, default=None)
    parser.add_argument(
        "-s", "--save", help="folder name to save the results", type=str, default=None
    )
    parser.add_argument(
        "-l", "--load", help="folder name to load results from", type=str, default=None
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="levels of print statements during training",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    return (
        args.outfile,
        args.epochs,
        args.lr,
        args.batch,
        args.seed,
        args.save,
        args.load,
        args.verbose,
    )


# Main
outfile, epochs, lr, batch_size, seed, save_folder, load_folder, verbose = handleArgs(sys.argv)

D = 2
N = 19
M = 3
num_images = 20

key = random.PRNGKey(time.time_ns() if (seed is None) else seed)

# get the conv_filters.
operators = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[M], ks=[0, 1], parities=[0], D=D, operators=operators)

# Get Training data
key, subkey = random.split(key)
train_scalar_layer, train_grad_layer = gen_data(N, num_images, subkey)

# Get Validation data
key, subkey = random.split(key)
val_scalar_layer, val_grad_layer = gen_data(N, batch_size, subkey)

# Test data
key, subkey = random.split(key)
test_scalar_layer, test_grad_layer = gen_data(N, batch_size, subkey)

# For each model, calculate the size of the parameters and add it to the tuple.
one_point = train_scalar_layer.get_subset(jnp.array([0]))
key, subkey = random.split(key)
div_net_f = partial(div_net, conv_filters=conv_filters)
params = ml.init_params(div_net_f, one_point, subkey)
print(f"Number of params: {ml.count_params(params)}")

key, subkey = random.split(key)
params, train_loss, val_loss = ml.train(
    train_scalar_layer,
    train_grad_layer,
    partial(map_and_loss, net_f=div_net_f),
    params,
    subkey,
    ml.ValLoss(patience=20, verbose=verbose),
    batch_size=batch_size,
    optimizer=optax.adam(
        optax.exponential_decay(
            lr,
            transition_steps=int(train_scalar_layer.get_L() / batch_size),
            decay_rate=0.995,
        )
    ),
    validation_X=val_scalar_layer,
    validation_Y=val_grad_layer,
)
print(f"train_loss: {train_loss[-1]}")
print(f"val_loss: {val_loss[-1]}")

key, subkey = random.split(key)
test_loss = map_and_loss(
    params, test_scalar_layer, test_grad_layer, subkey, train=False, net_f=div_net_f
)
print(f"test loss: {test_loss}")

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(32, 12))

train_x_img = geom.GeometricImage(
    train_scalar_layer[(0, 0)][0, 0, 2:-2, 2:-2], 0, D, is_torus=False
)
train_y_img = geom.GeometricImage(train_grad_layer[(1, 0)][0, 0, 2:-2, 2:-2], 0, D, is_torus=False)
train_recon_img = geom.GeometricImage(
    div_net_f(params, train_scalar_layer.get_subset(jnp.array([0])), subkey, False)[(1, 0)][
        0, 0, 2:-2, 2:-2
    ],
    0,
    D,
    is_torus=False,
)

utils.plot_image(
    train_x_img,
    ax=axs[0, 0],
    vmin=None,
    vmax=None,
    colorbar=True,
    title="Train Scalar Field",
)
utils.plot_image(train_y_img, ax=axs[0, 1], title="Train Ground Truth Grad")
utils.plot_image(train_recon_img, ax=axs[0, 2], title="Train Learned Grad")
utils.plot_image(train_y_img - train_recon_img, ax=axs[0, 3], title="Train Difference")

test_x_img = geom.GeometricImage(test_scalar_layer[(0, 0)][0, 0, 2:-2, 2:-2], 0, D, is_torus=False)
test_y_img = geom.GeometricImage(test_grad_layer[(1, 0)][0, 0, 2:-2, 2:-2], 0, D, is_torus=False)
test_recon_img = geom.GeometricImage(
    div_net_f(params, test_scalar_layer.get_subset(jnp.array([0])), subkey, False)[(1, 0)][
        0, 0, 2:-2, 2:-2
    ],
    0,
    D,
    is_torus=False,
)

utils.plot_image(
    test_x_img,
    ax=axs[1, 0],
    vmin=None,
    vmax=None,
    colorbar=True,
    title="Test Scalar Field",
)
utils.plot_image(test_y_img, ax=axs[1, 1], title="Test Ground Truth Grad")
utils.plot_image(test_recon_img, ax=axs[1, 2], title="Test Learned Grad")
utils.plot_image(test_y_img - test_recon_img, ax=axs[1, 3], title="Test Difference")

plt.savefig(outfile)
