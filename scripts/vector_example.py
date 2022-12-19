import numpy as np
import time
import argparse
import sys

import jax.numpy as jnp
from jax import value_and_grad, random
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml

def net(params, x, conv_filters, return_params=False):
    conv_layer_out = ml.conv_layer(x, conv_filters)
    linear_layer_out = ml.linear_layer(conv_layer_out)
    quad_layer_out = ml.quadratic_layer(conv_layer_out)

    final_layer_out, param_idx = ml.final_layer(params, 0, x, conv_filters, linear_layer_out + quad_layer_out)
    contracted_images, param_idx = ml.cascading_contractions(params, param_idx, x, final_layer_out)
    net_output = geom.linear_combination(contracted_images, params[param_idx:(param_idx + len(contracted_images))])

    return (net_output, param_idx + len(contracted_images)) if return_params else net_output

def loss(params, x, y, conv_filters):
    # Run the neural network, then calculate the MSE loss with the expected output y
    out = net(params, x, conv_filters)

    # calculate the mean squared error difference between y and out
    mse = jnp.sqrt(jnp.sum((y.data - out.data) ** 2) / out.data.size)
    return mse

def batch_loss(params, X, Y, conv_filters):
    # Loss function for batches, should be a way to vmap this.
    return jnp.mean(jnp.array([loss(params, x, y, conv_filters) for x,y in zip(X,Y)]))

def train(X, Y, conv_filters, batch_loss, params, epochs, learning_rate=0.1):
    # Train the model. Use a simple adaptive learning rate scheme, and go until the learning rate is small.
    batch_loss_grad = value_and_grad(batch_loss)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    for i in range(epochs):
        loss_val, grads = batch_loss_grad(params, X, Y, conv_filters)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if (i % (epochs // np.min([10,epochs])) == 0):
            print(f'Epoch {i}: {loss_val}')

    return params

def target_function(x, conv_filters):
    prod = x.convolve_with(conv_filters[1]) * x.convolve_with(conv_filters[2])
    return (prod).convolve_with(conv_filters[3]).contract(0,1)

def handleArgs(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('-epochs', help='number of epochs to run', type=int, default=10)
  parser.add_argument('-lr', help='learning rate', type=float, default=0.1)
  parser.add_argument('-num_images', help='number of images', type=int, default=3)

  args = parser.parse_args()

  return args.epochs, args.lr, args.num_images

# Main
epochs, learning_rate, num_images = handleArgs(sys.argv)

key = random.PRNGKey(time.time_ns())

D = 2
N = 64 #image size
k = 1
M = 3  #filter image size

group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    Ms=[M],
    ks=[0,1,2],
    parities=[0,1],
    D=D,
    operators=group_actions,
    return_list=True,
)

# construct num_images scalar images, (N,N)
key, subkey = random.split(key)
X = [geom.GeometricImage(data, 0, D) for data in random.normal(subkey, shape=((num_images,) + (N,)*D + (D,)*k))]
Y = [target_function(x, conv_filters) for x in X]

key, subkey = random.split(key)

huge_params = jnp.ones(ml.param_count(X[0], conv_filters, 2))
_, param_idx = net(huge_params, X[0], conv_filters, return_params=True)

params = random.normal(subkey, shape=(param_idx,))
params = train(
    X,
    Y,
    conv_filters,
    batch_loss,
    params,
    epochs=epochs,
    learning_rate=learning_rate,
)

print('train loss:', batch_loss(params, X, Y, conv_filters))

key, subkey = random.split(key)
X_test = [geom.GeometricImage(data, 0, D) for data in random.normal(subkey, shape=((num_images,) + (N,)*D + (D,)*k))]
Y_test = [target_function(x, conv_filters) for x in X_test]

print('test loss:', batch_loss(params, X_test, Y_test, conv_filters))

