import numpy as np
import jax.numpy as jnp
from jax import value_and_grad, jit, random
import geometricconvolutions.geometric as geom
import time
import itertools as it
import math
import optax

def net(params, x, conv_filters):
    # A simple neural net with 1 convolution layer. Then return a linear combination of the convolution outputs
    layer_out = []
    for c1_idx, c2_idx in it.combinations_with_replacement(range(len(conv_filters)), 2):
        c1 = conv_filters[c1_idx]
        c2 = conv_filters[c2_idx]

        layer_out.append(x.convolve_with(c1).convolve_with(c2))

    return geom.linear_combination(layer_out, params)

@jit
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
    return x.convolve_with(conv_filters[1]).convolve_with(conv_filters[2])

#Main
key = random.PRNGKey(time.time_ns())

D = 2
N = 64 #image size
M = 3  #filter image size
num_images = 10

group_actions = geom.make_all_operators(D)
# start with basic 3x3 scalar filters
conv_filters = geom.get_unique_invariant_filters(M=M, k=0, parity=0, D=D, operators=group_actions)

key, subkey = random.split(key)
X = [geom.GeometricImage(data, 0, D) for data in random.normal(subkey, shape=(num_images, N, N))]
Y = [target_function(x, conv_filters) for x in X]

key, subkey = random.split(key)
params = random.normal(subkey, shape=(len(conv_filters) + math.comb(len(conv_filters), 2),))

params = train(X, Y, conv_filters, batch_loss, params, epochs=1000)

print(batch_loss(params, X, Y, conv_filters))
print(params)

