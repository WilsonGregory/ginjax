import numpy as np
import jax.numpy as jnp
from jax import value_and_grad, jit, random, vmap
import geometricconvolutions.geometric as geom
import time

def conv_layer(x, conv_filters):
    # Given an input, a list of convolutional filters
    return [x.convolve_with(conv_filter) for conv_filter in conv_filters]

def net(params, x, conv_filters):
    # A simple neural net with 1 convolution layer. Then return a linear combination of the convolution outputs
    first_layer_out = conv_layer(x, conv_filters)

    second_layer_out = []
    for i in range(len(first_layer_out)):
        second_layer_out.extend(conv_layer(first_layer_out[i], conv_filters))

    return geom.GeometricImage.linear_combination(second_layer_out, params)

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

def train(X, Y, conv_filters, batch_loss, epochs, params, initial_lr=0.1, decay=0.005, min_lr=0.001):
    # Train the model. Use a simple adaptive learning rate scheme, and go until the learning rate is small.
    batch_loss_grad = value_and_grad(batch_loss)

    learning_rate = initial_lr

    for i in range(epochs):
        loss_val, grads = batch_loss_grad(params, X, Y, conv_filters)
        params = params - learning_rate * grads
        if ((min_lr is None) or (learning_rate > min_lr)):
            learning_rate = initial_lr * np.exp(-1*decay*i)

        if (i % (epochs // 10) == 0):
            print(loss_val, learning_rate)

    return params, loss_val


#Main
key = random.PRNGKey(0)

D = 2
N = 64 #image size
M = 3  #filter image size
num_images = 1

group_actions = geom.make_all_operators(D)
# start with basic 3x3 scalar filters
conv_filters = geom.get_unique_invariant_filters(M=M, k=0, parity=0, D=D, operators=group_actions)

key, subkey = random.split(key)
X = [geom.GeometricImage(data, 0, D) for data in random.normal(subkey, shape=(num_images, N, N))]
Y = [x.convolve_with(conv_filters[2]).convolve_with(conv_filters[1]) for x in X]

key, subkey = random.split(key)
params = random.normal(subkey, shape=(len(conv_filters)**2,))

params, loss_val = train(X, Y, conv_filters, batch_loss=batch_loss, epochs=2000, params = params)

print(batch_loss(params, X, Y, conv_filters))
print(params)

