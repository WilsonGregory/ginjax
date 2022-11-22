import numpy as np
import jax.numpy as jnp
from jax import value_and_grad, jit, random, vmap
import geometricconvolutions.geometric as geom

def initial_params(key, num_params):
    return random.normal(key, shape=(num_params,))

def conv_layer(params, x, conv_filters):
    # Given params, an input, a list of convolutional filters
    assert len(params) == len(conv_filters) #for right now, a single conv layer
    return [x.convolve_with(conv_filter).times_scalar(param) for param, conv_filter in zip(params, conv_filters)]

def net(params, x, conv_filters):
    # A simple neural net with 1 convolution layer. Then return a linear combination of the convolution outputs
    first_layer_out = conv_layer(params[:len(conv_filters)], x, conv_filters)

    second_layer_out = []
    for i in range(len(first_layer_out)):
        second_layer_out.extend(conv_layer(
            params[(i+1)*len(conv_filters):(i+2)*len(conv_filters)],
            first_layer_out[i],
            conv_filters,
        ))

    data = jnp.sum(jnp.array([out.data for out in second_layer_out]), axis=0)
    return geom.GeometricImage(data, x.parity, x.D)

def loss(params, x, y, conv_filters):
    # Run the neural network, then calculate the MSE loss with the expected output y
    out = net(params, x, conv_filters)

    # calculate the mean squared error difference between y and out
    mse = jnp.sqrt(jnp.sum((y.data - out.data) ** 2) / out.data.size)
    return mse

def batch_loss(params, X, Y, conv_filters):
    # Loss function for batches, should be a way to vmap this.
    lst = []
    res = jnp.array([])
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        res = jnp.append(res, loss(params, x, y, conv_filters))

    return jnp.mean(res)

def train(X, Y, conv_filters, batch_loss, epochs, params):
    # Train the model. Use a simple adaptive learning rate scheme, and go until the learning rate is small.
    batch_loss_grad = jit(value_and_grad(batch_loss))

    initial_lr = 0.1
    learning_rate = initial_lr
    decay = 0.01
    min_lr = 0.001

    for i in range(epochs):
        loss_val, grads = batch_loss_grad(params, X, Y, conv_filters)
        params = [param - learning_rate * grad for param, grad in zip(params, grads)]
        if (learning_rate > min_lr):
            learning_rate = initial_lr * np.exp(-1*decay*i)

        if (i % (epochs // 10) == 0):
            print(loss_val, learning_rate)

    return params, loss_val


#Main
key = random.PRNGKey(0)

D = 2
N = 64 #image size
M = 3  #filter image size
num_images = 5

group_actions = geom.make_all_operators(D)
# start with basic 3x3 scalar filters
conv_filters = geom.get_unique_invariant_filters(M=M, k=0, parity=0, D=D, operators=group_actions)

key, subkey = random.split(key)
X = [geom.GeometricImage(data, 0, D) for data in random.normal(subkey, shape=(num_images, N, N))]
Y = [x.convolve_with(conv_filters[2]).convolve_with(conv_filters[1]) for x in X]

key, subkey = random.split(key)
params = random.normal(subkey, shape=(len(conv_filters) + len(conv_filters)**2,))

params, loss_val = train(X, Y, conv_filters, batch_loss=batch_loss, epochs=2000, params = params)

print(batch_loss(params, X, Y, conv_filters))
print(params)

