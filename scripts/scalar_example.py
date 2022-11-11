import numpy as np
import jax.numpy as jnp
from jax import value_and_grad, jit, random
import geometricconvolutions.geometric as geom

def initial_params(num_params):
    key = random.PRNGKey(0)
    return random.normal(key, shape=(num_params,))

def conv_layer(params, x, conv_filters):
    # Given params, an input, a list of convolutional filters
    assert len(params) == len(conv_filters) #for right now, a single conv layer
    return [x.convolve_with(conv_filter).times_scalar(param) for param, conv_filter in zip(params, conv_filters)]

def net(params, x, conv_filters):
    # A simple neural net with 1 convolution layer. Then return a linear combination of the convolution outputs
    data = jnp.sum(jnp.array([out.data for out in conv_layer(params, x, conv_filters)]), axis=0)
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


def train(X, Y, conv_filters, batch_loss, epochs):
    # Train the model. Use a simple adaptive learning rate scheme, and go until the learning rate is small.
    params = initial_params(len(conv_filters))
    batch_loss_grad = jit(value_and_grad(batch_loss))

    learning_rate = 0.2

    loss_val = 1
    prev_loss_val = 100000
    while learning_rate > 0.0001:
        if 'grads' in locals():
            new_params = [param - learning_rate * grad for param, grad in zip(params, grads)]
        else:
            new_params = params

        loss_val, grads = batch_loss_grad(new_params, X, Y, conv_filters)

        # Update parameters via gradient descent
        if loss_val < prev_loss_val:
            print(loss_val, learning_rate)
            params = new_params
            learning_rate = 1.5 * learning_rate
            prev_loss_val = loss_val
        else:
            learning_rate = 0.5 * learning_rate

    return params, loss_val


#Main
D = 2

group_actions = geom.make_all_operators(D)
# start with basic 3x3 scalar filters
conv_filters = geom.get_unique_invariant_filters(M=3, k=0, parity=0, D=D, operators=group_actions)

X = [geom.GeometricImage(np.random.randn(64,64), 0, 2) for x in range(25)]
Y = [x.convolve_with(conv_filters[2]) for x in X]


params, loss_val = train(X, Y, conv_filters, batch_loss=batch_loss, epochs=10)

print(batch_loss(params, X, Y, conv_filters))
print(params)

