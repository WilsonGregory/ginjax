import itertools as it
import numpy as np
import jax.numpy as jnp
from jax import value_and_grad, jit, random, vmap
import geometricconvolutions.geometric as geom

def conv_layer(params, x, conv_filters):
    # Given params, an input, a list of convolutional filters
    assert len(params) == len(conv_filters) #for right now, a single conv layer
    return [x.convolve_with(conv_filter).times_scalar(param) for param, conv_filter in zip(params, conv_filters)]

@jit
def quadratic_layer(x, conv_filters):
    first_layer_out = conv_layer(np.ones(len(conv_filters)), x, conv_filters)

    prods = []
    for c1_idx, c2_idx in it.combinations(range(len(conv_filters)), 2):
        c1x = first_layer_out[c1_idx]
        c2x = first_layer_out[c2_idx]
        prod = c1x * c2x
        for conv_filter in conv_filters:
            if (
                ((prod.k + conv_filter.k - x.k) % 2 == 0) and
                ((prod.parity + conv_filter.parity - x.parity) % 2 == 0)
            ):
                #compatible filter
                prods.append(prod.convolve_with(conv_filter))

    return prods

def net(params, x, conv_filters):
    # A simple neural net with a quadratic convolution layer.
    prods = quadratic_layer(x, conv_filters)

    prods_by_k = { img.k: [] for img in prods }
    for img in prods:
        prods_by_k[img.k].append(img)

    i = 0
    final_list = []
    for lst in prods_by_k.values():
        sum_image_data = jnp.sum(jnp.array([out.data*param for out,param in zip(lst, params[i:i+len(lst)])]), axis=0)
        img = geom.GeometricImage(sum_image_data, x.parity, x.D)

        if img.k == 0:
            final_list.append(img)
        else:
            for idxs in geom.get_contraction_indices(img.k, x.k):
                final_list.append(img.multicontract(idxs))

        i += len(lst)

    data = jnp.sum(jnp.array([out.data*param for out,param in zip(final_list, params[i:i+len(final_list)])]), axis=0)
    return geom.GeometricImage(data, x.parity, x.D)

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

def get_batch(X, Y, batch_size):
    batch_indices = np.random.permutation(len(X))[:batch_size]
    return [X[idx] for idx in batch_indices], [Y[idx] for idx in batch_indices]

def train(X, Y, conv_filters, batch_loss, epochs, params, batch_size=16, initial_lr=0.01, decay=0.01, min_lr=None):
    # Train the model. Use a simple adaptive learning rate scheme, and go until the learning rate is small.
    batch_loss_grad = value_and_grad(batch_loss)

    learning_rate = initial_lr
    for i in range(epochs):
        X_batch, Y_batch = get_batch(X, Y, batch_size)
        loss_val, grads = batch_loss_grad(params, X_batch, Y_batch, conv_filters)
        params = params - learning_rate * grads
        if ((min_lr is None) or (learning_rate > min_lr)):
            learning_rate = initial_lr * jnp.exp(-1*decay*i)

        if (i % (epochs // np.min([10,epochs])) == 0):
            print(loss_val, learning_rate)

    return params, loss_val

#Main
key = random.PRNGKey(0)

D = 2
N = 64 #image size
k = 1
M = 3  #filter image size
num_images = 1

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
Y = [x.convolve_with(conv_filters[2]).convolve_with(conv_filters[1]) for x in X]

key, subkey = random.split(key)
params = random.normal(subkey, shape=(196+123,)) #for k=1

params, loss_val = train(X, Y, conv_filters, batch_loss=batch_loss, epochs=10, params=params)

print('train loss:', batch_loss(params, X, Y, conv_filters))

key, subkey = random.split(key)
X_test = [geom.GeometricImage(data, 0, D) for data in random.normal(subkey, shape=((num_images,) + (N,)*D + (D,)*k))]
Y_test = [x.convolve_with(conv_filters[2]).convolve_with(conv_filters[1]) for x in X_test]

print('test loss:', batch_loss(params, X_test, Y_test, conv_filters))

