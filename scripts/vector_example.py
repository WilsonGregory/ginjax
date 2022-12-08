import itertools as it
import numpy as np
import jax.numpy as jnp
from jax import value_and_grad, jit, random, vmap
import geometricconvolutions.geometric as geom
import time
import functools

def conv_layer(x, conv_filters):
    """
    For each conv_filter, apply it to the image x and return the list
    args:
        x (GeometricImage): image that we are applying the filters to
        conv_filters (list of GeometricFilter): the conv_filters we are applying to the image
    """
    return [x.convolve_with(conv_filter) for conv_filter in conv_filters]

def make_p_k_dict(images, filters=False):
    images_dict = {
        0: {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
        },
        1: {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
        },
    }

    for image in images:
        if filters:
            images_dict[image.parity % 2][image.k % 2].append(image)
        else:
            if image.k in {0,1}:
                image = image.anticontract(2)

            images_dict[image.parity % 2][image.k].append(image)

    return images_dict

def prod_layer(images, degree):
    """
    For the given degree, apply that many prods in all possible combinations with replacement of the images
    args:
        images (list of GeometricImage): images to prod, these should be a convolved image
        degree (int): degree of the prods
    """
    prods = []
    for idxs in it.combinations_with_replacement(range(len(images)), degree):
        prods.append(functools.reduce(lambda u,v: u * v, [images[idx] for idx in idxs]))

    return prods

@jit
def quadratic_layer(params, x, conv_filters):
    conv_layer_out = conv_layer(x, conv_filters)

    prod_layer_out = prod_layer(conv_layer_out, 2)

    prods_dict = make_p_k_dict(prod_layer_out)
    filters_dict = make_p_k_dict(conv_filters, filters=True)

    last_layer = []
    param_idx = 0
    for parity in [0,1]:
        for k in [0,1,2,3,4,5,6]:
            prods_group = prods_dict[parity][k]
            filter_group = filters_dict[(parity + x.parity) % 2][(k + x.k) % 2]

            if (len(filter_group) == 0 or len(prods_group) == 0):
                continue

            for conv_filter in filter_group:
                group_sum = geom.linear_combination(
                    prods_group,
                    params[param_idx:param_idx+len(prods_group)],
                )
                last_layer.append(group_sum.convolve_with(conv_filter))

                assert (last_layer[-1].k % 2) == (x.k % 2)
                assert (last_layer[-1].parity % 2) == (x.parity % 2)

                param_idx += len(prods_group)

    return last_layer

def net(params, x, conv_filters):
    # A simple neural net with a quadratic convolution layer.
    quad_results = quadratic_layer(params[:220], x, conv_filters)
    params = params[220:]

    quad_results_by_k = {}
    for img in quad_results:
        if img.k in quad_results_by_k:
            quad_results_by_k[img.k].append(img)
        else:
            quad_results_by_k[img.k] = [img]

    descending_k_dict = dict(reversed(sorted(quad_results_by_k.items())))

    param_idx = 0
    final_list = []
    for k in descending_k_dict.keys():
        images = descending_k_dict[k]
        for u,v in it.combinations(range(k), 2):
            group_sum = geom.linear_combination(
                images,
                params[param_idx:param_idx+len(images)],
            )
            contracted_img = group_sum.contract(u,v)
            # print(contracted_img.k)
            if contracted_img.k == x.k: #done contracting, add to the the final list
                final_list.append(contracted_img)
            else: #add the the next layer
                descending_k_dict[contracted_img.k].append(contracted_img)

            param_idx += len(images)

    return geom.linear_combination(final_list, params[param_idx:param_idx+len(final_list)])

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

def target_function(x, conv_filters):
    prod = x.convolve_with(conv_filters[2]) * x.convolve_with(conv_filters[1])
    return (prod).convolve_with(conv_filters[3]).contract(0,1)

#Main
key = random.PRNGKey(time.time_ns())

D = 2
N = 64 #image size
k = 1
M = 3  #filter image size
num_images = 3

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
params = random.normal(subkey, shape=(220 + 564 + 3,)) #for k=1

params, loss_val = train(
    X,
    Y,
    conv_filters,
    batch_loss=batch_loss,
    epochs=40,
    params=params,
    initial_lr=0.001,
    decay=0.1,
)

print('train loss:', batch_loss(params, X, Y, conv_filters))

key, subkey = random.split(key)
X_test = [geom.GeometricImage(data, 0, D) for data in random.normal(subkey, shape=((num_images,) + (N,)*D + (D,)*k))]
Y_test = [target_function(x, conv_filters) for x in X_test]

print('test loss:', batch_loss(params, X_test, Y_test, conv_filters))

