#generate gravitational field
import os
import numpy as np
import sys
from functools import partial
import argparse
import time
from scipy.special import binom

import jax.numpy as jnp
import jax.random as random
import jax.nn
from jax import vmap, jit
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml

# Copied from phase2vec

def load_dataset(data_path):
    # Load data
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))

    # Load labels
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))

    # Load pars
    p_train = np.load(os.path.join(data_path, 'p_train.npy'))
    p_test = np.load(os.path.join(data_path, 'p_test.npy'))
    return X_train, X_test, y_train, y_test, p_train, p_test

def library_size(dim, poly_order, include_sine=False, include_exp=False, include_constant=True):
    """
    Calculate library size for a given polynomial order. Taken from `https://github.com/kpchamp/SindyAutoencoders`

    """
    l = 0
    for k in range(poly_order+1):
        l += int(binom(dim+k-1,k))
    if include_sine:
        l += dim
    if include_exp:
        l+= dim
    if not include_constant:
        l -= 1
    return l

def sindy_library(X, poly_order, include_sine=False, include_exp=False):
    """
    Generate a library of polynomials of order poly_order. Taken from `https://github.com/kpchamp/SindyAutoencoders`
    """
    m,n = X.shape # samples x dim
    l = library_size(n, poly_order, include_sine=include_sine, include_exp=include_exp, include_constant=True)
    library = np.ones((m,l), dtype=X.dtype)
    index = 1
    
    library_terms = []

    for i in range(n):
        library[:,index] = X[:,i]
        index += 1
        library_terms.append(r'$x_{}$'.format(i))

    if poly_order > 1:
        for i in range(n):
            for j in range(i,n):
                library[:,index] = X[:,i]*X[:,j]
                index += 1
                library_terms.append(r'$x_{}x_{}$'.format(i,j))


    if poly_order > 2:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    library[:,index] = X[:,i]*X[:,j]*X[:,k]
                    index += 1
                    library_terms.append(r'$x_{}x_{}x_{}$'.format(i,j,k))

    if poly_order > 3:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]
                        index += 1
                        library_terms.append(r'$x_{}x_{}x_{}x_{}$'.format(i,j,k,q))

                    
    if poly_order > 4:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        for r in range(q,n):
                            library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]*X[:,r]
                            index += 1
                            library_terms.append(r'$x_{}x_{}x_{}x_{}x_{}$'.format(i,j,k,q,r))

    # Consolidate exponents
    for t, term in enumerate(library_terms):
        if 'sin' in term or 'e' in term: continue
        all_exps = []
        for i in range(n):
            all_exps.append(term.count(str(i)))
        new_term = ''
        for e, exp in enumerate(all_exps):
            if exp == 0: continue
            if exp == 1:
                new_term += 'x_{}'.format(e)
            else:
                new_term += 'x_{}^{}'.format(e,exp)
        new_term = r'${}$'.format(new_term)
        library_terms[t] = new_term
    # Add constant term
    library_terms.insert(0,'1')
    return library, library_terms

def fully_connected_layer(params, param_idx, x, width):
    weights_shape = (width, len(x))
    weights_size = width * len(x)
    W = params[param_idx:(param_idx + weights_size)].reshape(weights_shape)
    param_idx += weights_size
    return W @ x, param_idx

@partial(jit, static_argnums=1)
def batch_norm_1d(x, eps=1e-05):
    # x is shape (batch, width)
    return (x - jnp.mean(x, axis=0))/jnp.sqrt(jnp.var(x, axis=0) + eps)

@partial(jit, static_argnums=[1,3])
def dropout(x, p, key, train):
    """
    Dropout layer. If training, randomly set values of the layer to 0 at a rate of p
    args:
        x (jnp.array): input vector, shape (batch, width)
        p (float): dropout rate
        key (rand key): source of randomness
        train (bool): whether the network is being trained
    """
    if not train:
        return x 
    
    return x * (random.uniform(key, shape=x.shape) > p).astype(x.dtype)

def net(params, layer, conv_filters, ode_basis, key, train=True, return_params=False):
    # batch norm, dropout seem to make the network widely worse, not sure why.
    embedding_d = 100
    num_coeffs = ode_basis.shape[1]
    img_N = layer.N
    conv_filters = ml.make_layer(conv_filters)

    batch_conv_layer = vmap(
        ml.conv_layer_fixed_filters, 
        in_axes=((None,)*3 + (0,) + (None,)*7),
        out_axes=(0, None),
    )
    batch_relu_layer = vmap(ml.relu_layer)
    batch_fully_connected = vmap(fully_connected_layer, in_axes=(None, None, 0, None), out_axes=(0, None))

    # Convolution on the vector field generated by the ODE
    layer, param_idx = batch_conv_layer(
        params,
        0,
        conv_filters, 
        layer, #this is what is vmapped over
        4, #depth
        None, #target_k
        None, #max_k
        (2,)*layer.D, #stride
        'VALID', #padding
        None, #lhs_dilation
        None, #rhs_dilation
    )
    # layer = ml.batch_norm(layer)
    layer = batch_relu_layer(layer)

    layer, param_idx = batch_conv_layer(
        params,
        int(param_idx),
        conv_filters, 
        layer, 
        4, #depth
        None, #target_k
        None, #max_k
        (2,)*layer.D, #stride
        'VALID', #padding
        None, #lhs_dilation
        None, #rhs_dilation
    )
    # layer = ml.batch_norm(layer)
    layer = batch_relu_layer(layer)

    layer, param_idx = batch_conv_layer(
        params,
        int(param_idx),
        conv_filters, 
        layer, 
        4, #depth
        None, #target_k
        None, #max_k
        (2,)*layer.D, #stride
        'VALID', #padding
        None, #lhs_dilation
        None, #rhs_dilation
    )
    # layer = ml.batch_norm(layer)
    layer = batch_relu_layer(layer)

    # Embed the ODE in a d=100 vector
    embedded_ode = jnp.zeros(embedding_d)
    for image in layer.values():
        vectorized_image = image.reshape(layer.L,-1)
        layer_out, param_idx = batch_fully_connected(params, int(param_idx), vectorized_image, embedding_d)
        embedded_ode += layer_out

    # Pass the embedded ode through a 2 layer MLP to get the coefficients of poly ode
    layer_vec, param_idx = batch_fully_connected(params, int(param_idx), embedded_ode, 128)
    # layer_vec = batch_norm_1d(layer_vec)
    # layer_vec = jax.nn.relu(layer_vec)
    # key, subkey = random.split(key)
    # layer_vec = dropout(layer_vec, 0.1, subkey ,train)

    layer_vec, param_idx = batch_fully_connected(params, int(param_idx), layer_vec, 128)
    # layer_vec = batch_norm_1d(layer_vec)
    # layer_vec = jax.nn.relu(layer_vec)
    # key, subkey = random.split(key)
    # layer_vec = dropout(layer_vec, 0.1, subkey, train)

    coeffs, param_idx = batch_fully_connected(params, int(param_idx), layer_vec, num_coeffs * layer.D)
    coeffs = coeffs.reshape((layer.L, num_coeffs, layer.D))

    # multiply the functions by the coefficients
    vmap_mul = vmap(lambda coeffs_x: ode_basis @ coeffs_x)
    recon_x = vmap_mul(coeffs).reshape((layer.L,) + (img_N,)*layer.D + (layer.D,))
    return (recon_x, coeffs, param_idx) if return_params else (recon_x, coeffs)

def phase2vec_loss(recon_x, layer_y, coeffs, eps=1e-5, beta=1e-3):
    """
    Loss used in the phase2vec paper, combines a pointwise normalized l2 loss with a sparsity l1 loss
    """
    y = layer_y[1][:,0,...]
    m_gt = jnp.linalg.norm(y, axis=layer_y.D+1) #this maybe only works when k=1, its a vector field
    den = jnp.expand_dims(m_gt + eps, axis=layer_y.D+1) #ensure division is broadcast correctly
    # den shape will be (batch, N, N, 1)
    # recon shape will be (batch, N, N, D), since k=1

    #should be sum rather than mean?
    batch_loss = jnp.sqrt(vmap(jnp.mean)(((recon_x - y)**2) / den)) #normalize pointwise by gt norm
    normalized_loss = jnp.mean(batch_loss) #average the loss over the batch

    sparsity_loss = jnp.mean(jnp.abs(coeffs))
    return normalized_loss + beta * sparsity_loss

partial(jit, static_argnums=[4,5,6])
def map_and_loss(params, layer_x, layer_y, conv_filters, ode_basis, key, train, eps=1e-5, beta=1e-3):
    recon, coeffs = net(params, layer_x, conv_filters, ode_basis, key, train=train)
    return phase2vec_loss(recon, layer_y, coeffs, eps, beta)

def baseline_net(params, layer, ode_basis, key, train=True, return_params=False):
    embedding_d = 100
    num_coeffs = ode_basis.shape[1]
    img_N = layer.N
    # for the baseline model, the vector just becomes 2 channels
    layer = geom.BatchLayer({ 0: jnp.moveaxis(layer[1][:,0,...], -1, 1) }, layer.D, layer.is_torus)
    assert layer[0].shape[1:] == (2, img_N, img_N)

    batch_conv_layer = vmap(
        ml.conv_layer_free_filters, 
        in_axes=((None,)*2 + (0,) + (None,)*9),
        out_axes=(0, None),
    )
    batch_relu_layer = vmap(ml.relu_layer)
    batch_fully_connected = vmap(fully_connected_layer, in_axes=(None, None, 0, None), out_axes=(0, None))

    # Convolution on the vector field generated by the ODE
    layer, param_idx = batch_conv_layer(
        params,
        0, # param_idx
        layer, 
        128, # depth
        3, # M
        None,
        None,
        None,
        (2,)*layer.D, #stride
        'VALID', #padding
        None,
        None,
    )
    # layer = ml.batch_norm(layer)
    layer = batch_relu_layer(layer)

    layer, param_idx = batch_conv_layer(
        params,
        int(param_idx),
        layer, 
        128, # depth
        3, # M
        None,
        None,
        None,
        (2,)*layer.D, #stride
        'VALID', #padding
        None,
        None,
    )
    # layer = ml.batch_norm(layer)
    layer = batch_relu_layer(layer)

    layer, param_idx =batch_conv_layer(
        params,
        int(param_idx),
        layer, 
        128, # depth
        3, # M
        None,
        None,
        None,
        (2,)*layer.D, #stride
        'VALID', #padding
        None,
        None,
    )
    # layer = ml.batch_norm(layer)
    layer = batch_relu_layer(layer)
    
    # Embed the ODE in a d=100 vector
    embedded_ode = jnp.zeros(embedding_d)
    for image in layer.values():
        vectorized_image = image.reshape(layer.L,-1)
        layer_out, param_idx = batch_fully_connected(params, int(param_idx), vectorized_image, embedding_d)
        embedded_ode += layer_out

    # Pass the embedded ode through a 2 layer MLP to get the coefficients of poly ode
    layer_vec, param_idx = batch_fully_connected(params, int(param_idx), embedded_ode, 128)
    # layer_vec = batch_norm_1d(layer_vec)
    # layer_vec = jax.nn.relu(layer_vec)
    # key, subkey = random.split(key)
    # layer_vec = dropout(layer_vec, 0.1, subkey, train)

    layer_vec, param_idx = batch_fully_connected(params, int(param_idx), layer_vec, 128)
    # layer_vec = batch_norm_1d(layer_vec)
    # layer_vec = jax.nn.relu(layer_vec)
    # key, subkey = random.split(key)
    # layer_vec = dropout(layer_vec, 0.1, subkey, train)

    coeffs, param_idx = batch_fully_connected(params, int(param_idx), layer_vec, num_coeffs * layer.D)
    coeffs = coeffs.reshape((layer.L, num_coeffs, layer.D))

    # multiply the functions by the coefficients
    vmap_mul = vmap(lambda coeffs_x: ode_basis @ coeffs_x)
    recon_x = vmap_mul(coeffs).reshape((layer.L,) + (img_N,)*layer.D + (layer.D,))
    return (recon_x, coeffs, param_idx) if return_params else (recon_x, coeffs)

partial(jit, static_argnums=[4,5,6])
def baseline_map_and_loss(params, layer_x, layer_y, ode_basis, key, train, eps=1e-5, beta=1e-3):
    recon, coeffs = baseline_net(params, layer_x, ode_basis, key, train=train)
    return phase2vec_loss(recon, layer_y, coeffs, eps, beta)

def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', help='number of epochs', type=float, default=200)
    parser.add_argument('-lr', help='learning rate', type=float, default=0.1)
    parser.add_argument('-batch', help='batch size', type=int, default=1)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-s', '--save', help='file name to save the results', type=str, default=None)
    parser.add_argument('-l', '--load', help='file name to load results from', type=str, default=None)
    parser.add_argument('-v', '--verbose', help='levels of print statements during training', type=int, default=1)

    args = parser.parse_args()

    return (
        args.epochs,
        args.lr,
        args.batch,
        args.seed,
        args.save,
        args.load,
        args.verbose,
    )

# Main
epochs, lr, batch_size, seed, save_file, load_file, verbose = handleArgs(sys.argv)

D = 2

key = random.PRNGKey(seed if seed else time.time_ns())

# start with basic 3x3 scalar, vector, and 2nd order tensor images
conv_filters = geom.get_invariant_filters(
    Ms=[3],
    ks=[0,1],
    parities=[0],
    D=D,
    operators=geom.make_all_operators(D),
    return_list=True,
)

train_data_path = '../phase2vec/output/data/polynomial'
X_train_data, X_val_data, y_train, y_val, p_train, p_val = load_dataset(train_data_path)
X_train = geom.BatchLayer(
    { 1: jnp.expand_dims(X_train_data.transpose(0,2,3,1), axis=1) }, #(batch, channel, (N,)*D, (D,)*k)
    D,
    False,
)
X_val = geom.BatchLayer(
    { 1: jnp.expand_dims(X_train_data.transpose(0,2,3,1), axis=1) },
    D,
    False,
)

# generate function library
spatial_coords = [jnp.linspace(mn, mx, X_train.N) for (mn, mx) in zip([-1.,-1.], [1.,1.])]
mesh = jnp.meshgrid(*spatial_coords)
L = jnp.concatenate([ms[..., None] for ms in mesh], axis=-1)
ode_basis, _ = sindy_library(L.reshape(X_train.N**D,D), 3)

#its only a single point, but net expects a BatchLayer so that is what we do.
one_point = geom.BatchLayer(
    { 1: jnp.expand_dims(X_train[1][0], axis=0) }, 
    D,
    False,
)

huge_params = jnp.ones(100000000) #hundred million
_, _, num_params_model = net(
    huge_params,
    one_point,
    conv_filters,
    ode_basis,
    key,
    train=True,
    return_params=True,
)
print(f'Model Params: {num_params_model}')

_, _, num_params_baseline = baseline_net(
    huge_params,
    one_point,
    ode_basis,
    key,
    train=True,
    return_params=True,
)
print(f'Baseline Params {num_params_baseline}')

models = [
    (
        'GI-Net',
        partial(map_and_loss, conv_filters=conv_filters, ode_basis=ode_basis),
        num_params_model,
    ),
    (
        'Baseline',
        partial(baseline_map_and_loss, ode_basis=ode_basis),
        num_params_baseline,
    ),
]

for model_name, model, num_params in models:

    key, subkey = random.split(key)
    params = 0.1*random.normal(subkey, shape=(num_params,))

    key, subkey = random.split(key)
    params, _, _ = ml.train(
        X_train,
        X_train, #reconstruction, so we want to get back to the input
        model,
        params,
        key,
        ml.EpochStop(epochs, verbose=verbose),
        batch_size=batch_size,
        optimizer=optax.adam(lr),
        save_params=save_file,
    )