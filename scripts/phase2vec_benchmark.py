#generate gravitational field
import os
import numpy as np
import sys
from functools import partial
import argparse
import time
from scipy.special import binom
import math
from collections import defaultdict

import jax.numpy as jnp
import jax.random as random
import jax.nn
from jax import vmap, jit
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml

DENSE = 'dense'
BATCH_NORM_1D = 'batch_norm_1d'

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

def dense_layer(params, x, width, bias=True, mold_params=False):
    params_idx, this_params = ml.get_layer_params(params, mold_params, DENSE)
    if mold_params:
        this_params[ml.SCALE] = jnp.ones((width, len(x)))

    out_vec = this_params[ml.SCALE] @ x

    if bias:
        if mold_params: 
            this_params[ml.BIAS] = jnp.ones(width)

        out_vec = out_vec + this_params[ml.BIAS]

    params = ml.update_params(params, params_idx, this_params, mold_params)

    return out_vec, params

def dense_layer_init(rand_key, tree):
    bound = 1/jnp.sqrt(tree[ml.SCALE].shape[1])
    rand_key, subkey = random.split(rand_key)
    params = { ml.SCALE: random.uniform(subkey, shape=tree[ml.SCALE].shape, minval=-bound, maxval=bound) }
    if (ml.BIAS in tree):
        rand_key, subkey = random.split(rand_key)
        params[ml.BIAS] = random.uniform(subkey, shape=tree[ml.BIAS].shape, minval=-bound, maxval=bound)

    return params

@partial(jit, static_argnums=[2,5,6,7])
def batch_norm_1d(params, x, train, running_mean, running_var, momentum=0.1, eps=1e-05, mold_params=False):
    params_idx, this_params = ml.get_layer_params(params, mold_params, BATCH_NORM_1D)
    if mold_params:
        width = x.shape[1]
        this_params[ml.SCALE] = jnp.ones(width)
        this_params[ml.BIAS] = jnp.ones(width)

    if (train):
        mean = jnp.mean(x, axis=0)
        var = jnp.var(x, axis=0)

        if ((running_mean is None) and (running_var is None)):
            running_mean = mean
            running_var = var
        else:
            running_mean = (1 - momentum)*running_mean + momentum*mean
            running_var = (1 - momentum)*running_var + momentum*var
    else: # not train, use the final value from training
        mean = running_mean 
        var = running_var

    layer_vec = (x - mean)/jnp.sqrt(var + eps)
    layer_vec = vmap(lambda x: x * this_params[ml.SCALE])(layer_vec)
    layer_vec = vmap(lambda x: x + this_params[ml.BIAS])(layer_vec)

    params = ml.update_params(params, params_idx, this_params, mold_params)

    return layer_vec, params, running_mean, running_var

@partial(jit, static_argnums=[1,3])
def dropout_layer(x, p, key, train):
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
    
    return (x * (random.uniform(key, shape=x.shape) > p).astype(x.dtype))*(1/(1-p))

def net(params, layer, key, train, batch_stats, conv_filters, ode_basis, batch_norm=True, dropout=True, return_params=False):
    # batch norm, dropout seem to make the network widely worse, not sure why.
    embedding_d = 100
    num_coeffs = ode_basis.shape[1]
    depth = 4
    num_conv_layers = 3
    num_hidden_layers = 2
    img_N = layer.N
    if (batch_stats is None):
        batch_stats = { num: { 'mean': None, 'var': None } for num in range(5) }

    batch_dense_layer = vmap(dense_layer, in_axes=(None, 0, None, None, None), out_axes=(0, None))

    # Convolution on the vector field generated by the ODE
    batch_stats_idx = 0
    for _ in range(num_conv_layers):
        layer, params = ml.batch_conv_layer(
            params,
            layer,
            { 'type': 'fixed', 'filters': conv_filters }, 
            depth, 
            mold_params=return_params,
            stride=(2,)*layer.D, 
            padding='VALID',
        )
        # layer, params = ml.batch_layer_norm(params, layer, mold_params=return_params)
        layer = ml.batch_relu_layer(layer)
        batch_stats_idx += 1

    # Embed the ODE in a d=100 vector
    layer_vec = jnp.zeros(embedding_d)
    for image in layer.values():
        vectorized_image = image.reshape(layer.L,-1)
        layer_out, params = batch_dense_layer(params, vectorized_image, embedding_d, True, return_params)
        layer_vec += layer_out

    # Pass the embedded ode through a 2 layer MLP to get the coefficients of poly ode
    for _ in range(num_hidden_layers):
        layer_vec, params = batch_dense_layer(params, layer_vec, 128, True, return_params)
        if batch_norm:
            layer_vec, params, mean, var = batch_norm_1d(
                params, 
                layer_vec, 
                train,
                batch_stats[batch_stats_idx]['mean'],
                batch_stats[batch_stats_idx]['var'],
                mold_params=return_params,
            )
            batch_stats[batch_stats_idx] = { 'mean': mean, 'var': var }
        layer_vec = jax.nn.relu(layer_vec)
        if dropout:
            key, subkey = random.split(key)
            layer_vec = dropout_layer(layer_vec, 0.1, subkey ,train)

        batch_stats_idx += 1

    coeffs, params = batch_dense_layer(params, layer_vec, num_coeffs * layer.D, True, return_params)
    coeffs = coeffs.reshape((layer.L, num_coeffs, layer.D))

    # multiply the functions by the coefficients
    vmap_mul = vmap(lambda coeffs_x: ode_basis @ coeffs_x)
    recon_x = vmap_mul(coeffs).reshape((layer.L,) + (img_N,)*layer.D + (layer.D,))
    return (recon_x, coeffs, batch_stats, params) if return_params else (recon_x, coeffs, batch_stats)

def phase2vec_loss(recon_x, y, coeffs, eps=1e-5, beta=1e-3, reduce=True):
    """
    Loss used in the phase2vec paper, combines a pointwise normalized l2 loss with a sparsity l1 loss
    args:
        recon_x (jnp.array): image block of (batch, N, N, D)
        y (jnp.array): image_block of (batch, N, N, D)
    """
    '''Euclidean distance between two arrays with spatial dimensions. Normalizes pointwise across the spatial dimension by the norm of the second argument'''
    m_gt = jnp.expand_dims(jnp.linalg.norm(y, axis=3), axis=3)
    den = m_gt + eps 

    # normalize pointwise by the gt norm
    if reduce:
        recon_loss = jnp.mean(jnp.sqrt(vmap(jnp.mean)(((recon_x - y)**2) / den)))
    else:
        recon_loss = jnp.mean(jnp.sqrt(((recon_x - y)**2) / den))

    return recon_loss if (beta is None) else recon_loss + beta * jnp.mean(jnp.abs(coeffs))

@partial(jit, static_argnums=[4,8,9,10,11])
def map_and_loss(params, layer_x, layer_y, key, train, aux_data, conv_filters, ode_basis, batch_norm, dropout, eps=1e-5, beta=1e-3):
    recon, coeffs, batch_stats = net(params, layer_x, key, train, aux_data, conv_filters, ode_basis, batch_norm, dropout)
    return phase2vec_loss(recon, layer_y[1][:,0,...], coeffs, eps, beta), batch_stats

def baseline_net(params, layer, key, train, batch_stats, ode_basis, batch_norm, dropout, return_params=False):
    embedding_d = 100
    num_coeffs = ode_basis.shape[1]
    num_conv_layers = 3
    num_hidden_layers = 2
    img_N = layer.N
    # for the baseline model, the vector just becomes 2 channels
    layer = geom.BatchLayer({ 0: jnp.moveaxis(layer[1][:,0,...], -1, 1) }, layer.D, layer.is_torus)
    assert layer[0].shape[1:] == (2, img_N, img_N)
    if (batch_stats is None):
        batch_stats = { num: { 'mean': None, 'var': None } for num in range(5) }

    batch_dense_layer = vmap(dense_layer, in_axes=(None, 0, None, None, None), out_axes=(0, None))

    # Convolution on the vector field generated by the ODE
    batch_stats_idx = 0
    for _ in range(num_conv_layers):
        layer, params = ml.batch_conv_layer(
            params,
            layer, 
            { 'type': 'free', 'M': 3, 'filter_k_set': (0,) },
            depth=128,
            bias=True,
            mold_params=return_params,
            stride=(2,)*layer.D,
            padding='VALID',
        )
        if batch_norm:
            layer, params, mean, var = ml.batch_norm(
                params, 
                layer, 
                train,
                batch_stats[batch_stats_idx]['mean'],
                batch_stats[batch_stats_idx]['var'],
                mold_params=return_params,
            )
            batch_stats[batch_stats_idx] = { 'mean': mean, 'var': var }
        layer = ml.batch_relu_layer(layer)
        batch_stats_idx += 1
    
    # Embed the ODE in a d=100 vector
    layer_vec, params = batch_dense_layer(params, layer[0].reshape(layer.L,-1), embedding_d, True, return_params)

    # Pass the embedded ode through a 2 layer MLP to get the coefficients of poly ode
    for _ in range(num_hidden_layers):
        layer_vec, params = batch_dense_layer(params, layer_vec, 128, True, return_params)
        layer_vec, params, mean, var = batch_norm_1d(
            params, 
            layer_vec, 
            train,
            batch_stats[batch_stats_idx]['mean'],
            batch_stats[batch_stats_idx]['var'],
            mold_params=return_params,
        )
        batch_stats[batch_stats_idx] = { 'mean': mean, 'var': var }
        layer_vec = jax.nn.relu(layer_vec)
        if dropout:
            key, subkey = random.split(key)
            layer_vec = dropout_layer(layer_vec, 0.1, subkey, train)

        batch_stats_idx += 1

    coeffs, params = batch_dense_layer(params, layer_vec, num_coeffs * layer.D, True, return_params)
    coeffs = coeffs.reshape((layer.L, num_coeffs, layer.D))

    # multiply the functions by the coefficients
    vmap_mul = vmap(lambda coeffs_x: ode_basis @ coeffs_x)
    recon_x = vmap_mul(coeffs).reshape((layer.L,) + (img_N,)*layer.D + (layer.D,))
    return (recon_x, coeffs, batch_stats, params) if return_params else (recon_x, coeffs, batch_stats)

@partial(jit, static_argnums=[4,7,8,9,10])
def baseline_map_and_loss(params, layer_x, layer_y, key, train, aux_data, ode_basis, batch_norm, dropout, eps=1e-5, beta=1e-3):
    recon, coeffs, batch_stats = baseline_net(params, layer_x, key, train, aux_data, ode_basis, batch_norm, dropout)
    return phase2vec_loss(recon, layer_y[1][:,0,...], coeffs, eps, beta), batch_stats

def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', help='number of epochs', type=float, default=200)
    parser.add_argument('-lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('-batch', help='batch size', type=int, default=64)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-s', '--save', help='folder name to save the results', type=str, default=None)
    parser.add_argument('-l', '--load', help='folder name to load results from', type=str, default=None)
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
epochs, lr, batch_size, seed, save_folder, load_folder, verbose = handleArgs(sys.argv)

D = 2

key = random.PRNGKey(time.time_ns() if (seed is None) else seed)

# start with basic 3x3 scalar, vector, and 2nd order tensor images
operators = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[3], ks=[0,1], parities=[0], D=D, operators=operators)

# Get Training data
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

# Get testing data. We add the validation data set to the test data, as they do in the paper
test_data_path = '../phase2vec/output/data/classical'
X_test1, X_test2, y_test1, y_test2, p_test1, p_test2 = load_dataset(test_data_path)
X_test_data = jnp.concatenate([X_test1, X_test2, X_val_data])
y_test = jnp.concatenate([y_test1, y_test2, 16 * np.ones_like(y_val)])
p_test = jnp.concatenate([p_test1, p_test2, p_val])
X_test = geom.BatchLayer(
    { 1: jnp.expand_dims(X_test_data.transpose(0,2,3,1), axis=1) },
    D,
    False,
)

# generate function library
spatial_coords = [jnp.linspace(mn, mx, X_train.N) for (mn, mx) in zip([-1.,-1.], [1.,1.])]
mesh = jnp.meshgrid(*spatial_coords)
L = jnp.concatenate([ms[..., None] for ms in mesh], axis=-1)
ode_basis, _ = sindy_library(L.reshape(X_train.N**D,D), 3)

# Define the models that we are benchmarking
models = [
    (
        'gi_net',
        partial(map_and_loss, conv_filters=conv_filters, ode_basis=ode_basis, batch_norm=True, dropout=True),
        partial(net, conv_filters=conv_filters, ode_basis=ode_basis, batch_norm=True, dropout=True),
    ),
    (
        'gi_net_no_batch_norm',
        partial(map_and_loss, conv_filters=conv_filters, ode_basis=ode_basis, batch_norm=False, dropout=False),
        partial(net, conv_filters=conv_filters, ode_basis=ode_basis, batch_norm=False, dropout=False),
    ),
    (
        'baseline_beta',
        partial(baseline_map_and_loss, ode_basis=ode_basis, batch_norm=True, dropout=True),
        partial(baseline_net, ode_basis=ode_basis, batch_norm=True, dropout=True),
    ),
    (
        'baseline_no_batch_norm', # to compare to gi_net, no batch norm after the cnn layers
        partial(baseline_map_and_loss, ode_basis=ode_basis, batch_norm=False, dropout=False),
        partial(baseline_net, ode_basis=ode_basis, batch_norm=False, dropout=False),
    ),
]

# For each model, calculate the size of the parameters and add it to the tuple.
one_point = X_train.get_subset(jnp.array([0]))
models_init = []
for model_name, model_map_and_loss, model in models:
    get_params = ml.init_params(
        partial(model, batch_stats=None), 
        one_point, 
        key, 
        return_func=True, 
        override_initializers={
            DENSE: dense_layer_init,
            BATCH_NORM_1D: lambda _,tree: { ml.SCALE: jnp.ones(tree[ml.SCALE].shape), ml.BIAS: jnp.zeros(tree[ml.BIAS].shape) },
        }
    )
    print(f'{model_name}: {ml.count_params(get_params(key))} params')
    models_init.append((model_name, model_map_and_loss, model, get_params))

labels = ['saddle_node 0',
    'saddle_node 1',
    'pitchfork 0',
    'pitchfork 1',
    'transcritical 0',
    'transcritical 1',
    'selkov 0',
    'selkov 1',
    'homoclinic 0',
    'homoclinic 1',
    'vanderpol 0',
    'simple_oscillator 0',
    'simple_oscillator 1',
    'fitzhugh_nagumo 0',
    'fitzhugh_nagumo 1',
    'lotka_volterra 0',
    'polynomial',
]

for model_name, model, eval_net, get_params in models_init:
    print(f'Model: {model_name}')

    if load_folder:
        params = jnp.load(f'{load_folder}/{model_name}_s{seed}.npy')
    else:
        key, subkey = random.split(key)
        # might want to try to do kaiming initialization here
        params = get_params(subkey)

        key, subkey = random.split(key)
        params, batch_stats, _, _ = ml.train(
            X_train,
            X_train, #reconstruction, so we want to get back to the input
            model,
            params,
            subkey,
            ml.EpochStop(epochs, verbose=verbose),
            batch_size=batch_size,
            optimizer=optax.adam(lr),
            validation_X=X_val,
            validation_Y=X_val, # reconstruction, reuse X_val as the Y
            has_aux=True,
        )

        if save_folder:
            jnp.save(f'{save_folder}/{model_name}_s{seed}.npy', params)

    par_losses = []
    recon_losses = []
    recon_dict = {}
    coeffs_dict = {}
    for label in jnp.unique(y_test):
        class_layer = X_test.get_subset(y_test == label)
        class_pars = p_test[y_test == label]
        class_data = class_layer[1][:,0,...]

        full_recon = None
        full_coeffs = None
        for i in range(math.ceil(len(class_data) / batch_size)): #split into batches if its too big
            batch_layer = class_layer.get_subset(jnp.arange(
                batch_size*i, 
                min(batch_size*(i+1), len(class_data)),
            ))

            key, subkey = random.split(key)
            recon, coeffs, _ = eval_net(
                params, 
                batch_layer, 
                key=subkey, 
                train=False,
                batch_stats=batch_stats,
            )

            if full_recon is None:
                full_recon = recon
                full_coeffs = coeffs
            else:
                full_recon = jnp.concatenate([full_recon, recon])
                full_coeffs = jnp.concatenate([full_coeffs, coeffs])

        recon_dict[int(label)] = np.array(full_recon)
        coeffs_dict[int(label)] = np.array(full_coeffs)

        assert full_coeffs.shape == class_pars.shape
        assert full_recon.shape == class_data.shape
        par_loss = ml.mse_loss(full_coeffs, class_pars)
        recon_loss = phase2vec_loss(full_recon, class_data, full_coeffs, beta=None, reduce=False)

        print(f'{labels[label]}, par: {par_loss:0.5f} --- recon: {recon_loss:0.5f}')

        par_losses.append(par_loss)
        recon_losses.append(recon_loss)

    print('Mean par loss: ', jnp.mean(jnp.stack(par_losses)))
    print('Mean recon loss: ', jnp.mean(jnp.stack(recon_losses)))
