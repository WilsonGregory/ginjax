import os
import numpy as np
from functools import partial
from scipy.special import binom

import jax.numpy as jnp
import jax.random as random
import jax.nn
from jax import vmap, jit

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml

DENSE = 'dense'
BATCH_NORM_1D = 'batch_norm_1d'

# Copied from phase2vec

def load_dataset(D, data_path, as_layer=True):
    assert D == 2 # layer construction transpose is hardcoded, so make sure this is 2
    # Load data
    X_train_data = np.load(os.path.join(data_path, 'X_train.npy'))
    X_test_data = np.load(os.path.join(data_path, 'X_test.npy'))

    # Load labels
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))

    # Load pars
    p_train = np.load(os.path.join(data_path, 'p_train.npy'))
    p_test = np.load(os.path.join(data_path, 'p_test.npy'))

    if as_layer:
        X_train = geom.BatchLayer(
            { (1,0): jnp.expand_dims(X_train_data.transpose(0,2,3,1), axis=1) }, #(batch, channel, (N,)*D, (D,)*k)
            D,
            False,
        )
        X_test = geom.BatchLayer(
            { (1,0): jnp.expand_dims(X_test_data.transpose(0,2,3,1), axis=1) },
            D,
            False,
        )
        return X_train, X_test, y_train, y_test, p_train, p_test
    else:
        return X_train_data, X_test_data, y_train, y_test, p_train, p_test

def load_and_combine_data(D, train_data_path, test_data_path):
    X_train_data, X_val_data, y_train, y_val, p_train, p_val = load_dataset(D, train_data_path, as_layer=False)
    X_test_data1, X_test_data2, y_test1, y_test2, p_test1, p_test2 = load_dataset(D, test_data_path, as_layer=False)

    X_train = geom.BatchLayer(
        { (1,0): jnp.expand_dims(X_train_data.transpose(0,2,3,1), axis=1) }, #(batch, channel, (N,)*D, (D,)*k)
        D,
        False,
    )
    X_val = geom.BatchLayer(
        { (1,0): jnp.expand_dims(X_val_data.transpose(0,2,3,1), axis=1) },
        D,
        False,
    )

    # add validation data to the test data set, as they do in the phase2vec paper
    X_test_data = jnp.concatenate([X_test_data1, X_test_data2, X_val_data])
    y_test = jnp.concatenate([y_test1, y_test2, 16 * np.ones_like(y_val)])
    p_test = jnp.concatenate([p_test1, p_test2, p_val])
    X_test = geom.BatchLayer(
        { (1,0): jnp.expand_dims(X_test_data.transpose(0,2,3,1), axis=1) },
        D,
        False,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, p_train, p_val, p_test

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

def get_ode_basis(D, N, min_xy, max_xy, poly_order, as_vector_images=False):
    spatial_coords = [jnp.linspace(mn, mx, N) for (mn, mx) in zip(min_xy, max_xy)]
    mesh = jnp.meshgrid(*spatial_coords, indexing='ij')
    L = jnp.concatenate([ms[..., None] for ms in mesh], axis=-1)
    ode_basis, _ = sindy_library(L.reshape(N**D,D), poly_order)

    if as_vector_images:
        num_coeffs = ode_basis.shape[1]
        basis_img = ode_basis.transpose((1,0)).reshape((num_coeffs,N,N,1))

        # for the number of dimensions, copy the basis in that coordinate of the vector, all
        # others are 0. So there are num_coeffs * D elements of the ode basis
        filled_img = jnp.full((num_coeffs,N,N,D),basis_img)
        ode_basis = jnp.concatenate([jnp.eye(D)[i,:].reshape((1,1,1,D))*filled_img for i in range(D)])

    return ode_basis

# handles a batch input layer
def pinv_baseline(layer, key, library):
    N = layer.N
    D = layer.D
    # library is 4096 x 10
    batch_img = jnp.transpose(layer[(1,0)], [0,4,1,2,3]).reshape(layer.L, layer.D,-1)
    library_pinv = jnp.linalg.pinv(library.T)

    batch_mul = vmap(lambda batch_arr,single_arr: batch_arr @ single_arr, in_axes=(0,None))

    batch_coeffs = batch_mul(batch_img, library_pinv) # L,2,10
    batch_recon = batch_mul(batch_coeffs, library.T) # L, 2, 4096

    return (
        jnp.moveaxis(batch_recon.reshape((layer.L,D) + (N,)*D), 1, -1),
        jnp.moveaxis(batch_coeffs, 1, -1), 
        None, # this is the "batch_stats" for convenience in this script
    )

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

@partial(jit, static_argnums=[3,8,9])
def gi_net(
    params, 
    layer, 
    key, 
    train, 
    batch_stats, 
    conv_filters, 
    ode_basis,
    maps_to_coeffs=None,
    return_params=False, 
    return_embedding=False,
):
    num_coeffs = ode_basis.shape[1]
    depth = 30
    num_conv_layers = 3
    img_N = layer.N

    batch_dense_layer = vmap(dense_layer, in_axes=(None, 0, None, None, None), out_axes=(0, None))

    # Convolution on the vector field generated by the ODE
    for _ in range(num_conv_layers):
        layer, params = ml.batch_conv_contract(
            params,
            layer,
            conv_filters,
            depth,
            ((0,0),(1,0)),
            mold_params=return_params,
            stride=(2,)*layer.D,
            padding='VALID',
        )
        layer = ml.batch_scalar_activation(layer, jax.nn.leaky_relu)

    layer, params = ml.batch_conv_contract(
        params,
        layer,
        conv_filters,
        depth,
        ((0,0),(1,0),(2,0)),
        mold_params=return_params,
        padding='VALID',
    ) # down to 5x5
    layer, params = ml.batch_channel_collapse(params, layer, 1, mold_params=return_params)

    # Embed the ODE in a d=175 vector
    embedding = layer.to_vector()

    if maps_to_coeffs is None:
        coeffs, params = batch_dense_layer(params, embedding, num_coeffs * layer.D, True, return_params)
    else:
        coeffs, params = ml.batch_equiv_dense_layer(params, embedding, maps_to_coeffs, mold_params=return_params)

    coeffs = coeffs.reshape((layer.L, num_coeffs, layer.D))

    # multiply the functions by the coefficients
    vmap_mul = vmap(lambda coeffs_x: ode_basis @ coeffs_x)
    recon_x = vmap_mul(coeffs).reshape((layer.L,) + (img_N,)*layer.D + (layer.D,))

    output = (recon_x, coeffs, batch_stats)
    if return_embedding:
        output += (embedding,)
    if return_params:
        output += (params,)

    return output

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

def map_and_loss(params, layer_x, layer_y, key, train, aux_data, net, eps=1e-5, beta=1e-3):
    recon, coeffs, batch_stats = net(params, layer_x, key, train, aux_data)
    return phase2vec_loss(recon, layer_y[(1,0)][:,0,...], coeffs, eps, beta), batch_stats

@partial(jit, static_argnums=[3,6,7,8,9,10,11])
def baseline_net(
    params, 
    layer, 
    key, 
    train, 
    batch_stats, 
    ode_basis, 
    batch_norm, 
    dropout, 
    relu, 
    num_hidden_layers,
    return_params=False,
    return_embedding=False,
):
    embedding_d = 100
    num_coeffs = ode_basis.shape[1]
    num_conv_layers = 3
    img_N = layer.N
    # for the baseline model, the vector just becomes 2 channels
    layer = geom.BatchLayer({ (0,0): jnp.moveaxis(layer[(1,0)][:,0,...], -1, 1) }, layer.D, layer.is_torus)
    assert layer[(0,0)].shape[1:] == (2, img_N, img_N)
    if (batch_stats is None):
        batch_stats = { num: { 'mean': None, 'var': None } for num in range(5) }

    batch_dense_layer = vmap(dense_layer, in_axes=(None, 0, None, None, None), out_axes=(0, None))

    # Convolution on the vector field generated by the ODE
    batch_stats_idx = 0
    for _ in range(num_conv_layers):
        layer, params = ml.batch_conv_layer(
            params,
            layer, 
            { 'type': 'free', 'M': 3, 'filter_key_set': { (0,0) } },
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
    layer_vec, params = batch_dense_layer(params, layer[(0,0)].reshape(layer.L,-1), embedding_d, True, return_params)
    embedding = layer_vec

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
        if relu:
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

    output = (recon_x, coeffs, batch_stats)
    if return_embedding:
        output += (embedding,)
    if return_params:
        output += (params,)

    return output
