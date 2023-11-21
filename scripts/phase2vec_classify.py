import os
import numpy as np
import sys
from functools import partial
import argparse
import time
import math
from scipy.special import binom
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report

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

def load_dataset(data_path, combine=False):
    # Load data
    X_train_data = np.load(os.path.join(data_path, 'X_train.npy'))
    X_test_data = np.load(os.path.join(data_path, 'X_test.npy'))

    # Load labels
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))

    # Load pars
    p_train = np.load(os.path.join(data_path, 'p_train.npy'))
    p_test = np.load(os.path.join(data_path, 'p_test.npy'))

    if combine:
        all_data = jnp.concatenate([X_train_data, X_test_data], axis=0)
        X_train = geom.BatchLayer(
            { (1,0): jnp.expand_dims(all_data.transpose(0,2,3,1), axis=1) }, #(batch, channel, (N,)*D, (D,)*k)
            D,
            False,
        )
        y_train = np.concatenate([y_train, y_test], axis=0)
        p_train = np.concatenate([p_train, p_test], axis=0)
        return X_train, y_train, p_train
    else:   
        X_train = geom.BatchLayer(
            { (1,0): jnp.expand_dims(X_train_data.transpose(0,2,3,1), axis=1) },
            D,
            False,
        ) 
        X_test = geom.BatchLayer(
            { (1,0): jnp.expand_dims(X_test_data.transpose(0,2,3,1), axis=1) },
            D,
            False,
        )

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

@partial(jit, static_argnums=[3,7,8])
def gi_net(
    params, 
    layer, 
    key,
    train, 
    batch_stats, 
    conv_filters, 
    ode_basis, 
    return_params=False, 
    return_embedding=False,
):
    # batch norm, dropout seem to make the network widely worse, not sure why.
    embedding_d = 100
    num_coeffs = ode_basis.shape[1]
    depth = 4
    num_conv_layers = 3
    num_hidden_layers = 2
    img_N = layer.N

    batch_dense_layer = vmap(dense_layer, in_axes=(None, 0, None, None, None), out_axes=(0, None))

    # Convolution on the vector field generated by the ODE
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
        layer = ml.batch_relu_layer(layer)

    # Embed the ODE in a d=100 vector
    layer_vec = jnp.zeros(embedding_d)
    for image in layer.values():
        vectorized_image = image.reshape(layer.L,-1)
        layer_out, params = batch_dense_layer(params, vectorized_image, embedding_d, True, return_params)
        layer_vec += layer_out

    embedding = layer_vec # save the embedding

    # Pass the embedded ode through a 2 layer MLP to get the coefficients of poly ode
    for _ in range(num_hidden_layers):
        layer_vec, params = batch_dense_layer(params, layer_vec, 128, True, return_params)

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

@partial(jit, static_argnums=[3,6,7,8,9,10])
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
    return_params=False,
    return_embedding=False,
):
    embedding_d = 100
    num_coeffs = ode_basis.shape[1]
    num_conv_layers = 3
    num_hidden_layers = 2
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
    embedding = layer_vec # save the embedding

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

# handles a batch input layer
def eval_baseline(data, key, library):
    _, _, _, _, (X_eval_train, X_eval_test), (Y_eval_train, Y_eval_test) = data 

    # library is 4096 x 10
    library_pinv = jnp.linalg.pinv(library.T)
    batch_mul = vmap(lambda batch_arr,single_arr: batch_arr @ single_arr, in_axes=(0,None))

    batch_train_img = jnp.transpose(X_eval_train[(1,0)], [0,4,1,2,3]).reshape(X_eval_train.L, X_eval_train.D, -1)
    z_train = batch_mul(batch_train_img, library_pinv).reshape(X_eval_train.L,-1) # L,20

    batch_test_img = jnp.transpose(X_eval_test[(1,0)], [0,4,1,2,3]).reshape(X_eval_test.L, X_eval_test.D, -1)
    z_test = batch_mul(batch_test_img, library_pinv).reshape(X_eval_test.L,-1) # L,20

    num_c = 11 # default value
    k = 10 # default value

    clf_params = {
        'cv': KFold(n_splits=int(len(Y_eval_train) / float(k))),
        # 'random_state': 0,
        'dual': False,
        'solver': 'lbfgs',
        'class_weight': 'balanced',
        'multi_class': 'ovr',
        'refit': True,
        'scoring': 'accuracy',
        'tol': 1e-2,
        'max_iter': 5000,
        'verbose': 0,
        'Cs': np.logspace(-5, 5, num_c),
        'penalty': 'l2',
    }

    clf = LogisticRegressionCV(**clf_params).fit(z_train, Y_eval_train)

    for nm, lb_true, lb_pred in zip(
        ['train', 'test'], 
        [Y_eval_train, Y_eval_test], 
        [clf.predict(z_train), clf.predict(z_test)],
    ):
        report = classification_report(lb_true, lb_pred, output_dict=True)
        print(f"{nm}: {report['accuracy']}")

def train_and_eval(data, rand_key, net, batch_size, lr):
    X_train, Y_train, X_val, Y_val, (X_eval_train, X_eval_test), (Y_eval_train, Y_eval_test) = data 

    rand_key, subkey = random.split(rand_key)
    init_params = ml.init_params(
        partial(net, batch_stats=None),
        X_train.get_subset(jnp.array([0])), 
        subkey, 
        override_initializers={
            DENSE: dense_layer_init,
            BATCH_NORM_1D: lambda _,tree: { ml.SCALE: jnp.ones(tree[ml.SCALE].shape), ml.BIAS: jnp.zeros(tree[ml.BIAS].shape) },
        },
    )

    rand_key, subkey = random.split(rand_key)
    params, batch_stats, _, _ = ml.train(
        X_train,
        Y_train, 
        partial(map_and_loss, net=net),
        init_params,
        subkey,
        ml.EpochStop(epochs, verbose=verbose),
        batch_size=batch_size,
        optimizer=optax.adam(lr),
        validation_X=X_val,
        validation_Y=Y_val,
        has_aux=True,
    )

    # get the embeddings of x_train and x_test, using batches if necessary
    z_train = None
    for i in range(math.ceil(X_eval_train.L / batch_size)): #split into batches if its too big
        batch_layer = X_eval_train.get_subset(jnp.arange(
            batch_size*i, 
            min(batch_size*(i+1), X_eval_train.L),
        ))

        rand_key, subkey = random.split(rand_key)
        batch_z_train = net(params, batch_layer, subkey, train=False, batch_stats=batch_stats, return_embedding=True)[3]
        z_train = batch_z_train if z_train is None else jnp.concatenate([z_train,batch_z_train], axis=0)

    z_test = None
    for i in range(math.ceil(X_eval_test.L / batch_size)): #split into batches if its too big
        batch_layer = X_eval_test.get_subset(jnp.arange(
            batch_size*i, 
            min(batch_size*(i+1), X_eval_test.L),
        ))

        rand_key, subkey = random.split(rand_key)
        batch_z_test = net(params, batch_layer, subkey, train=False, batch_stats=batch_stats, return_embedding=True)[3]
        z_test = batch_z_test if z_test is None else jnp.concatenate([z_test,batch_z_test], axis=0)

    num_c = 11 # default value
    k = 10 # default value

    clf_params = {
        'cv': KFold(n_splits=int(len(Y_eval_train) / float(k))),
        # 'random_state': 0,
        'dual': False,
        'solver': 'lbfgs',
        'class_weight': 'balanced',
        'multi_class': 'ovr',
        'refit': True,
        'scoring': 'accuracy',
        'tol': 1e-2,
        'max_iter': 5000,
        'verbose': 0,
        'Cs': np.logspace(-5, 5, num_c),
        'penalty': 'l2',
    }

    clf = LogisticRegressionCV(**clf_params).fit(z_train, Y_eval_train)

    for nm, lb_true, lb_pred in zip(
        ['train', 'test'], 
        [Y_eval_train, Y_eval_test], 
        [clf.predict(z_train), clf.predict(z_test)],
    ):
        report = classification_report(lb_true, lb_pred, output_dict=True)
        print(f"{nm}: {report['accuracy']}")

        # for (key, value) in report.items():
        #     print(key + f': {value}')

def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('save_folder', help='where to save the image', type=str)
    parser.add_argument('-e', '--epochs', help='number of epochs', type=float, default=200)
    parser.add_argument('-lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('-batch', help='batch size', type=int, default=64)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-v', '--verbose', help='levels of print statements during training', type=int, default=1)
    parser.add_argument('-t', '--trials', help='number of trials', type=int, default=1)
    parser.add_argument(
        '-b', 
        '--benchmark', 
        help='what to benchmark over', 
        type=str, 
        choices=['masking_noise', 'gaussian_noise', 'parameter_noise'], 
        default='gaussian_noise',
    )
    parser.add_argument('-benchmark_steps', help='the number of steps in the benchmark_range', type=int, default=10)

    args = parser.parse_args()

    return (
        args.save_folder,
        args.epochs,
        args.lr,
        args.batch,
        args.seed,
        args.verbose,
        args.trials,
        args.benchmark,
        args.benchmark_steps,
    )

# Main
save_folder, epochs, lr, batch_size, seed, verbose, trials, benchmark, benchmark_steps = handleArgs(sys.argv)

D = 2
N = 64 

key = random.PRNGKey(time.time_ns() if (seed is None) else seed)

# start with basic 3x3 scalar, vector, and 2nd order tensor images
operators = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[3], ks=[0,1], parities=[0], D=D, operators=operators)

# Get data. Update these data paths to your own.
train_data_path = '../phase2vec/output/data/polynomial'
test_data_path = '../phase2vec/output/data/linear'
# test_data_path = '../phase2vec/output/data/conservative_vs_nonconservative'
# test_data_path = '../phase2vec/output/data/incompressible_vs_compressible'
X_train, X_val, y_train, y_val, p_train, p_val = load_dataset(train_data_path)
X_eval_train, X_eval_test, Y_eval_train, Y_eval_test, p_eval_train, p_eval_test = load_dataset(test_data_path)

# generate function library
spatial_coords = [jnp.linspace(mn, mx, X_train.N) for (mn, mx) in zip([-1.,-1.], [1.,1.])]
mesh = jnp.meshgrid(*spatial_coords, indexing='ij')
L = jnp.concatenate([ms[..., None] for ms in mesh], axis=-1)
ode_basis, _ = sindy_library(L.reshape(X_train.N**D,D), 3)

# Define the models that we are benchmarking
models = [
    (
        'gi_net', 
        partial(
            train_and_eval, 
            net=partial(gi_net, conv_filters=conv_filters, ode_basis=ode_basis), 
            batch_size=batch_size, 
            lr=lr,
        ),
    ),
    (
        'baseline', # identical to the paper architecture
        partial(
            train_and_eval,
            net=partial(baseline_net, ode_basis=ode_basis, batch_norm=True, dropout=True, relu=True),
            batch_size=batch_size,
            lr=lr,
        ),
    ),
    (
        'baseline_no_extras', # paper architecture, but no batchnorm or dropout, only relu in cnn
        partial(
            train_and_eval,
            net=partial(baseline_net, ode_basis=ode_basis, batch_norm=False, dropout=False, relu=False),
            batch_size=batch_size,
            lr=lr,
        ),
    ),
    ('pinv', partial(eval_baseline, library=ode_basis)),
]

for k, (model_name, model) in enumerate(models):
    print(f'{model_name}')

    key, subkey = random.split(key)
    res = model(
        (X_train, X_train, X_val, X_val, (X_eval_train, X_eval_test), (Y_eval_train, Y_eval_test)), 
        subkey,
    )
