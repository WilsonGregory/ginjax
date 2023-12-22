import numpy as np
import sys
from functools import partial
import argparse
import time
import math

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report

import jax.numpy as jnp
import jax.random as random
import jax
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml
import p2v_models

DENSE = 'dense'
BATCH_NORM_1D = 'batch_norm_1d'

# handles a batch input layer
def eval_baseline(data, key, library):
    _, _, _, _, (X_eval_train, X_eval_test), (Y_eval_train, Y_eval_test) = data 

    # library is 4096 x 10
    library_pinv = jnp.linalg.pinv(library.T)
    batch_mul = jax.vmap(lambda batch_arr,single_arr: batch_arr @ single_arr, in_axes=(0,None))

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

def train_and_eval(data, rand_key, net, batch_size, lr, epochs, verbose):
    X_train, Y_train, X_val, Y_val, (X_eval_train, X_eval_test), (Y_eval_train, Y_eval_test) = data 

    rand_key, subkey = random.split(rand_key)
    init_params = ml.init_params(
        partial(net, batch_stats=None),
        X_train.get_subset(jnp.array([0])), 
        subkey, 
        override_initializers={
            DENSE: p2v_models.dense_layer_init,
            BATCH_NORM_1D: lambda _,tree: { ml.SCALE: jnp.ones(tree[ml.SCALE].shape), ml.BIAS: jnp.zeros(tree[ml.BIAS].shape) },
        },
    )

    rand_key, subkey = random.split(rand_key)
    params, batch_stats, _, _ = ml.train(
        X_train,
        Y_train, 
        partial(p2v_models.map_and_loss, net=net),
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

def handleArgs(argv):
    test_datasets = ['linear', 'conservative_vs_nonconservative', 'incompressible_vs_compressible']
    parser = argparse.ArgumentParser()
    parser.add_argument('save_folder', help='where to save the image', type=str)
    parser.add_argument('-e', '--epochs', help='number of epochs', type=float, default=200)
    parser.add_argument('-lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('-batch', help='batch size', type=int, default=64)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-v', '--verbose', help='levels of print statements during training', type=int, default=1)
    parser.add_argument('-test_dataset', help='test data set', type=str, choices=test_datasets, default='linear')

    args = parser.parse_args()

    return (
        args.save_folder,
        args.epochs,
        args.lr,
        args.batch,
        args.seed,
        args.verbose,
        args.test_dataset,
    )

# Main
save_folder, epochs, lr, batch_size, seed, verbose, test_dataset = handleArgs(sys.argv)

D = 2
N = 64 

key = random.PRNGKey(time.time_ns() if (seed is None) else seed)

# start with basic 3x3 scalar, vector, and 2nd order tensor images
operators = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[3], ks=[0,1,2,3], parities=[0], D=D, operators=operators)

# Get data. Update these data paths to your own.
train_data_path = '../phase2vec/output/data/polynomial'
test_data_path = f'../phase2vec/output/data/{test_dataset}'
X_train, X_val, _, _, _, _ = p2v_models.load_dataset(D, train_data_path)
X_eval_train, X_eval_test, Y_eval_train, Y_eval_test, _, _ = p2v_models.load_dataset(D, test_data_path)

# generate function library
ode_basis = p2v_models.get_ode_basis(D, N, [-1.,-1.], [1.,1.], 3)

embedding_N = 5
embedding_layer = geom.Layer(
    {
        (0,0): jnp.zeros((1,) + (embedding_N,)*D),
        (1,0): jnp.zeros((1,) + (embedding_N,)*D + (D,)),
        (2,0): jnp.zeros((1,) + (embedding_N,)*D + (D,D)),
    },
    D, 
    X_train.is_torus,
)
small_ode_basis = p2v_models.get_ode_basis(D, 5, [-1.,-1.], [1.,1.], 3) # (N**D, 10)
Bd_equivariant_maps = geom.get_equivariant_map_to_coeffs(embedding_layer, operators, small_ode_basis)

# Define the models that we are benchmarking
models = [
    (
        'gi_net', 
        partial(
            train_and_eval, 
            net=partial(
                p2v_models.gi_net, 
                conv_filters=conv_filters, 
                ode_basis=ode_basis, 
                maps_to_coeffs=Bd_equivariant_maps,
            ), 
            batch_size=batch_size, 
            lr=lr,
            epochs=epochs,
            verbose=verbose,
        ),
    ),
    (
        'gi_net_full', # last layer is also rotation/reflection equivariant
        partial(
            train_and_eval, 
            net=partial(
                p2v_models.gi_net, 
                conv_filters=conv_filters, 
                ode_basis=ode_basis, 
                maps_to_coeffs=Bd_equivariant_maps,
            ), 
            batch_size=batch_size, 
            lr=lr,
            epochs=epochs,
            verbose=verbose,
        ),
    ),
    (
        'baseline', # identical to the paper architecture
        partial(
            train_and_eval,
            net=partial(
                p2v_models.baseline_net, 
                ode_basis=ode_basis, 
                batch_norm=True, 
                dropout=True, 
                relu=True, 
                num_hidden_layers=2,
            ),
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            verbose=verbose,
        ),
    ),
    (
        'baseline_no_extras', # paper architecture, but no batchnorm or dropout, only relu in cnn
        partial(
            train_and_eval,
            net=partial(
                p2v_models.baseline_net, 
                ode_basis=ode_basis, 
                batch_norm=False, 
                dropout=False,
                relu=False,
                num_hidden_layers=0,
            ),
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            verbose=verbose,
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
