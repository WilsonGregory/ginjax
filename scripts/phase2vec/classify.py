import numpy as np
import sys
from functools import partial, reduce
import argparse
import time

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

def classify_eval(z_train, Y_eval_train, z_test, Y_eval_test):
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

    train_accuracy = classification_report(Y_eval_train, clf.predict(z_train), output_dict=True)['accuracy']
    test_accuracy = classification_report(Y_eval_test, clf.predict(z_test), output_dict=True)['accuracy']
    return train_accuracy, test_accuracy

# handles a batch input layer
def eval_baseline(data, key, model_name, func):
    *_, X_test, Y_test, eval_data = data 

    key, subkey = random.split(key)
    par_loss, recon_loss = p2v_models.get_par_recon_loss(
        X_test, 
        Y_test, 
        subkey, 
        func, 
        X_test.get_L(), # batch_size, can just do it in one batch
    )

    results = []
    for X_eval_train, X_eval_test, Y_eval_train, Y_eval_test in eval_data:
        key, subkey = random.split(key)
        z_train = func(X_eval_train, subkey)[1] # L,10,2
        z_train = z_train.reshape((len(z_train),-1)) # L,20

        key, subkey = random.split(key)
        z_test = func(X_eval_test, subkey)[1] # L,10,2
        z_test = z_test.reshape((len(z_test),-1)) # L,20

        train_accuracy, test_accuracy = classify_eval(z_train, Y_eval_train, z_test, Y_eval_test)

        results.append(train_accuracy)
        results.append(test_accuracy)

    return (par_loss, recon_loss) + tuple(results) 

def train_and_eval(data, rand_key, model_name, net, batch_size, lr, epochs, verbose):
    X_train, Y_train, X_val, Y_val, X_test, Y_test, eval_data = data 

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

    rand_key, subkey = random.split(rand_key)
    par_loss, recon_loss = p2v_models.get_par_recon_loss(
        X_test,
        Y_test,
        subkey,
        partial(net, params=params, train=False, batch_stats=batch_stats),
        batch_size,
    )

    results = []
    for X_eval_train, X_eval_test, Y_eval_train, Y_eval_test in eval_data:
        # get the embeddings of x_train and x_test, using batches if necessary
        def embedding_net(params, layer, key, train, batch_stats):
            res = partial(net, return_embedding=True)(params, layer, key, train, batch_stats)
            return res[3],res[2] # return embedding, batch_stats

        rand_key, subkey = random.split(rand_key)
        z_train_list = ml.map_in_batches(embedding_net, params, X_eval_train, batch_size, subkey, False, True, batch_stats)
        embedding_d = z_train_list[0].shape[-1]
        z_train = reduce(
            lambda carry, val: jnp.concatenate([carry, val.reshape((-1,embedding_d))]), # (gpus,batch/gpus,embedding_d)
            z_train_list, 
            jnp.ones((0,embedding_d)),
        )

        rand_key, subkey = random.split(rand_key)
        z_test_list = ml.map_in_batches(embedding_net, params, X_eval_test, batch_size, subkey, False, True, batch_stats)
        z_test = reduce(
            lambda carry, val: jnp.concatenate([carry, val.reshape((-1,embedding_d))]), # (gpus,batch/gpus,embedding_d)
            z_test_list, 
            jnp.ones((0,embedding_d)),
        )

        train_accuracy, test_accuracy = classify_eval(z_train, Y_eval_train, z_test, Y_eval_test)

        results.append(train_accuracy)
        results.append(test_accuracy)
    
    return (par_loss, recon_loss) + tuple(results)

def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('save_folder', help='where to save the results array', type=str)
    parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=200)
    parser.add_argument('-lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('-batch', help='batch size', type=int, default=64)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-v', '--verbose', help='levels of print statements during training', type=int, default=1)
    parser.add_argument('-t', '--trials', help='number of trials', type=int, default=1)

    args = parser.parse_args()

    return (
        args.save_folder,
        args.epochs,
        args.lr,
        args.batch,
        args.seed,
        args.verbose,
        args.trials,
    )

# Main
save_folder, epochs, lr, batch_size, seed, verbose, trials = handleArgs(sys.argv)

D = 2
N = 64 

key = random.PRNGKey(time.time_ns() if (seed is None) else seed)

# start with basic 3x3 scalar, vector, and 2nd order tensor images
operators = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[3], ks=[0,1,2,3], parities=[0], D=D, operators=operators)

# Get data. Update these data paths to your own.
train_data_path = '../phase2vec/output/data/polynomial'
test_data_path = '../phase2vec/output/data/classical'
X_train, X_val, X_test, _, _, y_test, _, _, p_test = p2v_models.load_and_combine_data(
    D, 
    train_data_path, 
    test_data_path,
)

eval_data = []
for test_dataset in ['linear', 'conservative_vs_nonconservative', 'incompressible_vs_compressible']:
    test_data_path = f'../phase2vec/output/data/{test_dataset}'
    X_eval_train, X_eval_test, Y_eval_train, Y_eval_test, _, _ = p2v_models.load_dataset(D, test_data_path)
    eval_data.append((X_eval_train, X_eval_test, Y_eval_train, Y_eval_test))

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
            net=partial(p2v_models.gi_net, conv_filters=conv_filters, ode_basis=ode_basis), 
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
        'baseline_simple', # paper architecture, but no batchnorm or dropout, only relu in cnn
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
    # ('pinv', partial(eval_baseline, func=partial(p2v_models.pinv_baseline, library=ode_basis))),
    ('lasso', partial(eval_baseline, func=partial(p2v_models.lasso, library=ode_basis))),
]

key, subkey = random.split(key)
results = ml.benchmark(
    lambda _,_2: (X_train, X_train, X_val, X_val, X_test, (X_test, y_test, p_test), eval_data),
    models,
    subkey,
    '',
    [0], 
    num_trials=trials,
    num_results=8,
)

jnp.save(f'{save_folder}/results_s{seed}_e{epochs}_t{trials}', results)

print('results', results)
print('mean_results', jnp.mean(results, axis=0)) # this is the mean over trials
