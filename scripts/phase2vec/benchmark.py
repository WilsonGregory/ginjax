import numpy as np
import sys
from functools import partial
import argparse
import time
import math
import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax.random as random
import jax
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml
import p2v_models

DENSE = 'dense'
BATCH_NORM_1D = 'batch_norm_1d'

def get_par_recon_loss(X_test, Y_test_tuple, rand_key, eval_net, batch_size, print_errs=0):
    labels = [
        'saddle_node 0',
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
        'lotka_volterra 0', #idx 15, skip this one cause its bugged
        'polynomial',
    ]

    Y_test, true_labels, p_test = Y_test_tuple

    par_losses = []
    recon_losses = []
    recon_dict = {}
    coeffs_dict = {}
    for label in jnp.unique(true_labels):
        if (label == 15): # skip this one cause its bugged
            continue

        X_class = X_test.get_subset(true_labels == label)
        Y_class = Y_test.get_subset(true_labels == label)
        class_pars = p_test[true_labels == label]
        Y_class_img = Y_class[(1,0)][:,0,...]
        class_size = len(class_pars)

        full_recon = None
        full_coeffs = None
        for i in range(math.ceil(class_size / batch_size)): #split into batches if its too big
            batch_layer = X_class.get_subset(jnp.arange(
                batch_size*i, 
                min(batch_size*(i+1), class_size),
            ))

            rand_key, subkey = random.split(rand_key)
            recon, coeffs, _ = eval_net(layer=batch_layer, key=subkey)

            full_recon = recon if (full_recon is None) else jnp.concatenate([full_recon, recon])
            full_coeffs = coeffs if (full_coeffs is None) else jnp.concatenate([full_coeffs, coeffs])

        recon_dict[int(label)] = np.array(full_recon)
        coeffs_dict[int(label)] = np.array(full_coeffs)

        assert full_coeffs.shape == class_pars.shape
        assert full_recon.shape == Y_class_img.shape
        par_loss = ml.mse_loss(full_coeffs, class_pars)
        recon_loss = p2v_models.phase2vec_loss(full_recon, Y_class_img, full_coeffs, beta=None, reduce=False)

        if (print_errs >= 2):
            print(f'{labels[label]}, par: {par_loss:0.5f} --- recon: {recon_loss:0.5f}')

        par_losses.append(par_loss)
        recon_losses.append(recon_loss)

    mean_par_loss = jnp.mean(jnp.stack(par_losses))
    mean_recon_loss = jnp.mean(jnp.stack(recon_losses))
    if (print_errs >= 1):
        print('Mean par loss: ', mean_par_loss)
        print('Mean recon loss: ', mean_recon_loss)

    return mean_par_loss, mean_recon_loss

def train_and_eval(data, rand_key, net, batch_size, lr, epochs, verbose, print_errs=0, get_param_count=False):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = data 

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

    if get_param_count:
        return ml.count_params(init_params)

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
    return get_par_recon_loss(
        X_test,
        Y_test,
        subkey,
        partial(net, params=params, train=False, batch_stats=batch_stats),
        batch_size,
        print_errs=print_errs,
    )

def pinv_baseline_map(data, rand_key, library, print_errs=False, get_param_count=False):
    _, _, _, _, X_test, Y_test = data 

    if get_param_count:
        return 0

    rand_key, subkey = random.split(rand_key)
    return get_par_recon_loss(
        X_test, 
        Y_test, 
        subkey, 
        partial(p2v_models.pinv_baseline, library=library), 
        len(X_test[(1,0)]), # batch_size, can just do it in one batch
        print_errs,
    )

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
    parser.add_argument(
        '-print_result', 
        help='if >= 1, print mean par loss and mean recon loss. if >= 2, print per class mean loss', 
        type=int, 
        default=0,
    )

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
        args.print_result,
    )

# Main
save_folder, epochs, lr, batch_size, seed, verbose, trials, benchmark, benchmark_steps, print_result = handleArgs(sys.argv)

D = 2
N = 64 

key = random.PRNGKey(time.time_ns() if (seed is None) else seed)

# start with basic 3x3 scalar, vector, and 2nd order tensor images
operators = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[3], ks=[0,1,2,3], parities=[0], D=D, operators=operators)

# Get data
train_data_path = '../phase2vec/output/data/polynomial'
test_data_path = '../phase2vec/output/data/classical'
X_train, X_val, X_test, y_train, y_val, y_test, p_train, p_val, p_test = p2v_models.load_and_combine_data(
    D, 
    train_data_path, 
    test_data_path,
)

print(X_train)
print(X_train.get_subset(jnp.array([0,1])))
res = geom.get_equivariant_maps(X_train.get_subset(jnp.array([0,1])), operators)
exit()

# generate function library
ode_basis = p2v_models.get_ode_basis(D, N, [-1.,-1.], [1.,1.], 3)

# apply noise to the test data inputs
if benchmark == 'masking_noise':
    def apply_noise(masked_percent, rand_key, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        X_test_out = X_test.empty()
        for (k,parity),val in X_test.items():
            rand_key, subkey = random.split(rand_key)
            X_test_out.append(k, parity, val * (1.*(random.uniform(subkey, shape=(val.shape)) > masked_percent)))

        return X_train, Y_train, X_val, Y_val, X_test_out, Y_test

elif benchmark == 'gaussian_noise':
    def apply_noise(stdev_scale, rand_key, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        X_test_out = X_test.empty()
        for (k,parity),val in X_test.items():
            rand_key, subkey = random.split(rand_key)
            noise_std = jnp.std(val) * stdev_scale # magnitude scales the existing stdev
            X_test_out.append(k, parity, val + noise_std * random.normal(subkey, shape=val.shape))

        return X_train, Y_train, X_val, Y_val, X_test_out, Y_test

elif benchmark == 'parameter_noise': 
    def apply_noise(noise_stdev, rand_key, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        _, _, p_test = Y_test

        X_test_out = X_test.empty()
        for (k,parity),val in X_test.items():
            rand_key, subkey = random.split(rand_key)
            p_test_noisy = p_test + noise_stdev * random.normal(subkey, shape=p_test.shape)
            vmap_mul = jax.vmap(lambda basis, coeffs: basis @ coeffs, in_axes=(None, 0))
            X_test_out.append(k, parity, vmap_mul(ode_basis, p_test_noisy).reshape(val.shape))

        return X_train, Y_train, X_val, Y_val, X_test_out, Y_test

get_data = partial(
    apply_noise, 
    X_train=X_train, 
    Y_train=X_train, 
    X_val=X_val, 
    Y_val=X_val,
    X_test=X_test,
    Y_test=(X_test, y_test, p_test),
)

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
            print_errs=print_result,
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
            print_errs=print_result,
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
            print_errs=print_result,
        ),
    ),
    ('pinv_baseline', partial(pinv_baseline_map, library=ode_basis, print_errs=print_result)),
]

for model_name, model in models:
    key, subkey = random.split(key)
    print(f'{model_name} params: {model(get_data(0, subkey), subkey, print_errs=False, get_param_count=True)}')

benchmark_range = np.linspace(0,0.3,benchmark_steps) # should be 20
key, subkey = random.split(key)
results = ml.benchmark(
    get_data,
    models,
    subkey,
    benchmark,
    benchmark_range, 
    num_trials=trials,
    num_results=2,
)
num_metrics = results.shape[-1]

# Plot the results
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'STIXGeneral'
plt.tight_layout()

fig, axs = plt.subplots(ncols=num_metrics, figsize=(num_metrics*8,6))
for k, (model_name, _) in enumerate(models):
    for i in range(num_metrics):
        loss_mean = np.mean(results[:,:,k,i], axis=0) # average over the trials
        axs[i].plot(benchmark_range, loss_mean, label=model_name)

        if trials > 1:
            loss_stdev = np.std(results[:,:,k,i], axis=0) # take the stdev over the trials
            axs[i].fill_between(benchmark_range, loss_mean - loss_stdev, loss_mean + loss_stdev, alpha=0.2)

plt.legend()
# Just gonna hard code these names
axs[0].set_title(f'Param Loss vs. {benchmark}')
axs[0].set_ylabel('Parameter Loss')
axs[0].set_xlabel(f'{benchmark} Magnitude')
axs[1].set_title(f'Recon Loss vs. {benchmark}')
axs[1].set_ylabel('Reconstruction Loss')
axs[1].set_xlabel(f'{benchmark} Magnitude')
plt.savefig(f'{save_folder}benchmark_{benchmark}_t{trials}_s{seed}_steps{benchmark_steps}.png')
