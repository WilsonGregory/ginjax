import sys
import time
import argparse
import numpy as np
from functools import partial
import h5py
import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax
import jax.random as random
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.data as gc_data
import geometricconvolutions.ml as ml
import geometricconvolutions.models as models
import geometricconvolutions.utils as utils

def read_one_h5(filename: str) -> tuple:
    """
    Given a filename and a type of data (train, test, or validation), read the data and return as jax arrays.
    args:
        filename (str): the full file path
    returns: force, particle, and velocity
    """
    data_dict = h5py.File(filename) # keys 'force', 'particles', 't', 'velocity'
    # 4 runs, 1000 time points, 512x512 grid
    force = jax.device_put(jnp.array(data_dict['force'][()]), jax.devices('cpu')[0]) # (4,512,512,2)
    particles = jax.device_put(jnp.array(data_dict['particles'][()]), jax.devices('cpu')[0]) # (4,1000,512,512,1)
    # these are advected particles, might be a proxy of density?
    velocity = jax.device_put(jnp.array(data_dict['velocity'][()]), jax.devices('cpu')[0]) # (4,1000,512,512,2)

    data_dict.close()

    return force, particles[...,0], velocity

def get_data(
    filename: str, 
    num_train_traj: int, 
    num_val_traj: int, 
    num_test_traj: int, 
    past_steps: int,
    rollout_steps: int,
    subsample: int = 1,
    downsample: int = 0,
) -> tuple:
    """
    Get train, val, and test data sets.
    args:
        infile (str): directory of data
        num_train_traj (int): number of training trajectories
        num_val_traj (int): number of validation trajectories
        num_test_traj (int): number of testing trajectories
        past_steps (int): length of the lookback to predict the next step
        rollout_steps (int): number of steps of rollout to compare against
        subsample (int): can subsample for longer timesteps, default 1 (no subsampling)
        downsample (int): number of times to spatial downsample, defaults to 0 (no downsampling)
    """
    D = 2
    force, particles, velocity = read_one_h5(filename)

    start = 0
    stop = num_train_traj
    train_X, train_Y = gc_data.times_series_to_layers(
        D, 
        { (0,0): particles[start:stop], (1,0): velocity[start:stop] },
        { (1,0): force[start:stop] },
        False,
        past_steps,
        1,
        0,
        subsample,
        downsample,
    )
    start = stop
    stop = stop + num_val_traj
    val_X, val_Y = gc_data.times_series_to_layers(
        D, 
        { (0,0): particles[start:stop], (1,0): velocity[start:stop] },
        { (1,0): force[start:stop] },
        False, # is_torus
        past_steps,
        1, # future_steps
        0, # skip_initial
        subsample,
        downsample,
    )
    start = stop
    stop = stop + num_test_traj
    test_single_X, test_single_Y = gc_data.times_series_to_layers(
        D, 
        { (0,0): particles[start:stop], (1,0): velocity[start:stop] },
        { (1,0): force[start:stop] },
        False, # is_torus
        past_steps,
        1, # future_steps
        0, # skip_initial
        subsample,
        downsample,
    )
    test_rollout_X, test_rollout_Y = gc_data.times_series_to_layers(
        D, 
        { (0,0): particles[start:stop], (1,0): velocity[start:stop] },
        { (1,0): force[start:stop] },
        False, # is_torus
        past_steps,
        rollout_steps,
        0, # skip_initial
        subsample,
        downsample,
    )
    
    return train_X, train_Y, val_X, val_Y, test_single_X, test_single_Y, test_rollout_X, test_rollout_Y
    
def plot_layer(test_layer, actual_layer, save_loc, future_steps):
    test_images = [geom.GeometricImage(img, 0, test_layer.D, test_layer.is_torus) for img in test_layer[(1,0)][0,...,0]]
    actual_images = [geom.GeometricImage(img, 0, test_layer.D, test_layer.is_torus) for img in actual_layer[(1,0)][0,...,0]]

    # figsize is 8 per col, 6 per row, (cols,rows)
    fig, axes = plt.subplots(nrows=3, ncols=future_steps, figsize=(8*future_steps,6*3))
    for col in range(future_steps):
        utils.plot_image(test_images[col], ax=axes[0,col], title=f'{col}', colorbar=True)
        utils.plot_image(actual_images[col], ax=axes[1,col], title=f'{col}', colorbar=True)
        utils.plot_image((actual_images[col] - test_images[col]).norm(), ax=axes[2,col], title=f'{col}', colorbar=True)

    plt.savefig(save_loc)
    plt.close(fig)

def map_and_loss(
    params, 
    layer_x, 
    layer_y, 
    key, 
    train, 
    aux_data=None, 
    net=None, 
    has_aux=False, 
    future_steps=1,
    return_map_only=False,
):
    assert net is not None
    curr_layer = layer_x
    out_layer = layer_y.empty()
    for _ in range(future_steps):
        key, subkey = random.split(key)
        if has_aux:
            learned_x, aux_data = net(params, curr_layer, subkey, train, batch_stats=aux_data)
        else:
            learned_x = net(params, curr_layer, subkey, train)

        out_layer = out_layer.concat(learned_x, axis=1)
        next_layer = curr_layer.empty()
        next_layer.append(0, 0, jnp.concatenate([curr_layer[(0,0)][:,1:], learned_x[(0,0)]], axis=1))
        vec_img = curr_layer[(1,0)]
        # drop the first channel, learned step becomes last channel, followed by forcing field channel
        next_layer.append(1, 0, jnp.concatenate([vec_img[:,1:-1], learned_x[(1,0)], vec_img[:,-1:]], axis=1))

        curr_layer = next_layer

    loss = ml.smse_loss(out_layer, layer_y)
    # loss = ml.pointwise_normalized_loss(out_layer, layer_y)
    if return_map_only:
        return out_layer
    else:
        return (loss, aux_data) if has_aux else loss

def train_and_eval(
    data, 
    key, 
    model_name, 
    net, 
    lr,
    batch_size, 
    epochs, 
    save_params, 
    images_dir, 
    noise_stdev=None,
    has_aux=False,
    verbose=2,
):
    train_X, train_Y, val_X, val_Y, test_single_X, test_single_Y, test_rollout_X, test_rollout_Y = data

    key, subkey = random.split(key)
    params = ml.init_params(
        net,
        train_X.get_one(),
        subkey,
    )
    print(f'Model params: {ml.count_params(params):,}')

    steps_per_epoch = int(np.ceil(train_X.get_L() / batch_size))

    key, subkey = random.split(key)
    results = ml.train(
        train_X,
        train_Y,
        partial(map_and_loss, net=net, has_aux=has_aux),
        params,
        subkey,
        stop_condition=ml.EpochStop(epochs, verbose=verbose),
        batch_size=batch_size,
        optimizer=optax.adamw(
            optax.warmup_cosine_decay_schedule(1e-8, lr, 5*steps_per_epoch, 50*steps_per_epoch, 1e-7),
            weight_decay=1e-5,
        ),
        validation_X=val_X,
        validation_Y=val_Y,
        noise_stdev=noise_stdev,
        has_aux=has_aux,
    )

    if has_aux:
        params, batch_stats, train_loss, val_loss = results
    else:
        params, train_loss, val_loss = results
        batch_stats = None

    if save_params is not None:
        jnp.save(
            f'{save_params}{model_name}_L{train_X.get_L()}_e{epochs}_params.npy', 
            { 'params': params, 'batch_stats': None if (batch_stats is None) else dict(batch_stats) },
        )

    key, subkey = random.split(key)
    test_loss = ml.map_loss_in_batches(
        partial(map_and_loss, net=net, has_aux=has_aux), 
        params, 
        test_single_X, 
        test_single_Y, 
        batch_size, 
        subkey, 
        False,
        has_aux=has_aux,
        aux_data=batch_stats,
    )
    print(f'Test Loss: {test_loss}')

    key, subkey = random.split(key)
    test_rollout_loss = ml.map_loss_in_batches(
        partial(map_and_loss, net=net, has_aux=has_aux, future_steps=5), 
        params, 
        test_rollout_X, 
        test_rollout_Y, 
        batch_size, 
        subkey, 
        False,
        has_aux=has_aux,
        aux_data=batch_stats,
    )
    print(f'Test Rollout Loss: {test_rollout_loss}')

    if images_dir is not None:
        key, subkey = random.split(key)
        rollout_layer = map_and_loss(
            params, 
            test_rollout_X.get_one(), 
            test_rollout_Y.get_one(), 
            subkey, 
            False, # train
            batch_stats,
            net, 
            has_aux, 
            future_steps=5, 
            return_map_only=True,
        )
        plot_layer(rollout_layer.get_one(), test_rollout_Y.get_one(), images_dir + f'/{model_name}_rollout.png', 5)

    return train_loss[-1], val_loss[-1], test_loss, test_rollout_loss

def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='the directory where the .h5 files are located', type=str)
    parser.add_argument('-e', '--epochs', help='number of epochs to run', type=int, default=50)
    parser.add_argument('-lr', help='learning rate', type=float, default=2e-4)
    parser.add_argument('-batch', help='batch size', type=int, default=16)
    parser.add_argument('-train_traj', help='number of training trajectories', type=int, default=1)
    parser.add_argument('-val_traj', help='number of validation trajectories, defaults to 1', type=int, default=1)
    parser.add_argument('-test_traj', help='number of testing trajectories, defaults to 1', type=int, default=1)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-s', '--save', help='file name to save the params', type=str, default=None)
    parser.add_argument('-l', '--load', help='file name to load params from', type=str, default=None)
    parser.add_argument(
        '-images_dir', 
        help='directory to save images, or None to not save',
        type=str, 
        default=None,
    )
    parser.add_argument('-subsample', help='how many timesteps per model step, default 1', type=int, default=1)
    parser.add_argument('-downsample', help='spatial downsampling, number of times to divide by 2', type=int, default=0)
    parser.add_argument('-t', '--trials', help='number of trials to run', type=int, default=1)
    parser.add_argument('-v', '--verbose', help='verbose argument passed to trainer', type=int, default=1)

    args = parser.parse_args()

    return (
        args.filename,
        args.epochs,
        args.lr,
        args.batch,
        args.train_traj,
        args.val_traj,
        args.test_traj,
        args.seed,
        args.save,
        args.load,
        args.images_dir,
        args.subsample,
        args.downsample,
        args.trials,
        args.verbose,
    )

#Main
(
    filename, 
    epochs, 
    lr, 
    batch_size, 
    train_traj, 
    val_traj, 
    test_traj, 
    seed, 
    save_file, 
    load_file, 
    images_dir,
    subsample,
    downsample,
    trials,
    verbose,
) = handleArgs(sys.argv)

D = 2
past_steps = 4 # how many steps to look back to predict the next step
rollout_steps = 5
key = random.PRNGKey(time.time_ns()) if (seed is None) else random.PRNGKey(seed)

train_X, train_Y, val_X, val_Y, test_single_X, test_single_Y, test_rollout_X, test_rollout_Y = get_data(
    filename, 
    train_traj, 
    val_traj, 
    test_traj, 
    past_steps,
    rollout_steps,
    subsample,
    downsample,
)

group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[3], ks=[0,1,2], parities=[0,1], D=D, operators=group_actions)
upsample_filters = geom.get_invariant_filters(Ms=[2], ks=[0,1,2], parities=[0,1], D=D, operators=group_actions)

output_keys = tuple(train_Y.keys())
train_and_eval = partial(
    train_and_eval, 
    lr=lr,
    batch_size=batch_size, 
    epochs=epochs, 
    save_params=save_file, 
    images_dir=images_dir,
    verbose=verbose,
)

models = [
    (
        'do_nothing', 
        partial(
            train_and_eval, 
            net=partial(models.do_nothing, idxs={ (1,0): past_steps-1, (0,0): past_steps-1 }),
        ),
    ),
    (
        'dil_resnet',
        partial(
            train_and_eval, 
            net=partial(models.dil_resnet, depth=64, activation_f=jax.nn.gelu, output_keys=output_keys),
        ),
    ),
    (
        'dil_resnet_equiv', # test_loss is better, but rollout is worse
        partial(
            train_and_eval, 
            net=partial(
                models.dil_resnet, 
                depth=32, 
                activation_f=ml.VN_NONLINEAR, # takes more memory, hmm
                equivariant=True, 
                conv_filters=conv_filters,
                output_keys=output_keys,
            ),
        ),
    ),
    (
        'resnet',
        partial(
            train_and_eval, 
            net=partial(models.resnet, output_keys=output_keys, depth=64),
        ),   
    ),
    (
        'resnet_equiv', # better across the board
        partial(
            train_and_eval, 
            net=partial(
                models.resnet, 
                output_keys=output_keys, 
                equivariant=True, 
                conv_filters=conv_filters,
                activation_f=ml.VN_NONLINEAR,
                depth=32,
            ),
        ),  
    ),
    (
        'unet2015',
        partial(
            train_and_eval,
            net=partial(models.unet2015, output_keys=output_keys),
            has_aux=True,
        ),
    ),
    (
        'unet2015_equiv',
        partial(
            train_and_eval,
            net=partial(
                models.unet2015, 
                equivariant=True,
                conv_filters=conv_filters, 
                upsample_filters=upsample_filters,
                output_keys=output_keys,
                activation_f=ml.VN_NONLINEAR,
                depth=32, # 64=41M, 48=23M, 32=10M
            ),
        ),
    ),
    (
        'unetBase',
        partial(
            train_and_eval,
            net=partial(
                models.unetBase, 
                output_keys=output_keys, 
            ),
        ),
    ),
    (
        'unetBase_equiv', # works best
        partial(
            train_and_eval,
            net=partial(
                models.unetBase, 
                output_keys=output_keys,
                equivariant=True,
                conv_filters=conv_filters,
                activation_f=ml.VN_NONLINEAR,
                depth=32,
                upsample_filters=upsample_filters,
            ),
        ),
    ),
]

key, subkey = random.split(key)
results = ml.benchmark(
    lambda _, _2: (train_X, train_Y, val_X, val_Y, test_single_X, test_single_Y, test_rollout_X, test_rollout_Y),
    models,
    subkey,
    'Nothing',
    [0],
    num_results=4,
    num_trials=trials,
)

print(results)
