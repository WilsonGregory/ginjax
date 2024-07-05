import sys
import time
import argparse
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import h5py

import jax.numpy as jnp
import jax
import jax.random as random
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml
import geometricconvolutions.utils as utils
import geometricconvolutions.data as gc_data
import geometricconvolutions.models as models

def read_one_h5(filename: str, num_trajectories: int) -> tuple:
    """
    Given a filename and a type of data (train, test, or validation), read the data and return as jax arrays.
    args:
        filename (str): the full file path
        data_class (str): either 'train', 'test', or 'valid'
    returns: u, vxy as jax arrays
    """
    data_dict = h5py.File(filename)

    # all of these are shape (num_trajectories, t, x, y) = (10K, 21, 128, 128)
    density = jax.device_put(jnp.array(data_dict['density'][:num_trajectories][()]), jax.devices('cpu')[0])
    pressure = jax.device_put(jnp.array(data_dict['pressure'][:num_trajectories][()]), jax.devices('cpu')[0])
    vx = jax.device_put(jnp.array(data_dict['Vx'][:num_trajectories][()]), jax.devices('cpu')[0])
    vy = jax.device_put(jnp.array(data_dict['Vy'][:num_trajectories][()]), jax.devices('cpu')[0])
    vxy = jnp.stack([vx, vy], axis=-1)

    data_dict.close()

    return density, pressure, vxy

def get_data(
    D: int, 
    filename: str, 
    n_train: int,
    n_val: int, 
    n_test: int,
    past_steps: int, 
    rollout_steps: int, 
    normalize: bool = True,
):
    density, pressure, velocity = read_one_h5(filename, n_train + n_val + n_test)

    if normalize:
        density = (density - jnp.mean(density[:(n_train + n_val)])) / jnp.std(density[:(n_train + n_val)])
        pressure = (pressure - jnp.mean(pressure[:(n_train + n_val)])) / jnp.std(pressure[:(n_train + n_val)])
        velocity = velocity / jnp.std(velocity[:(n_train + n_val)]) # this one I am not so sure about

    start = 0
    stop = n_train
    train_X, train_Y = gc_data.times_series_to_layers(
        D, 
        { (0,0): density[start:stop], (1,0): velocity[start:stop] }, 
        {}, 
        True, 
        past_steps, 
        1,
    )
    train_pressure_X, train_pressure_Y = gc_data.times_series_to_layers(
        D, 
        { (0,0): pressure[start:stop] }, 
        {},
        True, 
        past_steps, 
        1,
    )
    train_X = train_X.concat(train_pressure_X, axis=1)
    train_Y = train_Y.concat(train_pressure_Y, axis=1)

    start = start + n_train
    stop = start + n_val
    val_X, val_Y = gc_data.times_series_to_layers(
        D, 
        { (0,0): density[start:stop], (1,0): velocity[start:stop] }, 
        {}, 
        True, 
        past_steps, 
        1,
    )
    val_pressure_X, val_pressure_Y = gc_data.times_series_to_layers(
        D, 
        { (0,0): pressure[start:stop] }, 
        {},
        True, 
        past_steps, 
        1,
    )
    val_X = val_X.concat(val_pressure_X, axis=1)
    val_Y = val_Y.concat(val_pressure_Y, axis=1)

    start = start + n_val
    stop = start + n_test
    test_X, test_Y = gc_data.times_series_to_layers(
        D, 
        { (0,0): density[start:stop], (1,0): velocity[start:stop] }, 
        {}, 
        True, 
        past_steps, 
        1,
    )
    test_pressure_X, test_pressure_Y = gc_data.times_series_to_layers(
        D, 
        { (0,0): pressure[start:stop] }, 
        {},
        True, 
        past_steps, 
        1,
    )
    test_X = test_X.concat(test_pressure_X, axis=1)
    test_Y = test_Y.concat(test_pressure_Y, axis=1)

    test_rollout_X, test_rollout_Y = gc_data.times_series_to_layers(
        D, 
        { (0,0): density[start:stop], (1,0): velocity[start:stop] }, 
        {}, 
        True, 
        past_steps, 
        rollout_steps,
    )
    test_rollout_pressure_X, test_rollout_pressure_Y = gc_data.times_series_to_layers(
        D, 
        { (0,0): pressure[start:stop] }, 
        {},
        True, 
        past_steps, 
        rollout_steps,
    )
    test_rollout_X = test_rollout_X.concat(test_rollout_pressure_X, axis=1)
    test_rollout_Y = test_rollout_Y.concat(test_rollout_pressure_Y, axis=1)

    return train_X, train_Y, val_X, val_Y, test_X, test_Y, test_rollout_X, test_rollout_Y

def plot_layer(test_layer, actual_layer, save_loc, future_steps):
    # scale the images so that they range from -1 to 1
    data = jnp.concatenate([test_layer[(1,0)][0,...,0].reshape(-1), actual_layer[(1,0)][0,...,0].reshape(-1)])
    max_val = jnp.max(jnp.abs(data))
    test_layer = test_layer / max_val
    actual_layer = actual_layer / max_val

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

    past_steps = layer_x[(1,0)].shape[1]
    out_layer = layer_y.empty()
    for _ in range(future_steps):
        key, subkey = random.split(key)
        if has_aux:
            learned_x, aux_data = net(params, layer_x, subkey, train, batch_stats=aux_data)
        else:
            learned_x = net(params, layer_x, subkey, train)

        layer_x, out_layer = ml.autoregressive_step(layer_x, learned_x, out_layer, past_steps)

    loss = ml.smse_loss(out_layer, layer_y)
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
    load_params,
    images_dir, 
    noise_stdev=None,
    has_aux=False,
    verbose=1,
):
    train_X, train_Y, val_X, val_Y, test_single_X, test_single_Y, test_rollout_X, test_rollout_Y = data

    key, subkey = random.split(key)
    params = ml.init_params(
        net,
        train_X.get_one(),
        subkey,
    )
    print(f'Model params: {ml.count_params(params):,}')

    if load_params is None:
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
                optax.warmup_cosine_decay_schedule(1e-8, lr, 5*steps_per_epoch, epochs*steps_per_epoch, 1e-7),
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
                f'{save_params}{model_name}_L{train_X.L}_e{epochs}_params.npy', 
                { 'params': params, 'batch_stats': None if (batch_stats is None) else dict(batch_stats) },
            )
    else:
        dict = jnp.load(
            f'{load_params}{model_name}_L{train_X.L}_e{epochs}_params.npy', 
            allow_pickle=True,
        ).item()
        params = dict['params']
        batch_stats = dict['batch_stats']

        # so it doesn't break
        train_loss = [0]
        val_loss = [0]


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
        plot_layer(
            rollout_layer.get_one(), 
            test_rollout_Y.get_one(), 
            f'{images_dir}/{model_name}_L{train_X.L}_e{epochs}_rollout.png',
            5,
        )

    return train_loss[-1], val_loss[-1], test_loss, test_rollout_loss

def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='the data .hdf5 file', type=str)
    parser.add_argument('-e', '--epochs', help='number of epochs to run', type=int, default=50)
    parser.add_argument('-batch', help='batch size', type=int, default=8)
    parser.add_argument('-n_train', help='number of training trajectories', type=int, default=100)
    parser.add_argument('-n_val', help='number of validation trajectories, defaults to batch', type=int, default=None)
    parser.add_argument('-n_test', help='number of testing trajectories, defaults to batch', type=int, default=None)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-s', '--save', help='file name to save the params', type=str, default=None)
    parser.add_argument('-l', '--load', help='file name to load params from', type=str, default=None)
    parser.add_argument(
        '-images_dir', 
        help='directory to save images, or None to not save',
        type=str, 
        default=None,
    )
    parser.add_argument('-v', '--verbose', help='verbose argument passed to trainer', type=int, default=1)

    args = parser.parse_args()

    return (
        args.data,
        args.epochs,
        args.batch,
        args.n_train,
        args.n_val,
        args.n_test,
        args.seed,
        args.save,
        args.load,
        args.images_dir,
        args.verbose,
    )

#Main
args = handleArgs(sys.argv)
data_path, epochs, batch_size, n_train, n_val, n_test, seed, save_file, load_file, images_dir, verbose = args

D = 2
N = 128
output_keys = ((0,0), (1,0))
output_depth = ( ((0,0),2), ((1,0),1) )

past_steps = 4 # how many steps to look back to predict the next step
rollout_steps = 5
key = random.PRNGKey(time.time_ns()) if (seed is None) else random.PRNGKey(seed)

# an attempt to reduce recompilation, but I don't think it actually is working
if n_test is None:
    n_test = batch_size
if n_val is None:
    n_val = batch_size

data = get_data(D, data_path, n_train, n_val, n_test, past_steps, rollout_steps)

group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[3], ks=[0,1,2], parities=[0,1], D=D, operators=group_actions)
upsample_filters = geom.get_invariant_filters(Ms=[2], ks=[0,1,2], parities=[0,1], D=D, operators=group_actions)

train_and_eval = partial(
    train_and_eval, 
    batch_size=batch_size, 
    epochs=epochs, 
    save_params=save_file, 
    load_params=load_file,
    images_dir=images_dir,
    verbose=verbose,
)

model_list = [
    (
        'unetBase',
        partial(
            train_and_eval,
            net=partial(
                models.unetBase, 
                output_keys=output_keys, 
                output_depth=output_depth,
            ),
            lr=8e-4,
        ),
    ),
    (
        'unetBase_equiv64', # works best
        partial(
            train_and_eval,
            net=partial(
                models.unetBase, 
                output_keys=output_keys,
                output_depth=output_depth,
                equivariant=True,
                conv_filters=conv_filters,
                activation_f=ml.VN_NONLINEAR,
                use_group_norm=False,
                upsample_filters=upsample_filters,
            ),
            lr=3e-4, 
        ),
    ),
    (
        'unetBase_equiv48', # works best
        partial(
            train_and_eval,
            net=partial(
                models.unetBase, 
                output_keys=output_keys,
                output_depth=output_depth,
                depth=48,
                equivariant=True,
                conv_filters=conv_filters,
                activation_f=ml.VN_NONLINEAR,
                use_group_norm=False,
                upsample_filters=upsample_filters,
            ),
            lr=5e-4, 
        ),
    ),
]

key, subkey = random.split(key)

# # Use this for benchmarking over different learning rates
# results = ml.benchmark(
#     lambda _: data,
#     model_list,
#     subkey,
#     'lr',
#     [4e-4, 5e-4, 6e-4],
#     benchmark_type=ml.BENCHMARK_MODEL,
#     num_trials=3,
#     num_results=4,
# )

# Use this for benchmarking the models with known learning rates.
results = ml.benchmark(
    lambda _: data,
    model_list,
    subkey,
    '',
    [0],
    benchmark_type=ml.BENCHMARK_NONE,
    num_trials=3,
    num_results=4,
)

print(results)
print('Mean', jnp.mean(results, axis=0), sep='\n')
