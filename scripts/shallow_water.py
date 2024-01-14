import sys
import os
import time
import argparse
import numpy as np
from functools import partial
import xarray as xr

import jax.numpy as jnp
import jax
import jax.random as random
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml
import geometricconvolutions.data as gc_data
import geometricconvolutions.models as models

def read_one_seed(data_dir: str, data_class: str, seed: str) -> tuple:
    target_file = f'{data_dir}/{data_class}/{seed}/all_data.npy'

    # net-cdf load keeps on breaking, so we try to save it as a .npy
    if len(list(filter(lambda f: f == 'all_data.npy', os.listdir(f'{data_dir}/{data_class}/{seed}/')))) > 0:
        dataset = jnp.load(target_file, allow_pickle=True).item()
        u = jax.device_put(dataset['u'], jax.devices('cpu')[0]) # velocity in x direction
        v = jax.device_put(dataset['v'], jax.devices('cpu')[0]) # velocity in y direction
        pres = jax.device_put(dataset['pres'], jax.devices('cpu')[0]) # pressure scalar
        vor = jax.device_put(dataset['vor'], jax.devices('cpu')[0]) #vorticity pseudoscalar
    else:
        datals = os.path.join(data_dir, data_class, seed, 'run*', 'output.nc')
        dataset = xr.open_mfdataset(datals, concat_dim="b", combine="nested", parallel=True) # dict
        u = jax.device_put(jnp.array(dataset['u'].to_numpy()), jax.devices('cpu')[0]) # velocity in x direction
        v = jax.device_put(jnp.array(dataset['v'].to_numpy()), jax.devices('cpu')[0]) # velocity in y direction
        pres = jax.device_put(jnp.array(dataset['pres'].to_numpy()), jax.devices('cpu')[0]) # pressure scalar
        vor = jax.device_put(jnp.array( dataset['vor'].to_numpy()), jax.devices('cpu')[0]) #vorticity pseudoscalar

        jnp.save(target_file, { 'u': u, 'v': v, 'pres': pres, 'vor': vor })

    uv = jnp.stack([u[:,:,0,...],v[:,:,0,...]], axis=-1)
    return uv, pres, vor[:,:,0,...]

def get_data_layers(
    data_dir: str, 
    num_trajectories: int, 
    data_class: str, 
    window: int, 
    velocity_form: bool = True,
    is_torus: bool = True,
) -> tuple:
    """
    Given a specified dataset, load the data into layers where the layer_X has a channel per image in the
    lookback window, and the layer_Y has just the single next image.
    args:
        data_dir (str): directory of the data
        seeds (list of str): seeds for the data
        data_class (str): type of data, either train, valid, or test
        window (int): the lookback window, how many steps we look back to predict the next one
        velocity_form (bool): whether to use the velocity/pressure of the output, or the pressure/vorticity form
        is_torus (bool): whether to create the layers on the torus
    """
    all_seeds = sorted(os.listdir(f'{data_dir}/{data_class}/'))

    spatial_dims = (96,192)
    D = 2
    all_uv = jnp.zeros((0,88) + spatial_dims + (D,))
    all_pres = jnp.zeros((0,88) + spatial_dims)
    all_vor = jnp.zeros((0,88) + spatial_dims)
    for seed in all_seeds:
        uv, pres, vor = read_one_seed(data_dir, data_class, seed)

        all_uv = jnp.concatenate([all_uv, uv])
        all_pres = jnp.concatenate([all_pres, pres])
        all_vor = jnp.concatenate([all_vor, vor])

        if len(all_uv) >= num_trajectories:
            break

    if len(all_uv) < num_trajectories:
        print(
            f'WARNING get_data_layers: wanted {num_trajectories} {data_class} trajectories, ' \
            f'but only found {len(all_uv)}',
        )
        num_trajectories = len(all_uv)

    # all_uv.shape[1] -1 because the last one is the output
    window_idx = gc_data.rolling_window_idx(all_uv.shape[1]-1, window)
    input_uv = all_uv[:num_trajectories, window_idx].reshape((-1, window) + spatial_dims + (D,))
    input_pres = all_pres[:num_trajectories, window_idx].reshape((-1, window) + spatial_dims)
    # input_vor = all_vor[:num_trajectories, window_idx].reshape((-1, window) + spatial_dims)

    output_uv = all_uv[:num_trajectories, window:].reshape((-1, 1) + spatial_dims + (D,))
    output_pres = all_pres[:num_trajectories, window:].reshape((-1, 1) + spatial_dims)
    output_vor = all_vor[:num_trajectories, window:].reshape((-1, 1) + spatial_dims)

    layer_X = geom.BatchLayer({ (0,0): input_pres, (1,0): input_uv }, D, is_torus)
    if velocity_form:
        layer_Y = geom.BatchLayer({ (0,0): output_pres, (1,0): output_uv }, D, is_torus)
    else:
        layer_Y = geom.BatchLayer({ (0,0): output_pres, (0,1): output_vor }, D, is_torus)

    return layer_X, layer_Y

def get_data(data_dir: str, num_train_traj: int, num_val_traj: int, num_test_traj: int, window: int) -> tuple:
    """
    Get train, val, and test data sets.
    args:
        data_dir (str): directory of data
        num_train_traj (int): number of training trajectories
        num_val_traj (int): number of validation trajectories
        num_test_traj (int): number of testing trajectories
        window (int): length of the lookback to predict the next step
    """
    train_X, train_Y = get_data_layers(data_dir, num_train_traj, 'train', window)
    val_X, val_Y = get_data_layers(data_dir, num_val_traj, 'valid', window)
    test_X, test_Y = get_data_layers(data_dir, num_test_traj, 'test', window)

    return train_X, train_Y, val_X, val_Y, test_X, test_Y

def map_and_loss(params, layer_x, layer_y, key, train, aux_data=None, net=None, has_aux=False):
    assert net is not None
    if has_aux:
        learned_x, batch_stats = net(params, layer_x, key, train, batch_stats=aux_data)
    else:
        learned_x = net(params, layer_x, key, train)

    spatial_size = np.multiply.reduce(geom.parse_shape(layer_x[(1,0)].shape[2:], layer_x.D)[0])
    batch_smse = jax.vmap(lambda x,y: ml.l2_squared_loss(x.to_vector(), y.to_vector())/spatial_size)

    if has_aux:
        return jnp.mean(batch_smse(learned_x, layer_y)), batch_stats
    else:
        return jnp.mean(batch_smse(learned_x, layer_y))

def train_and_eval(
    data, 
    key, 
    net, 
    model_name, 
    lr, 
    batch_size, 
    epochs, 
    save_params, 
    images_dir, 
    noise_stdev=None,
    has_aux=False,
):
    train_X, train_Y, val_X, val_Y, test_X, test_Y = data

    key, subkey = random.split(key)
    params = ml.init_params(
        net,
        train_X.get_one(),
        subkey,
    )
    print(f'Model params: {ml.count_params(params)}')

    steps_per_epoch = int(np.ceil(train_X.L / batch_size))

    key, subkey = random.split(key)
    results = ml.train(
        train_X,
        train_Y,
        partial(map_and_loss, net=net, has_aux=has_aux),
        params,
        subkey,
        stop_condition=ml.EpochStop(epochs, verbose=2),
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

    key, subkey = random.split(key)
    test_loss = ml.map_in_batches(
        partial(map_and_loss, net=net, has_aux=has_aux), 
        params, 
        test_X, 
        test_Y, 
        batch_size, 
        subkey, 
        False,
        has_aux=has_aux,
        aux_data=batch_stats,
    )
    print(f'Test Loss: {test_loss}')

    return train_loss[-1], val_loss[-1], test_loss

def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='the directory where the .h5 files are located', type=str)
    parser.add_argument('-e', '--epochs', help='number of epochs to run', type=int, default=50)
    parser.add_argument('-lr', help='learning rate', type=float, default=2e-4)
    parser.add_argument('-batch', help='batch size', type=int, default=16)
    parser.add_argument('-train_traj', help='number of training trajectories', type=int, default=100)
    parser.add_argument('-val_traj', help='number of validation trajectories, defaults to batch', type=int, default=None)
    parser.add_argument('-test_traj', help='number of testing trajectories, defaults to batch', type=int, default=None)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-s', '--save', help='file name to save the params', type=str, default=None)
    parser.add_argument('-l', '--load', help='file name to load params from', type=str, default=None)
    parser.add_argument(
        '-images_dir', 
        help='directory to save images, or None to not save',
        type=str, 
        default=None,
    )

    args = parser.parse_args()

    return (
        args.data_dir,
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
    )

#Main
args = handleArgs(sys.argv)
data_dir, epochs, lr, batch_size, train_traj, val_traj, test_traj, seed, save_file, load_file, images_dir = args

D = 2
N = 128
window = 2 # how many steps to look back to predict the next step
key = random.PRNGKey(time.time_ns()) if (seed is None) else random.PRNGKey(seed)

# an attempt to reduce recompilation, but I don't think it actually is working
if test_traj is None:
    test_traj = batch_size
if val_traj is None:
    val_traj = batch_size

train_X, train_Y, val_X, val_Y, test_X, test_Y = get_data(data_dir, train_traj, val_traj, test_traj, window)

group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[3], ks=[0,1,2], parities=[0,1], D=D, operators=group_actions)
upsample_filters = geom.get_invariant_filters(Ms=[2], ks=[0,1,2], parities=[0,1], D=D, operators=group_actions)

train_and_eval = partial(
    train_and_eval, 
    lr=lr,
    batch_size=batch_size, 
    epochs=epochs, 
    save_params=save_file, 
    images_dir=images_dir,
)

models = [
    # (
    #     'Dil-ResNet',
    #     partial(
    #         train_and_eval, 
    #         net=models.dil_resnet, 
    #         model_name='dil_resnet',
    #     ),
    # ),
    # (
    #     'Dil-ResNet Equiv',
    #     partial(
    #         train_and_eval, 
    #         net=partial(models.dil_resnet, equivariant=True, conv_filters=conv_filters),
    #         model_name='dil_resnet_equiv',
    #     ),
    # ),
    (
        'U-Net 2015',
        partial(
            train_and_eval,
            net=models.unet2015,
            model_name='unet2015',
            has_aux=True,
        ),
    ),
    (
        'U-Net 2015 equiv',
        partial(
            train_and_eval,
            net=partial(
                models.unet2015_equiv, 
                conv_filters=conv_filters, 
                upsample_filters=upsample_filters,
                depth=32, # 64=41M, 48=23M, 32=10M
            ),
            model_name='unet2015_equiv',
        ),
    ),
]

key, subkey = random.split(key)
results = ml.benchmark(
    lambda _, _2: (train_X, train_Y, val_X, val_Y, test_X, test_Y),
    models,
    subkey,
    'Nothing',
    [0],
    num_results=3,
)

print(results)
