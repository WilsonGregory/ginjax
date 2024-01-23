import sys
import os
import time
import argparse
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import imageio.v2 as imageio
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

def read_one_h5(filename: str, data_class: str) -> tuple:
    """
    Given a filename and a type of data (train, test, or validation), read the data and return as jax arrays.
    args:
        filename (str): the full file path
        data_class (str): either 'train', 'test', or 'valid'
    returns: u, vxy as jax arrays
    """
    file = h5py.File(filename)
    data_dict = file[data_class]

    # all of these are shape (num_trajectories, t, x, y) = (100, 14, 128, 128)
    u = jax.device_put(jnp.array(data_dict['u'][()]), jax.devices('cpu')[0])
    vx = jax.device_put(jnp.array(data_dict['vx'][()]), jax.devices('cpu')[0])
    vy = jax.device_put(jnp.array(data_dict['vy'][()]), jax.devices('cpu')[0])
    vxy = jnp.stack([vx, vy], axis=-1)

    file.close()

    return u, vxy

def merge_h5s_into_layer(data_dir: str, num_trajectories: int, data_class: str, window: int) -> tuple:
    """
    Given a specified dataset, load the data into layers where the layer_X has a channel per image in the
    lookback window, and the layer_Y has just the single next image.
    args:
        data_dir (str): directory of the data
        seeds (list of str): seeds for the data
        data_class (str): type of data, either train, valid, or test
        window (int): the lookback window, how many steps we look back to predict the next one
    """
    all_files = sorted(filter(lambda file: f'NavierStokes2D_{data_class}' in file, os.listdir(data_dir)))

    N = 128
    D = 2
    all_u = jnp.zeros((0,14,N,N))
    all_vxy = jnp.zeros((0,14,N,N,D))
    for filename in all_files:
        u, vxy = read_one_h5(f'{data_dir}/{filename}', data_class)

        all_u = jnp.concatenate([all_u, u])
        all_vxy = jnp.concatenate([all_vxy, vxy])

        if len(all_u) >= num_trajectories:
            break

    if len(all_u) < num_trajectories:
        print(
            f'WARNING merge_h5s_into_layer: wanted {num_trajectories} {data_class} trajectories, ' \
            f'but only found {len(all_u)}',
        )
        num_trajectories = len(all_u)

    # all_u.shape[1] -1 because the last one is the output
    window_idx = gc_data.rolling_window_idx(all_u.shape[1]-1, window)
    input_u = all_u[:num_trajectories, window_idx].reshape((-1, window, N, N))
    input_vxy = all_vxy[:num_trajectories, window_idx].reshape((-1, window, N, N, D))

    output_u = all_u[:num_trajectories, window:].reshape(-1, 1, N, N)
    output_vxy = all_vxy[:num_trajectories, window:].reshape(-1, 1, N, N, D)

    layer_X = geom.BatchLayer({ (0,0): input_u, (1,0): input_vxy }, D, False)
    layer_Y = geom.BatchLayer({ (0,0): output_u, (1,0): output_vxy }, D, False)

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
    train_X, train_Y = merge_h5s_into_layer(data_dir, num_train_traj, 'train', window)
    val_X, val_Y = merge_h5s_into_layer(data_dir, num_val_traj, 'valid', window)
    test_X, test_Y = merge_h5s_into_layer(data_dir, num_test_traj, 'test', window)

    return train_X, train_Y, val_X, val_Y, test_X, test_Y

def plot_layer(test_layer, actual_layer, save_loc):
    test_rho_img = geom.GeometricImage(test_layer[(0,0)][0,0], 0, test_layer.D, test_layer.is_torus)
    test_vel_img = geom.GeometricImage(test_layer[(1,0)][0,0], 0, test_layer.D, test_layer.is_torus)

    actual_rho_img = geom.GeometricImage(actual_layer[(0,0)][0,0], 0, actual_layer.D, actual_layer.is_torus)
    actual_vel_img = geom.GeometricImage(actual_layer[(1,0)][0,0], 0, actual_layer.D, actual_layer.is_torus)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24,12)) # 8 per col, 6 per row, (cols,rows)
    utils.plot_image(test_rho_img, ax=axes[0,0], title='Predicted Rho', colorbar=True)
    utils.plot_image(actual_rho_img, ax=axes[0,1], title='Actual Rho', colorbar=True)
    utils.plot_image(actual_rho_img - test_rho_img, ax=axes[0,2], title='Difference', colorbar=True)

    utils.plot_image(test_vel_img, ax=axes[1,0], title='Predicted Velocity')
    utils.plot_image(actual_vel_img, ax=axes[1,1], title='Actual Velocity')
    utils.plot_image(actual_vel_img - test_vel_img, ax=axes[1,2], title='Difference')

    plt.savefig(save_loc)
    plt.close(fig)

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
    model_name, 
    net, 
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
            f'{save_params}{model_name}_trajectories{train_X.L // 10}_e{epochs}_params.npy', 
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

    # keeping this for now
    # i = 0
    # err = 0
    # while (err < 100):
    #     key, subkey = random.split(key)
    #     delta_img = net(params, test_img, subkey, False)

    #     # advance the rollout from the current image, plus the calculated delta.
    #     # The actual image we compare it against is the true next step.
    #     test_img = geom.BatchLayer.from_vector(test_img.to_vector() + delta_img.to_vector(), test_img)
    #     actual_img = test_X.get_one(i+1)

    #     err = ml.rmse_loss(test_img.to_vector(), actual_img.to_vector())
    #     print(f'Rollout Loss step {i}: {err}')
    #     if ((images_dir is not None) and err < 1):
    #         plot_layer(test_img, actual_img, f'{images_dir}{model_name}_err_{i}.png')
    #         imax = i

    #     i += 1

    # if images_dir is not None:
    #     with imageio.get_writer(f'{images_dir}{model_name}_error.gif', mode='I') as writer:
    #         for j in range(imax+1):
    #             image = imageio.imread(f'{images_dir}{model_name}_err_{j}.png')
    #             writer.append_data(image)

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
window = 4 # how many steps to look back to predict the next step
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
    (
        'dil_resnet',
        partial(
            train_and_eval, 
            net=models.dil_resnet, 
        ),
    ),
    (
        'dil_resnet_equiv',
        partial(
            train_and_eval, 
            net=partial(models.dil_resnet, equivariant=True, conv_filters=conv_filters),
        ),
    ),
    (
        'unet2015',
        partial(
            train_and_eval,
            net=models.unet2015,
            has_aux=True,
        ),
    ),
    (
        'unet2015_equiv',
        partial(
            train_and_eval,
            net=partial(
                models.unet2015_equiv, 
                conv_filters=conv_filters, 
                upsample_filters=upsample_filters,
                depth=32, # 64=41M, 48=23M, 32=10M
            ),
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

