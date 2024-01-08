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
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

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
    u = jnp.array(data_dict['u'][()])
    vx = jnp.array(data_dict['vx'][()])
    vy = jnp.array(data_dict['vy'][()])
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

@partial(jax.jit, static_argnums=[3,5,6,7])
def gi_net(params, layer, key, train, conv_filters, depth, use_odd_parity=False, return_params=False):
    assert layer.D == 2
    num_blocks = 4

    target_keys = ((0,0),(0,1),(1,0),(1,1)) if use_odd_parity else ((0,0),(1,0))

    # encoder
    layer, params = ml.batch_conv_contract(
        params, 
        layer, 
        conv_filters, 
        depth, 
        target_keys, 
        mold_params=return_params,
    )

    if use_odd_parity: # get the residual to line up...
        layer, params = ml.batch_conv_contract(
            params, 
            layer, 
            conv_filters, 
            depth, 
            target_keys, 
            mold_params=return_params,
        )

    for _ in range(num_blocks):

        residual_layer = layer.copy()
        # dCNN block
        for dilation in [1,2,4,8,4,2,1]:
            layer, params = ml.batch_conv_contract(
                params, 
                layer, 
                conv_filters, 
                depth, 
                target_keys, 
                mold_params=return_params,
                rhs_dilation=(dilation,)*D,
            )
            if use_odd_parity:
                layer = ml.batch_scalar_activation(layer, jax.nn.tanh)
            else:
                layer = ml.batch_scalar_activation(layer, jax.nn.relu)

        layer = geom.BatchLayer.from_vector(layer.to_vector() + residual_layer.to_vector(), layer)

    # decoder
    layer, params = ml.batch_conv_contract(
        params, 
        layer, 
        conv_filters, 
        1, 
        ((0,0),(1,0)), #output is always the same
        mold_params=return_params,
    )

    layer, params = ml.batch_channel_collapse(params, layer, mold_params=return_params)

    return (layer, params) if return_params else layer

@partial(jax.jit, static_argnums=[3,4])
def dil_resnet(params, layer, key, train, return_params=False):
    assert layer.D == 2
    depth = 48
    num_blocks = 4

    spatial_dims,_ = geom.parse_shape(layer[(1,0)].shape[2:], layer.D)
    layer = geom.BatchLayer(
        {
            (0,0): jnp.concatenate([
                layer[(0,0)],
                layer[(1,0)].transpose((0,1,4,2,3)).reshape((layer.L,-1) + spatial_dims),
            ], axis=1),
        },
        layer.D,
        layer.is_torus,
    )

    # encoder
    layer, params = ml.batch_conv_layer(
        params, 
        layer,
        { 'type': 'free', 'M': 3, 'filter_key_set': { (0,0) } },
        depth, 
        bias=True,
        mold_params=return_params,
    )

    for _ in range(num_blocks):

        residual_layer = layer.copy()
        # dCNN block
        for dilation in [1,2,4,8,4,2,1]:
            layer, params = ml.batch_conv_layer(
                params, 
                layer,
                { 'type': 'free', 'M': 3, 'filter_key_set': { (0,0) } },
                depth, 
                bias=True,
                mold_params=return_params,
                rhs_dilation=(dilation,)*D,
            )
            layer = ml.batch_relu_layer(layer)

        layer = geom.BatchLayer.from_vector(layer.to_vector() + residual_layer.to_vector(), layer)

    # decoder
    layer, params = ml.batch_conv_layer(
        params, 
        layer,
        { 'type': 'free', 'M': 3, 'filter_key_set': { (0,0) } },
        3, 
        bias=True,
        mold_params=return_params,
    )

    # swap the vector back to the vector img
    layer = geom.BatchLayer(
        {
            (0,0): jnp.expand_dims(layer[(0,0)][:,0], axis=1),
            (1,0): jnp.expand_dims(layer[(0,0)][:,1:].transpose((0,2,3,1)), axis=1),
        },
        layer.D,
        layer.is_torus,
    )

    return (layer, params) if return_params else layer

def map_and_loss(params, layer_x, layer_y, key, train, net, aux_data=None):
    learned_x, batch_stats = net(params, layer_x, key, train, batch_stats=aux_data)
    # return learned_x[(0,0)][0,0,0,0], None

    # jax.debug.visualize_array_sharding(learned_x[(0,0)][:,:,0,0])
    # jax.debug.visualize_array_sharding(layer_y[(0,0)][:,:,0,0])
    # print(batch_stats)

    # return jnp.mean(learned_x[(1,0)] - layer_y[(1,0)]), None

    spatial_size = np.multiply.reduce(geom.parse_shape(layer_x[(1,0)].shape[2:], layer_x.D)[0])
    batch_smse = jax.vmap(lambda x,y: ml.l2_squared_loss(x.to_vector(), y.to_vector())/spatial_size)
    return jnp.mean(batch_smse(learned_x, layer_y)), batch_stats

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
    sharding=None,
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

    # # X_batches, Y_batches = ml.get_batch_layer(train_X, train_Y, batch_size, subkey, sharding)
    # X_batch = train_X.get_subset(jnp.arange(batch_size)).device_put(sharding, num_gpus).get_subset(jnp.arange(batch_size))
    # Y_batch = train_Y.get_subset(jnp.arange(batch_size)).device_put(sharding, num_gpus).get_subset(jnp.arange(batch_size))
    # params = jax.device_put(params, sharding.replicate())
    # # ml.print_params(params)
    # # jax.debug.visualize_array_sharding(X_batches[0][(0,0)][:,:,0,0])
    # # jax.debug.visualize_array_sharding(Y_batches[0][(0,0)][:,:,0,0])
    # # jax.debug.visualize_array_sharding(params[(0, 'conv')]['free'][(0,0)][(0,0)][:,:,0,0])
    # # jax.debug.visualize_array_sharding(params)
    # (loss_val, _), _2 = jax.value_and_grad(map_and_loss, has_aux=has_aux)(
    #     params, 
    #     X_batch,
    #     Y_batch,
    #     subkey, 
    #     True, 
    #     net,
    # )
    # print(loss_val)
    # exit()

    key, subkey = random.split(key)
    results = ml.train(
        train_X,
        train_Y,
        partial(map_and_loss, net=net),
        params,
        subkey,
        stop_condition=ml.EpochStop(epochs, verbose=2),
        batch_size=batch_size,
        # optimizer=optax.adam(lr),
        # optimizer=optax.adamw(lr, weight_decay=1e-5),
        optimizer=optax.adamw(
            optax.warmup_cosine_decay_schedule(1e-8, lr, 5*steps_per_epoch, epochs*steps_per_epoch, 1e-7),
            weight_decay=1e-5,
        ),
        validation_X=val_X,
        validation_Y=val_Y,
        noise_stdev=noise_stdev,
        has_aux=has_aux,
        sharding=sharding,
    )

    if has_aux:
        params, batch_stats, train_loss, val_loss = results
        trained_net = partial(net, batch_stats=batch_stats)
    else:
        params, train_loss, val_loss = results
        batch_stats = None
        trained_net = net

    if save_params is not None:
        jnp.save(
            save_params + model_name + '_params.npy', 
            { 'params': params, 'batch_stats': None if (batch_stats is None) else dict(batch_stats) },
        )

    key, subkey = random.split(key)
    test_loss = ml.map_in_batches(
        partial(map_and_loss, net=trained_net), 
        params, 
        test_X, 
        test_Y, 
        batch_size, 
        subkey, 
        False,
        has_aux,
        batch_stats,
        sharding,
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
    parser.add_argument('-val_traj', help='number of validation trajectories', type=int, default=25)
    parser.add_argument('-test_traj', help='number of testing trajectories', type=int, default=25)
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

train_X, train_Y, val_X, val_Y, test_X, test_Y = get_data(data_dir, train_traj, val_traj, test_traj, window)

group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[3], ks=[0,1,2], parities=[0,1], D=D, operators=group_actions)

# This section below demonstrates how to shard across multiple GPUs. 
# Create a Sharding object to distribute a value across devices:
num_gpus = jax.device_count()
print('Num gpus: ', num_gpus)
sharding = PositionalSharding(mesh_utils.create_device_mesh((num_gpus,)))
conv_filters = conv_filters.device_replicate(sharding)

train_and_eval = partial(
    train_and_eval, 
    lr=lr,
    batch_size=batch_size, 
    epochs=epochs, 
    save_params=save_file, 
    images_dir=images_dir,
    sharding=sharding,
)

models = [
    # (
    #     'GI-Net',
    #     partial(
    #         train_and_eval, 
    #         net=partial(gi_net, conv_filters=conv_filters, depth=10), 
    #         model_name='gi_net',
    #     ),
    # ),
    # (
    #     'GI-Net Odd',
    #     partial(
    #         train_and_eval, 
    #         net=partial(gi_net, conv_filters=conv_filters, depth=10, use_odd_parity=True),
    #         model_name='gi_net_odd',
    #     ),
    # ),
    # (
    #     'Dil-ResNet',
    #     partial(
    #         train_and_eval, 
    #         net=dil_resnet, 
    #         model_name='dil_resnet',
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
    )
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

