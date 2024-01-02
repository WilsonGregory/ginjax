import sys
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

train_seeds = [
    '126288',
    '142103',
    '144307',
    '153264',
    '198010',
    '234461',
    '238834',
    '296423',
    '337393',
    '350040',
    '391455',
    '392020',
    '403221',
    '415374',
    '429773',
    '431752',
    '433143',
    '445636',
    '452245',
    '453445',
    '471651',
    '496019',
    '509578',
    '516076',
    '540888',
    '585598',
    '588419',
    '597635',
    '597861',
    '600790',
    '61749',
    '61991',
    '646044',
    '666597',
    '766153',
    '772245',
    '788727',
    '792179',
    '795128',
    '795933',
    '806439',
    '857042',
    '857466',
    '874947',
    '891593',
    '901448',
    '946191',
    '950647',
    '954413',
    '97473',
    '975533',
    '991670',
]

def get_old_data(D, data_dir, train_steps, val_steps, skip_initial):
    rho = jnp.load(data_dir + 'rho.npy')[skip_initial:] # (807-skip_initial, 64, 64)
    vel = jnp.load(data_dir + 'vel.npy')[skip_initial:] # (807-skip_initial, 64, 64, 2)

    layer = geom.BatchLayer(
        {
            (0,0): jnp.expand_dims(rho, 1),
            (1,0): jnp.expand_dims(vel, 1),
        },
        D,
        True,
    )

    train_X = layer.get_subset(jnp.arange(train_steps-1))
    # train_Y is the delta from train_X to the next step
    train_Y = geom.BatchLayer.from_vector(
        layer.get_subset(jnp.arange(1,train_steps)).to_vector() - train_X.to_vector(),
        train_X,
    )

    val_X = layer.get_subset(jnp.arange(train_steps, train_steps+val_steps-1))
    val_Y = geom.BatchLayer.from_vector(
        layer.get_subset(jnp.arange(train_steps+1,train_steps+val_steps)).to_vector() - val_X.to_vector(),
        val_X,
    )

    test_X = layer.get_subset(jnp.arange(train_steps+val_steps,len(rho)-1))
    test_Y = geom.BatchLayer.from_vector(
        layer.get_subset(jnp.arange(train_steps+val_steps+1,len(rho))).to_vector() - test_X.to_vector(),
        test_X,
    )

    return train_X, train_Y, val_X, val_Y, test_X, test_Y

def read_one_h5(filename, data_class):
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

def merge_h5s_into_layer(D: int, data_dir: str, seeds: list, data_class: str, window: int) -> geom.BatchLayer:
    """
    args:
        window (int): the lookback window, how many steps we look back to predict the next one
    """
    N = 128
    all_u = jnp.zeros((0,14,N,N))
    all_vxy = jnp.zeros((0,14,N,N,2))
    for seed in seeds:
        if data_class == 'train':
            filename = f'{data_dir}NavierStokes2D_{data_class}_{seed}_0.50000_100.h5'
        else:
            filename = f'{data_dir}NavierStokes2D_{data_class}_{seed}_0.50000.h5'

        u, vxy = read_one_h5(filename, data_class)

        all_u = jnp.concatenate([all_u, u])
        all_vxy = jnp.concatenate([all_vxy, vxy])

    # all_u.shape[1] -1 because the last one is the output
    window_idx = utils.rolling_window_idx(all_u.shape[1]-1, window)
    input_u = all_u[:,window_idx].reshape((-1, window, N, N))
    input_vxy = all_vxy[:,window_idx].reshape((-1, window, N, N, 2))

    output_u = all_u[:,window:].reshape(-1, 1, N, N)
    output_vxy = all_vxy[:,window:].reshape(-1, 1, N, N, 2)

    layer_X = geom.BatchLayer({ (0,0): input_u, (1,0): input_vxy }, D, False)
    layer_Y = geom.BatchLayer({ (0,0): output_u, (1,0): output_vxy }, D, False)

    return layer_X, layer_Y

def get_data(D: int, data_dir: str, window: int) -> tuple:
    train_X, train_Y = merge_h5s_into_layer(D, data_dir, train_seeds[:1], 'train', window)
    val_X, val_Y = merge_h5s_into_layer(D, data_dir, ['126388'], 'valid', window)
    test_X, test_Y = merge_h5s_into_layer(D, data_dir, ['126488'], 'test', window)

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
                layer = ml.scalar_activation(layer, jax.nn.tanh)
            else:
                layer = ml.batch_relu_layer(layer)

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

def map_and_loss(params, layer_x, layer_y, key, train, net):
    learned_x = net(params, layer_x, key, train)
    spatial_size = np.multiply.reduce(geom.parse_shape(layer_x[(1,0)].shape[2:], layer_x.D)[0])
    batch_smse = jax.vmap(lambda x,y: ml.l2_squared_loss(x.to_vector(), y.to_vector())/spatial_size)
    return jnp.mean(batch_smse(learned_x, layer_y))

def train_and_eval(data, key, net, model_name, lr, batch_size, epochs, max_rollout, save_params, images_dir, noise_stdev=None):
    train_X, train_Y, val_X, val_Y, test_X, test_Y = data

    key, subkey = random.split(key)
    params = ml.init_params(
        net,
        train_X.get_subset(jnp.arange(batch_size)), # so we don't have to recompile
        subkey,
    )
    print(f'Model params: {ml.count_params(params)}')

    steps_per_epoch = int(np.ceil(train_X.L / batch_size))

    key, subkey = random.split(key)
    params, train_loss, val_loss = ml.train(
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
            optax.warmup_cosine_decay_schedule(1e-8, lr, 5*steps_per_epoch, 30*steps_per_epoch, 1e-8),
            weight_decay=1e-5,
        ),
        validation_X=val_X,
        validation_Y=val_Y,
        noise_stdev=noise_stdev,
    )

    if save_params is not None:
        jnp.save(save_params + model_name + '_params.npy', params)

    test_img = test_X.get_one()
    if (max_rollout is None) or (max_rollout > test_X.L - 1):
        max_rollout = test_X.L - 1

    i = 0
    err = 0
    while ((i < max_rollout) and (i == 0 or err < 100)):
        key, subkey = random.split(key)
        delta_img = net(params, test_img, subkey, False)

        # advance the rollout from the current image, plus the calculated delta.
        # The actual image we compare it against is the true next step.
        test_img = geom.BatchLayer.from_vector(test_img.to_vector() + delta_img.to_vector(), test_img)
        actual_img = test_X.get_one(i+1)

        err = ml.rmse_loss(test_img.to_vector(), actual_img.to_vector())
        print(f'Rollout Loss step {i}: {err}')
        if ((images_dir is not None) and err < 1):
            plot_layer(test_img, actual_img, f'{images_dir}{model_name}_err_{i}.png')
            imax = i

        i += 1

    if images_dir is not None:
        with imageio.get_writer(f'{images_dir}{model_name}_error.gif', mode='I') as writer:
            for j in range(imax+1):
                image = imageio.imread(f'{images_dir}{model_name}_err_{j}.png')
                writer.append_data(image)

    return train_loss[-1], val_loss[-1], i

def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', help='number of epochs to run', type=int, default=50)
    parser.add_argument('-lr', help='learning rate', type=float, default=3e-4)
    parser.add_argument('-batch', help='batch size', type=int, default=16)
    parser.add_argument('-train_steps', help='number of images to train on', type=int, default=100)
    parser.add_argument('-val_steps', help='number of steps in the validation set', type=int, default=50)
    parser.add_argument('-skip_initial', help='number of steps in the beginning to skip', type=int, default=0)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-s', '--save', help='file name to save the params', type=str, default=None)
    parser.add_argument('-l', '--load', help='file name to load params from', type=str, default=None)
    parser.add_argument('-noise', help='whether to add gaussian noise', default=False, action='store_true')
    parser.add_argument('-max_rollout', help='the max number of rollout steps', type=int, default=None)
    parser.add_argument(
        '-images_dir', 
        help='directory to save images, or None to not save',
        type=str, 
        default=None,
    )

    args = parser.parse_args()

    return (
        args.epochs,
        args.lr,
        args.batch,
        args.train_steps,
        args.val_steps,
        args.skip_initial,
        args.seed,
        args.save,
        args.load,
        args.noise,
        args.max_rollout,
        args.images_dir,
    )

#Main
args = handleArgs(sys.argv)
epochs, lr, batch_size, train_steps, val_steps, skip_initial, seed, save_file, load_file, noise, max_rollout, images_dir = args

D = 2
N = 128
window = 4 # how many steps to look back to predict the next step
key = random.PRNGKey(time.time_ns()) if (seed is None) else random.PRNGKey(seed)

data_dir = '../NavierStokes-2D/'
train_X, train_Y, val_X, val_Y, test_X, test_Y = get_data(D, data_dir, window)

group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[3], ks=[0,1,2], parities=[0,1], D=D, operators=group_actions)

models = [
    # (
    #     'GI-Net',
    #     partial(
    #         train_and_eval, 
    #         net=partial(gi_net, conv_filters=conv_filters, depth=28), 
    #         lr=lr, 
    #         batch_size=batch_size, 
    #         epochs=epochs, 
    #         max_rollout=max_rollout, 
    #         images_dir=images_dir,
    #     ),
    # ),
    # (
    #     'GI-Net Odd noise',
    #     partial(
    #         train_and_eval, 
    #         net=partial(gi_net, conv_filters=conv_filters, depth=10, use_odd_parity=True),
    #         lr=3e-4, 
    #         batch_size=batch_size, 
    #         epochs=epochs, 
    #         max_rollout=max_rollout, 
    #         save_params=None if save_file is None else save_file + 'gi_net_odd_noise_',
    #         images_dir=None if images_dir is None else images_dir + 'gi_net_odd_noise_',
    #         noise_stdev=0.01,
    #     ),
    # ),
    # (
    #     'GI-Net Odd no noise',
    #     partial(
    #         train_and_eval, 
    #         net=partial(gi_net, conv_filters=conv_filters, depth=10, use_odd_parity=True),
    #         lr=1e-3, 
    #         batch_size=batch_size, 
    #         epochs=epochs, 
    #         max_rollout=max_rollout, 
    #         images_dir=images_dir,
    #     ),
    # ),
    (
        'Dil-ResNet',
        partial(
            train_and_eval, 
            net=dil_resnet, 
            model_name='dil_resnet',
            lr=1e-4, 
            batch_size=batch_size, 
            epochs=epochs, 
            max_rollout=max_rollout, 
            save_params=save_file,
            images_dir=images_dir,
            # noise_stdev=0.01,
        ),
    ),
]

key, subkey = random.split(key)
results = ml.benchmark(
    lambda _, _2: (train_X, train_Y, val_X, val_Y, test_X, test_Y),
    models,
    subkey,
    'Train Steps',
    [train_steps],
    num_results=3,
)

