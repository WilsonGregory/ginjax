import sys
import os
import time
import argparse
import numpy as np
from functools import partial
import xarray as xr
from typing_extensions import Optional, Union

import jax.numpy as jnp
import jax
import jax.random as random
from jaxtyping import ArrayLike
import optax
import equinox as eqx

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml
import geometricconvolutions.data as gc_data
import geometricconvolutions.models as models


def read_orography(
    D: int, is_torus: Union[bool, tuple[bool, ...]], root_dir: str, file: str = "orographyT63.nc"
) -> geom.MultiImage:
    """
    Read the orography data from the file and construct a multi image from it.

    args:
        D: dimension of the space
        is_torus: toroidal structure of the image
        root_dir: root directory of the orography file
        file: filename of the orography file

    returns:
        A multi image of the orography data, shape (1,spatial).
    """
    orography = xr.open_mfdataset(root_dir + "/" + file)
    # loads as (96,192), swap to (192,96)
    orography_arr = jax.device_put(jnp.array(orography["orog"].to_numpy()), jax.devices("cpu")[0]).T
    return geom.MultiImage({(0, 0): orography_arr.reshape((1, 192, 96))}, D, is_torus)


def read_one_seed(
    data_dir: str, data_class: str, seed: str
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Read all the runs of one seed and combine them into blocks of data. The runs from the directory
    data_dir/data_class/seed/ will be read and made into uv, pres, vor, and lat data blocks. If
    there is an `all_data.npy` file in that directory, that is loaded. Otherwise, the separate
    mfd_datasets are loaded, then saved to an `all_data.npy` file. This will allow the data to be
    loaded much quicker on the next run.

    Note that the data is stored as (lat,lon) or (96,192), but we swap it to (lon,lat) or (192,96)
    because multi images are expected to by an (x,y) grid.

    args:
        data_dir: the data directory, probably something/ShallowWater-2D
        data_class: one of "train", "valid", or "test"
        seed: the particular seed we are reading, probably

    returns:
        The jax data arrays, uv, pres, vor, and lats. Their shapes are as follows:
        uv: (batch,timesteps,spatial,D)
        pres: (batch,timesteps,spatial)
        vor: (batch,timesteps,spatial)
        lats: (96,)
    """
    target_file = f"{data_dir}/{data_class}/{seed}/all_data.npy"
    if "all_data.npy" in os.listdir(f"{data_dir}/{data_class}/{seed}/"):
        dataset = jnp.load(target_file, allow_pickle=True).item()
        u = jax.device_put(dataset["u"], jax.devices("cpu")[0])  # velocity in x direction
        v = jax.device_put(dataset["v"], jax.devices("cpu")[0])  # velocity in y direction
        pres = jax.device_put(dataset["pres"], jax.devices("cpu")[0])  # pressure scalar
        vor = jax.device_put(dataset["vor"], jax.devices("cpu")[0])  # vorticity pseudoscalar

        lat = jax.device_put(dataset["lat"], jax.devices("cpu")[0])  # latitude degrees
        # lon = jax.device_put(dataset["lon"], jax.devices("cpu")[0])  # longitude degrees
        # div = jax.device_put(dataset["div"], jax.devices("cpu")[0])  # divergence
    else:
        datals = os.path.join(data_dir, data_class, seed, "run*", "output.nc")
        dataset = xr.open_mfdataset(datals, concat_dim="b", combine="nested", parallel=True)  # dict
        # all have shape (batch, timesteps, lev, lat, lon) = (25,88,1,96,192)
        # u: zonal velocity, in x direction
        u = jax.device_put(jnp.array(dataset["u"].to_numpy()), jax.devices("cpu")[0])
        # v: meridonial velocity, in y direction
        v = jax.device_put(jnp.array(dataset["v"].to_numpy()), jax.devices("cpu")[0])
        # pressure scalar, does not have lev dim
        pres = jax.device_put(jnp.array(dataset["pres"].to_numpy()), jax.devices("cpu")[0])
        # vorticity pseudoscalar
        vor = jax.device_put(jnp.array(dataset["vor"].to_numpy()), jax.devices("cpu")[0])

        # all have shape (96,)
        # latitudes
        lat = jax.device_put(jnp.array(dataset["lat"].to_numpy()), jax.devices("cpu")[0])
        # longitudes
        lon = jax.device_put(jnp.array(dataset["lon"].to_numpy()), jax.devices("cpu")[0])
        # divergence?
        div = jax.device_put(jnp.array(dataset["div"].to_numpy()), jax.devices("cpu")[0])

        jnp.save(
            target_file,
            {"u": u, "v": v, "pres": pres, "vor": vor, "lat": lat, "lon": lon, "div": div},
        )

    # Data is loaded with shape (96,192), or (latitude,longitude) or (y,x). This means that the
    # normal way we think of arrays, by rows then columns it will be layed out like a typical map.
    # However, this is backwards of when we think of accessing elements [x,y] where the x-coordinate
    # is the first and the y-coordinate is second. So we will flip them.
    u = jnp.moveaxis(u, -2, -1)
    v = jnp.moveaxis(v, -2, -1)
    pres = jnp.moveaxis(pres, -2, -1)
    vor = jnp.moveaxis(vor, -2, -1)

    uv = jnp.stack([u[:, :, 0, ...], v[:, :, 0, ...]], axis=-1)
    return uv, pres, vor[:, :, 0, ...], lat


def read_all_seeds(
    data_dir: str, n_trajectories: int, data_class: str
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Given a specific dataset and data class (train, valid, or test), read seeds until we get to
    n_trajectories, and then return the data.

    args:
        data_dir: directory of the data
        n_trajectories: number of trajectories
        data_class: type of data, either train, valid, or test

    returns:
        The jax data arrays, uv, pres, vor, and lats. Their shapes are as follows:
        uv: (batch,timesteps,spatial,D)
        pres: (batch,timesteps,spatial)
        vor: (batch,timesteps,spatial)
        lats: (96,)
    """
    all_seeds = sorted(os.listdir(f"{data_dir}/{data_class}/"))
    all_seeds = filter(lambda path: "seed=" in path, all_seeds)

    spatial_dims = (192, 96)  # (lon,lat) (x,y)
    D = 2
    total_steps = 88
    all_uv = jnp.zeros((0, total_steps) + spatial_dims + (D,))
    all_pres = jnp.zeros((0, total_steps) + spatial_dims)
    all_vor = jnp.zeros((0, total_steps) + spatial_dims)
    lats = jnp.zeros(96)  # assume lats are the same for all the data
    for seed in all_seeds:
        uv, pres, vor, lats = read_one_seed(data_dir, data_class, seed)

        all_uv = jnp.concatenate([all_uv, uv])
        all_pres = jnp.concatenate([all_pres, pres])
        all_vor = jnp.concatenate([all_vor, vor])

        if len(all_uv) >= n_trajectories:
            break

    if len(all_uv) < n_trajectories:
        print(
            f"WARNING get_data_layers: wanted {n_trajectories} {data_class} trajectories, "
            f"but only found {len(all_uv)}",
        )
        n_trajectories = len(all_uv)

    # (b,timesteps,spatial,tensor)
    all_uv = all_uv[:n_trajectories]
    all_pres = all_pres[:n_trajectories]
    all_vor = all_vor[:n_trajectories]

    return all_uv, all_pres, all_vor, lats


def make_constant_fields(
    orography: Optional[geom.MultiImage],
    lats: jax.Array,
    is_torus: Union[bool, tuple[bool, ...]],
    spatial_dims: tuple[int, ...],
    include_lats: bool,
    include_metric_tensor: bool,
) -> geom.MultiImage:
    """
    Build the constant fields from orography and latitudes depending on the arguments.

    args:
        orography: a multi image of the mountains, shape (1,spatial)
        lats: array of the latitudes in degrees, (96,)
        is_torus: toroidal structure of the images
        spatial_dims: spatial dimensions of the images
        include_lats: whether to include the raw latitude values
        include_metric_tensor: whether to include the metric tensor as an input field

    returns:
        a multi image of the constant fields, shape (channels,spatial,tensor).
    """
    constant_fields = geom.MultiImage({}, D, is_torus)
    if orography is not None:
        constant_fields = orography
    if include_lats:
        constant_fields.append(0, 0, jnp.full((1,) + spatial_dims, lats[None, None]))
    if include_metric_tensor:
        sin_squared_lats = (jnp.sin((jnp.pi / 180) * lats) ** 2).reshape((96, 1, 1))
        bottom = sin_squared_lats * jnp.array([[0, 0], [0, 1]]).reshape((1, 2, 2))
        top = jnp.ones((96, 1, 1)) * jnp.array([[1, 0], [0, 0]]).reshape((1, 2, 2))
        metric = (top + bottom).reshape((1, 1, 96, 2, 2))  # (96,2,2) -> (1,1,96,2,2)
        constant_fields.append(2, 0, jnp.full((1,) + spatial_dims + (2, 2), metric))

    return constant_fields


def get_data_multi_images(
    uv: jax.Array,
    pres: jax.Array,
    vor: jax.Array,
    constant_fields: geom.MultiImage,
    pres_vor_form: bool,
    total_steps: int,
    past_steps: int,
    future_steps: int,
    skip_initial: int = 4,
    subsample: int = 1,
    is_torus: Union[bool, tuple[bool, ...]] = (True, False),
) -> tuple[geom.MultiImage, geom.MultiImage]:
    """
    Construct the multi images from the data.

    args:
        uv: velocity data array, shape (batch,timesteps,spatial,tensor)
        pres: pressure data array, shape (batch,timesteps,spatial,tensor)
        vor: vorticity data array, shape (batch,timesteps,spatial,tensor)
        constant_fields: fields that do not vary by timestep, shape (channels,spatial,tensor)
        pres_vor_form: use pressure/vorticity form instead
        total_steps: the number of timesteps
        past_steps: the lookback window, how many steps we look back to predict the next one
        future_steps: the number of steps in the future to compare against
        skip_initial: how many initial timesteps to skip
        subsample: timesteps are 6 simulation hours, can subsample for longer timesteps
        is_torus: toroidal structure of the images

    returns:
        the input and output multi images
    """
    # add the batch dimension
    batch_const_fields = constant_fields.empty()
    for (k, p), image in constant_fields.items():
        batch_const_fields.append(k, p, jnp.full((len(uv),) + image.shape, image[None]))

    if pres_vor_form:
        multi_image = geom.MultiImage({(0, 0): pres, (0, 1): vor, (1, 0): uv}, D, is_torus)
        multi_image_x, multi_image_y = gc_data.batch_time_series(
            multi_image,
            batch_const_fields,
            total_steps,
            past_steps,
            future_steps,
            skip_initial,
            subsample,
        )
        del multi_image_x.data[(0, 1)]  # remove vorticity from input
    else:
        multi_image = geom.MultiImage({(0, 0): pres, (1, 0): uv}, D, is_torus)
        multi_image_x, multi_image_y = gc_data.batch_time_series(
            multi_image,
            batch_const_fields,
            total_steps,
            past_steps,
            future_steps,
            skip_initial,
            subsample,
        )

    return multi_image_x, multi_image_y


def get_data(
    data_dir: str,
    n_train: int,
    n_val: int,
    n_test: int,
    past_steps: int,
    rollout_steps: int,
    normalize: int = 0,
    pres_vor_form: bool = False,
    subsample: int = 1,
    include_lats: bool = False,
    include_metric_tensor: bool = False,
) -> tuple[
    tuple[
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
    ],
    dict[tuple[int, int], int],
]:
    """
    Get train, val, and test data sets.

    args:
        data_dir: directory of data
        n_train: number of training trajectories
        n_val: number of validation trajectories
        n_test: number of testing trajectories
        past_steps: length of the lookback to predict the next step
        rollout_steps: number of steps of rollout to compare against
        normalize: whether normalize the pressure/vorticity, 0 No, 1 single_value, 2 pixelwise
        pres_vor_form: use pressure/vorticity form instead, defaults to False
        subsample: timesteps are 6 simulation hours, can subsample for longer timesteps
        include_lats: include the raw latitudes as a scalar field
        include_metric_tensor: include the metric tensor as an input field

    returns:
        train, val, test one step, and test rollout input and output multi images. Also a
        dictionary specifying what constant fields there are of each image type.
    """
    train_uv, train_pres, train_vor, lats = read_all_seeds(data_dir, n_train, "train")
    val_uv, val_pres, val_vor, _ = read_all_seeds(data_dir, n_val, "valid")
    test_uv, test_pres, test_vor, _ = read_all_seeds(data_dir, n_test, "test")

    if normalize:
        pres_mean = jnp.mean(jnp.concatenate([train_pres, val_pres]))
        pres_std = jnp.std(jnp.concatenate([train_pres, val_pres]))
        train_pres = (train_pres - pres_mean) / pres_std
        val_pres = (val_pres - pres_mean) / pres_std
        test_pres = (test_pres - pres_mean) / pres_std

        vor_std = jnp.std(jnp.concatenate([train_vor, val_vor]))
        train_vor = train_vor / vor_std
        val_vor = val_vor / vor_std
        test_vor = test_vor / vor_std

        uv_std = jnp.std(jnp.concatenate([train_uv, val_uv]))
        train_uv = train_uv / uv_std
        val_uv = val_uv / uv_std
        test_uv = test_uv / uv_std

    D = 2
    is_torus = (True, False)
    total_steps = train_uv.shape[1]
    spatial_dims = train_uv.shape[2:4]

    orography = read_orography(D, is_torus, data_dir)
    constant_fields = make_constant_fields(
        orography, lats, is_torus, spatial_dims, include_lats, include_metric_tensor
    )

    train_X, train_Y = get_data_multi_images(
        train_uv,
        train_pres,
        train_vor,
        constant_fields,
        pres_vor_form,
        total_steps,
        past_steps,
        1,
        subsample=subsample,
        is_torus=is_torus,
    )
    val_X, val_Y = get_data_multi_images(
        val_uv,
        val_pres,
        val_vor,
        constant_fields,
        pres_vor_form,
        total_steps,
        past_steps,
        1,
        subsample=subsample,
        is_torus=is_torus,
    )
    test_single_X, test_single_Y = get_data_multi_images(
        test_uv,
        test_pres,
        test_vor,
        constant_fields,
        pres_vor_form,
        total_steps,
        past_steps,
        1,
        subsample=subsample,
        is_torus=is_torus,
    )
    test_rollout_X, test_rollout_Y = get_data_multi_images(
        test_uv,
        test_pres,
        test_vor,
        constant_fields,
        pres_vor_form,
        total_steps,
        past_steps,
        rollout_steps,
        subsample=subsample,
        is_torus=is_torus,
    )
    constant_fields_dict = {k: n_channels for k, n_channels in constant_fields.get_signature()}

    return (
        train_X,
        train_Y,
        val_X,
        val_Y,
        test_single_X,
        test_single_Y,
        test_rollout_X,
        test_rollout_Y,
    ), constant_fields_dict


@eqx.filter_jit
def map_and_loss(
    model: models.MultiImageModule,
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    aux_data: Optional[eqx.nn.State] = None,
    future_steps: int = 1,
    return_map: bool = False,
    constant_fields: dict[tuple[int, int], int] = {},
) -> Union[
    tuple[jax.Array, Optional[eqx.nn.State], geom.MultiImage],
    tuple[jax.Array, Optional[eqx.nn.State]],
]:
    assert (0, 1) not in multi_image_y, "Cannot handle pressure/vorticity form yet"

    vmap_autoregressive = jax.vmap(
        ml.autoregressive_map,
        in_axes=(None, 0, None, None, None, None),
        out_axes=(0, None),
        axis_name="batch",
    )
    out, aux_data = vmap_autoregressive(
        model,
        multi_image_x,
        aux_data,
        multi_image_x[(1, 0)].shape[1],  # past_steps
        future_steps,
        constant_fields,
    )

    loss = ml.timestep_smse_loss(out, multi_image_y, future_steps)
    loss = loss[0] if future_steps == 1 else loss

    return (loss, aux_data, out) if return_map else (loss, aux_data)


def train_and_eval(
    data: tuple[geom.MultiImage, ...],
    key: ArrayLike,
    model_name: str,
    model: models.MultiImageModule,
    lr: float,
    batch_size: int,
    epochs: int,
    rollout_steps: int,
    save_model: Optional[str],
    load_model: Optional[str],
    images_dir: Optional[str],
    has_aux: bool = False,
    verbose: int = 1,
    plot_component: int = 0,
    constant_fields: dict[tuple[int, int], int] = {},
) -> tuple[Optional[ArrayLike], ...]:
    (
        train_X,
        train_Y,
        val_X,
        val_Y,
        test_single_X,
        test_single_Y,
        test_rollout_X,
        test_rollout_Y,
    ) = data
    batch_stats = eqx.nn.State(model) if has_aux else None

    print(f"Model params: {models.count_params(model):,}")

    map_and_loss_f = partial(map_and_loss, constant_fields=constant_fields)

    if load_model is None:
        steps_per_epoch = int(np.ceil(train_X.get_L() / batch_size))
        key, subkey = random.split(key)
        model, batch_stats, train_loss, val_loss = ml.train(
            train_X,
            train_Y,
            map_and_loss_f,
            model,
            subkey,
            stop_condition=ml.EpochStop(epochs, verbose=verbose),
            batch_size=batch_size,
            optimizer=optax.adamw(
                optax.warmup_cosine_decay_schedule(
                    1e-8, lr, 5 * steps_per_epoch, epochs * steps_per_epoch, 1e-7
                ),
                weight_decay=1e-5,
            ),
            validation_X=val_X,
            validation_Y=val_Y,
            aux_data=batch_stats,
        )

        if save_model is not None:
            # TODO: need to save batch_stats as well
            ml.save(f"{save_model}{model_name}_L{train_X.get_L()}_e{epochs}_model.eqx", model)
    else:
        model = ml.load(f"{load_model}{model_name}_L{train_X.get_L()}_e{epochs}_model.eqx", model)

        key, subkey1, subkey2 = random.split(key, num=3)
        train_loss = ml.map_loss_in_batches(
            map_and_loss_f,
            model,
            train_X,
            train_Y,
            batch_size,
            subkey1,
            aux_data=batch_stats,
        )
        val_loss = ml.map_loss_in_batches(
            map_and_loss_f,
            model,
            val_X,
            val_Y,
            batch_size,
            subkey2,
            aux_data=batch_stats,
        )

    key, subkey = random.split(key)
    test_loss = ml.map_loss_in_batches(
        map_and_loss_f,
        model,
        test_single_X,
        test_single_Y,
        batch_size,
        subkey,
        aux_data=batch_stats,
    )
    print(f"Test Loss: {test_loss}")

    key, subkey = random.split(key)
    test_rollout_loss, rollout_multi_image = ml.map_plus_loss_in_batches(
        partial(map_and_loss_f, future_steps=rollout_steps, return_map=True),
        model,
        test_rollout_X,
        test_rollout_Y,
        batch_size,
        subkey,
        aux_data=batch_stats,
    )
    print(f"Test Rollout Loss: {test_rollout_loss}, Sum: {jnp.sum(test_rollout_loss)}")

    return train_loss, val_loss, test_loss, *test_rollout_loss


def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="the data .hdf5 file", type=str)
    parser.add_argument("-e", "--epochs", help="number of epochs to run", type=int, default=50)
    parser.add_argument("-batch", help="batch size", type=int, default=8)
    parser.add_argument("-n_train", help="number of training trajectories", type=int, default=100)
    parser.add_argument(
        "-n_val",
        help="number of validation trajectories, defaults to batch",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-n_test",
        help="number of testing trajectories, defaults to batch",
        type=int,
        default=None,
    )
    parser.add_argument("-t", "--n_trials", help="number of trials to run", type=int, default=1)
    parser.add_argument("-seed", help="the random number seed", type=int, default=None)
    parser.add_argument(
        "-s", "--save_model", help="file name to save the params", type=str, default=None
    )
    parser.add_argument(
        "-l", "--load_model", help="file name to load params from", type=str, default=None
    )
    parser.add_argument(
        "-images_dir",
        help="directory to save images, or None to not save",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="verbose argument passed to trainer",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--normalize",
        help="normalize input data, equivariantly",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include_lats",
        help="include the latitudes as a scalar field input",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--include_metric_tensor",
        help="include the metric tensor as an input tensor field",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--plot_component",
        help="which component to plot, one of 0-3",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
    )
    parser.add_argument(
        "--rollout_steps",
        help="number of steps to rollout in test",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-pres_vor_form",
        help="toggle to use pressure/vorticity form, rather than pressure/velocity form",
        action="store_true",
    )
    parser.add_argument(
        "-subsample",
        help="how many timesteps per model step, default 1",
        type=int,
        default=1,
    )

    return parser.parse_args()


# Main
args = handleArgs(sys.argv)

D = 2

past_steps = 2  # how many steps to look back to predict the next step
key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)

# an attempt to reduce recompilation, but I don't think it actually is working
n_val = args.batch if args.n_val is None else args.n_val
n_test = args.batch if args.n_test is None else args.n_test

data, constant_fields = get_data(
    args.data,
    args.n_train,
    n_val,
    n_test,
    past_steps,
    args.rollout_steps,
    args.normalize,
    args.pres_vor_form,
    args.subsample,
    args.include_lats,
    args.include_metric_tensor,
)

input_keys = data[0].get_signature()
output_keys = data[1].get_signature()

group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    Ms=[3], ks=[0, 1, 2], parities=[0, 1], D=D, operators=group_actions
)
upsample_filters = geom.get_invariant_filters(
    Ms=[2], ks=[0, 1, 2], parities=[0, 1], D=D, operators=group_actions
)

train_and_eval = partial(
    train_and_eval,
    batch_size=args.batch,
    epochs=args.epochs,
    rollout_steps=args.rollout_steps,
    save_model=args.save_model,
    load_model=args.load_model,
    images_dir=args.images_dir,
    verbose=args.verbose,
    plot_component=args.plot_component,
    constant_fields=constant_fields,
)

key, *subkeys = random.split(key, num=13)
models_ls = [
    (
        "dil_resnet64",
        partial(
            train_and_eval,
            model=models.DilResNet(
                D,
                input_keys,
                output_keys,
                depth=64,
                equivariant=False,
                kernel_size=3,
                key=subkeys[0],
            ),
            lr=2e-3,
        ),
    ),
    (
        "dil_resnet_equiv20",
        partial(
            train_and_eval,
            model=models.DilResNet(
                D,
                input_keys,
                output_keys,
                depth=20,
                conv_filters=conv_filters,
                key=subkeys[1],
            ),
            lr=1e-3,
        ),
    ),
    # (
    #     "dil_resnet_equiv48",
    #     partial(
    #         train_and_eval,
    #         model=models.DilResNet(
    #             D,
    #             input_keys,
    #             output_keys,
    #             depth=48,
    #             conv_filters=conv_filters,
    #             key=subkeys[2],
    #         ),
    #         lr=1e-3,
    #     ),
    # ),
    # (
    #     "resnet",
    #     partial(
    #         train_and_eval,
    #         model=models.ResNet(
    #             D,
    #             input_keys,
    #             output_keys,
    #             depth=128,
    #             equivariant=False,
    #             kernel_size=3,
    #             key=subkeys[3],
    #         ),
    #         lr=1e-3,
    #     ),
    # ),
    # (
    #     "resnet_equiv_groupnorm_42",
    #     partial(
    #         train_and_eval,
    #         model=models.ResNet(
    #             D,
    #             input_keys,
    #             output_keys,
    #             depth=42,
    #             conv_filters=conv_filters,
    #             key=subkeys[4],
    #         ),
    #         lr=7e-4,
    #     ),
    # ),
    # (
    #     "resnet_equiv_groupnorm_100",
    #     partial(
    #         train_and_eval,
    #         model=models.ResNet(
    #             D,
    #             input_keys,
    #             output_keys,
    #             depth=100,  # very slow at 100
    #             conv_filters=conv_filters,
    #             key=subkeys[5],
    #         ),
    #         lr=7e-4,
    #     ),
    # ),
    # (
    #     "unetBase",
    #     partial(
    #         train_and_eval,
    #         model=models.UNet(
    #             D,
    #             input_keys,
    #             output_keys,
    #             depth=64,
    #             use_bias=True,
    #             activation_f=jax.nn.gelu,
    #             equivariant=False,
    #             kernel_size=3,
    #             use_group_norm=True,
    #             key=subkeys[6],
    #         ),
    #         lr=8e-4,
    #     ),
    # ),
    # (
    #     "unetBase_equiv20",
    #     partial(
    #         train_and_eval,
    #         model=models.UNet(
    #             D,
    #             input_keys,
    #             output_keys,
    #             depth=20,
    #             conv_filters=conv_filters,
    #             upsample_filters=upsample_filters,
    #             key=subkeys[7],
    #         ),
    #         lr=6e-4,  # 4e-4 to 6e-4 works, larger sometimes explodes
    #     ),
    # ),
    # (
    #     "unetBase_equiv48",
    #     partial(
    #         train_and_eval,
    #         model=models.UNet(
    #             D,
    #             input_keys,
    #             output_keys,
    #             depth=48,
    #             activation_f=jax.nn.gelu,
    #             conv_filters=conv_filters,
    #             upsample_filters=upsample_filters,
    #             key=subkeys[8],
    #         ),
    #         lr=4e-4,  # 4e-4 to 6e-4 works, larger sometimes explodes
    #     ),
    # ),
    # (
    #     "unet2015",
    #     partial(
    #         train_and_eval,
    #         model=models.UNet(
    #             D,
    #             input_keys,
    #             output_keys,
    #             depth=64,
    #             use_bias=False,
    #             equivariant=False,
    #             kernel_size=3,
    #             use_batch_norm=True,
    #             key=subkeys[9],
    #         ),
    #         lr=8e-4,
    #         has_aux=True,
    #     ),
    # ),
    # (
    #     "unet2015_equiv20",
    #     partial(
    #         train_and_eval,
    #         model=models.UNet(
    #             D,
    #             input_keys,
    #             output_keys,
    #             depth=20,
    #             use_bias=False,
    #             conv_filters=conv_filters,
    #             upsample_filters=upsample_filters,
    #             key=subkeys[10],
    #         ),
    #         lr=7e-4,  # sometimes explodes for larger values
    #     ),
    # ),
    # (
    #     "unet2015_equiv48",
    #     partial(
    #         train_and_eval,
    #         model=models.UNet(
    #             D,
    #             input_keys,
    #             output_keys,
    #             depth=48,
    #             use_bias=False,
    #             conv_filters=conv_filters,
    #             upsample_filters=upsample_filters,
    #             key=subkeys[11],
    #         ),
    #         lr=3e-4,
    #     ),
    # ),
]

key, subkey = random.split(key)
results = ml.benchmark(
    lambda _: data,
    models_ls,
    subkey,
    "",
    [0],
    benchmark_type=ml.BENCHMARK_NONE,
    num_results=4,
    num_trials=args.n_trials,
)

print(results)
