import time
import argparse
import pathlib
import h5py

import jax.numpy as jnp
import jax
import jax.random as random

import ginjax.geometric as geom
import ginjax.utils as utils
import ginjax.data as gc_data
import ginjax.models as models
from . import anyd_helpers


def read_one_h5(D: int, filename: pathlib.Path, num_trajectories: int) -> tuple:
    """
    Given a dimension and filename, read the data and return as jax arrays.

    args:
        D: the dimension of the space
        filename: the full file path
        num_trajectories: number of trajectories to read

    returns:
        density, pressure, and velocity fields
    """
    data_dict = h5py.File(filename)

    # all of these are shape (num_trajectories, timesteps, spatial, tensor)
    # 1D: (10K,101,1024,tensor)
    # 2D: (1K,21,512,512,tensor) or (10K,21,128,128,tensor)
    # 2D: my janky homebrew is (9990,21,128,128,tensor)
    # 3D: (100,21,128,128,128,tensor)
    density = jax.device_put(
        jnp.array(data_dict["density"][:num_trajectories][()]), jax.devices("cpu")[0]
    )
    pressure = jax.device_put(
        jnp.array(data_dict["pressure"][:num_trajectories][()]), jax.devices("cpu")[0]
    )

    velocities = []
    for vkey in ["Vx", "Vy", "Vz"][:D]:
        velocities.append(
            jax.device_put(jnp.array(data_dict[vkey][:num_trajectories][()]), jax.devices("cpu")[0])
        )

    velocity = jnp.stack(velocities, axis=-1)

    data_dict.close()

    return density, pressure, velocity


def get_data(
    D: int,
    n_train: int,
    n_val: int,
    n_test: int,
    past_steps: int,
    data_dir: str,
) -> tuple[
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    float,
]:
    """
    TODO: maybe I can load data as needed? one batch at a time? probably needed for 3d data
    Get data of a particular dimension from a preset list of files in the specified folder.

    args:
        D: the dimension of the data to get
        n_train: number of training trajectories
        n_val: number of validation trajectories
        n_test: number of test trajectories
        past_steps: number of historical steps to use to predict the next 1 step
        data_dir: the direcory where the data is stored

    returns:
        input and output pairs of multi images for train, val, and test, plus data generation time
            for train
    """
    data_dir_path = pathlib.Path(data_dir)
    is_torus = True

    fnames = {
        1: "1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5",
        2: "2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_128_Train.hdf5",
        3: "3D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_Train.hdf5",
    }

    fpath = data_dir_path / f"cfd_{D}d" / fnames[D]
    n_traj = n_train + n_val + n_test

    density, pressure, velocity = read_one_h5(D, fpath, n_traj)

    constant_fields = geom.MultiImage({}, D, is_torus)

    # (batch,2,timesteps,spatial)
    if n_traj == 0:
        density_pressure = jnp.zeros((n_traj, 2 * density.shape[1]) + density.shape[2:])
    else:
        density_pressure = jnp.concatenate([density[:, None], pressure[:, None]], axis=1)
        # (batch,2*timesteps,spatial)
        density_pressure = density_pressure.reshape(
            (len(density_pressure), -1) + density_pressure.shape[3:]
        )

    start = 0
    stop = n_train
    train_X, train_Y = gc_data.batch_time_series(
        geom.MultiImage(
            {(0, 0): density_pressure[start:stop], (1, 0): velocity[start:stop]},
            D,
            is_torus,
        ),
        constant_fields,
        velocity.shape[1],
        past_steps,
        1,
    )

    start = start + n_train
    stop = start + n_val
    val_X, val_Y = gc_data.batch_time_series(
        geom.MultiImage(
            {(0, 0): density_pressure[start:stop], (1, 0): velocity[start:stop]},
            D,
            is_torus,
        ),
        constant_fields,
        velocity.shape[1],
        past_steps,
        1,
    )

    start = start + n_val
    stop = start + n_test

    test_X, test_Y = gc_data.batch_time_series(
        geom.MultiImage(
            {(0, 0): density_pressure[start:stop], (1, 0): velocity[start:stop]}, D, is_torus
        ),
        constant_fields,
        velocity.shape[1],
        past_steps,
        1,
    )

    data_generation_time = 0.0  # TODO: fix this

    return train_X, train_Y, val_X, val_Y, test_X, test_Y, data_generation_time


def handleArgs() -> argparse.Namespace:
    parser = utils.get_common_parser()
    parser.add_argument(
        "--n-tune-range",
        help="the number of data points in the tuning set",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default="0,1,4,32,128",
    )
    parser.add_argument(
        "--past-steps", help="number of past steps to use as input", type=int, default=4
    )
    parser.add_argument(
        "--find-train-lr",
        help="benchmark trained model over the lr",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--find-tune-lr",
        help="benchmark tuned model over the lr",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    # need do to --train-wandb or --tune-wandb to activate
    parser.add_argument("--wandb-project", help="the wandb project", type=str, default="cfd-anyd")
    parser.add_argument(
        "--train-wandb",
        help="whether to use wandb during training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--tune-wandb",
        help="whether to use wandb during tuning",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    return parser.parse_args()


# MAIN
args = handleArgs()
if args.wandb:
    print("Use --train-wandb or --test-wandb to control these individually. Exiting.")
    exit()

if args.load_model or args.save_model:
    print("Use --model-dir and possibly --overwrite-save-model instead of --save-model")
    exit()

key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)
lr_range = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]

# D=1 doesn't make sense for a vector field, so we restrict the problem to only this case
train_D = 2
test_D = 3
n_results = 5

train_kwargs = {
    "train_loss_f": geom.Losses.NRMSE,
    "residual": False,
    "batch_size": args.batch,
    "epochs": args.epochs,
    "model_dir": pathlib.Path(args.model_dir) if args.model_dir else None,
    "overwrite_save_model": args.overwrite_save_model,
    "images_dir": None,
    "verbose": args.verbose,
    "is_wandb": args.train_wandb,
}

test_kwargs = {
    "train_loss_f": geom.Losses.NRMSE,
    "residual": False,
    "batch_size": args.batch,
    "epochs": args.epochs,
    "model_dir": pathlib.Path(args.model_dir) if args.model_dir else None,
    "overwrite_save_model": args.overwrite_save_model,
    "images_dir": None,
    "verbose": args.verbose,
    "is_wandb": args.tune_wandb,
}

upsample_filters_dict = {}
free_filters_dict = {}

full_D_range = [train_D, test_D]
for D in [train_D, test_D]:
    group_actions = geom.make_all_operators(D)
    upsample_filters_dict[D] = geom.get_invariant_filters(
        Ms=[2],
        ks=[0, 1, 2],
        parities=[0],
        D=D,
        operators=group_actions,
        scale=geom.FilterScaling.ONE,  # for N=2, all pixels are equidistant
    )
    free_filters_dict[D] = geom.get_invariant_filters(
        Ms=[3],
        ks=[0, 1, 2],
        parities=[0],
        D=D,
        operators=group_actions,
        scale=geom.FilterScaling.ONE,
    )

print("Define the models!")
model_list_d = {}
for D in full_D_range:
    key, subkey = random.split(key)
    train_x0, train_xt, _, _, _, _, _ = get_data(D, 1, 0, 0, args.past_steps, args.data)
    input_keys = train_x0.get_signature()
    output_keys = train_xt.get_signature()

    key, *subkeys = random.split(key, num=10)
    model_list = [
        (
            f"unetBase_equiv48_D{D}",
            {  # train_kwargs
                "model": models.UNet(
                    D,
                    input_keys,
                    output_keys,
                    depth=48,
                    activation_f=jax.nn.gelu,
                    conv_filters=free_filters_dict[D],
                    upsample_filters=upsample_filters_dict[D],
                    key=subkeys[2],
                ),
                "lr": {2: {128: 1e-4}, 3: {0: 1e-4, 1: 1e-4, 4: 1e-4, 32: 1e-4, 128: 1e-4}},
                # D=3, for all of them (and tuning) its just 1e-4
                **train_kwargs,
            },
            {  # tune and eval kwargs
                "lr": {(2, 3): {0: 1e-4, 1: 1e-4, 4: 1e-4, 32: 1e-4, 128: 1e-4}},
                "rescale": geom.Rescaling.COMPAT_FLEX,
                "conv_filters_dict": free_filters_dict,
                "upsample_filters_dict": upsample_filters_dict,
                **test_kwargs,
            },
        ),
    ]
    model_list_d[D] = model_list


# extended lambda function
def get_data_lambda(D: int, n_train: int, n_val: int, n_test: int, key: jax.Array) -> tuple[
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    float,
]:
    return get_data(D, n_train, n_val, n_test, args.past_steps, args.data)


key, subkey = random.split(key)
anyd_helpers.run_anyd(
    (train_D,), (test_D,), args.n_tune_range, args, subkey, get_data_lambda, model_list_d, lr_range
)
