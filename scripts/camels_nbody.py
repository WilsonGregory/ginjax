import sys
import os
import time
import argparse
import h5py
import hdf5plugin

import jax.numpy as jnp
import jax
import jax.random as random

import ginjax.data as gc_data


def read_one_h5(filename: str) -> tuple:
    """
    Given a filename and a type of data (train, test, or validation), read the data and return as jax arrays.
    args:
        filename (str): the full file path
    returns: force, particle, and velocity
    """
    data_dict = h5py.File(filename)["PartType1"]  # keys 'Coordinates', 'ParticleIDs', 'Velocities'
    print(data_dict.keys())
    print(data_dict["Coordinates"].shape)
    print(data_dict["ParticleIDs"].shape)
    print(data_dict["Velocities"].shape)
    exit()
    # 4 runs, 1000 time points, 512x512 grid
    # time ranges from 0 to 5, presumably seconds
    force = jax.device_put(
        jnp.array(data_dict["force"][()]), jax.devices("cpu")[0]
    )  # (4,512,512,2)
    particles = jax.device_put(
        jnp.array(data_dict["particles"][()]), jax.devices("cpu")[0]
    )  # (4,1000,512,512,1)
    # these are advected particles, might be a proxy of density?
    velocity = jax.device_put(
        jnp.array(data_dict["velocity"][()]), jax.devices("cpu")[0]
    )  # (4,1000,512,512,2)
    data_dict.close()

    return force, particles[..., 0], velocity


def read_data(data_dir: str, num_trajectories: int, downsample: int = 0) -> tuple:
    """
    Load data from the multiple .hd5 files
    args:
        data_dir (str): directory of the data
        num_trajectories (int): total number of trajectories to read in
    """
    N = int(512 / (2**downsample))
    D = 2
    all_files = filter(lambda file: f"snapshot_" in file, os.listdir(data_dir))

    all_force = jnp.zeros((0, N, N, D))
    all_particles = jnp.zeros((0, 1000, N, N))
    all_velocity = jnp.zeros((0, 1000, N, N, D))
    for filename in all_files:
        force, particles, velocity = read_one_h5(f"{data_dir}/{filename}")

        all_force = jnp.concatenate([all_force, force])
        all_particles = jnp.concatenate([all_particles, particles])
        all_velocity = jnp.concatenate([all_velocity, velocity])

        if len(all_force) >= num_trajectories:
            break

    if len(all_force) < num_trajectories:
        print(
            f"WARNING read_data: wanted {num_trajectories} trajectories, but only found {len(all_force)}"
        )
        num_trajectories = len(all_force)

    all_force = all_force[:num_trajectories]
    all_particles = all_particles[:num_trajectories]
    all_velocity = all_velocity[:num_trajectories]

    return all_force, all_particles, all_velocity


def get_data(
    data_dir: str,
    num_train_traj: int,
    num_val_traj: int,
    num_test_traj: int,
    past_steps: int,
    rollout_steps: int,
    delta_t: int = 1,
    downsample: int = 0,
    skip_initial: int = 0,
) -> tuple:
    """
    Get train, val, and test data sets.
    args:
        data_dir (str): directory of data
        num_train_traj (int): number of training trajectories
        num_val_traj (int): number of validation trajectories
        num_test_traj (int): number of testing trajectories
        past_steps (int): length of the lookback to predict the next step
        rollout_steps (int): number of steps of rollout to compare against
        delta_t (int): number of timesteps per model step, default 1
        downsample (int): number of times to spatial downsample, defaults to 0 (no downsampling)
        skip_initial (int): number of initial steps to skip, default to 0
    """
    D = 2
    force, particles, velocity = read_data(
        data_dir,
        num_train_traj + num_val_traj + num_test_traj,
        downsample,  # downsampling handled prior to time_series_to_multi_images
    )

    start = 0
    stop = num_train_traj
    train_X, train_Y = gc_data.times_series_to_multi_images(
        D,
        {(0, 0): particles[start:stop], (1, 0): velocity[start:stop]},
        {(1, 0): force[start:stop]},
        False,
        past_steps,
        1,
        skip_initial,
        delta_t,
    )
    start = stop
    stop = stop + num_val_traj
    val_X, val_Y = gc_data.times_series_to_multi_images(
        D,
        {(0, 0): particles[start:stop], (1, 0): velocity[start:stop]},
        {(1, 0): force[start:stop]},
        False,  # is_torus
        past_steps,
        1,  # future_steps
        skip_initial,
        delta_t,
    )
    start = stop
    stop = stop + num_test_traj
    test_single_X, test_single_Y = gc_data.times_series_to_multi_images(
        D,
        {(0, 0): particles[start:stop], (1, 0): velocity[start:stop]},
        {(1, 0): force[start:stop]},
        False,  # is_torus
        past_steps,
        1,  # future_steps
        skip_initial,
        delta_t,
    )
    test_rollout_X, test_rollout_Y = gc_data.times_series_to_multi_images(
        D,
        {(0, 0): particles[start:stop], (1, 0): velocity[start:stop]},
        {(1, 0): force[start:stop]},
        False,  # is_torus
        past_steps,
        rollout_steps,
        skip_initial,
        delta_t,
    )

    return (
        train_X,
        train_Y,
        val_X,
        val_Y,
        test_single_X,
        test_single_Y,
        test_rollout_X,
        test_rollout_Y,
    )


def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="the directory where the .h5 files are located", type=str)
    parser.add_argument("-e", "--epochs", help="number of epochs to run", type=int, default=50)
    parser.add_argument("-lr", help="learning rate", type=float, default=2e-4)
    parser.add_argument("-batch", help="batch size", type=int, default=16)
    parser.add_argument("-train_traj", help="number of training trajectories", type=int, default=1)
    parser.add_argument(
        "-val_traj",
        help="number of validation trajectories, defaults to 1",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-test_traj",
        help="number of testing trajectories, defaults to 1",
        type=int,
        default=1,
    )
    parser.add_argument("-seed", help="the random number seed", type=int, default=None)
    parser.add_argument("-s", "--save", help="file name to save the params", type=str, default=None)
    parser.add_argument(
        "-l", "--load", help="file name to load params from", type=str, default=None
    )
    parser.add_argument(
        "-images_dir",
        help="directory to save images, or None to not save",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-delta_t",
        help="how many timesteps per model step, default 1",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-downsample",
        help="spatial downsampling, number of times to divide by 2",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-skip_initial",
        help="beginning steps of each trajectory to skip",
        type=int,
        default=0,
    )
    parser.add_argument("-t", "--trials", help="number of trials to run", type=int, default=1)
    parser.add_argument(
        "-v",
        "--verbose",
        help="verbose argument passed to trainer",
        type=int,
        default=1,
    )

    return parser.parse_args()


# Main
args = handleArgs(sys.argv)

D = 2
past_steps = 4  # how many steps to look back to predict the next step
rollout_steps = 5
key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)

data = get_data(
    args.data_dir,
    args.train_traj,
    args.val_traj,
    args.test_traj,
    past_steps,
    rollout_steps,
    args.delta_t,
    args.downsample,
    args.skip_initial,
)
