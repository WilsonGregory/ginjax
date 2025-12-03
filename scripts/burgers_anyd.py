import argparse
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import time
from typing_extensions import Self

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
import apebench

import ginjax.geometric as geom
from ginjax import models
from ginjax import ml
import ginjax.data as gc_data
from ginjax import utils


def get_data(
    data_dir: str,
    train_D: int,
    range_test_D: list[int],
    N: int,
    n_train: int,
    n_val: int,
    n_test: int,
    subsample: int,
    past_steps: int,
    key: jax.Array,
) -> tuple[
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    list[geom.MultiImage],
    list[geom.MultiImage],
]:
    if train_D != range_test_D[0]:
        raise ValueError()

    is_torus = True
    n_timesteps = 50
    n_timesteps_int = n_timesteps * subsample  # integrator time steps
    scenario = "diff_burgers"
    diffusion_gamma = 1.5  # this is the default for diff_burgers
    convection_delta = -1.5  # default for diff_burgers

    train_name = f"{train_D}D_{scenario}_N{N}_timesteps{n_timesteps_int}_diffusion{diffusion_gamma}"
    train_path = pathlib.Path(f"{data_dir}") / f"{train_name}_train.npy"
    val_path = pathlib.Path(f"{data_dir}") / f"{train_name}_test.npy"
    if not train_path.is_file() or not val_path.is_file():
        print(f"Generating train data D={train_D}")
        key, subkey = random.split(key)
        train_seed, test_seed = random.randint(subkey, shape=(2,), minval=0, maxval=10000)

        apebench.scraper.scrape_data_and_metadata(
            data_dir,
            scenario=scenario,
            name=train_name,
            num_spatial_dims=train_D,
            num_train_samples=n_train,
            num_test_samples=n_val,
            num_points=N,
            train_seed=int(train_seed),
            test_seed=int(test_seed),
            train_temporal_horizon=n_timesteps_int - 1,
            test_temporal_horizon=n_timesteps_int - 1,
            diffusion_gamma=diffusion_gamma,
        )

    cpu = jax.devices("cpu")[0]
    # (batch,timesteps,tensor,spatial) -> (batch,timesteps,spatial,tensor)
    train_data = jnp.moveaxis(jax.device_put(jnp.load(train_path)[:, ::subsample], cpu), 2, -1)
    val_data = jnp.moveaxis(jax.device_put(jnp.load(val_path)[:, ::subsample], cpu), 2, -1)
    # subsample here for memory efficiency

    constant_fields = geom.MultiImage({}, train_D, is_torus)
    train_X, train_Y = gc_data.batch_time_series(
        geom.MultiImage({(1, 0): train_data}, train_D, is_torus),
        constant_fields,
        n_timesteps,
        past_steps,
        1,
    )
    val_X, val_Y = gc_data.batch_time_series(
        geom.MultiImage({(1, 0): val_data}, train_D, is_torus),
        constant_fields,
        n_timesteps,
        past_steps,
        1,
    )

    test_Xs = []
    test_Ys = []
    key, subkey = random.split(key)
    test_seeds = random.randint(subkey, shape=(len(range_test_D),), minval=0, maxval=10000)
    for D, test_seed in zip(range_test_D, test_seeds):
        # if D=3, N=128, baseline timesteps=50, subsample=8, that takes 10Gb of memory, so split it up
        batch = 1 if D == 3 else n_test

        test_data = jax.device_put(jnp.zeros((0, n_timesteps) + (N,) * D + (D,)), cpu)
        for i in range(n_test // batch):
            # diff_burgers scales diffusion_gamma and convection_delta by D, so we unscale them so
            # that the equation is the same across dimensions.

            # a little awkward because it will say test_test at the end
            test_name = f"{D}D_{scenario}_N{N}_timesteps{n_timesteps_int}_diffusion{diffusion_gamma * D / 2}_i{i}_test"
            test_path = pathlib.Path(f"{data_dir}") / f"{test_name}_test.npy"
            if not test_path.is_file():
                print(f"Generating test data, D={D}")
                key, subkey = random.split(key)
                train_seed, test_seed = random.randint(subkey, shape=(2,), minval=0, maxval=10000)

                apebench.scraper.scrape_data_and_metadata(
                    data_dir,
                    scenario=scenario,
                    name=test_name,
                    num_spatial_dims=D,
                    num_train_samples=0,
                    num_test_samples=batch,
                    num_points=N,
                    test_seed=int(test_seed),
                    train_temporal_horizon=n_timesteps_int - 1,
                    test_temporal_horizon=n_timesteps_int - 1,
                    diffusion_gamma=diffusion_gamma * D / 2,  # may have to scale relative to D
                    convection_delta=convection_delta * D / 2,
                )

            # subsample here for memory efficiency
            test_data_i = jnp.moveaxis(
                jax.device_put(jnp.load(test_path)[:, ::subsample], cpu), 2, -1
            )
            test_data = jnp.concatenate([test_data, test_data_i], axis=0)

        if D == 1:
            test_data = test_data[..., 0]

        constant_fields = geom.MultiImage({}, D, is_torus)

        test_X, test_Y = gc_data.batch_time_series(
            geom.MultiImage({(1, 0): test_data}, D, is_torus),
            constant_fields,
            n_timesteps,
            past_steps,
            1,
        )

        test_Xs.append(test_X)
        test_Ys.append(test_Y)

    return train_X, train_Y, val_X, val_Y, test_Xs, test_Ys


def plot_multi_image(
    input_multi_image: geom.MultiImage,
    actual_multi_image: geom.MultiImage,
    test_multi_image: geom.MultiImage,
    save_loc: str,
    title: str = "",
):
    """
    Plot vector x and y components of two MultiImages, and the differences between them. Each row
    is a component, and the columns are actual, test, and diff. If the image is 3D, plot the middle
    slice.

    args:
        input_multi_image_full: the input image, with past_steps number of timesteps
        actual_multi_image: the ground truth MultiImage
        test_multi_image: the predicted MultiImage
        save_loc: file location to save the image
        title: additional str to add to title, will be "test {title} {col}"
            "actual {title} {col}"
    """
    print(
        f"Printed image relative error: {ml.l2_rel_error(test_multi_image, actual_multi_image):.4f}%"
    )

    if input_multi_image.get_n_leading() == 2:
        input_multi_image = input_multi_image.get_one(keepdims=False)

    if actual_multi_image.get_n_leading() == 2:
        actual_multi_image = actual_multi_image.get_one(keepdims=False)

    if test_multi_image.get_n_leading() == 2:
        test_multi_image = test_multi_image.get_one(keepdims=False)

    # images now no longer have batch dimension

    N = input_multi_image.get_spatial_dims()[0]
    if input_multi_image.D == 3:
        img_arr = jnp.concatenate(
            [
                input_multi_image[((False,), 0)][:, :, N // 2],
                test_multi_image[((False,), 0)][:, :, N // 2],
                actual_multi_image[((False,), 0)][:, :, N // 2],
            ]
        )
    else:
        img_arr = jnp.concatenate(
            [
                input_multi_image.to_vector(),
                test_multi_image.to_vector(),
                actual_multi_image.to_vector(),
            ]
        )

    vmax = float(jnp.max(jnp.abs(img_arr)))
    vmin = -1 * vmax

    timesteps = len(input_multi_image[((False,), 0)])

    nrows = test_multi_image.D
    ncols = timesteps + 3
    # figsize is 6 per col, 6 per row, (cols,rows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))
    for component in range(test_multi_image.D):
        comp_name = ["x", "y", "z"][component]

        input_multi_image_comp = input_multi_image.get_component(component, timesteps)
        for i, input_image in enumerate(input_multi_image_comp.to_images()):
            if input_image.D == 3:
                input_image = geom.GeometricImage(input_image.data[N // 2], input_image.parity, 2)

            input_image.plot(
                axes[component, i],
                title=f"input {(i+1)-timesteps} {title} {comp_name}",
                vmin=vmin,
                vmax=vmax,
                colorbar=True,
            )

        actual_multi_image_comp = actual_multi_image.get_component(component)
        test_multi_image_comp = test_multi_image.get_component(component)

        actual_image = actual_multi_image_comp.to_images()[0]
        test_image = test_multi_image_comp.to_images()[0]

        if actual_image.D == 3:
            actual_image = geom.GeometricImage(actual_image.data[N // 2], actual_image.parity, 2)
        if test_image.D == 3:
            test_image = geom.GeometricImage(test_image.data[N // 2], test_image.parity, 2)

        diff = actual_image - test_image
        diff_max = float(jnp.max(jnp.abs(diff.data)))
        diff_l2 = jnp.mean(diff.data**2)
        diff_rel_err = jnp.mean(jnp.abs(diff.data / actual_image.data)) * 100
        diff_title = f"diff (l2: {diff_l2:.3e}, rel. err: {diff_rel_err:.2f}%)"

        actual_image.plot(
            axes[component, timesteps],
            title=f"output {title} {comp_name}",
            vmin=vmin,
            vmax=vmax,
            colorbar=True,
        )
        test_image.plot(
            axes[component, timesteps + 1],
            title=f"pred {title} {comp_name}",
            vmin=vmin,
            vmax=vmax,
            colorbar=True,
        )
        diff.plot(
            axes[component, timesteps + 2],
            title=diff_title,
            vmin=-diff_max,
            vmax=diff_max,
            colorbar=True,
        )

    plt.tight_layout()
    plt.savefig(save_loc)
    plt.close(fig)


@eqx.filter_jit
def map_residual(
    model: models.MultiImageModule,
    multi_image_x: geom.MultiImage,
    aux_data: eqx.nn.State | None = None,
) -> tuple[geom.MultiImage, eqx.nn.State | None]:
    residual, aux_data = jax.vmap(model, in_axes=(0, None), out_axes=(0, None), axis_name="batch")(
        multi_image_x, aux_data
    )

    # add the last timestep to the residual
    pred_y = residual.empty()
    for ((k, parity), img_in), img_resid in zip(multi_image_x.items(), residual.values()):
        pred_y.append(k, parity, img_in[:, -1:] + img_resid)

    return pred_y, aux_data


@eqx.filter_jit
def map_and_loss_rel_error(
    model: models.MultiImageModule,
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    aux_data: eqx.nn.State | None = None,
) -> tuple[jax.Array, eqx.nn.State | None]:
    """
    Calculates both the smse_loss and the rel_error.
    """
    pred_y, aux_data = map_residual(model, multi_image_x, aux_data)

    loss = ml.smse_loss(pred_y, multi_image_y)
    rel_error = ml.l2_rel_error(pred_y, multi_image_y)
    return jnp.stack([loss, rel_error]), aux_data


@eqx.filter_jit
def map_and_loss(
    model: models.MultiImageModule,
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    aux_data: eqx.nn.State | None = None,
) -> tuple[jax.Array, eqx.nn.State | None]:
    pred_y, aux_data = map_residual(model, multi_image_x, aux_data)

    return ml.smse_loss(pred_y, multi_image_y), aux_data


def train_and_eval(
    data: tuple[
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
        list[geom.MultiImage],
        list[geom.MultiImage],
    ],
    key: jax.Array,
    model_name: str,
    model: models.AnyDimensionalModel,
    lr: float,
    test_conv_filters: list[tuple[geom.MultiImage, geom.MultiImage]],
    batch_size: int,
    test_batch_size: int,
    epochs: int,
    save_model: str | None,
    load_model: str | None,
    images_dir: str | None,
    has_aux: bool = False,
    verbose: int = 1,
    is_wandb: bool = False,
) -> tuple[jax.Array | None, ...]:
    train_X, train_Y, val_X, val_Y, test_d_X, test_d_Y = data
    N = train_X.get_spatial_dims()[0]
    batch_stats = eqx.nn.State(model) if has_aux else None
    model_name_extended = f"{model_name}_L{train_X.get_L()}_N{N}_e{epochs}"

    print(f"Model params: {models.count_params(model):,}")

    if load_model is None:
        steps_per_epoch = int(np.ceil(train_X.get_L() / batch_size))
        key, subkey = random.split(key)
        trained_model, batch_stats, train_loss, val_loss = ml.train(
            train_X,
            train_Y,
            map_and_loss,
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
            is_wandb=is_wandb,
        )

        if save_model is not None:
            # TODO: need to save batch_stats as well
            ml.save(f"{save_model}{model_name_extended}_model.eqx", trained_model)
    else:
        trained_model = ml.load(f"{load_model}{model_name_extended}_model.eqx", model)

        key, subkey1, subkey2 = random.split(key, num=3)
        train_loss = ml.map_loss_in_batches(
            map_and_loss,
            trained_model,
            train_X,
            train_Y,
            batch_size,
            subkey1,
            aux_data=batch_stats,
        )
        val_loss = ml.map_loss_in_batches(
            map_and_loss,
            trained_model,
            val_X,
            val_Y,
            batch_size,
            subkey2,
            aux_data=batch_stats,
        )

    assert isinstance(trained_model, models.AnyDimensionalModel)
    test_losses = []
    for test_X, test_Y, (conv_filters, upsample_filters) in zip(
        test_d_X, test_d_Y, test_conv_filters
    ):
        if test_X.D == train_X.D:
            trained_model_d = trained_model
        else:
            key, subkey = random.split(key)
            # rescale can be true or false when using zero_sum, the effect is small
            trained_model_d = trained_model.convertD(
                conv_filters, False, subkey, upsample_filters=upsample_filters
            )

        key, subkey = random.split(key)
        test_loss = ml.map_loss_in_batches(
            map_and_loss_rel_error,
            trained_model_d,
            test_X,
            test_Y,
            test_batch_size,
            subkey,
            aux_data=batch_stats,
        )
        print(f"Test Loss D={test_X.D}: {test_loss[0]}")
        print(f"Test Relative Error D={test_X.D}: {test_loss[1]:.4f}%")

        test_losses.append(test_loss[0])
        test_losses.append(test_loss[1])

        if images_dir:
            pred_y, _ = map_residual(trained_model_d, test_X.get_one(), batch_stats)

            plot_multi_image(
                test_X.get_one(),
                test_Y.get_one(),
                pred_y.get_one(),
                f"{images_dir}{model_name_extended}_D{test_X.D}.png",
                "burgers vector",
            )

    return train_loss, val_loss, *test_losses


def handleArgs() -> argparse.Namespace:
    parser = utils.get_common_parser()
    parser.add_argument(
        "--train-D", help="dimension of data to train on", choices=[1, 2, 3], default=2, type=int
    )
    parser.add_argument(
        "--max-test-D",
        help="maximum dimension of data to test on",
        choices=[1, 2, 3],
        default=3,
        type=int,
    )
    parser.add_argument("-N", help="spatial size", type=int, default=128)
    parser.add_argument(
        "--past-steps", help="number of past steps to use as input", type=int, default=4
    )
    parser.add_argument(
        "--subsample", help="how much to subsample the trajectories", type=int, default=8
    )
    parser.add_argument("--test-batch", help="batch size for test data", type=int, default=1)
    # need do to --wandb to activate, also need --wandb-entity your_wandb_name_here
    parser.add_argument(
        "--wandb-project", help="the wandb project", type=str, default="burgers-anyd"
    )

    return parser.parse_args()


# Main
args = handleArgs()
assert args.train_D <= args.max_test_D

range_test_D = list(range(args.train_D, args.max_test_D + 1))

key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)

key, subkey = random.split(key)
data = get_data(
    args.data,
    args.train_D,
    range_test_D,
    args.N,
    args.n_train,
    args.n_val,
    args.n_test,
    args.subsample,
    args.past_steps,
    subkey,
)

input_keys = data[0].get_signature()
output_keys = data[1].get_signature()

group_actions = geom.make_all_operators(args.train_D)
conv_filters = geom.get_invariant_filters(
    Ms=[3],
    ks=[0, 1, 2],
    parities=[0],
    D=args.train_D,
    operators=group_actions,
    scale=geom.FilterScaling.ZERO_SUM,
)
upsample_filters = geom.get_invariant_filters(
    Ms=[2],
    ks=[0, 1, 2],
    parities=[0],
    D=args.train_D,
    operators=group_actions,
    scale=geom.FilterScaling.ZERO_SUM,
)

test_conv_filters = []
for D in range_test_D:
    group_actions_d = geom.make_all_operators(D)
    conv_filters_d = geom.get_invariant_filters(
        Ms=[3],
        ks=[0, 1, 2],
        parities=[0],
        D=D,
        operators=group_actions_d,
        scale=geom.FilterScaling.ZERO_SUM,
    )
    upsample_filters_d = geom.get_invariant_filters(
        Ms=[2],
        ks=[0, 1, 2],
        parities=[0],
        D=D,
        operators=group_actions_d,
        scale=geom.FilterScaling.ZERO_SUM,
    )
    test_conv_filters.append((conv_filters_d, upsample_filters_d))


train_kwargs = {
    "test_conv_filters": test_conv_filters,
    "batch_size": args.batch,
    "test_batch_size": args.test_batch,
    "epochs": args.epochs,
    "save_model": args.save_model,
    "load_model": args.load_model,
    "images_dir": args.images_dir,
    "verbose": args.verbose,
    "is_wandb": args.wandb,
}

padding_mode = "CIRCULAR" if data[0].is_torus == (True,) * args.train_D else "ZEROS"
key, *subkeys = random.split(key, num=13)
model_list = [
    (
        "unetBase_equiv48",
        train_and_eval,
        {
            "model": models.UNet(
                args.train_D,
                input_keys,
                output_keys,
                depth=48,
                num_downsamples=3 if args.N <= 64 else 4,
                activation_f=jax.nn.gelu,
                conv_filters=conv_filters,
                upsample_filters=upsample_filters,
                key=subkeys[0],
            ),
            "lr": 4e-4,  # 4e-4 to 6e-4 works, larger sometimes explodes
            **train_kwargs,
        },
    ),
    (
        "lastStepIdentity",
        train_and_eval,
        {"model": models.LastStepIdentity(residual=True), "lr": 1, **train_kwargs},
    ),
    (
        "resnet_equiv_42",
        train_and_eval,
        {
            "model": models.ResNet(
                args.train_D,
                input_keys,
                output_keys,
                depth=42,
                conv_filters=conv_filters,
                use_group_norm=False,
                key=subkeys[1],
            ),
            "lr": 7e-4,
            **train_kwargs,
        },
    ),
]

key, subkey = random.split(key)

# Use this for benchmarking the models with known learning rates.
results = ml.benchmark(
    lambda _: data,
    model_list,
    subkey,
    "",
    [0],
    benchmark_type=ml.BENCHMARK_NONE,
    num_trials=args.n_trials,
    num_results=2 + len(range_test_D) * 2,  # train,val, l2 and rel error per test
    is_wandb=args.wandb,
    wandb_project=args.wandb_project,
    wandb_entity=args.wandb_entity,
)  # (trials,benchmark,models,outputs)

print(results)
mean_results = jnp.mean(results, axis=0)  # (benchmark_vals,models,outputs)
std_results = jnp.std(results, axis=0)
print("Mean over trials", mean_results, sep="\n")
print("Stdev over trials", std_results, sep="\n")
