import argparse
import numpy as np
import time
import matplotlib.pyplot as plt

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
    train_D: int,
    range_test_D: list[int],
    N: int,
    n_train: int,
    n_val: int,
    n_test: int,
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
    n_timesteps = 21

    key, subkey = random.split(key)
    train_seed, test_seed = random.randint(subkey, shape=(2,), minval=0, maxval=10000)
    # (batch,timesteps,tensor,spatial), timesteps defaults to 51 for train, 201 for test
    train_data, val_data, _ = apebench.scraper.scrape_data_and_metadata(
        None,
        scenario="diff_burgers",
        num_spatial_dims=train_D,
        num_train_samples=n_train,
        num_test_samples=n_val,
        num_points=N,
        train_seed=train_seed,
        test_seed=test_seed,
        train_temporal_horizon=n_timesteps - 1,
        test_temporal_horizon=n_timesteps - 1,
    )
    # -> (batch,timesteps,spatial,tensor)
    train_data = jax.device_put(jnp.moveaxis(train_data, 2, -1), jax.devices("cpu")[0])
    val_data = jax.device_put(jnp.moveaxis(val_data, 2, -1), jax.devices("cpu")[0])

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
        print(f"Generating test data, D={D}")
        _, test_data, _ = apebench.scraper.scrape_data_and_metadata(
            None,
            scenario="diff_burgers",
            num_spatial_dims=D,
            num_train_samples=0,
            num_test_samples=n_test,
            num_points=N,
            test_seed=test_seed,
            train_temporal_horizon=n_timesteps - 1,
            test_temporal_horizon=n_timesteps - 1,
        )
        test_data = jax.device_put(jnp.moveaxis(test_data, 2, -1), jax.devices("cpu")[0])

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
    test_multi_image: geom.MultiImage,
    actual_multi_image: geom.MultiImage,
    save_loc: str,
    title: str = "",
    minimal: bool = False,
):
    """
    Plot vector x and y components of two MultiImages, and the differences between them. Each row
    is is a component, and the columns are test, actual, and diff

    args:
        test_multi_image: the predicted MultiImage
        actual_multi_image: the ground truth MultiImage
        save_loc: file location to save the image
        title: additional str to add to title, will be "test {title} {col}"
            "actual {title} {col}"
        minimal: if minimal, no titles, colorbars, or axes labels
    """
    print(f"in print: {ml.l1_rel_error(test_multi_image, actual_multi_image):.4f}%")

    if test_multi_image.get_n_leading() == 2:
        test_multi_image = test_multi_image.get_one(keepdims=False)

    if actual_multi_image.get_n_leading() == 2:
        actual_multi_image = actual_multi_image.get_one(keepdims=False)

    img_arr = jnp.stack([test_multi_image[((False,), 0)], actual_multi_image[((False,), 0)]])
    vmax = float(jnp.max(jnp.abs(img_arr)))
    vmin = -1 * vmax

    nrows = test_multi_image.D
    ncols = 3
    # figsize is 6 per col, 6 per row, (cols,rows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))
    for component in range(test_multi_image.D):

        test_multi_image_comp = test_multi_image.get_component(component)
        actual_multi_image_comp = actual_multi_image.get_component(component)

        test_image = test_multi_image_comp.to_images()[0]
        actual_image = actual_multi_image_comp.to_images()[0]

        diff = actual_image - test_image
        diff_max = float(jnp.max(jnp.abs(diff.data)))
        diff_l2 = jnp.mean(diff.data**2)
        diff_rel_err = jnp.mean(jnp.abs(diff.data / actual_image.data)) * 100

        if minimal:
            test_title = ""
            actual_title = ""
            diff_title = ""
            colorbar = False
        else:
            test_title = f"test {title} {'y' if component else 'x'}"
            actual_title = f"actual {title} {'y' if component else 'x'}"
            diff_title = f"diff (l2: {diff_l2:.3e}, rel. err: {diff_rel_err:.2f}%)"
            colorbar = True

        test_image.plot(
            axes[component, 0], title=test_title, vmin=vmin, vmax=vmax, colorbar=colorbar
        )
        actual_image.plot(
            axes[component, 1], title=actual_title, vmin=vmin, vmax=vmax, colorbar=colorbar
        )
        diff.plot(
            axes[component, 2], title=diff_title, vmin=-diff_max, vmax=diff_max, colorbar=colorbar
        )

    plt.tight_layout()
    plt.savefig(save_loc)
    plt.close(fig)


@eqx.filter_jit
def map_and_rel_error(
    model: models.MultiImageModule,
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    aux_data: eqx.nn.State | None = None,
) -> tuple[jax.Array, eqx.nn.State | None]:
    residual, aux_data = jax.vmap(model, in_axes=(0, None), out_axes=(0, None), axis_name="batch")(
        multi_image_x, aux_data
    )

    # add the last timestep to the residual
    pred_y = residual.empty()
    for ((k, parity), img_in), img_resid in zip(multi_image_x.items(), residual.values()):
        pred_y.append(k, parity, img_in[:, -1:] + img_resid)

    return ml.l1_rel_error(pred_y, multi_image_y), aux_data


@eqx.filter_jit
def map_and_loss(
    model: models.MultiImageModule,
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    aux_data: eqx.nn.State | None = None,
) -> tuple[jax.Array, eqx.nn.State | None]:
    residual, aux_data = jax.vmap(model, in_axes=(0, None), out_axes=(0, None), axis_name="batch")(
        multi_image_x, aux_data
    )

    # add the last timestep to the residual. Maybe I already have a function for this?
    pred_y = residual.empty()
    for ((k, parity), img_in), img_resid in zip(multi_image_x.items(), residual.values()):
        pred_y.append(k, parity, img_in[:, -1:] + img_resid)

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
            ml.save(
                f"{save_model}{model_name}_L{train_X.get_L()}_N{N}_e{epochs}_model.eqx",
                trained_model,
            )
    else:
        trained_model = ml.load(
            f"{load_model}{model_name}_L{train_X.get_L()}_N{N}_e{epochs}_model.eqx", model
        )

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
            trained_model_d = trained_model.convertD(
                conv_filters, False, subkey, upsample_filters=upsample_filters
            )

        key, subkey = random.split(key)
        test_loss = ml.map_loss_in_batches(
            map_and_loss,
            trained_model_d,
            test_X,
            test_Y,
            test_batch_size,
            subkey,
            aux_data=batch_stats,
        )
        print(f"Test Loss D={test_X.D}: {test_loss}")
        test_losses.append(test_loss)

        key, subkey = random.split(key)
        test_rel_error = ml.map_loss_in_batches(
            map_and_rel_error,
            trained_model_d,
            test_X,
            test_Y,
            test_batch_size,
            subkey,
            aux_data=batch_stats,
        )
        print(f"Test Relative Error D={test_X.D}: {test_rel_error:.4f}%")
        test_losses.append(test_loss)

        if test_X.D == train_X.D:
            trained_model_rescale_d = trained_model
        else:
            key, subkey = random.split(key)
            trained_model_rescale_d = trained_model.convertD(
                conv_filters, True, subkey, upsample_filters=upsample_filters
            )

        key, subkey = random.split(key)
        test_loss = ml.map_loss_in_batches(
            map_and_loss,
            trained_model_rescale_d,
            test_X,
            test_Y,
            test_batch_size,
            subkey,
            aux_data=batch_stats,
        )
        print(f"Test Loss rescale D={test_X.D}: {test_loss}")
        test_losses.append(test_loss)

        key, subkey = random.split(key)
        test_rel_error = ml.map_loss_in_batches(
            map_and_rel_error,
            trained_model_rescale_d,
            test_X,
            test_Y,
            test_batch_size,
            subkey,
            aux_data=batch_stats,
        )
        print(f"Test Relative Error rescale D={test_X.D}: {test_rel_error:.4f}%")
        test_losses.append(test_loss)

    if images_dir:
        pred_y, _ = jax.vmap(trained_model, in_axes=(0, None), out_axes=(0, None))(
            val_X.get_one(), batch_stats
        )
        plot_multi_image(
            pred_y,
            val_Y.get_one(),
            f"{images_dir}{model_name}_L{train_X.get_L()}_e{epochs}.png",
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
        "--rollout-steps",
        help="number of steps to rollout in test",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--past-steps", help="number of past steps to use as input", type=int, default=4
    )
    parser.add_argument("--test-batch", help="batch size for test data", type=int, default=1)
    # need do to --wandb to activate, also need --wandb-entity your_wandb_name_here
    parser.add_argument("--wandb-project", help="the wandb project", type=str, default="cfd-anyd")

    return parser.parse_args()


# Main
args = handleArgs()
assert args.train_D <= args.max_test_D

range_test_D = list(range(args.train_D, args.max_test_D + 1))

key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)

key, subkey = random.split(key)
data = get_data(
    args.train_D,
    range_test_D,
    args.N,
    args.n_train,
    args.n_val,
    args.n_test,
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
    scale=geom.FilterScaling.ZERO_SUM,  # don't want zero sum for M=2
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
        scale=geom.FilterScaling.ZERO_SUM,  # don't want zero sum for M=2
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
                key=subkeys[8],
            ),
            "lr": 4e-4,  # 4e-4 to 6e-4 works, larger sometimes explodes
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
    num_results=2 + len(range_test_D),
    is_wandb=args.wandb,
    wandb_project=args.wandb_project,
    wandb_entity=args.wandb_entity,
)

rollout_res = results[..., 3:]
non_rollout_res = jnp.concatenate(
    [results[..., :3], jnp.sum(rollout_res, axis=-1, keepdims=True)], axis=-1
)
print(non_rollout_res)
mean_results = jnp.mean(
    non_rollout_res, axis=0
)  # includes the sum of rollout. (benchmark_vals,models,outputs)
std_results = jnp.std(non_rollout_res, axis=0)
print("Mean", mean_results, sep="\n")
