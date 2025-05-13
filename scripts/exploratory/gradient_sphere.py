import time
import argparse
import functools
import numpy as np
from typing_extensions import Callable, Optional

import jax.numpy as jnp
import jax
import jax.random as random
from jaxtyping import ArrayLike
import optax
import equinox as eqx

import ginjax.geometric as geom
import ginjax.ml as ml
import ginjax.models as models
import ginjax.utils as utils


def f(p: jax.Array, coeffs: jax.Array) -> jax.Array:
    """
    Scalar function on the 2d torus at point p

    args:
        p: 2d array of x,y coordinates
        coeffs: 2*D + (2*D choose 2) (=14 for D=2) coefficients to construct our function with
    """
    trig = jnp.concatenate([jnp.sin(p), jnp.cos(p)])
    trig_squared = jnp.einsum("i,j->ij", trig, trig)[jnp.triu_indices(len(trig))]
    library = jnp.concatenate([trig, trig_squared])  # 14 total

    return jnp.einsum("i,i->", library, coeffs)


def metric_f(p: jax.Array) -> jax.Array:
    """
    Metric tensor at point p of the typical round metric on a sphere.

    args:
        p: the point on the surface

    returns:
        2D metric tensor g_ij
    """
    x, _ = p
    return jnp.array([[1, 0], [0, jnp.sin(x) ** 2]])


def metric_inv_f(p: jax.Array) -> jax.Array:
    """
    Inverse metric tensor at point p of the typical round metric on a sphere.

    args:
        p: the point on the surface

    returns:
        2D inverse metric tensor g^ij
    """
    x, _ = p
    return jnp.array([[1, 0], [0, 1 / jnp.sin(x) ** 2]])


def christoffel_f(p: jax.Array) -> jax.Array:
    """
    Christoffel symbols at point p on a surface with a metric tensor given by metric_f

    args:
        p: the point on the surface

    returns:
        Christoffel symbol Lambda^k_ij
    """
    metric_partial = jax.jacfwd(metric_f)(p)
    assert isinstance(metric_partial, jax.Array)
    metric_sum = (
        metric_partial.transpose(1, 2, 0) + metric_partial.transpose(2, 0, 1) - metric_partial
    )
    metric_inv = metric_inv_f(p)
    return 0.5 * jnp.einsum("kl,ijl->kij", metric_inv, metric_sum)


def covariant_derivative(p: jax.Array, f: Callable) -> jax.Array:
    """
    Calculate the covariant derivative at point p of function f

    args:
        p: the point on the surface
        f: the scalar function defined on the surface

    returns:
        the covariant derivative of the gradient, (delta_f)_c;a
    """
    # covariant derivative of a covariant vector, grad
    grad_f = jax.grad(f)
    hessian_f = jax.jacfwd(grad_f)

    grad = grad_f(p)
    hessian = hessian_f(p)
    christoffel = christoffel_f(p)
    return hessian - jnp.einsum("bca,b->ca", christoffel, grad)


def get_dataset(D: int, n_images: int, key: jax.Array) -> tuple[geom.MultiImage, geom.MultiImage]:
    """
    Calculate a dataset of scalar function -> gradient of the scalar function

    args:
        D: the dimension of the space
        n_images: the number of images to generate
        key: random key

    returns:
        input scalar image and output gradient image
    """
    N = 16
    spatial_dims = (N, N)

    # skip 0 because 0 and 2pi are equal
    theta_range = jnp.linspace(0, 2 * jnp.pi, num=spatial_dims[0] + 1)[1:]
    phi_range = jnp.linspace(0, 2 * jnp.pi, num=spatial_dims[1] + 1)[1:]

    # avoid singularities at theta=0,pi,2pi
    theta_range += theta_range[0] / 2
    phi_range += phi_range[0] / 2

    grid = jnp.stack(jnp.meshgrid(theta_range, phi_range, indexing="ij"), axis=-1)
    grid_flattened = grid.reshape((-1, D))

    scalar_images = jnp.zeros((0, 1) + spatial_dims)
    grad_images = jnp.zeros((0, 1) + spatial_dims + (D,))
    covariant_deriv_images = jnp.zeros((0, 1) + spatial_dims + (D, D))

    for _ in range(n_images):
        key, subkey = random.split(key)
        # 130 for full library
        coeffs = random.normal(subkey, shape=(14,))  # hardcoded library size
        f_partial = functools.partial(f, coeffs=coeffs)
        # could maybe start from grad instead of scalar?
        scalar_image = jax.vmap(f_partial)(grid_flattened).reshape((1, 1) + spatial_dims)
        scalar_images = jnp.concatenate([scalar_images, scalar_image])

        grad_image = jax.vmap(jax.grad(f_partial))(grid_flattened).reshape(
            (1, 1) + spatial_dims + (D,)
        )
        grad_images = jnp.concatenate([grad_images, grad_image])

        covariant_deriv_image = jax.vmap(covariant_derivative, in_axes=(0, None))(
            grid_flattened, f_partial
        ).reshape((1, 1) + spatial_dims + (D, D))
        covariant_deriv_images = jnp.concatenate(
            [
                covariant_deriv_images,
                covariant_deriv_image,
            ]
        )

    metric_tensor = geom.GeometricImage(
        jax.vmap(metric_f)(grid_flattened).reshape(spatial_dims + (D, D)),
        0,
        D,
        (False, True),
        (True, True),
    )
    metric_tensor_inv = geom.GeometricImage(
        jax.vmap(metric_inv_f)(grid_flattened).reshape(spatial_dims + (D, D)),
        0,
        D,
        (False, True),
        (True, True),
    )

    X = geom.MultiImage(
        {((), 0): scalar_images},
        D,
        (True, True),
        # metric_tensor,
        # metric_tensor_inv,
    )
    # Y = geom.MultiImage(
    #     {((True, True), 0): covariant_deriv_images},
    #     D,
    #     (True, True),
    #     metric_tensor,
    #     metric_tensor_inv,
    # )
    Y = geom.MultiImage(
        {((True,), 0): grad_images},
        D,
        (True, True),
        # (False, True),
        # metric_tensor,
        # metric_tensor_inv,
    )

    return X, Y


def get_data(
    D: int, n_train: int, n_val: int, n_test: int, normalize: bool, key: jax.Array
) -> tuple[
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
]:
    key, subkey1, subkey2, subkey3 = random.split(key, num=4)
    train_X, train_Y = get_dataset(D, n_train, subkey1)
    val_X, val_Y = get_dataset(D, n_val, subkey2)
    test_X, test_Y = get_dataset(D, n_test, subkey3)

    if normalize:

        for (k, parity), train_block in train_X.items():
            non_test_X = jnp.concatenate([train_block, val_X[(k, parity)]])

            if k == ():
                mean_X = jnp.mean(non_test_X, axis=(0,) + tuple(range(2, 2 + D)))
                std_X = jnp.std(non_test_X, axis=(0,) + tuple(range(2, 2 + D)))
            else:
                mean_X = jnp.zeros((1, non_test_X.shape[1]) + (1,) * D)
                norm = jnp.linalg.norm(
                    non_test_X, axis=tuple(range(non_test_X.ndim - len(k), non_test_X.ndim))
                )
                assert norm.shape == non_test_X.shape[: non_test_X.ndim - len(k)]
                std_X = jnp.std(norm, axis=(0,) + tuple(range(2, 2 + D)))

            train_X[((), 0)] = (train_X[((), 0)] - mean_X) / std_X
            val_X[((), 0)] = (val_X[((), 0)] - mean_X) / std_X
            test_X[((), 0)] = (test_X[((), 0)] - mean_X) / std_X

    return train_X, train_Y, val_X, val_Y, test_X, test_Y


@eqx.filter_jit
def map_and_loss(
    model: models.MultiImageModule,
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    aux_data: Optional[eqx.nn.State] = None,
) -> tuple[jax.Array, Optional[eqx.nn.State]]:
    vmap_model = jax.vmap(model, in_axes=(0, None), out_axes=(0, None))
    pred_y, aux_data = vmap_model(multi_image_x, aux_data)
    loss = ml.smse_loss(pred_y, multi_image_y)

    return loss, aux_data


def train_and_eval(
    data: tuple[geom.MultiImage, ...],
    key: jax.Array,
    model_name: str,
    model: models.MultiImageModule,
    lr: float,
    batch_size: int,
    epochs: int,
    save_model: Optional[str],
    load_model: Optional[str],
    has_aux: bool = False,
    verbose: int = 1,
    is_wandb: bool = False,
) -> tuple[Optional[ArrayLike], Optional[ArrayLike], jax.Array]:
    (
        train_X,
        train_Y,
        val_X,
        val_Y,
        test_X,
        test_Y,
    ) = data
    batch_stats = eqx.nn.State(model) if has_aux else None

    print(f"Model params: {models.count_params(model):,}")

    if load_model is None:
        steps_per_epoch = int(np.ceil(train_X.get_L() / batch_size))
        key, subkey = random.split(key)
        model, batch_stats, train_loss, val_loss = ml.train(
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
            ml.save(f"{save_model}{model_name}_L{train_X.get_L()}_e{epochs}_model.eqx", model)
    else:
        model = ml.load(f"{load_model}{model_name}_L{train_X.get_L()}_e{epochs}_model.eqx", model)

        key, subkey1, subkey2 = random.split(key, num=3)
        train_loss = ml.map_loss_in_batches(
            map_and_loss,
            model,
            train_X,
            train_Y,
            batch_size,
            subkey1,
            aux_data=batch_stats,
        )
        val_loss = ml.map_loss_in_batches(
            map_and_loss,
            model,
            val_X,
            val_Y,
            batch_size,
            subkey2,
            aux_data=batch_stats,
        )

    key, subkey = random.split(key)
    test_loss = ml.map_loss_in_batches(
        map_and_loss,
        model,
        test_X,
        test_Y,
        batch_size,
        subkey,
        aux_data=batch_stats,
    )
    print(f"Test Loss: {test_loss}")

    vmap_model = jax.vmap(model, in_axes=(0, None), out_axes=(0, None))
    out = None
    for gg in geom.make_all_operators(D):
        gg_out = vmap_model(test_X.times_gg_precise(gg), None)[0].times_gg_precise(gg.T)
        out = gg_out if out is None else out + gg_out

    assert out is not None
    out = out / len(geom.make_all_operators(D))
    test_loss1 = ml.smse_loss(out, test_Y)
    print(f"Group Average Test Loss Manual: {test_loss1}")

    return train_loss, val_loss, test_loss


def handleArgs() -> argparse.Namespace:
    parser = utils.get_common_parser()
    parser.add_argument(
        "--wandb-project", help="the wandb project", type=str, default="gradient-sphere"
    )
    return parser.parse_args()


# Main
args = handleArgs()
D = 2

key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)

key, subkey = random.split(key)
data = get_data(D, args.n_train, args.n_val, args.n_test, args.normalize, subkey)

input_keys = data[0].get_signature()
output_keys = data[1].get_signature()

group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    Ms=[3], ks=[0, 1, 2], parities=[0, 1], D=D, operators=group_actions
)
assert conv_filters is not None

train_kwargs = {
    "batch_size": args.batch,
    "epochs": args.epochs,
    "save_model": args.save_model,
    "load_model": args.load_model,
    "verbose": args.verbose,
    "is_wandb": args.wandb,
}

key, *subkeys = random.split(key, num=12)
model_list = [
    (
        "resnet",
        train_and_eval,
        {
            "model": models.ResNet(
                D,
                input_keys,
                output_keys,
                depth=128,
                equivariant=False,
                kernel_size=3,
                key=subkeys[0],
            ),
            "lr": 1e-3,
            **train_kwargs,
        },
    ),
    (
        "resnet_equiv_42",
        train_and_eval,
        {
            "model": models.ResNet(
                D,
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
    num_results=3,
    is_wandb=args.wandb,
    wandb_project=args.wandb_project,
    wandb_entity=args.wandb_entity,
)
