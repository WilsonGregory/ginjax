import jax
import jax.random as random
import jax.numpy as jnp

import ginjax.geometric as geom
import ginjax.ml as ml

# This experiments tests whether a convolution conv(A_up) == conv(A_down)

D = 2
M = 3
N = 5
in_c = 3

key = random.PRNGKey(0)

key, subkey1, subkey2 = random.split(key, num=3)
eigvecs = random.orthogonal(subkey1, D, shape=(N,) * D)
eigvals = jax.vmap(jnp.diag)(random.uniform(subkey2, shape=(N**D, D)) + 0.5)
metric_tensor_data = jnp.einsum(
    "...ij,...jk,...kl->...il",
    eigvecs,
    eigvals.reshape((N,) * D + (D, D)),
    jnp.moveaxis(eigvecs, -1, -2),
)
metric_tensor = geom.GeometricImage(metric_tensor_data, 0, D, covariant_axes=(True, True))
metric_tensor_inv = geom.get_metric_inverse(metric_tensor)

# filters we only want contravariant indices
operators = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters([M], [0, 1, 2], [0, 1], D, operators)

# small test
key, subkey = random.split(key)
multi_image1 = geom.MultiImage(
    {((False,), 0): random.normal(subkey, shape=(in_c,) + (N,) * D + (D,))},
    D,
    True,
    metric_tensor,
    metric_tensor_inv,
)

multi_image2 = multi_image1.lower_all(jax.lax.Precision.HIGHEST)
# multi_image2[((True,), 0)] = multi_image1.lower_all(jax.lax.Precision.HIGHEST)[((False,), 0)]

key, subkey = random.split(key)
# define ConvContract lower -> upper
conv = ml.ConvContract(
    multi_image2.get_signature(),
    multi_image1.get_signature(),
    conv_filters,
    use_bias=False,
    key=subkey,
)

# this conv goes upper -> upper
conv_upper = ml.ConvContract(
    multi_image1.get_signature(),
    multi_image1.get_signature(),
    conv_filters,
    use_bias=False,
    key=subkey,
)
# set the weights so they are the same
conv_upper.weights[((False,), 0)][((False,), 0)] = conv.weights[((True,), 0)][((False,), 0)]
first = conv_upper(multi_image1)
second = conv(multi_image2)
print(jnp.max(jnp.abs(first.to_vector() - second.to_vector())))

exit()

for gg in operators:
    first = conv(multi_image1.times_gg_precise(gg))
    second = conv(multi_image1).times_gg_precise(gg)
    assert first.get_signature() == second.get_signature() == out_signature
    assert first.__eq__(second, 1e-4, 1e-4)  # should break?

# big test
ks = [(), (True,), (False,), (True, True), (True, False), (False, True), (False, False)]
parities = [0, 1]
ks_ps_prod = list(it.product(ks, parities))
key, *subkeys = random.split(key, num=len(ks_ps_prod) + 1)
multi_image2 = geom.MultiImage(
    {
        (k, p): random.normal(subkeys[i], shape=(in_c,) + (N,) * D + (D,) * len(k))
        for i, (k, p) in enumerate(ks_ps_prod)
    },
    D,
    True,
    metric_tensor,
    metric_tensor_inv,
)

key, subkey = random.split(key)
conv = ml.ConvContract(
    multi_image2.get_signature(),
    multi_image2.get_signature(),
    conv_filters,
    use_bias=False,
    key=subkey,
)

for gg in operators:
    first = conv(multi_image2.times_gg_precise(gg))
    second = conv(multi_image2).times_gg_precise(gg)
    assert first.__eq__(second, 1e-4, 1e-4)  # should break?
