import jax
import jax.random as random
import jax.numpy as jnp

import ginjax.geometric as geom
import ginjax.ml as ml

D = 2
N = 3
M = N
in_c = 3

key = random.PRNGKey(3)

# key, subkey1, subkey2 = random.split(key, num=3)
# eigvecs = random.orthogonal(subkey1, D, shape=(N,) * D)
# metric_tensor_data = jax.vmap(jnp.diag)(
#     random.uniform(subkey2, shape=(N**D, D)) * 10 + 0.5
# ).reshape((N,) * D + (D, D))
# # eigvals = jax.vmap(jnp.diag)(random.uniform(subkey2, shape=(N**D, D)) * 10 + 0.5)
# # metric_tensor_data = jnp.einsum(
# #     "...ij,...jk,...kl->...il",
# #     eigvecs,
# #     eigvals.reshape((N,) * D + (D, D)),
# #     jnp.moveaxis(eigvecs, -1, -2),
# # )
# metric_tensor = geom.GeometricImage(metric_tensor_data, 0, D, covariant_axes=(True, True))
# metric_tensor_inv = geom.GeometricImage(
#     geom.get_metric_inverse(metric_tensor_data), 0, D, covariant_axes=(False, False)
# )

# # filters we only want contravariant indices
# operators = geom.make_all_operators(D)
# conv_filters = geom.get_invariant_filters([M], [0, 1, 2], [0, 1], D, operators)

# # small test
# key, subkey = random.split(key)
# multi_image1 = geom.MultiImage(
#     {((False,), 0): random.normal(subkey, shape=(in_c,) + (N,) * D + (D,))},
#     D,
#     True,
#     metric_tensor,
#     metric_tensor_inv,
# )
# key, subkey = random.split(key)
# out_signature = geom.Signature(((((True,), 0), in_c),))
# conv = ml.ConvContract(
#     multi_image1.get_signature(), out_signature, conv_filters, use_bias=False, key=subkey
# )

# print(jnp.max(jnp.abs(multi_image1.to_vector() - conv(multi_image1).to_vector())))

# multi_image2 = multi_image1.empty()
# multi_image2[((True,), 0)] = multi_image1[((False,), 0)]

# # print(multi_image1)
# # print(multi_image2)

# # for gg in operators:
# #     first = multi_image1.times_gg_precise(gg)
# #     second = multi_image2.times_gg_precise(gg)
# #     print(jnp.max(jnp.abs(first.to_vector() - second.to_vector())))

# # first = conv(multi_image1.times_gg_precise(gg))
# # second = conv(multi_image1).times_gg_precise(gg)
# # assert first.get_signature() == second.get_signature() == out_signature
# # assert first.__eq__(second, 1e-4, 1e-4)  # should break?

# key, subkey = random.split(key)
# data = random.normal(subkey, shape=(N,) * D + (D,))
# image1 = geom.GeometricImage(data, 0, D, covariant_axes=True)
# image2 = geom.GeometricImage(data, 0, D, covariant_axes=False)
# print(image1)
# print(image2)
# for gg in operators:
#     first = image1.times_gg_precise(gg, metric_tensor)
#     second = image2.times_gg_precise(gg, metric_tensor)
#     # print(first.data)
#     # print(second.data)
#     print(jnp.max(jnp.abs(first.data - second.data)))

#     # print(metric_tensor[0, 0] @ gg @ metric_tensor_inv[0, 0])
#     print(gg)


# key, subkey = random.split(key)
# Q = random.orthogonal(subkey, D)
# print(metric_tensor[0, 0])
# print(metric_tensor_inv[0, 0])
# print(metric_tensor[0, 0] @ Q @ metric_tensor_inv[0, 0])
# print(Q)


metric = jnp.array([[1, 0], [0, 0.5]])
metric_inv = jnp.array([[1, 0], [0, 2]])
key, subkey = random.split(key)
Q = random.orthogonal(subkey, 2)
rot_metric = Q.T @ metric @ Q
rot_metric_inv = Q.T @ metric_inv @ Q
print(rot_metric)
print(rot_metric_inv)
print(rot_metric @ rot_metric_inv)
print(rot_metric_inv @ rot_metric)
