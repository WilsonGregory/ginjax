import jax
import jax.numpy as jnp
import jax.random as random

import ginjax.geometric as geom

key = random.PRNGKey(0)
D = 2
operators = geom.make_all_operators(D)

metric = jnp.array([[1, 0], [0, 0.5]])
H = jnp.array([[1, 0], [0, 1 / jnp.sqrt(2)]])
H_inv = jnp.array([[1, 0], [0, jnp.sqrt(2)]])
assert jnp.allclose(metric, H.T @ H)
assert jnp.allclose(jnp.eye(D), H @ H_inv)
assert jnp.allclose(jnp.eye(D), H_inv @ H)

key, subkey = random.split(key)
a_upper = random.normal(subkey, shape=(D,))
key, subkey = random.split(key)
b_upper = random.normal(subkey, shape=(D,))
b_lower = metric @ b_upper

print(a_upper @ b_lower, a_upper @ metric @ b_upper)

key, subkey = random.split(key)
Q = random.orthogonal(subkey, D)

print(
    "inner prod: ",
    # (Q @ a_upper).T @ (Q @ b_lower),
    # (H_inv @ Q @ a_upper).T @ metric @ (H_inv @ Q @ b_lower),
    a_upper @ metric @ b_upper,
    (H_inv @ Q @ H @ a_upper).T @ metric @ (H_inv @ Q @ H @ b_upper),  # <- this is what I want?
    # (Q @ H @ a_upper).T @ (Q @ H @ b_upper),
)

key, subkey = random.split(key)
c_upper_lower = random.normal(subkey, shape=(D, D))
print(
    "matrix mult:",
    Q @ c_upper_lower @ a_upper,
    Q @ c_upper_lower @ Q.T @ Q @ a_upper,
)

inv_filters = geom.get_invariant_filters_list([3], [1], [0], D, operators, scale="one")
print(len(inv_filters))

# filter1 = geom.GeometricImage(inv_filters[0].data, inv_filters[0].parity, D, False, metric)
# rot_90 = operators[5]

# print("filter1")
# print(filter1.data)
# rot_filter1 = filter1.times_group_element(rot_90, jax.lax.Precision.HIGHEST)
# print(rot_filter1.data)

# H_inv_filter1 = geom.GeometricImage(
#     jnp.einsum("ij,...j->...i", H_inv, filter1.data),
#     filter1.parity,
#     D,
#     filter1.is_torus,
#     filter1.metric,
# )
# print("H_inv_filter1")
# print(H_inv_filter1.data)
# rot_H_inv_filter1 = H_inv_filter1.times_group_element(rot_90, jax.lax.Precision.HIGHEST)
# print(rot_H_inv_filter1.data)

# key, subkey = random.split(key)
# N = 3
# image = geom.GeometricImage(random.normal(subkey, shape=(N,) * D + (D,)), 0, D)
