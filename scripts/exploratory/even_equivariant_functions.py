import time
import math
import itertools as it
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random

import ginjax.geometric as geom

N = 4
D = 2
spatial_dims = (2, 3)
spatial_size = math.prod(spatial_dims)

key = random.PRNGKey(time.time_ns())
key, subkey = random.split(key)

translations = jnp.array(
    np.array([key for key in it.product(*list(range(N) for N in spatial_dims))])
)

translations_matrices = []
# (N**D,N**D)
possible_images = [
    geom.GeometricImage(image_basis.reshape(spatial_dims), 0, D, is_torus=True)
    for image_basis in jnp.eye(spatial_size)
]

for tau in translations:
    tau_matrix = []
    for image in possible_images:
        tau_matrix.append(image.translate(tau).data.ravel())

    translations_matrices.append(jnp.stack(tau_matrix))

gg_matrices = []
for gg in geom.make_all_operators(D):
    gg_matrix = []
    for image in possible_images:
        gg_matrix.append(image.times_gg_precise(gg).data.ravel())

    gg_matrices.append(jnp.stack(gg_matrix))

linear_maps = []
for linear_map in jnp.eye(spatial_size * spatial_size):
    linear_map = linear_map.reshape((spatial_size, spatial_size))
    linear_map_orbit = []
    for tau_matrix in translations_matrices:
        assert jnp.allclose(tau_matrix @ tau_matrix.T, jnp.eye(spatial_size))
        # print(tau_matrix @ tau_matrix.T) # these are currently permutations, so transpose inverts
        linear_map_orbit.append(tau_matrix @ linear_map @ tau_matrix.T)

    linear_maps.append(jnp.sum(jnp.stack(linear_map_orbit), axis=0))

od_linear_maps = []
for linear_map in linear_maps:
    linear_map_orbit = []
    for gg_matrix in gg_matrices:
        linear_map_orbit.append(gg_matrix @ linear_map @ gg_matrix.T)

    od_linear_maps.append(jnp.sum(jnp.stack(linear_map_orbit), axis=0))


od_linear_maps = jnp.stack(od_linear_maps)

_, s, vt = jnp.linalg.svd(od_linear_maps.reshape((len(od_linear_maps), -1)), full_matrices=False)
print(s)
sbig = s > geom.TINY


# normalize the amplitudes so they max out at +/- 1.
amps = vt[sbig] / jnp.max(jnp.abs(vt[sbig]), axis=1, keepdims=True)
# make sure the amps are positive, generally
amps = jnp.round(amps, decimals=5) + 0.0
signs = jnp.sign(jnp.sum(amps, axis=1, keepdims=True))
signs = jnp.where(
    signs == 0, jnp.ones(signs.shape), signs
)  # if signs is 0, just want to multiply by 1
amps *= signs
# make sure that the zeros are zeros.
amps = jnp.round(amps, decimals=5) + 0.0

for row in amps:
    square_map = row.reshape(spatial_size, spatial_size)
    print(square_map)
    print(jnp.arange(spatial_size).reshape(spatial_dims))
    print((square_map @ jnp.arange(spatial_size)).reshape(spatial_dims))
    print(" ")

print(f"Number of maps: {len(amps)}")
