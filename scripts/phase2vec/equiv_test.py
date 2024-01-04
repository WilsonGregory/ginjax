import time
import itertools as it

import jax.numpy as jnp
import jax.random as random
import jax

import geometricconvolutions.geometric as geom
import p2v_models

# Main
D = 2
N = 64
seed = None 

key = random.PRNGKey(time.time_ns() if (seed is None) else seed)

# start with basic 3x3 scalar, vector, and 2nd order tensor images
operators = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[3], ks=[0,1,2,3], parities=[0,1], D=D, operators=operators)

# generate function library
ode_basis = p2v_models.get_ode_basis(D, N, [-1.,-1.], [1.,1.], 3) # (N**D, 10)
num_coeffs = ode_basis.shape[1]
small_ode_basis = p2v_models.get_ode_basis(D, 5, [-1.,-1.], [1.,1.], 3)
operators_on_coeffs = geom.get_operators_on_coeffs(D, operators, small_ode_basis)

key, subkey = random.split(key)
rand_coeffs = random.normal(subkey, shape=(num_coeffs, D))

vec_img = (ode_basis @ rand_coeffs).reshape((N,)*D + (D,))

for gg, gg_coeffs in zip(operators, operators_on_coeffs):
    rotated_coeffs = (gg_coeffs @ rand_coeffs.reshape((num_coeffs*D,))).reshape((num_coeffs,D))
    first = geom.times_group_element(D, vec_img, 0, gg, precision=jax.lax.Precision.HIGHEST)
    second = (ode_basis @ rotated_coeffs).reshape((N,)*D + (D,))
    assert jnp.allclose(first, second, atol=geom.TINY, rtol=geom.TINY), f'{jnp.max(first - second)}'

# if these all pass, then that indicates to me that the process of finding coefficients itself is 
# an equivariant process. That is the following are equivalent:
# generate coeffs -> make image -> rotate image
# take generated coeffs -> rotate coeffs -> make image
print('All rotation/reflection tests passed!')

# Now we want to do the same for translations

n = 2
N = 2*n + 1
large_small_ode_basis = jnp.round(p2v_models.get_ode_basis(D, N*3, [-(n+N), -(n+N)], [n+N, n+N], 3))

"""
shape when (10,2)
[1, 1]
[x, x]
[y, y]
[x^2, x^2]
[xy, xy]
[y^2, y^2]
[x^3, x^3]
[x^2y, x^2y]
[xy^2, xy^2]
[y^3, y^3]

shape when (20,)
[1, 1, x, x, y, y, x^2, x^2, xy, xy, y^2, y^2, x^3, x^3, x^2y, x^2y, xy^2, xy^2, y^3, y^3]
"""

def get_translation_op(a,b):
    """
    Given coefficients for a function
    """
    return jnp.array([
        [1, -a, -b, a**2, a*b, b**2,  -a**3, -a**2*b, -b**2*a,  -b**3],
        [0,  1,  0, -2*a,  -b,    0, 3*a**2,   2*a*b,    b**2,      0],
        [0,  0,  1,    0,  -a, -2*b,      0,    a**2,   2*a*b, 3*b**2],
        [0,  0,  0,    1,   0,    0,   -3*a,      -b,       0,      0],
        [0,  0,  0,    0,   1,    0,      0,    -2*a,    -2*b,      0],
        [0,  0,  0,    0,   0,    1,      0,       0,      -a,   -3*b],
        [0,  0,  0,    0,   0,    0,      1,       0,       0,      0],
        [0,  0,  0,    0,   0,    0,      0,       1,       0,      0],
        [0,  0,  0,    0,   0,    0,      0,       0,       1,      0],
        [0,  0,  0,    0,   0,    0,      0,       0,       0,      1],
    ])

key, subkey = random.split(key)
rand_coeffs = random.normal(subkey, shape=(num_coeffs,D))

for (a,b) in it.product(range(-N,N+1),range(-N,N+1)):
    translation_gg = get_translation_op(a,b)
    # build image from coefficients, then apply the translation
    first = (large_small_ode_basis @ rand_coeffs).reshape((3*N,)*D + (D,))[N-a:2*N-a,N-b:2*N-b]
    # apply the translation, then build the image from the coefficients
    second = (large_small_ode_basis @ translation_gg @ rand_coeffs).reshape((3*N,)*D + (D,))[N:2*N,N:2*N]

    assert jnp.allclose(first, second, atol=geom.TINY, rtol=geom.TINY), f'{a},{b}: {jnp.max(first - second)}'

print('All translation tests passed!')