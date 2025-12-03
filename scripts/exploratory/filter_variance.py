import time

import jax
import jax.numpy as jnp
import jax.random as random
import jaxopt

import ginjax.geometric as geom


N = 3
D = 3

key = random.PRNGKey(time.time_ns())
key, subkey = random.split(key)

data = random.normal(subkey, shape=(N,) * D)
A = geom.GeometricImage(data, 0, D, True)

ff = None
if D == 1:
    ff = geom.GeometricFilter(jnp.array([1, -2, 1]), 0, D)
elif D == 2:
    ff1 = geom.GeometricFilter(jnp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), 0, D)
    ff2 = geom.GeometricFilter(jnp.array([[1, 0, 1], [0, -4, 0], [1, 0, 1]]), 0, D)

    x = 4 / 9
    y = -2 / 9
    ff = (ff1 * x) + (ff2 * y)
    ff_prime = ff + geom.GeometricFilter(jnp.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]]), 0, D)
elif D == 3:
    ff1 = geom.GeometricFilter(
        jnp.array(
            [
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
                [
                    [0, 1, 0],
                    [1, -6, 1],
                    [0, 1, 0],
                ],
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
            ]
        ),
        0,
        D,
    )

    ff2 = geom.GeometricFilter(
        jnp.array(
            [
                [
                    [0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0],
                ],
                [
                    [1, 0, 1],
                    [0, -12, 0],
                    [1, 0, 1],
                ],
                [
                    [0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0],
                ],
            ]
        ),
        0,
        D,
    )
    ff3 = geom.GeometricFilter(
        jnp.array(
            [
                [
                    [1, 0, 1],
                    [0, 0, 0],
                    [1, 0, 1],
                ],
                [
                    [0, 0, 0],
                    [0, -8, 0],
                    [0, 0, 0],
                ],
                [
                    [1, 0, 1],
                    [0, 0, 0],
                    [1, 0, 1],
                ],
            ]
        ),
        0,
        D,
    )

    x = 8 / 27
    y = -4 / 27
    z = 2 / 27
    ff = (ff1 * x) + (ff2 * y) + (ff3 * z)
    ff_prime = ff + geom.GeometricFilter(
        jnp.array(
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ]
        ),
        0,
        D,
    )

assert isinstance(ff, geom.GeometricFilter)
einstr = f"{geom.LETTERS[:D]},{geom.LETTERS[D:2*D]}"
print(f"E[C^2]={jnp.mean(jnp.einsum(einstr, ff.data, ff.data))} (in tensor product sense)")

A_ff = A.convolve_with(ff)
A_ff_prime = A.convolve_with(ff_prime)
print("var(A) + E[(A*C)^2] + 2E[A x A*C]: (here multiplication is pointwise)")
print(
    f"{jnp.var(A.data):.3f} + {jnp.mean((A_ff * A_ff).data):.3f} + {2 * jnp.mean((A * A_ff).data):.3f} = {jnp.var(A.data) + jnp.mean((A_ff * A_ff).data) + 2 * jnp.mean((A * A_ff).data):.3f}"
)
print(jnp.var((A + A_ff).data))
