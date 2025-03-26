import jax.numpy as jnp

import ginjax.geometric as geom
import ginjax.ml as ml

D = 2
N = 3
past_steps = 2

x = geom.MultiImage(
    {
        (0, 0): jnp.arange(1, past_steps + 1).reshape((past_steps, 1, 1)) * jnp.ones((1, N, N)),
        (1, 0): jnp.arange(1, past_steps + 1).reshape((past_steps, 1, 1, 1))
        * jnp.ones((1, N, N, D)),
    },
    D,
)
constant_fields = geom.MultiImage(
    {(0, 0): -jnp.ones((1, N, N)), (0, 1): -2 * jnp.ones((1, N, N))}, D
)
x = x.concat(constant_fields)

dynamic_x, const_x = x.concat_inverse({(0, 0): 1, (0, 1): 1})
print(dynamic_x.expand(0, past_steps)[(0, 0)])
print(const_x[(0, 0)])

out_x = x.empty()  # assume out matches D and is_torus
for i in range(5):
    pred_x = geom.MultiImage(
        {
            (0, 0): (i + 1) * 100 * jnp.ones((1, N, N)),
            (1, 0): (i + 1) * 10 * jnp.ones((1, N, N, D)),
        },
        D,
    )
    # pred_x, aux_data = model(x, aux_data)
    x = ml.autoregressive_step(x, pred_x, past_steps, {(0, 0): 1, (0, 1): 1})

    out_x = out_x.concat(pred_x.expand(axis=0, size=1), axis=1)
    print(i)
    dynamic_x, const_x = x.concat_inverse({(0, 0): 1, (0, 1): 1})
    print(dynamic_x.expand(0, past_steps)[(0, 0)])
    print("const scalar")
    print(const_x[(0, 0)])
    print("const pseudoscalar")
    print(const_x[(0, 1)])
    print("out")
    print(out_x)
    print(out_x[(0, 0)])

out_x.combine_axes((0, 1))
