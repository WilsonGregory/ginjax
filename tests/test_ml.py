from typing_extensions import Optional, Self, Union

import jax.numpy as jnp
from jax import random
import jax
import equinox as eqx

import ginjax.geometric as geom
import ginjax.ml as ml
import ginjax.models as models


# dummy module for testing autoregressive map
class DummyModule(models.MultiImageModule):
    def __call__(
        self: Self, x: geom.MultiImage, aux_data: Optional[eqx.nn.State] = None
    ) -> tuple[geom.MultiImage, Optional[eqx.nn.State]]:
        out = x.empty()
        for (k, parity), image_block in x.items():
            out.append(k, parity, image_block[:1] + image_block[-1:])

        return out, aux_data


class TestMachineLearning:

    def testGetBatches(self):
        num_devices = 1  # since it can only see the cpu
        cpu = [jax.devices("cpu")[0]]
        key = random.PRNGKey(0)
        N = 5
        D = 2
        k = 0

        X = geom.MultiImage({(k, 0): random.normal(key, shape=((10, 1) + (N,) * D + (D,) * k))}, D)
        Y = geom.MultiImage({(k, 0): random.normal(key, shape=((10, 1) + (N,) * D + (D,) * k))}, D)

        batch_size = 2
        X_batches, Y_batches = ml.get_batches(
            (X, Y), batch_size=batch_size, rand_key=key, devices=cpu
        )
        assert len(X_batches) == len(Y_batches) == 5
        for X_batch, Y_batch in zip(X_batches, Y_batches):
            assert (
                X_batch[((False,) * k, 0)].shape
                == Y_batch[((False,) * k, 0)].shape
                == (num_devices, batch_size, 1) + (N,) * D + (D,) * k
            )

        X = geom.MultiImage(
            {
                (0, 0): random.normal(key, shape=((20, 1) + (N,) * D + (D,) * 0)),
                (1, 0): random.normal(key, shape=((20, 1) + (N,) * D + (D,) * 1)),
            },
            D,
        )
        Y = geom.MultiImage(
            {
                (0, 0): random.normal(key, shape=((20, 1) + (N,) * D + (D,) * 0)),
                (1, 0): random.normal(key, shape=((20, 1) + (N,) * D + (D,) * 1)),
            },
            D,
        )

        # batching when the multi_image has multiple channels at different values of k
        batch_size = 5
        X_batches, Y_batches = ml.get_batches(
            (X, Y), batch_size=batch_size, rand_key=key, devices=cpu
        )
        assert len(X_batches) == len(Y_batches) == 4
        for X_batch, Y_batch in zip(X_batches, Y_batches):
            assert (
                X_batch[((), 0)].shape
                == Y_batch[((), 0)].shape
                == (num_devices, batch_size, 1) + (N,) * D + (D,) * 0
            )
            assert (
                X_batch[((False,), 0)].shape
                == Y_batch[((False,), 0)].shape
                == (num_devices, batch_size, 1) + (N,) * D + (D,) * 1
            )

        X = geom.MultiImage(
            {
                (0, 0): random.normal(key, shape=((20, 2) + (N,) * D + (D,) * 0)),
                (1, 0): random.normal(key, shape=((20, 1) + (N,) * D + (D,) * 1)),
            },
            D,
        )
        Y = geom.MultiImage(
            {
                (0, 0): random.normal(key, shape=((20, 2) + (N,) * D + (D,) * 0)),
                (1, 0): random.normal(key, shape=((20, 1) + (N,) * D + (D,) * 1)),
            },
            D,
        )

        # batching when multi_image has multiple channels for one value of k
        batch_size = 5
        X_batches, Y_batches = ml.get_batches(
            (X, Y), batch_size=batch_size, rand_key=key, devices=cpu
        )
        assert len(X_batches) == len(Y_batches) == 4
        for X_batch, Y_batch in zip(X_batches, Y_batches):
            assert (
                X_batch[((), 0)].shape
                == Y_batch[((), 0)].shape
                == (num_devices, batch_size, 2) + (N,) * D + (D,) * 0
            )
            assert (
                X_batch[((False,), 0)].shape
                == Y_batch[((False,), 0)].shape
                == (num_devices, batch_size, 1) + (N,) * D + (D,) * 1
            )

    def testAutoregressiveStep(self):
        past_steps = 4
        future_steps = 1
        N = 5
        D = 2

        key = random.PRNGKey(0)
        key1, key2, key3, key4, key5, key6, key7 = random.split(key, 7)

        # Basic example with scalar field
        data1 = random.normal(key1, shape=(past_steps,) + (N,) * D)
        input1 = geom.MultiImage({(0, 0): data1}, D)
        one_step1 = geom.MultiImage(
            {(0, 0): random.normal(key2, shape=(future_steps,) + (N,) * D)}, D
        )

        new_input = ml.training.autoregressive_step(input1, one_step1, past_steps)
        assert jnp.allclose(
            new_input[((), 0)],
            jnp.concatenate([input1[((), 0)][future_steps:], one_step1[((), 0)]]),
        )

        # Example with scalar and vector field
        data2 = random.normal(key3, shape=(2 * past_steps,) + (N,) * D + (D,))
        input2 = geom.MultiImage({(0, 0): data1, (1, 0): data2}, D)
        one_step2 = geom.MultiImage(
            {
                (0, 0): random.normal(key4, shape=(1 * future_steps,) + (N,) * D),
                (1, 0): random.normal(key5, shape=(2 * future_steps,) + (N,) * D + (D,)),
            },
            D,
        )

        new_input = ml.training.autoregressive_step(input2, one_step2, past_steps)
        assert jnp.allclose(
            new_input[((), 0)],
            jnp.concatenate([input2[((), 0)][future_steps:], one_step2[((), 0)]]),
        )
        new_input_exp = new_input.expand(0, past_steps)[
            ((False,), 0)
        ]  # (c,past_steps,spatial,tensor)
        input_exp = input2.expand(0, past_steps)[((False,), 0)]
        step_exp = one_step2.expand(0, future_steps)[((False,), 0)]
        assert jnp.allclose(new_input_exp[:, :-future_steps], input_exp[:, future_steps:])
        assert jnp.allclose(new_input_exp[:, -future_steps:], step_exp)

        # Example with constant scalar and vector fields
        constant_field1 = random.normal(key6, shape=(1,) + (N,) * D)
        constant_field2 = random.normal(key7, shape=(1,) + (N,) * D + (D,))

        input3 = input2.concat(
            geom.MultiImage({(0, 0): constant_field1, (1, 0): constant_field2}, D)
        )
        new_input = ml.training.autoregressive_step(
            input3, one_step2, past_steps, {((), 0): 1, ((False,), 0): 1}
        )
        assert jnp.allclose(
            new_input[((), 0)],
            jnp.concatenate(
                [input3[((), 0)][future_steps:-future_steps], one_step2[((), 0)], constant_field1]
            ),
        )
        input_dynamic_fields, _ = input3.concat_inverse({((), 0): 1, ((False,), 0): 1})
        new_dynamic_fields, new_const_fields = new_input.concat_inverse(
            {((), 0): 1, ((False,), 0): 1}
        )

        new_input_exp = new_dynamic_fields.expand(0, past_steps)[
            ((False,), 0)
        ]  # (c,past_steps,spatial,tensor)
        input_exp = input_dynamic_fields.expand(0, past_steps)[((False,), 0)]
        step_exp = one_step2.expand(0, future_steps)[((False,), 0)]
        assert jnp.allclose(new_input_exp[:, :-future_steps], input_exp[:, future_steps:])
        assert jnp.allclose(new_input_exp[:, -future_steps:], step_exp)
        assert jnp.allclose(new_const_fields[((False,), 0)], constant_field2)

        # test when there is a field which is only constant
        input4 = input1.concat(
            geom.MultiImage({(0, 0): constant_field1, (1, 0): constant_field2}, D)
        )
        new_input = ml.training.autoregressive_step(
            input4, one_step1, past_steps, {((), 0): 1, ((False,), 0): 1}
        )
        assert jnp.allclose(
            new_input[((), 0)],
            jnp.concatenate(
                [input4[((), 0)][future_steps:-future_steps], one_step1[((), 0)], constant_field1]
            ),
        )
        new_dynamic_fields, new_const_fields = new_input.concat_inverse(
            {((), 0): 1, ((False,), 0): 1}
        )
        assert ((False,), 0) not in new_dynamic_fields
        assert jnp.allclose(new_const_fields[((False,), 0)], constant_field2)

    def testAutoregressiveMap(self):
        past_steps = 4
        N = 5
        D = 2

        key = random.PRNGKey(0)

        # Test an input image with constant fields for each image type
        key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)
        x = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=(past_steps,) + (N,) * D),
                (1, 0): random.normal(subkey2, shape=(past_steps,) + (N,) * D + (D,)),
            },
            D,
        )
        constant_fields = geom.MultiImage(
            {
                (0, 0): random.normal(subkey3, shape=(1,) + (N,) * D),
                (1, 0): random.normal(subkey4, shape=(1,) + (N,) * D + (D,)),
            },
            D,
        )
        x = x.concat(constant_fields)

        model = DummyModule()
        out, _ = ml.autoregressive_map(
            model, x, None, past_steps, 5, {((), 0): 1, ((False,), 0): 1}
        )
        assert len(out[((), 0)]) == 5
        assert jnp.allclose(out[((), 0)][0], x[((), 0)][0] + constant_fields[((), 0)])
        assert jnp.allclose(out[((), 0)][1], x[((), 0)][1] + constant_fields[((), 0)])
        assert jnp.allclose(out[((), 0)][2], x[((), 0)][2] + constant_fields[((), 0)])
        assert jnp.allclose(out[((), 0)][3], x[((), 0)][3] + constant_fields[((), 0)])
        assert jnp.allclose(
            out[((), 0)][4], x[((), 0)][0] + constant_fields[((), 0)] + constant_fields[((), 0)]
        )

        assert len(out[((False,), 0)]) == 5
        assert jnp.allclose(
            out[((False,), 0)][0], x[((False,), 0)][0] + constant_fields[((False,), 0)]
        )
        assert jnp.allclose(
            out[((False,), 0)][1], x[((False,), 0)][1] + constant_fields[((False,), 0)]
        )
        assert jnp.allclose(
            out[((False,), 0)][2], x[((False,), 0)][2] + constant_fields[((False,), 0)]
        )
        assert jnp.allclose(
            out[((False,), 0)][3], x[((False,), 0)][3] + constant_fields[((False,), 0)]
        )
        assert jnp.allclose(
            out[((False,), 0)][4],
            x[((False,), 0)][0] + constant_fields[((False,), 0)] + constant_fields[((False,), 0)],
        )

        # Test when constant fields only has 1 constant field
        key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)
        x = geom.MultiImage(
            {
                (0, 0): random.normal(subkey1, shape=(past_steps,) + (N,) * D),
                (1, 0): random.normal(subkey2, shape=(past_steps,) + (N,) * D + (D,)),
            },
            D,
        )
        constant_fields = geom.MultiImage(
            {(0, 0): random.normal(subkey3, shape=(1,) + (N,) * D)}, D
        )
        x = x.concat(constant_fields)

        model = DummyModule()
        out, _ = ml.autoregressive_map(model, x, None, past_steps, 5, {((), 0): 1})
        assert len(out[((), 0)]) == 5
        assert jnp.allclose(out[((), 0)][0], x[((), 0)][0] + constant_fields[((), 0)])
        assert jnp.allclose(out[((), 0)][1], x[((), 0)][1] + constant_fields[((), 0)])
        assert jnp.allclose(out[((), 0)][2], x[((), 0)][2] + constant_fields[((), 0)])
        assert jnp.allclose(out[((), 0)][3], x[((), 0)][3] + constant_fields[((), 0)])
        assert jnp.allclose(
            out[((), 0)][4], x[((), 0)][0] + constant_fields[((), 0)] + constant_fields[((), 0)]
        )

        assert len(out[((False,), 0)]) == 5
        assert jnp.allclose(out[((False,), 0)][0], x[((False,), 0)][0] + x[((False,), 0)][3])
        assert jnp.allclose(
            out[((False,), 0)][1], x[((False,), 0)][1] + x[((False,), 0)][0] + x[((False,), 0)][3]
        )
        assert jnp.allclose(
            out[((False,), 0)][2],
            x[((False,), 0)][2] + x[((False,), 0)][1] + x[((False,), 0)][0] + x[((False,), 0)][3],
        )
        assert jnp.allclose(
            out[((False,), 0)][3],
            x[((False,), 0)][3]
            + x[((False,), 0)][2]
            + x[((False,), 0)][1]
            + x[((False,), 0)][0]
            + x[((False,), 0)][3],
        )
        assert jnp.allclose(
            out[((False,), 0)][4],
            x[((False,), 0)][0]
            + x[((False,), 0)][3]
            + x[((False,), 0)][3]
            + x[((False,), 0)][2]
            + x[((False,), 0)][1]
            + x[((False,), 0)][0]
            + x[((False,), 0)][3],
        )
