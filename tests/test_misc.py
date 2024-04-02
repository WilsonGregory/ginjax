import geometricconvolutions.geometric as geom
import geometricconvolutions.data as gc_data
import pytest
import jax.numpy as jnp
import jax.random as random
import jax
import math
import time

import sys
sys.path.append('scripts/phase2vec/')
import p2v_models

class TestMisc:

    def testPermutationParity(self):
        assert geom.permutation_parity([0]) == 1
        assert geom.permutation_parity((0,1)) == 1
        assert geom.permutation_parity((1,0)) == -1
        assert geom.permutation_parity([1,0]) == -1
        assert geom.permutation_parity([1,1]) == 0
        assert geom.permutation_parity([0,1,2]) == 1
        assert geom.permutation_parity([0,2,1]) == -1
        assert geom.permutation_parity([1,2,0]) == 1
        assert geom.permutation_parity([1,0,2]) == -1
        assert geom.permutation_parity([2,1,0]) == -1
        assert geom.permutation_parity([2,0,1]) == 1
        assert geom.permutation_parity([2,1,1]) == 0

    def testLeviCivitaSymbol(self):
        with pytest.raises(AssertionError):
            geom.LeviCivitaSymbol.get(1)

        assert (geom.LeviCivitaSymbol.get(2) == jnp.array([[0, 1], [-1, 0]], dtype=int)).all()
        assert (geom.LeviCivitaSymbol.get(3) == jnp.array(
            [
                [[0,0,0], [0,0,1], [0,-1,0]],
                [[0,0,-1], [0,0,0], [1,0,0]],
                [[0,1,0], [-1,0,0], [0,0,0]],
            ],
            dtype=int)).all()

        assert geom.LeviCivitaSymbol.get(2) is geom.LeviCivitaSymbol.get(2) #test that we aren't remaking them

    def testGroupSize(self):
        for d in range(2,7):
            operators = geom.make_all_operators(d)

            # test the group size
            assert len(operators) == 2*(2**(d-1))*math.factorial(d)

    def testGetContractionIndices(self):
        idxs = geom.get_contraction_indices(3,1)
        known_list = [((0,1),), ((0,2),), ((1,2),)]
        assert len(idxs) == len(known_list)
        for pair in known_list:
            assert pair in idxs

        idxs = geom.get_contraction_indices(3,1,((0,1),))
        known_list = [((0,1),), ((0,2),)]
        assert len(idxs) == len(known_list)
        for pair in known_list:
            assert pair in idxs

        idxs = geom.get_contraction_indices(5,3)
        known_list = [
            ((0,1),), ((0,2),), ((0,3),), ((0,4),), ((1,2),), ((1,3),), ((1,4),), ((2,3),), ((2,4),), ((3,4),)
        ]
        assert len(idxs) == len(known_list)
        for pair in known_list:
            assert pair in idxs

        idxs = geom.get_contraction_indices(5,1)
        known_list = [
            ((0,1),(2,3)), ((0,1),(2,4)), ((0,1),(3,4)), ((0,2),(1,3)), ((0,2),(1,4)), ((0,2),(3,4)),
            ((0,3),(1,2)), ((0,3),(1,4)), ((0,3),(2,4)), ((0,4),(1,2)), ((0,4),(1,3)), ((0,4),(2,3)),
            ((1,2),(3,4)), ((1,3),(2,4)), ((1,4), (2,3)),
        ]
        assert len(idxs) == len(known_list)
        for pair in known_list:
            assert pair in idxs

        idxs = geom.get_contraction_indices(5,1, ((0,1),))
        known_list = [
            ((0,1),(2,3)), ((0,1),(2,4)), ((0,1),(3,4)), ((0,2),(1,3)), ((0,2),(1,4)), ((0,2),(3,4)),
            ((0,3),(1,4)), ((0,3),(2,4)), ((0,4),(2,3)),
        ]
        assert len(idxs) == len(known_list)
        for pair in known_list:
            assert pair in idxs

        idxs = geom.get_contraction_indices(5,1, ((0,1),(2,3)))
        known_list = [
            ((0,1),(2,3)), ((0,1),(2,4)), ((0,2),(1,3)), ((0,2),(1,4)), ((0,2),(3,4)), ((0,4),(2,3)),
        ]
        assert len(idxs) == len(known_list)
        for pair in known_list:
            assert pair in idxs

    def testGetInvariantImage(self):
        N = 5

        D = 2
        operators = geom.make_all_operators(D)
        for k in [0,2,4]:
            invariant_basis = geom.get_invariant_image(N, D, k, 0, data_only=False)

            for gg in operators:
                assert invariant_basis == invariant_basis.times_group_element(gg)

        D = 3
        operators = geom.make_all_operators(D)
        for k in [0,2,4]: 
            invariant_basis = geom.get_invariant_image(N, D, k, 0, data_only=False)

            for gg in operators:
                assert invariant_basis == invariant_basis.times_group_element(gg)

    def testGetOperatorsOnCoeffs(self):
        # Ensure that this representation has orthogonal group elements are orthogonal
        key = random.PRNGKey(0)
        D = 2
        library_N = 5
        operators = jnp.stack(geom.make_all_operators(D))
        library = p2v_models.get_ode_basis(D, library_N, [-1.,-1.], [1.,1.], 3)
        library_pinv = jnp.linalg.pinv(library) # library is (N**D, num_coeffs), pinv is (num_coeffs, N**D)
        num_coeffs = library.shape[1]

        operators_on_coeffs = geom.get_operators_on_coeffs(D, operators, library)

        # Assert that all the operators are orthogonal
        for gg in operators_on_coeffs:
            assert jnp.allclose(gg @ gg.T, jnp.eye(len(gg))), f'{jnp.max(gg @ gg.T - jnp.eye(len(gg)))}'
            assert jnp.allclose(gg.T @ gg, jnp.eye(len(gg))), f'{jnp.max(gg.T @ gg - jnp.eye(len(gg)))}'

        rand_coeffs = random.normal(key, shape=(num_coeffs * D,))

        # Assert that left multiplying by the operator on coeffs is equivalent to rotating the 
        for gg_coeff, gg in zip(operators_on_coeffs, operators):
            vec_img = (library @ rand_coeffs.reshape((num_coeffs, D))).reshape((library_N,)*D + (D,)) 
            rotated_img = geom.times_group_element(D, vec_img, 0, gg, jax.lax.Precision.HIGHEST).reshape((library_N**D,D))
            rotated_coeffs = (library_pinv @ rotated_img).reshape(-1) # (num_coeffs, D) -> (num_coeffs*D,)

            assert jnp.allclose(rotated_coeffs, gg_coeff @ rand_coeffs, rtol=geom.TINY, atol=geom.TINY)

    def testGetOperatorsOnLayer(self):
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5
        operators = jnp.stack(geom.make_all_operators(D))

        layer = geom.Layer(
            {
                (0,0): jnp.ones((1,) + (N,)*D),
                (1,0): jnp.ones((1,) + (N,)*D + (D,)),
                (2,0): jnp.ones((1,) + (N,)*D + (D,D)),
            },
            D, 
            False,
        )

        operators_on_layer = geom.get_operators_on_layer(operators, layer)

        # Assert that operators are orthogonal
        for gg_layer in operators_on_layer:
            assert jnp.allclose(gg_layer @ gg_layer.T, jnp.eye(len(gg_layer)))
            assert jnp.allclose(gg_layer.T @ gg_layer, jnp.eye(len(gg_layer)))

        rand_layer = layer.empty()
        for (k, parity), img in layer.items():
            key, subkey = random.split(key)
            rand_layer.append(k, parity, random.normal(subkey, shape=img.shape))

        # Assert that using the operator_on_layer is equivalent to using the times_group_element
        for gg_layer, gg in zip(operators_on_layer, operators):
            assert jnp.allclose(
                rand_layer.times_group_element(gg, precision=jax.lax.Precision.HIGHEST).to_vector(),
                gg_layer @ rand_layer.to_vector(),
            )

    def testGetEquivariantMapToCoeffs(self):
        # Ensure that the maps are indeed equivariant
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5
        operators = jnp.stack(geom.make_all_operators(D))
        library = p2v_models.get_ode_basis(D, 5, [-1.,-1.], [1.,1.], 3)

        key, subkey1 = random.split(key)
        key, subkey2 = random.split(key)
        key, subkey3 = random.split(key)

        rand_layer = geom.Layer(
            {
                (0,0): random.normal(subkey1, shape=(1,) + (N,)*D),
                (1,0): random.normal(subkey2, shape=(1,) + (N,)*D + (D,)),
                (2,0): random.normal(subkey3, shape=(1,) + (N,)*D + (D,D)),
            },
            D, 
            False,
        )

        operators_on_layer = geom.get_operators_on_layer(operators, rand_layer)
        operators_on_coeffs = geom.get_operators_on_coeffs(D, operators, library)
        equiv_maps = geom.get_equivariant_map_to_coeffs(rand_layer, operators, library)

        key,subkey = random.split(key)
        rand_map = jnp.sum(random.normal(subkey, shape=(len(equiv_maps),1,1))*equiv_maps, axis=0)

        # For this random layer and random map, ensure that it is equivariant to the group.
        for gg_layer, coeffs_gg in zip(operators_on_layer, operators_on_coeffs):
            print(jnp.max(rand_map @ gg_layer @ rand_layer.to_vector() - coeffs_gg @ rand_map @ rand_layer.to_vector()))
            assert jnp.allclose(
                rand_map @ gg_layer @ rand_layer.to_vector(),
                coeffs_gg @ rand_map @ rand_layer.to_vector(),
                rtol=geom.TINY,
                atol=geom.TINY,
            )

    def testTimeSeriesIdxsHardcoded(self):
        past_steps = 2
        future_steps = 1
        num_channels = 5
        delta_t = 1
        input_idxs, output_idxs = gc_data.time_series_idxs(past_steps, future_steps, delta_t, num_channels)
        assert len(input_idxs) == 3
        assert jnp.allclose(input_idxs[0], jnp.array([0,1]))
        assert jnp.allclose(input_idxs[1], jnp.array([1,2]))
        assert jnp.allclose(input_idxs[2], jnp.array([2,3]))

        assert len(output_idxs) == 3
        assert jnp.allclose(output_idxs[0], jnp.array([2]))
        assert jnp.allclose(output_idxs[1], jnp.array([3]))
        assert jnp.allclose(output_idxs[2], jnp.array([4]))

        num_channels = 10
        delta_t = 2
        input_idxs, output_idxs = gc_data.time_series_idxs(past_steps, future_steps, delta_t, num_channels)
        assert len(input_idxs) == 6
        assert jnp.allclose(input_idxs[0], jnp.array([0,2]))
        assert jnp.allclose(input_idxs[1], jnp.array([1,3]))
        assert jnp.allclose(input_idxs[2], jnp.array([2,4]))
        assert jnp.allclose(input_idxs[3], jnp.array([3,5]))
        assert jnp.allclose(input_idxs[4], jnp.array([4,6]))
        assert jnp.allclose(input_idxs[5], jnp.array([5,7]))

        assert len(output_idxs) == 6
        assert jnp.allclose(output_idxs[0], jnp.array([4]))
        assert jnp.allclose(output_idxs[1], jnp.array([5]))
        assert jnp.allclose(output_idxs[2], jnp.array([6]))
        assert jnp.allclose(output_idxs[3], jnp.array([7]))
        assert jnp.allclose(output_idxs[4], jnp.array([8]))
        assert jnp.allclose(output_idxs[5], jnp.array([9]))

        future_steps = 2
        input_idxs, output_idxs = gc_data.time_series_idxs(past_steps, future_steps, delta_t, num_channels)
        assert len(input_idxs) == 4
        assert jnp.allclose(input_idxs[0], jnp.array([0,2]))
        assert jnp.allclose(input_idxs[1], jnp.array([1,3]))
        assert jnp.allclose(input_idxs[2], jnp.array([2,4]))
        assert jnp.allclose(input_idxs[3], jnp.array([3,5]))

        assert len(output_idxs) == 4
        assert jnp.allclose(output_idxs[0], jnp.array([4,6]))
        assert jnp.allclose(output_idxs[1], jnp.array([5,7]))
        assert jnp.allclose(output_idxs[2], jnp.array([6,8]))
        assert jnp.allclose(output_idxs[3], jnp.array([7,9]))

    def testTimeSeriesIdxs(self):
        D = 2
        N = 3
        spatial_dims = (N,)*D
        batch = 10
        channels = 11
        key = random.PRNGKey(0)

        for past_steps in [1,2,4]:
            for future_steps in [1,5]:
                for k in [0,1,2]:
                    key, subkey = random.split(key)
                    img_data = random.normal(subkey, shape=((batch,channels) + spatial_dims + (D,)*k))
                    num_windows = channels-future_steps-past_steps+1

                    input_idxs, output_idxs = gc_data.time_series_idxs(past_steps, future_steps, 1, channels)
                    input_data = img_data[:, input_idxs]
                    assert input_data.shape == (batch,num_windows,past_steps) + spatial_dims + (D,)*k

                    output_data = img_data[:, output_idxs]
                    assert output_data.shape == (batch,num_windows,future_steps) + spatial_dims + (D,)*k

                    for b in range(batch):
                        for i in range(num_windows):
                            assert jnp.allclose(input_data[b,i], img_data[b,i:i + past_steps])
                            assert jnp.allclose(
                                output_data[b,i], 
                                img_data[b,i+past_steps:i+past_steps+future_steps],
                            )
                    
