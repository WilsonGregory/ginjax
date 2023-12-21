import geometricconvolutions.geometric as geom
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
        D = 2
        operators = jnp.stack(geom.make_all_operators(D))
        library = p2v_models.get_ode_basis(D, 4, [-1.,-1.], [1.,1.], 3)

        operators_on_coeffs = geom.get_operators_on_coeffs(D, operators, library)
        for gg in operators_on_coeffs:
            assert jnp.allclose(gg @ gg.T, jnp.eye(len(gg))), f'{jnp.max(gg @ gg.T - jnp.eye(len(gg)))}'
            assert jnp.allclose(gg.T @ gg, jnp.eye(len(gg))), f'{jnp.max(gg.T @ gg - jnp.eye(len(gg)))}'

    def testGetEquivariantMapToCoeffs(self):
        # Ensure that the maps are indeed equivariant
        key = random.PRNGKey(time.time_ns())
        D = 2
        N = 5
        operators = jnp.stack(geom.make_all_operators(D))
        library = p2v_models.get_ode_basis(D, 4, [-1.,-1.], [1.,1.], 3)

        key, subkey1 = random.split(key)
        key, subkey2 = random.split(key)
        key, subkey3 = random.split(key)

        rand_layer = geom.BatchLayer(
            {
                (0,0): random.normal(subkey1, shape=(1,1) + (N,)*D),
                (1,0): random.normal(subkey2, shape=(1,1) + (N,)*D + (D,)),
                (2,0): random.normal(subkey3, shape=(1,1) + (N,)*D + (D,D)),
            },
            D, 
            False,
        )

        # First, we construct basis of layer elements
        basis_len = rand_layer.size()
        layer_basis = jax.vmap(lambda e: rand_layer.__class__.from_vector(e, rand_layer))(jnp.eye(basis_len))

        # Now we use this basis to get the representation of each group element on the layer
        operators_on_layer = jax.vmap(
            jax.vmap(lambda gg, e: e.times_group_element(gg).to_vector(), in_axes=(None,0)), 
            in_axes=(0,None),
        )(operators, layer_basis)

        operators_on_coeffs = geom.get_operators_on_coeffs(D, operators, library)
        equiv_maps = geom.get_equivariant_map_to_coeffs(rand_layer, operators, library)

        key,subkey = random.split(key)
        rand_map = jnp.sum(random.normal(subkey, shape=(len(equiv_maps),1,1))*equiv_maps, axis=0)

        # For this random layer and random map, ensure that it is equivariant to the group.
        for gg_layer, coeffs_gg in zip(operators_on_layer, operators_on_coeffs):
            assert jnp.allclose(
                rand_map @ gg_layer @ rand_layer.to_vector(),
                coeffs_gg @ rand_map @ rand_layer.to_vector(),
            )
