import geometricconvolutions.geometric as geom
import pytest
import jax.numpy as jnp
import jax.random as random
import math

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
        print(idxs)
        known_list = [
            ((0,1),(2,3)), ((0,1),(2,4)), ((0,1),(3,4)), ((0,2),(1,3)), ((0,2),(1,4)), ((0,2),(3,4)),
            ((0,3),(1,4)), ((0,3),(2,4)), ((0,4),(2,3)),
            ((0,3),(1,2)), ((0,4),(1,2)), ((0,4),(1,3)), #bad ones, are not removed correctly
        ]
        assert len(idxs) == len(known_list)
        for pair in known_list:
            assert pair in idxs

        idxs = geom.get_contraction_indices(5,1, ((0,1),(2,3)))
        known_list = [
            ((0,1),(2,3)), ((0,1),(2,4)), ((0,2),(1,3)), ((0,2),(1,4)), ((0,2),(3,4)), ((0,4),(2,3)),
            ((0,4),(1,2)), #bad one, should be removed
        ]
        assert len(idxs) == len(known_list)
        for pair in known_list:
            assert pair in idxs
