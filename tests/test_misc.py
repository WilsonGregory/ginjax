import sys
sys.path.insert(0,'src/geometricconvolutions/')

from geometric import levi_civita_symbol
import pytest
import jax.numpy as jnp

TINY = 1.e-5

class TestMisc:

    def testLeviCivitaSymbol(self):
        with pytest.raises(AssertionError):
            levi_civita_symbol.get(1)

        assert (levi_civita_symbol.get(2) == jnp.array([[0, 1], [-1, 0]], dtype=int)).all()
        assert (levi_civita_symbol.get(3) == jnp.array(
            [
                [[0,0,0], [0,0,1], [0,-1,0]],
                [[0,0,-1], [0,0,0], [1,0,0]],
                [[0,1,0], [-1,0,0], [0,0,0]],
            ],
            dtype=int)).all()

        assert levi_civita_symbol.get(2) is levi_civita_symbol.get(2) #test that we aren't remaking them



