import sys
sys.path.insert(0,'src/geometricconvolutions/')
import itertools as it

from geometric import ktensor, geometric_filter, geometric_image
import jax.numpy as jnp
import jax.random as random

def levi_civita_contract_old(ktensor_obj, index):
    assert ktensor_obj.D in [2, 3] # BECAUSE WE SUCK
    assert (ktensor_obj.k + 1) >= ktensor_obj.D # so we have enough indices work on
    if ktensor_obj.D == 2 and not isinstance(index, tuple):
        index = (index,)

    if ktensor_obj.D == 2:
        index = index[0]
        otherdata = jnp.zeros_like(ktensor_obj.data)
        otherdata = otherdata.at[..., 0].set(-1. * jnp.take(ktensor_obj.data, 1, axis=index))
        otherdata = otherdata.at[..., 1].set(1. * jnp.take(ktensor_obj.data, 0, axis=index))
        return ktensor(otherdata, ktensor_obj.parity + 1, ktensor_obj.D)
    if ktensor_obj.D == 3:
        assert len(index) == 2
        i, j = index
        assert i < j
        otherdata = jnp.zeros_like(ktensor_obj.data[..., 0])
        otherdata = otherdata.at[..., 0].set(jnp.take(jnp.take(ktensor_obj.data, 2, axis=j), 1, axis=i) \
                          - jnp.take(jnp.take(ktensor_obj.data, 1, axis=j), 2, axis=i))
        otherdata = otherdata.at[..., 1].set(jnp.take(jnp.take(ktensor_obj.data, 0, axis=j), 2, axis=i) \
                          - jnp.take(jnp.take(ktensor_obj.data, 2, axis=j), 0, axis=i))
        otherdata = otherdata.at[..., 2].set(jnp.take(jnp.take(ktensor_obj.data, 1, axis=j), 0, axis=i) \
                          - jnp.take(jnp.take(ktensor_obj.data, 0, axis=j), 1, axis=i))
        return ktensor(otherdata, ktensor_obj.parity + 1, ktensor_obj.D)
    return

def testCrossProduct():
        a = ktensor(jnp.array([0,1,3]), 0, 3)
        b = ktensor(jnp.array([1,2,-1]), 0, 3)

        ab = a*b
        cross = ab.levi_civita_contract((0,1))
        assert (cross.data == jnp.array([-7,3,-1])).all()
        assert cross.parity == (ab.parity + 1) % 2
        assert cross.D == ab.D == a.D == b.D
        assert cross.k == (ab.k - ab.D + 2)

def testLeviCivitaContract():
    key = random.PRNGKey(0)

    for D in range(2,4):
        for k in range(D-1, D+2):
            key, subkey = random.split(key)
            ktensor1 = ktensor(random.uniform(key, shape=k*(D,)), 0, D)
            print(ktensor1.data)

            for indices in it.combinations(range(k), D-1):
                print(indices)
                ktensor1_contracted = ktensor1.levi_civita_contract(indices)
                print(ktensor1_contracted.data)
                print(levi_civita_contract_old(ktensor1, indices).data)
                assert (ktensor1_contracted.data == levi_civita_contract_old(ktensor1, indices).data).all()
                assert ktensor1_contracted.k == (ktensor1.k - ktensor1.D + 2)
                assert ktensor1_contracted.parity == (ktensor1.parity + 1) % 2
                assert ktensor1_contracted.D == ktensor1.D


# testLeviCivitaContract()
testCrossProduct()


# res = ktensor(jnp.array([2,1]),0,2).levi_civita_contract(0)
# print(res.data)

