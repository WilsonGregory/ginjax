import sys
sys.path.insert(0,'src/geometricconvolutions/')
import itertools as it

import geometric as geom
import jax.numpy as jnp
import jax.random as random

def testIK0_FK1():
    """
    Convolve with where the input is k=0, and the filter is k=1
    """
    image1 = geom.geometric_image(jnp.array([[2,1,0], [0,0,-3], [2,0,1]], dtype=int), 0, 2)
    filter_image = geom.geometric_filter(jnp.array([
        [[0,0], [0,1], [0,0]],
        [[-1,0],[0,0], [1,0]],
        [[0,0], [0,-1],[0,0]],
    ], dtype=int), 0, 2) #this is an invariant filter, hopefully not a problem?

    convolved_image = image1.convolve_with(filter_image)
    # print(convolved_image.data)

    assert convolved_image.D == image1.D
    assert convolved_image.N == image1.N
    assert convolved_image.k == image1.k + filter_image.k
    assert convolved_image.parity == (image1.parity + filter_image.parity) % 2
    assert (convolved_image.data == jnp.array([
        [[1,2],[-2,0],[1,4]],
        [[3,0],[-3,1],[0,-1]],
        [[-1,-2],[-1,-1],[2,-3]]
    ], dtype=int)).all()

def testIK0_FK0():
    #did these out by hand, hopefully, my arithmetic is correct...
    image1 = geom.geometric_image(jnp.array([[2,1,0], [0,0,-3], [2,0,1]], dtype=int), 0, 2)
    filter_image = geom.geometric_filter(jnp.array([[1,0,1], [0,0,0], [1,0,1]], dtype=int), 0, 2)

    convolved_image = image1.convolve_with(filter_image)
    assert convolved_image.D == image1.D
    assert convolved_image.N == image1.N
    assert convolved_image.k == image1.k + filter_image.k
    assert convolved_image.parity == (image1.parity + filter_image.parity) % 2
    assert (convolved_image.data == jnp.array([[-2,0,2], [2,5,5], [-2,-1,3]], dtype=int)).all()

    key = random.PRNGKey(0)
    image2 = geom.geometric_image(jnp.floor(10*random.uniform(key, shape=(5,5))), 0, 2)
    convolved_image2 = image2.convolve_with(filter_image)
    assert convolved_image2.D == image2.D
    assert convolved_image2.N == image2.N
    assert convolved_image2.k == image2.k + filter_image.k
    assert convolved_image2.parity == (image2.parity + filter_image.parity) % 2
    assert (convolved_image2.data == jnp.array(
        [
            [16,9,16,11,10],
            [15,19,15,13,28],
            [17,19,15,16,17],
            [16,12,13,13,18],
            [8,23,11,13,29],
        ],
    dtype=int)).all()

def testUniqueInvariantFilters():
    # ensure that all the filters are actually invariant
    key = random.PRNGKey(0)

    for D in [2]: #image dimension
        operators = geom.make_all_operators(D)
        for N in [3]: #filter size
            key, subkey = random.split(key)
            image = geom.geometric_image(random.uniform(key, shape=(N,N)), 0, D)
            for k in [1]: #tensor order of filter
                for parity in [0,1]:
                    filters = geom.get_unique_invariant_filters(N, k, parity, D, operators)

                    for gg in operators:
                        for geom_filter in filters:
                            print(gg)
                            print(geom_filter.data)

                            # test that the filters are invariant to the group operators
                            assert jnp.allclose(geom_filter.data, geom_filter.times_group_element(gg).data)

                            # test that the convolution with the invariant filters is equivariant to gg
                            img1 = image.convolve_with(geom_filter).times_group_element(gg).data
                            img2 = image.times_group_element(gg).convolve_with(geom_filter).data
                            print(img1)
                            print(img2)

                            assert jnp.allclose(img1, img2)

# testUniqueInvariantFilters()

testIK0_FK0()
testIK0_FK1()

# key = random.PRNGKey(0)

# #N=10, D=2, parity=0, k=2
# rand_img = geom.geometric_image(random.uniform(key, shape=(1000,1000,2)), parity=0, D=2)

# #N=3, D=2, parity=0, k=0
# geom_filter = geom.geometric_filter(jnp.array([[[1,1], [0,0], [1,1]], [[0,0],[0,0],[0,0]], [[1,1],[0,0],[1,1]]]), parity=0, D=2)

# convolved_img = rand_img.convolve_with(geom_filter)
# # print(convolved_img.data)
# print(convolved_img.shape())





