import sys
sys.path.insert(0,'src/geometricconvolutions/')

from geometric import ktensor
import jax.numpy as jnp
import jax.random as random

key = random.PRNGKey(0)

# image = geom.geometric_image(random.uniform(key, shape=(5,5,5,3,3,3)), 0, 3)
# contracted_image0_1 = image.levi_civita_contract((0,1))
# print(contracted_image0_1.data)
# print(contracted_image0_1.data.shape)
# contracted_image0_2 = image.levi_civita_contract((0,2))
# contracted_image1_2 = image.levi_civita_contract((1,2))
# contracted_image1_0 = image.levi_civita_contract((1,0))
# print()
# print((contracted_image1_0.data == contracted_image0_1.data).all())
# print(contracted_image.data.shape)

a = ktensor(jnp.array([0,1,3]), 0, 3)
b = ktensor(jnp.array([1,2,-1]), 0, 3)

ab = a*b
print(ab.data)
cross = ab.levi_civita_contract((0,1))
print(cross.data)



# filter_image = geom.geometric_filter.zeros(3, 0, 0, 2)
# filter_image[0,0] = filter_image[0,2] = filter_image[2,0] = filter_image[2,2] = 1
# # print(image)
# print(filter_image.data)
# print(image.data)
# convolved_image = image.convolve_with(filter_image)
# print(convolved_image.data)



# print(geom.get_levi_civita_symbol(4))
