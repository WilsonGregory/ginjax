import sys
sys.path.insert(0,'src/geometricconvolutions/')

import geometric as geom
import jax.numpy as jnp
import jax.random as random

key = random.PRNGKey(0)

image = geom.geometric_image(random.uniform(key, shape=(5,5)), 0, 2)
filter_image = geom.geometric_filter.zeros(3, 0, 0, 2)
filter_image[0,0] = filter_image[0,2] = filter_image[2,0] = filter_image[2,2] = 1
# print(image)
print(filter_image.data)
print(image.data)
convolved_image = image.convolve_with(filter_image)
print(convolved_image.data)

