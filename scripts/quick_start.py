import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

import ginet.geometric as geom

key = random.PRNGKey(0)
key, subkey = random.split(key)

# Suppose you have some data that forms a 3 by 3 vector image, so N=3, D=2, and k=1.
N = 3  # image side length. Currently only square or cube images are valid.
D = 2  # image dimensions. Currently, D=2 and D=3 are valid dimensions
k = 1  # pixel tensor order. Can be 1 or 0
parity = 0  # parity, how the image responds when it is reflected. This is used by pseudovectors like angular velocity.
data = random.normal(subkey, shape=((N,) * D + (D,) * k))

# We can form a GeometricImage from this data by specifying to data, the parity, and D. N and k are inferred from data.
image = geom.GeometricImage(data, parity=0, D=2)

# We can visualize this image with the plot function:
image.plot()

# Now we can do various operations on this geometric image
image2 = geom.GeometricImage.fill(
    N, parity, D, fill=jnp.array([1, 0])
)  # fill constructor, each pixel is fill

# pixel-wise addition
image + image2

# pixel-wise subtraction
image - image2

# pixel-wise tensor product
image * image2

# scalar multiplication
image * 3

# We can also apply a group action on the image. First we generate all the operators for dimension D, then we apply one
operators = geom.make_all_operators(D)
print("Number of operators:", len(operators))
image.times_group_element(operators[1])

# Now let us generate all 3 by 3 filters of tensor order k=0,1 and parity=0,1 that are invariant to the operators
invariant_filters = geom.get_invariant_filters_list(
    Ms=[3],
    ks=[0, 1],
    parities=[0, 1],
    D=D,
    operators=operators,
    scale="one",  # all the values of the filter are 1, can also 'normalize' so the norm of the tensor pixel is 1
)
print("Number of invariant filters N=3, k=0,1 parity=0,1:", len(invariant_filters))

# Using these filters, we can perform convolutions on our image. Since the filters are invariant, the convolution
# will be equivariant.
gg = operators[1]  # one operator, a flip over the y-axis
ff_k0 = invariant_filters[1]  # one filter, a non-trivial scalar filter
print(
    "Equivariant:",
    jnp.allclose(
        image.times_group_element(gg).convolve_with(ff_k0).data,
        image.convolve_with(ff_k0).times_group_element(gg).data,
        rtol=1e-2,
        atol=1e-2,
    ),
)

# When convolving with filters that have tensor order > 0, the resulting image have tensor order img.k + filter.k
ff_k1 = invariant_filters[5]
print("image k:", image.k)
print("filter k:", ff_k1.k)
convolved_image = image.convolve_with(ff_k1)
print("convolved image k:", convolved_image.k)

# After convolving, the image has tensor order 1+1=2 pixels. We can transpose the indices of the tensor:
convolved_image.transpose((1, 0))

# Since the tensor order is >= 2, we can perform a contraction on those indices which will reduce it to tensor order 0.
print("contracted image k:", convolved_image.contract(0, 1).k)

# Show images.
plt.show()
