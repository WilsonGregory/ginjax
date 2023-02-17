import itertools as it
import numpy as np
import math
import time
from functools import partial

import jax.numpy as jnp
import jax.random as random
from jax import jit

import geometricconvolutions.geometric as geom

def compute_number_of_maps(N, l):
    #l is the degree of the polynomial
    sum_res = 0
    for j in range(int(l/2)+1):
        sum_res += (l - 2*j + 1)*math.comb((N ** 2) + j - 2, j)

    return int(0.25*(math.comb(2*(N**2)+l - 1, l) + (-1)**(l+1)*sum_res))

def poly_filter(images, conv_filter):
    if len(images) == 1:
        return images[0]
    else:
        prod_image = images[0]
        for convolved_image in images[1:]:
            prod_image = prod_image * convolved_image

        return prod_image.convolve_with(conv_filter)

def all_kronecker_contractions(img, target_img, swappable_idxs=()):
    vector_images = []
    jit_contract_idxs = partial(jit, static_argnums=[0,1,2])(geom.get_contraction_indices)
    for idxs in jit_contract_idxs(img.k, target_img.k, swappable_idxs):
        idxs = tuple((int(x), int(y)) for x,y in idxs)
        img_contracted = img.multicontract(idxs)
        assert img_contracted.shape() == target_img.shape()
        vector_images.append(img_contracted.data.flatten())
    
    return vector_images

def get_vector_images(vector_image, conv_filters, degree):
    vector_images = []
    convolved_images = [vector_image.convolve_with(c) for c in conv_filters]
    for last_filter_idx in range(len(conv_filters)):
        last_filter = conv_filters[last_filter_idx]

        if degree == 1:
            if (
                ((last_filter.k)%2 == 0) and
                ((last_filter.parity)%2 == 0)
            ):
                print(last_filter_idx)
                img = vector_image.convolve_with(last_filter)
                if (last_filter.parity == 0 and last_filter.k == 2):
                    swappable_idxs = ((img.k-2, img.k-1),)
                else:
                    swappable_idxs = ()  
                vector_images.extend(all_kronecker_contractions(img, vector_image, swappable_idxs))
        else:
            for filter_idxs in it.combinations_with_replacement(range(len(conv_filters)), degree):
                image_set = [convolved_images[idx] for idx in filter_idxs]

                image_set_k = np.sum([image.k for image in image_set])
                image_set_parity = np.sum([image.parity for image in image_set])

                #conditions suitable for a sequence of kronecker contractions
                if (
                    ((image_set_k + last_filter.k - vector_image.k)%2 == 0) and
                    ((image_set_parity + last_filter.parity - vector_image.parity)%2 == 0)
                ):
                    img = poly_filter(image_set, last_filter)
                    swappable_idxs = ()
                    for i, filter_idx in enumerate(filter_idxs):
                        conv_filter_tmp = conv_filters[filter_idx]
                        if (conv_filter_tmp.parity == 0 and conv_filter_tmp.k == 2):
                            swappable_idxs = swappable_idxs + (((i*3)+1, (i*3)+2),)

                    if (last_filter.parity == 0 and last_filter.k == 2):
                        (img.k - 2, img.k - 1)

                    print(filter_idxs + (last_filter_idx,))
                    vector_images.extend(all_kronecker_contractions(img, vector_image, swappable_idxs))
                
    return jnp.array(vector_images) 

D = 2
N = 3
img_k = 1
group_operators = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    [N], 
    [1,2],
    [0], 
    D, 
    group_operators,
    scale='one', 
    return_list=True,
)

polynomial_degrees = [2]

#confirmed good:
#degree=1, N=3,5,7
#degree=2, N=3,5,7

#not yet good
#degree=3, N=3 I get 286/290

# Make an N side length, parity=0 geometric vector image on a D-torus
key = random.PRNGKey(time.time_ns())


for degree in polynomial_degrees:
    num_maps = compute_number_of_maps(N, degree)
    num_images = math.ceil(num_maps / ((N**D) * (D**img_k)))

    print('~~~~~~~~~~~~~~~~')
    print(f'Degree {degree}, expected maps: {num_maps}')

    key, subkey = random.split(key)
    img_shape = ((num_images,) + (N,)*D + (D,)*img_k)
    geom_img = geom.BatchGeometricImage(2*random.normal(subkey, shape=img_shape), 0, D)

    datablock = get_vector_images(geom_img, conv_filters, degree)
    print('Datablock', datablock.shape)
    unique_rows = np.unique(datablock, axis=0)
    print('Unique Rows', unique_rows.shape)
    np.save('../data/all_functions_unique_rows.npy', unique_rows)
    # unique_rows = np.load('../data/all_functions_unique_rows.npy')

    s = np.linalg.svd(unique_rows, full_matrices=False, compute_uv=False)

    np.save('../data/all_functions_s.npy',s)
    # s = np.load('../data/all_functions_s.npy')
    found_functions = np.sum(s > 100*geom.TINY)
    print("there are", found_functions, "different images")
    print('Matches formula:', found_functions == num_maps)

    print(s[found_functions-5:found_functions+5])


