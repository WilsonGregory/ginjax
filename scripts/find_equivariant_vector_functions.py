import itertools as it
import numpy as np
import math
import time

import jax.random as random
from jax import vmap

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml

def compute_number_of_maps(N, l):
    #l is the degree of the polynomial
    sum_res = 0
    for j in range(int(l/2)+1):
        sum_res += (l - 2*j + 1)*math.comb((N ** 2) + j - 2, j)

    return int(0.25*(math.comb(2*(N**2)+l - 1, l) + (-1)**(l+1)*sum_res))

def get_images_random(key, num_images, N, D, k):
    key, subkey = random.split(key)
    img_shape = ((num_images,) + (N,)*D + (D,)*k)

    return geom.BatchGeometricImage(random.uniform(subkey, shape=img_shape, minval=-100, maxval=100), 0, D)

def poly_filter(D, images):
    prod_image = images[0]
    for convolved_image in images[1:]:
        prod_image = geom.mul(D, prod_image, convolved_image)

    return prod_image

def get_swappable_indices(k_tuple, img_k, filter_k, res_k):
    swappable_idxs = ()
    curr_axis_idx = 0
    for tmp_k in k_tuple:
        if (tmp_k == (img_k + 2)): # we can swap those two axes
            swappable_idxs += ((curr_axis_idx + img_k, curr_axis_idx + img_k + 1),)

        curr_axis_idx += tmp_k

    if (filter_k == 2):
        swappable_idxs += ((res_k - 2, res_k - 1),)

    return swappable_idxs

def get_linear_functions(D, image_data, conv_filters):
    _, img_k = geom.parse_shape(image_data.shape, D)

    out_layer = {}
    for conv_filter in conv_filters:
        _, filter_k = geom.parse_shape(conv_filter.shape, D)
        if ((filter_k % 2) != 0):
            continue

        res = geom.convolve(D, image_data, conv_filter, True)
        res_k = img_k + filter_k 

        out_layer = ml.add_to_layer(out_layer, res_k, res.reshape((1,) + res.shape))

    all = ml.all_contractions(img_k, out_layer, D)
    print(all.shape)
    return all

def get_vector_images_fast(D, image_data_block, conv_filters, degree):
    #vmap over the multiple image
    # vmap_apply_funcs = vmap(apply_all_functions, in_axes=(None, 0, None, None))
    vmap_apply_funcs = vmap(apply_all_functions, in_axes=(None, 0, None, None))
    datablock = vmap_apply_funcs(D, image_data_block, [c.data for c in conv_filters], degree)
    return np.transpose(datablock, axes=(1,2,0)).reshape((datablock.shape[1],-1))

def apply_all_functions(D, image_data, conv_filters, degree):
    _, img_k = geom.parse_shape(image_data.shape, D)
    # image_data is a block of (N**D, D**k)
    # multiple images, one filter
    multi_image_convolve = vmap(geom.convolve, in_axes=(None, 0, None, None, None, None, None, None))
    multi_filter_convolve = vmap(
        multi_image_convolve, 
        in_axes=(None, None, 0, None, None, None, None, None),
    )

    conv_filter_layer = {}
    for conv_filter in conv_filters:
        _, filter_k = geom.parse_shape(conv_filter.shape, D) 
        conv_filter_layer = ml.add_to_layer(
            conv_filter_layer, 
            filter_k,
            conv_filter.reshape((1,) + conv_filter.shape),
        )

    convolved_images = [geom.convolve(D, image_data, c, True) for c in conv_filters]
    layer = {}

    # now do polynomials
    for filter_idxs in it.combinations_with_replacement(range(len(conv_filters)), degree):
        image_set = [convolved_images[idx] for idx in filter_idxs]
        poly_img = poly_filter(D, image_set)
        k_tuple = tuple([geom.parse_shape(img.shape, D)[1] for img in image_set])
        layer = ml.add_to_layer(layer, k_tuple, poly_img.reshape((1,) + poly_img.shape))

    out_layer = {}
    for k_tuple in layer.keys():
        prods_group = layer[k_tuple]
        for filter_k, filter_group in conv_filter_layer.items():
            _, k = geom.parse_shape(prods_group.shape[1:], D)
            if (((k + 1 - filter_k) % 2 != 0)):
                continue

            print(k_tuple, filter_k)

            res_k = k + filter_k
            res = multi_filter_convolve(
                D, 
                prods_group, 
                filter_group, 
                True, #is_torus
                None, #stride
                None, #padding
                None, #lhs_dilations
                None, #rhs_dilations
            )
            res = res.reshape((len(prods_group) * len(filter_group),) + res.shape[2:])

            swappable_idxs = get_swappable_indices(k_tuple, img_k, filter_k, res_k)
            print(swappable_idxs)

            #now contract with those swappable idxs
            idx_shift = 1 + D # layer plus N^D 
            for contract_idx in geom.get_contraction_indices(res_k, img_k, swappable_idxs):
                shifted_idx = tuple((i + idx_shift, j + idx_shift) for i,j in contract_idx)
                contracted_img = geom.multicontract(res, shifted_idx)

                out_layer = ml.add_to_layer(out_layer, img_k, contracted_img)

    return out_layer[img_k].reshape((out_layer[img_k].shape[0],-1))

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
#degree=3, N=3 I get 289/290

# Make an N side length, parity=0 geometric vector image on a D-torus
key = random.PRNGKey(time.time_ns())

for degree in polynomial_degrees:
    num_maps = compute_number_of_maps(N, degree)
    num_images = math.ceil(num_maps / ((N**D) * (D**img_k)))
    # need to increase the number of images for degee=2, N=7

    print('~~~~~~~~~~~~~~~~')
    print(f'Degree {degree}, expected maps: {num_maps}')

    geom_img = get_images_random(key, num_images, N, D, img_k)

    datablock = get_vector_images_fast(D, geom_img.data, conv_filters, degree)
    print('Datablock', datablock.shape)
    unique_rows = np.unique(datablock, axis=0)
    np.save(f'../data/all_functions_unique_rows_deg{degree}_n{N}.npy', unique_rows)
    # unique_rows = np.load(f'../data/all_functions_unique_rows_deg{degree}_n{N}.npy')
    print('Unique Rows', unique_rows.shape)

    s = np.linalg.svd(unique_rows, full_matrices=False, compute_uv=False)

    np.save(f'../data/all_functions_s_deg{degree}_n{N}.npy',s)
    # s = np.load(f'../data/all_functions_s_deg{degree}_n{N}.npy')
    found_functions = np.sum(s > 1)
    print("there are", found_functions, "different images")
    print('Matches formula:', found_functions == num_maps)

    print(s[found_functions-5:found_functions+5])
    print(s[num_maps-5:num_maps+5])


