import geometricconvolutions.geometric as geom
import geometricconvolutions.utils as utils
import jax.random as random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import pandas as pd

np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(float=lambda x: "%.3g" % x))

def getLargeRepresentation(X, g):
    lst = []
    for i in range(X.data.size):
        permutation = X.data.size*[0]
        permutation[i] = 1
        rotated_basis = geom.GeometricImage(jnp.array(permutation).reshape((3,3,2)), 0, 2).times_group_element(g)
        lst.append(rotated_basis.data.reshape((rotated_basis.data.size,1)))

    matrix_of_action = jnp.concatenate(lst, axis=1)
    return matrix_of_action

def getLargeRepConv(X, C):
    lst = []
    for i in range(X.data.size):
        permutation = X.data.size*[0]
        permutation[i] = 1
        convolved_basis = geom.GeometricImage(jnp.array(permutation).reshape((3,3,2)), 0, X.D).convolve_with(C)
        lst.append(convolved_basis.data.reshape((convolved_basis.data.size,1)))


    matrix_of_conv = jnp.concatenate(lst, axis=1)
    return matrix_of_conv

def getHrs(r, s, shape):
    assert r >= s
    Hrs = np.zeros(shape)
    Hrs[r, s] = 1
    Hrs[s, r] = 1
    return Hrs

def group_average_Hrs(Hrs, operators_big, basis_list, idxs_list):
    # print(len(basis_list))
    H = np.zeros(Hrs.shape)
    for g in operators_big:
        for i in range(len(basis_list)):
            g = (basis_list[i].T @ g @ basis_list[i])[idxs_list[i]]

        H = H + (g.T @ Hrs @ g)

    H = H / len(operators_big)

    for g in operators_big:
        for i in range(len(basis_list)):
            g = (basis_list[i].T @ g @ basis_list[i])[idxs_list[i]]

        # print(g @ H)
        # print(H @ g)

        assert np.allclose(g @ H, H @ g, rtol=geom.TINY, atol=geom.TINY)

    return H

def findH(shape, operators, basis_list, idxs_list):
    found_H = False
    for r in range(shape[0]):
        for s in range(r+1):
            Hrs = getHrs(r, s, shape)

            H = group_average_Hrs(Hrs, operators_big, basis_list, idxs_list)
            if (not np.allclose(H - np.diag(np.diagonal(H)), np.zeros(shape), atol=geom.TINY, rtol=geom.TINY)):
                found_H = True
                break

        if (found_H):
            break

    return found_H, H

def findPwithJordanDecomp(operators_big, basis_list, idxs_list, og_P):
    print(len(basis_list)*'.' + str(og_P.shape[1]))
    if og_P.shape[1] == 1: #there is a single column, so clearly it is irreducible
        return og_P

    big_shape = (og_P.shape[1], og_P.shape[1])
    found_H, H = findH(big_shape, operators_big, basis_list, idxs_list)

    if (not found_H): #its already irreducible
        # print('no H')
        return og_P

    evals, P = np.linalg.eigh(H) #P is a matrix, evals is a vector
    assert np.allclose(P @ np.diag(evals) @ P.T, H)
    # print(evals)

    basis_columns_section = og_P @ P

    p_cols = np.array([])
    start = 0
    i = 0
    for i in range(big_shape[0]):
        if not np.isclose(evals[start], evals[i], atol=geom.TINY, rtol=geom.TINY): #its a different value
            p_col = findPwithJordanDecomp(
                operators_big,
                basis_list + [P],
                idxs_list + [2*(slice(start, i),)],
                basis_columns_section[:, slice(start,i)], # put those columns in the new basis
            )
            p_cols = np.concatenate((p_cols, p_col), axis=1) if p_cols.size else p_col
            start = i

    i += 1
    # recurse with the last block at the end
    p_col = findPwithJordanDecomp(
        operators_big,
        basis_list + [P],
        idxs_list + [2*(slice(start, i),)],
        basis_columns_section[:, slice(start,i)], # put those columns in the new basis
    )
    p_cols = np.concatenate((p_cols, p_col), axis=1) if p_cols.size else p_col

    return p_cols


#Main
img_shape = (3,3,2)
img_shape_size = 18
D = 2

operators = geom.make_all_operators(D)
conv_filters = geom.get_unique_invariant_filters(3, 0, 0, 2, operators)

key = random.PRNGKey(0)
arr = random.normal(key, shape=img_shape)
X = geom.GeometricImage(arr, parity=0, D=D).normalize()
conv_large_reps = [getLargeRepConv(X, C) for C in conv_filters]

operators_big = [getLargeRepresentation(X, g) for g in operators]

P = findPwithJordanDecomp(
    operators_big,
    [np.eye(img_shape_size)], #basis list
    [(slice(0,img_shape_size), slice(0,img_shape_size))], #starting indices, all of it
    np.eye(img_shape_size), #starting basis
)

# P = np.around(P, decimals=3)

# print(P @ P.T)
print(np.allclose(P @ P.T, np.eye(P.shape[0]), rtol=geom.TINY, atol=geom.TINY))

for g in operators_big:
    assert np.allclose(g @ conv_large_reps[1], conv_large_reps[1] @ g)

# for i in range(len(operators_big)):
#     print(f'Original Operator {i}')
#     print(operators_big[i])
#     df = pd.DataFrame(np.around(P.T @ operators_big[i] @ P, decimals=2))
#     with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#         print(df)


print(f'Original Operator {6}')
print(operators_big[6])
df = pd.DataFrame(np.around(P.T @ operators_big[6] @ P, decimals=2))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)

print('~~~~~~~~~~~~~~~~~~~~~~~~')
print('Original Convolution 2')
print(conv_large_reps[2])
df = pd.DataFrame(np.around(P.T @ conv_large_reps[2] @ P, decimals=2))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)



print('ill0:', np.allclose(X.convolve_with(conv_filters[0]).data.reshape((18,1)), conv_large_reps[0]@X.data.reshape((18,1))))
print('ill1:', np.allclose(X.convolve_with(conv_filters[1]).data.reshape((18,1)), conv_large_reps[1]@X.data.reshape((18,1))))
print('ill2:', np.allclose(X.convolve_with(conv_filters[2]).data.reshape((18,1)), conv_large_reps[2]@X.data.reshape((18,1))))









