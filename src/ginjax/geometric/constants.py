import itertools as it
import numpy as np
from typing_extensions import Sequence

import jax.numpy as jnp

TINY = 1.0e-5
LETTERS = "abcdefghijklmnopqrstuvwxyxABCDEFGHIJKLMNOPQRSTUVWXYZ"


class KroneckerDeltaSymbol:
    """
    TODO: the KroneckerDelta should only be a 2-tensor, not a k-tensor.
    """

    symbol_dict = {}

    @classmethod
    def get(cls, D: int, k: int) -> np.ndarray:
        """
        Get the Levi Civita symbol for dimension D from the cache, or creating it on a cache miss
        args:
            D (int): dimension of the Kronecker symbol
            k (int): order of the Kronecker Delta symbol
        """
        assert D > 1
        assert k > 1
        if (D, k) not in cls.symbol_dict:
            arr = np.zeros((k * (D,)), dtype=int)
            for i in range(D):
                arr[(i,) * k] = 1
            cls.symbol_dict[(D, k)] = arr

        return cls.symbol_dict[(D, k)]


def permutation_parity(pi: Sequence[int]) -> int:
    """
    Code taken from Sympy Permutations: https://github.com/sympy/sympy/blob/26f7bdbe3f860e7b4492e102edec2d6b429b5aaf/sympy/combinatorics/permutations.py#L114
    Slightly modified to return 1 for even permutations, -1 for odd permutations, and 0 for repeated digits
    Permutations of length n must consist of numbers {0, 1, ..., n-1}

    args:
        pi: permutation sequence

    returns:
        parity of the permutation
    """
    if len(np.unique(pi)) != len(pi):
        return 0

    n = len(pi)
    a = [0] * n
    c = 0
    for j in range(n):
        if a[j] == 0:
            c += 1
            a[j] = 1
            i = j
            while pi[i] != j:
                i = pi[i]
                a[i] = 1

    # code originally returned 1 for odd permutations (we want -1) and 0 for even permutations (we want 1)
    return -2 * ((n - c) % 2) + 1


class LeviCivitaSymbol:
    """
    The Levi Civita symbol, the unique alternating D-tensor. The $i_{1},...,i_{D}$ entry of tensor is +1 for
    a positive parity sequence, -1 for a negative parity sequence, and 0 if any indices are repeated.
    """

    symbol_dict = {}

    @classmethod
    def get(cls, D: int) -> jnp.ndarray:
        """
        Get the Levi Civita symbol for dimension D from the cache, or creating it on a cache miss

        args:
            D: dimension of the Levi Civita symbol

        returns:
            the Levi Civita tensor
        """
        assert D > 1
        if D not in cls.symbol_dict:
            arr = np.zeros((D * (D,)), dtype=int)
            for index in it.product(range(D), repeat=D):
                arr[index] = permutation_parity(index)
            cls.symbol_dict[D] = jnp.array(arr)

        return cls.symbol_dict[D]
