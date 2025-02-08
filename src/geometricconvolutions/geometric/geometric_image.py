import itertools as it
import functools
import numpy as np
import matplotlib.pyplot as plt
from typing_extensions import Any, Callable, Generator, Optional, Self, Sequence, Union

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from geometricconvolutions.geometric.constants import LeviCivitaSymbol, KroneckerDeltaSymbol, TINY
from geometricconvolutions.geometric.functional_geometric_image import (
    average_pool,
    convolve,
    get_rotated_keys,
    hash,
    max_pool,
    mul,
    multicontract,
    norm,
    parse_shape,
    times_group_element,
)
import geometricconvolutions.utils as utils


@register_pytree_node_class
class GeometricImage:

    # Constructors

    @classmethod
    def zeros(
        cls,
        N: int,
        k: int,
        parity: int,
        D: int,
        is_torus: Union[bool, tuple[bool]] = True,
    ) -> Self:
        """
        Class method zeros to construct a geometric image of zeros
        args:
            N (int or tuple of ints): length of all sides if an int, otherwise a tuple of the side lengths
            k (int): the order of the tensor in each pixel, i.e. 0 (scalar), 1 (vector), 2 (matrix), etc.
            parity (int): 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D (int): dimension of the image, and length of vectors or side length of matrices or tensors.
            is_torus (bool): whether the datablock is a torus, used for convolutions. Defaults to true.
        """
        spatial_dims = N if isinstance(N, tuple) else (N,) * D
        assert len(spatial_dims) == D
        return cls(jnp.zeros(spatial_dims + (D,) * k), parity, D, is_torus)

    @classmethod
    def fill(
        cls,
        N: int,
        parity: int,
        D: int,
        fill: Union[jnp.ndarray, int, float],
        is_torus: Union[bool, tuple[bool]] = True,
    ) -> Self:
        """
        Class method fill constructor to construct a geometric image every pixel as fill
        args:
            N (int or tuple of ints): length of all sides if an int, otherwise a tuple of the side lengths
            parity (int): 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D (int): dimension of the image, and length of vectors or side length of matrices or tensors.
            fill (jnp.ndarray or number): tensor to fill the image with
            is_torus (bool): whether the datablock is a torus, used for convolutions. Defaults to true.
        """
        spatial_dims = N if isinstance(N, tuple) else (N,) * D
        assert len(spatial_dims) == D

        k = (
            len(fill.shape)
            if (isinstance(fill, jnp.ndarray) or isinstance(fill, np.ndarray))
            else 0
        )
        data = jnp.stack([fill for _ in range(np.multiply.reduce(spatial_dims))]).reshape(
            spatial_dims + (D,) * k
        )
        return cls(data, parity, D, is_torus)

    def __init__(
        self: Self,
        data: jnp.ndarray,
        parity: int,
        D: int,
        is_torus: Union[bool, tuple[bool]] = True,
    ) -> Self:
        """
        Construct the GeometricImage. It will be (N^D x D^k), so if N=100, D=2, k=1, then it's (100 x 100 x 2)
        args:
            data (array-like):
            parity (int): 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D (int): dimension of the image, and length of vectors or side length of matrices or tensors.
            is_torus (bool or tuple of bools): whether the datablock is a torus, used for convolutions.
                Takes either a tuple of bools of length D specifying whether each dimension is toroidal,
                or simply True or False which sets all dimensions to that value. Defaults to True.
        """
        self.D = D
        self.spatial_dims, self.k = parse_shape(data.shape, D)
        assert data.shape[D:] == self.k * (
            self.D,
        ), "GeometricImage: each pixel must be D cross D, k times"
        self.parity = parity % 2

        assert (isinstance(is_torus, tuple) and (len(is_torus) == D)) or isinstance(is_torus, bool)
        if isinstance(is_torus, bool):
            is_torus = (is_torus,) * D

        self.is_torus = is_torus

        self.data = jnp.copy(
            data
        )  # TODO: don't need to copy if data is already an immutable jnp array

    def copy(self: Self) -> Self:
        return self.__class__(self.data, self.parity, self.D, self.is_torus)

    # Getters, setters, basic info

    def hash(self: Self, indices: jnp.ndarray) -> jnp.ndarray:
        """
        Deals with torus by modding (with `np.remainder()`).
        args:
            indices (tuple of ints): indices to apply the remainder to
        """
        return hash(self.D, self.data, indices)

    def __getitem__(self: Self, key) -> jnp.ndarray:
        """
        Accessor for data values. Now you can do image[key] where k are indices or array slices and it will just work
        Note that JAX does not throw errors for indexing out of bounds
        args:
            key (index): JAX/numpy indexer, i.e. "0", "0,1,3", "4:, 2:3, 0" etc.
        """
        return self.data[key]

    def __setitem__(self: Self, key, val) -> Self:
        """
        Jax arrays are immutable, so this reconstructs the data object with copying, and is potentially slow
        """
        self.data = self.data.at[key].set(val)
        return self

    def shape(self: Self) -> tuple[int]:
        """
        Return the full shape of the data block
        """
        return self.data.shape

    def image_shape(self: Self, plus_Ns: Optional[tuple[int]] = None) -> tuple[int]:
        """
        Return the shape of the data block that is not the ktensor shape, but what comes before that.
        args:
            plus_Ns (tuple of ints): d-length tuple, N to add to each spatial dim
        """
        plus_Ns = (0,) * self.D if (plus_Ns is None) else plus_Ns
        return tuple(N + plus_N for N, plus_N in zip(self.spatial_dims, plus_Ns))

    def pixel_shape(self: Self) -> tuple[int]:
        """
        Return the shape of the data block that is the ktensor, aka the pixel of the image.
        """
        return self.k * (self.D,)

    def pixel_size(self: Self) -> int:
        """
        Get the size of the pixel shape, i.e. (D,D,D) = D**3
        """
        return self.D**self.k

    def __str__(self: Self) -> str:
        return "<{} object in D={} with spatial_dims={}, k={}, parity={}, is_torus={}>".format(
            self.__class__,
            self.D,
            self.spatial_dims,
            self.k,
            self.parity,
            self.is_torus,
        )

    def keys(self: Self) -> Sequence[Sequence[int]]:
        """
        Iterate over the keys of GeometricImage
        """
        return it.product(*list(range(N) for N in self.spatial_dims))

    def key_array(self: Self) -> jnp.ndarray:
        # equivalent to the old pixels function
        return jnp.array([key for key in self.keys()], dtype=int)

    def pixels(self: Self) -> Generator[jnp.ndarray]:
        """
        Iterate over the pixels of GeometricImage.
        """
        for key in self.keys():
            yield self[key]

    def items(self: Self) -> Generator[tuple[Any, jax.Array]]:
        """
        Iterate over the key, pixel pairs of GeometricImage.
        """
        for key in self.keys():
            yield (key, self[key])

    # Binary Operators, Complicated functions

    def __eq__(self: Self, other: Self) -> bool:
        """
        Equality operator, must have same shape, parity, and data within the TINY=1e-5 tolerance.
        """
        return (
            self.D == other.D
            and self.spatial_dims == other.spatial_dims
            and self.k == other.k
            and self.parity == other.parity
            and self.is_torus == other.is_torus
            and self.data.shape == other.data.shape
            and jnp.allclose(self.data, other.data, rtol=TINY, atol=TINY)
        )

    def __add__(self: Self, other: Self) -> Self:
        """
        Addition operator for GeometricImages. Both must be the same size and parity. Returns a new GeometricImage.
        args:
            other (GeometricImage): other image to add the the first one
        """
        assert self.D == other.D
        assert self.spatial_dims == other.spatial_dims
        assert self.k == other.k
        assert self.parity == other.parity
        assert self.is_torus == other.is_torus
        assert self.data.shape == other.data.shape
        return self.__class__(self.data + other.data, self.parity, self.D, self.is_torus)

    def __sub__(self: Self, other: Self) -> Self:
        """
        Subtraction operator for GeometricImages. Both must be the same size and parity. Returns a new GeometricImage.
        args:
            other (GeometricImage): other image to add the the first one
        """
        assert self.D == other.D
        assert self.spatial_dims == other.spatial_dims
        assert self.k == other.k
        assert self.parity == other.parity
        assert self.is_torus == other.is_torus
        assert self.data.shape == other.data.shape
        return self.__class__(self.data - other.data, self.parity, self.D, self.is_torus)

    def __mul__(self: Self, other: Union[Self, float, int]) -> Self:
        """
        If other is a scalar, do scalar multiplication of the data. If it is another GeometricImage, do the tensor
        product at each pixel. Return the result as a new GeometricImage.
        args:
            other (GeometricImage or number): scalar or image to multiply by
        """
        if isinstance(other, GeometricImage):
            assert self.D == other.D
            assert self.spatial_dims == other.spatial_dims
            assert self.is_torus == other.is_torus
            return self.__class__(
                mul(self.D, self.data, other.data),
                self.parity + other.parity,
                self.D,
                self.is_torus,
            )
        else:  # its an integer or a float, or something that can we can multiply a Jax array by (like a DeviceArray)
            return self.__class__(self.data * other, self.parity, self.D, self.is_torus)

    def __rmul__(self: Self, other: Union[Self, float, int]) -> Self:
        """
        If other is a scalar, multiply the data by the scalar. This is necessary for doing scalar * image, and it
        should only be called in that case.
        """
        return self * other

    def transpose(self: Self, axes_permutation: Sequence[int]) -> Self:
        """
        Transposes the axes of the tensor, keeping the image axes in the front the same
        args:
            axes_permutation (iterable of indices): new axes order
        """
        idx_shift = len(self.image_shape())
        new_indices = tuple(
            tuple(range(idx_shift)) + tuple(axis + idx_shift for axis in axes_permutation)
        )
        return self.__class__(
            jnp.transpose(self.data, new_indices), self.parity, self.D, self.is_torus
        )

    @functools.partial(jax.jit, static_argnums=[2, 3, 4, 5])
    def convolve_with(
        self: Self,
        filter_image: Self,
        stride: Optional[tuple[int]] = None,
        padding: Optional[tuple[int]] = None,
        lhs_dilation: Optional[tuple[int]] = None,
        rhs_dilation: Optional[tuple[int]] = None,
    ) -> Self:
        """
        See convolve for a description of this function.
        """
        convolved_array = convolve(
            self.D,
            self.data[None, None],  # add batch, in_channels axes
            filter_image.data[None, None],  # add out_channels, in_channels axes
            self.is_torus,
            stride,
            padding,
            lhs_dilation,
            rhs_dilation,
        )
        return self.__class__(
            convolved_array[0, 0],  # ignore batch, out_channels axes
            self.parity + filter_image.parity,
            self.D,
            self.is_torus,
        )

    @functools.partial(jax.jit, static_argnums=[1, 2])
    def max_pool(self: Self, patch_len: int, use_norm: bool = True) -> Self:
        """
         Perform a max pooling operation where the length of the side of each patch is patch_len. Max is determined
        by the norm of the pixel when use_norm is True. Note that for scalars, this will be the absolute value of
        the pixel. If you want to use the max instead, set use_norm to False (requires scalar images).
        args:
            patch_len (int): the side length of the patches, must evenly divide all spatial dims
        """
        return self.__class__(
            max_pool(self.D, self.data, patch_len, use_norm),
            self.parity,
            self.D,
            self.is_torus,
        )

    @functools.partial(jax.jit, static_argnums=1)
    def average_pool(self: Self, patch_len: int) -> Self:
        """
        Perform a average pooling operation where the length of the side of each patch is patch_len. This is
        equivalent to doing a convolution where each element of the filter is 1 over the number of pixels in the
        filter, the stride length is patch_len, and the padding is 'VALID'.
        args:
            patch_len (int): the side length of the patches, must evenly divide self.N
        """
        return self.__class__(
            average_pool(self.D, self.data, patch_len),
            self.parity,
            self.D,
            self.is_torus,
        )

    @functools.partial(jax.jit, static_argnums=1)
    def unpool(self: Self, patch_len: int) -> Self:
        """
        Each pixel turns into a (patch_len,)*self.D patch of that pixel. Also called "Nearest Neighbor" unpooling
        args:
            patch_len (int): side length of the patch of our unpooled images
        """
        grow_filter = GeometricImage(jnp.ones((patch_len,) * self.D), 0, self.D)
        return self.convolve_with(
            grow_filter,
            padding=((patch_len - 1,) * 2,) * self.D,
            lhs_dilation=(patch_len,) * self.D,
        )

    def times_scalar(self: Self, scalar: float) -> Self:
        """
        Scale the data by a scalar, returning a new GeometricImage object. Alias of the multiplication operator.
        args:
            scalar (number): number to scale everything by
        """
        return self * scalar

    @jax.jit
    def norm(self: Self) -> Self:
        """
        Calculate the norm pixel-wise. This becomes a 0 parity image.
        returns: scalar image
        """
        return self.__class__(norm(self.D, self.data), 0, self.D, self.is_torus)

    def normalize(self: Self) -> Self:
        """
        Normalize so that the max norm of each pixel is 1, and all other tensors are scaled appropriately
        """
        max_norm = jnp.max(self.norm().data)
        if max_norm > TINY:
            return self.times_scalar(1.0 / max_norm)
        else:
            return self.times_scalar(1.0)

    def activation_function(self: Self, function: Callable[[jnp.ndarray], jnp.ndarray]) -> Self:
        assert (
            self.k == 0
        ), "Activation functions only implemented for k=0 tensors due to equivariance"
        return self.__class__(function(self.data), self.parity, self.D, self.is_torus)

    @functools.partial(jax.jit, static_argnums=[1, 2])
    def contract(self: Self, i: int, j: int) -> Self:
        """
        Use einsum to perform a kronecker contraction on two dimensions of the tensor
        args:
            i (int): first index of tensor
            j (int): second index of tensor
        """
        assert self.k >= 2
        idx_shift = len(self.image_shape())
        return self.__class__(
            multicontract(self.data, ((i, j),), idx_shift),
            self.parity,
            self.D,
            self.is_torus,
        )

    @functools.partial(jax.jit, static_argnums=1)
    def multicontract(self: Self, indices: tuple[tuple[int]]) -> Self:
        """
        Use einsum to perform a kronecker contraction on two dimensions of the tensor
        args:
            indices (tuple of tuples of ints): indices to contract
        """
        assert self.k >= 2
        idx_shift = len(self.image_shape())
        return self.__class__(
            multicontract(self.data, indices, idx_shift),
            self.parity,
            self.D,
            self.is_torus,
        )

    def levi_civita_contract(self: Self, indices: tuple[tuple[int]]) -> Self:
        """
        Perform the Levi-Civita contraction. Outer product with the Levi-Civita Symbol, then perform D-1 contractions.
        Resulting image has k= self.k - self.D + 2
        args:
            indices (int, or tuple, or list): indices of tensor to perform contractions on
        """
        assert self.k >= (
            self.D - 1
        )  # so we have enough indices to work on since we perform D-1 contractions
        if self.D == 2 and not (isinstance(indices, tuple) or isinstance(indices, list)):
            indices = (indices,)
        assert len(indices) == self.D - 1

        levi_civita = LeviCivitaSymbol.get(self.D)
        outer = jnp.tensordot(self.data, levi_civita, axes=0)

        # make contraction index pairs with one of specified indices, and index (in order) from the levi_civita symbol
        idx_shift = len(self.image_shape())
        zipped_indices = tuple(
            (i + idx_shift, j + idx_shift)
            for i, j in zip(indices, range(self.k, self.k + len(indices)))
        )
        return self.__class__(
            multicontract(outer, zipped_indices), self.parity + 1, self.D, self.is_torus
        )

    def get_rotated_keys(self: Self, gg: np.ndarray) -> np.ndarray:
        """
        Slightly messier than with GeometricFilter because self.N-1 / 2 might not be an integer, but should work
        args:
            gg (jnp array-like): group operation
        """
        return get_rotated_keys(self.D, self.data, gg)

    def times_group_element(
        self: Self, gg: np.ndarray, precision: Optional[jax.lax.Precision] = None
    ) -> Self:
        """
        Apply a group element of SO(2) or SO(3) to the geometric image. First apply the action to the location of the
        pixels, then apply the action to the pixels themselves.
        args:
            gg (group operation matrix): a DxD matrix that rotates the tensor
            precision (jax.lax.Precision): precision level for einsum, for equality tests use Precision.HIGH
        """
        assert self.k < 14
        assert gg.shape == (self.D, self.D)

        return self.__class__(
            times_group_element(self.D, self.data, self.parity, gg, precision=precision),
            self.parity,
            self.D,
            self.is_torus,
        )

    def plot(
        self: Self,
        ax: Optional[plt.Axes] = None,
        title: str = "",
        boxes: bool = False,
        fill: bool = True,
        symbols: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = False,
        vector_scaling: float = 0.5,
    ) -> None:
        # plot functions should fail gracefully
        if self.D != 2 and self.D != 3:
            print(
                f"GeometricImage::plot: Can only plot dimension 2 or 3 images, but got D={self.D}"
            )
            return
        if self.k > 2:
            print(
                f"GeometricImage::plot: Can only plot tensor order 0,1, or 2 images, but got k={self.k}"
            )
            return
        if self.k == 2 and self.D == 3:
            print(f"GeometricImage::plot: Cannot plot D=3, k=2 geometric images.")
            return

        ax = utils.setup_plot() if ax is None else ax

        # This was breaking earlier with jax arrays, not sure why. I really don't want plotting to break,
        # so I am will swap to numpy arrays just in case.
        xs, ys, *zs = np.array(self.key_array()).T
        if self.D == 3:
            xs = xs + utils.XOFF * zs
            ys = ys + utils.YOFF * zs

        pixels = np.array(list(self.pixels()))

        if self.k == 0:
            vmin = np.min(pixels) if vmin is None else vmin
            vmax = np.max(pixels) if vmax is None else vmax
            utils.plot_scalars(
                ax,
                self.spatial_dims,
                xs,
                ys,
                pixels,
                boxes=boxes,
                fill=fill,
                symbols=symbols,
                vmin=vmin,
                vmax=vmax,
                colorbar=colorbar,
            )
        elif self.k == 1:
            vmin = 0.0 if vmin is None else vmin
            vmax = 2.0 if vmax is None else vmax
            utils.plot_vectors(
                ax,
                xs,
                ys,
                pixels,
                boxes=boxes,
                fill=fill,
                vmin=vmin,
                vmax=vmax,
                scaling=vector_scaling,
            )
        else:  # self.k == 2
            utils.plot_tensors(ax, xs, ys, pixels, boxes=boxes)

        utils.finish_plot(ax, title, xs, ys, self.D)

    def tree_flatten(
        self: Self,
    ) -> tuple[tuple[jnp.ndarray], dict[str, Union[int, Union[bool, tuple[bool]]]]]:
        """
        Helper function to define GeometricImage as a pytree so jax.jit handles it correctly. Children and aux_data
        must contain all the variables that are passed in __init__()
        """
        children = (self.data,)  # arrays / dynamic values
        aux_data = {
            "D": self.D,
            "parity": self.parity,
            "is_torus": self.is_torus,
        }  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Helper function to define GeometricImage as a pytree so jax.jit handles it correctly.
        """
        return cls(*children, **aux_data)


@register_pytree_node_class
class GeometricFilter(GeometricImage):

    def __init__(
        self: Self,
        data: jnp.ndarray,
        parity: int,
        D: int,
        is_torus: Union[bool, tuple[bool]] = True,
    ) -> Self:
        super(GeometricFilter, self).__init__(data, parity, D, is_torus)
        assert (
            self.spatial_dims == (self.spatial_dims[0],) * self.D
        ), "GeometricFilter: Filters must be square."  # I could remove  this requirement in the future

    @classmethod
    def from_image(cls, geometric_image: GeometricImage) -> Self:
        """
        Constructor that copies a GeometricImage and returns a GeometricFilter
        """
        return cls(
            geometric_image.data,
            geometric_image.parity,
            geometric_image.D,
            geometric_image.is_torus,
        )

    def __str__(self: Self) -> str:
        return "<geometric filter object in D={} with spatial_dims={}, k={}, parity={}, and is_torus={}>".format(
            self.D, self.spatial_dims, self.k, self.parity, self.is_torus
        )

    def bigness(self: Self) -> float:
        """
        Gives an idea of size for a filter, sparser filters are smaller while less sparse filters are larger
        """
        norms = self.norm().data
        numerator = 0.0
        for key in self.key_array():
            numerator += jnp.linalg.norm(key * norms[tuple(key)], ord=2)

        denominator = jnp.sum(norms)
        return numerator / denominator

    def rectify(self: Self) -> Self:
        """
        Filters form an equivalence class up to multiplication by a scalar, so if its negative we want to flip the sign
        """
        if self.k == 0:
            if jnp.sum(self.data) < 0:
                return self.times_scalar(-1)
        elif self.k == 1:
            if self.parity % 2 == 0:
                if np.sum([np.dot(np.array(key), self[key]) for key in self.keys()]) < 0:
                    return self.times_scalar(-1)
            elif self.D == 2:
                if np.sum([np.cross(np.array(key), self[key]) for key in self.keys()]) < 0:
                    return self.times_scalar(-1)
        return self

    def plot(
        self: Self,
        ax: Optional[plt.Axes] = None,
        title: str = "",
        boxes: bool = True,
        fill: bool = True,
        symbols: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = False,
        vector_scaling: float = 0.33,
    ) -> None:
        if self.k == 0:
            vmin = -3.0 if vmin is None else vmin
            vmax = 3.0 if vmax is None else vmax
        else:
            vmin = 0.0 if vmin is None else vmin
            vmax = 3.0 if vmax is None else vmax

        super(GeometricFilter, self).plot(
            ax, title, boxes, fill, symbols, vmin, vmax, colorbar, vector_scaling
        )


def get_kronecker_delta_image(N: int, D: int, k: int) -> GeometricImage:
    return GeometricImage(
        jnp.stack([KroneckerDeltaSymbol.get(D, k) for _ in range(N**D)]).reshape(
            ((N,) * D + (D,) * k)
        ),
        0,
        D,
    )
