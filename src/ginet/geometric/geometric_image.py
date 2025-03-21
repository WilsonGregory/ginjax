import itertools as it
import functools
import numpy as np
import matplotlib.axes
from typing_extensions import Any, Callable, Generator, Optional, Self, Sequence, Union

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import ArrayLike

from ginet.geometric.constants import LeviCivitaSymbol, KroneckerDeltaSymbol, TINY
from ginet.geometric.functional_geometric_image import (
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
import ginet.utils as utils


@register_pytree_node_class
class GeometricImage:
    """
    One of the main classes of the package. This class is a single geometric image, a.k.a. an image
    where every pixel is a k,p tensor. This class is primarily used for simple operations on
    geometric images and plotting.
    """

    D: int
    spatial_dims: tuple[int, ...]
    k: int
    data: jax.Array
    parity: int
    is_torus: tuple[bool, ...]

    # Constructors

    @classmethod
    def zeros(
        cls,
        N: Union[int, tuple[int, ...]],
        k: int,
        parity: int,
        D: int,
        is_torus: Union[bool, tuple[bool]] = True,
    ) -> Self:
        """
        Zero constructor for GeometricImage.

        args:
            N: length of all sides if an int, otherwise a tuple of the side lengths
            k: the order of the tensor in each pixel, i.e. 0 (scalar), 1 (vector), 2 (matrix), etc.
            parity: 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D: dimension of the image, and length of vectors or side length of matrices or tensors.
            is_torus: whether the datablock is a torus, used for convolutions

        returns:
            constructed GeometricImage
        """
        spatial_dims = N if isinstance(N, tuple) else (N,) * D
        assert len(spatial_dims) == D
        return cls(jnp.zeros(spatial_dims + (D,) * k), parity, D, is_torus)

    @classmethod
    def fill(
        cls,
        N: Union[int, tuple[int, ...]],
        parity: int,
        D: int,
        fill: Union[jax.Array, float],
        is_torus: Union[bool, tuple[bool, ...]] = True,
    ) -> Self:
        """
        Fill constructor to construct a geometric image every pixel as fill

        args:
            N: length of all sides if an int, otherwise a tuple of the side lengths
            parity: 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D: dimension of the image, and length of vectors or side length of matrices or tensors.
            fill: tensor to fill the image with
            is_torus: whether the datablock is a torus, used for convolutions. Defaults to true.

        returns:
            Constructed GeometricImage
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
        is_torus: Union[bool, tuple[bool, ...]] = True,
    ) -> None:
        """
        Constructor for GeometricImage. It will be (N^D x D^k), so if N=100, D=2, k=1, then it's
        (100 x 100 x 2). The spatial dimensions don't have to be square.

        args:
            data: image data, shape (spatial,tensor)
            parity: 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D: dimension of the image, and length of vectors or side length of matrices or tensors.
            is_torus: whether the datablock is a torus, used for convolutions.
                Takes either a tuple of bools of length D specifying whether each dimension is toroidal,
                or simply True or False which sets all dimensions to that value.
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
        """
        Copy the geometric image.
        """
        return self.__class__(self.data, self.parity, self.D, self.is_torus)

    # Getters, setters, basic info

    def hash(self: Self, indices: ArrayLike) -> tuple[jax.Array, ...]:
        """
        Converts an array of indices to their pixels on the torus by modding the indices with the
        spatial dimensions.

        args:
            indices: array of indices, shape (num_idx, D) to apply the remainder to

        returns:
            the pixel indices as a d-tuple of jax arrays
        """
        return hash(self.D, self.data, indices)

    def __getitem__(self: Self, key: Any) -> jax.Array:
        """
        Accessor for data values. Now you can do image[key] where k are indices or array slices and it will just work
        Note that JAX does not throw errors for indexing out of bounds

        args:
            key: JAX/numpy indexer, i.e. "0", "0,1,3", "4:, 2:3, 0" etc.

        returns:
            data from the specified index or slice.
        """
        return self.data[key]

    def __setitem__(self: Self, key: Any, val: Any) -> Self:
        """
        Set the jax array data to the specified value. Jax arrays are immutable, so this
        reconstructs the data object with copying, and is potentially slow.

        args:
            key: index or slice to access data
            val: value to set the data to

        returns:
            the geometric image
        """
        self.data = self.data.at[key].set(val)
        return self

    def shape(self: Self) -> tuple[int, ...]:
        """
        Return the full shape of the data block

        returns:
            The shape of the data block
        """
        return self.data.shape

    def image_shape(self: Self, plus_Ns: Optional[tuple[int, ...]] = None) -> tuple[int, ...]:
        """
        Return the shape of the data block that is not the ktensor shape, but what comes before that.

        args:
            plus_Ns: d-length tuple, N to add to each spatial dim

        returns:
            the shape of the image, modified by plus_Ns
        """
        plus_Ns = (0,) * self.D if (plus_Ns is None) else plus_Ns
        return tuple(N + plus_N for N, plus_N in zip(self.spatial_dims, plus_Ns))

    def pixel_shape(self: Self) -> tuple[int, ...]:
        """
        Return the shape of the data block that is the ktensor, aka the pixel of the image.

        returns:
            the shape of the pixel
        """
        return self.k * (self.D,)

    def pixel_size(self: Self) -> int:
        """
        Get the size of the pixel shape, i.e. (D,D,D) = D**3

        returns:
            the size of the pixels
        """
        return self.D**self.k

    def __str__(self: Self) -> str:
        """
        returns:
            the string representation of the GeometricImage
        """
        return "<{} object in D={} with spatial_dims={}, k={}, parity={}, is_torus={}>".format(
            self.__class__,
            self.D,
            self.spatial_dims,
            self.k,
            self.parity,
            self.is_torus,
        )

    # itertools does not have type hints, but it will be a product[tuple[int,...]]
    def keys(self: Self) -> Any:
        """
        Iterate over the keys of GeometricImage
        """
        return it.product(*list(range(N) for N in self.spatial_dims))

    def key_array(self: Self) -> jax.Array:
        """
        returns:
            the pixel indices as a jax array
        """
        # equivalent to the old pixels function
        return jnp.array([key for key in self.keys()], dtype=int)

    def pixels(self: Self) -> Generator[jax.Array]:
        """
        Iterate over the pixels of GeometricImage.

        returns:
            a generator of the pixels
        """
        for key in self.keys():
            yield self[key]

    def items(self: Self) -> Generator[tuple[Any, jax.Array]]:
        """
        Iterate over the key, pixel pairs of GeometricImage.

        returns:
            a generator of pairs of the pixel index and its pixel
        """
        for key in self.keys():
            yield (key, self[key])

    # Binary Operators, Complicated functions

    def __eq__(self: Self, other: object) -> bool:
        """
        Equality operator, must have same shape, parity, and data within the TINY=1e-5 tolerance.

        args:
            other: an object to compare to this GeometricImage

        returns:
            true if they are equal, false otherwise
        """
        if isinstance(other, GeometricImage):
            return (
                self.D == other.D
                and self.spatial_dims == other.spatial_dims
                and self.k == other.k
                and self.parity == other.parity
                and self.is_torus == other.is_torus
                and self.data.shape == other.data.shape
                and bool(jnp.allclose(self.data, other.data, rtol=TINY, atol=TINY))
            )
        else:
            return False

    def __add__(self: Self, other: Self) -> Self:
        """
        Addition operator for GeometricImages. Both must be the same size and parity. Returns a new GeometricImage.

        args:
            other: other image to add the the first one

        returns:
            a new GeometricImage that is the sum of this one and the other one
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
            other: other image to add the the first one

        returns:
            a new GeometricImage that is the difference of this GeometricImage and the other one
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

        returns:
            a new GeometricImage that is the product of this GeometricImage with other
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

        args:
            other (GeometricImage or number): scalar or image to multiply by

        returns:
            a new GeometricImage that is the product of this GeometricImage with other
        """
        return self * other

    def transpose(self: Self, axes_permutation: Sequence[int]) -> Self:
        """
        Transposes the axes of the tensor, keeping the image axes in the front the same

        args:
            axes_permutation: new axes order

        returns:
            a new GeometricImage that has been transposed
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
        stride: Union[int, tuple[int, ...]] = 1,
        padding: Optional[tuple[tuple[int, int]]] = None,
        lhs_dilation: Optional[tuple[int, ...]] = None,
        rhs_dilation: Union[int, tuple[int, ...]] = 1,
    ) -> Self:
        """
        See [convolve](functional_geometric_image.md#ginet.geometric.functional_geometric_image.convolve)
        for a description of this function.

        args:
            filter_image: the convolution filter, shape (out_c,in_c,spatial,tensor)
            stride: convolution stride, defaults to (1,)*self.D
            padding: either 'TORUS','VALID', 'SAME', or D length tuple of (upper,lower) pairs,
                defaults to 'TORUS' if image.is_torus, else 'SAME'
            lhs_dilation: amount of dilation to apply to image in each dimension D, also transposed conv
            rhs_dilation: amount of dilation to apply to filter in each dimension D, defaults to 1

        returns:
            convolved_image of shape (batch,out_c,spatial,tensor)
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
            patch_len: the side length of the patches, must evenly divide all spatial dims
            use_norm: whether to use norm to calculate the max

        returns:
            a new GeometricImage with the max pool applied
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
            patch_len: the side length of the patches, must evenly divide self.N

        returns:
            a new GeometricImage with the average pool applied
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
        Each pixel turns into a (patch_len,)*self.D patch of that pixel. Also called
        "Nearest Neighbor" unpooling.

        args:
            patch_len: side length of the patch of our unpooled images

        returns:
            a new GeometricImage with the unpool applied
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
            scalar: number to scale everything by

        returns:
            a new GeometricImage scaled by the scalar
        """
        return self * scalar

    @jax.jit
    def norm(self: Self) -> Self:
        """
        Calculate the norm pixel-wise. This becomes a scalar image.

        returns:
            a new GeoemtricImage of all the pixels normed.
        """
        return self.__class__(norm(self.D, self.data), 0, self.D, self.is_torus)

    def normalize(self: Self) -> Self:
        """
        Normalize so that the max norm of each pixel is 1, and all other tensors are scaled appropriately

        returns:
            a new GeometricImage scaled by the max norm
        """
        max_norm = float(jnp.max(self.norm().data))
        if max_norm > TINY:
            return self.times_scalar(1.0 / max_norm)
        else:
            return self.times_scalar(1.0)

    def activation_function(self: Self, function: Callable[[jnp.ndarray], jnp.ndarray]) -> Self:
        """
        Apply the specified activation function to the GeometricImage

        args:
            function: the activation function

        returns:
            a new GeometricImage with the activation function applied
        """
        assert (
            self.k == 0
        ), "Activation functions only implemented for k=0 tensors due to equivariance"
        return self.__class__(function(self.data), self.parity, self.D, self.is_torus)

    @functools.partial(jax.jit, static_argnums=[1, 2])
    def contract(self: Self, i: int, j: int) -> Self:
        """
        Use einsum to perform a kronecker contraction on two dimensions of the tensor

        args:
            i: first index of tensor
            j: second index of tensor

        returns:
            a new GeometricImage contracted by those indices
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
            indices: indices to contract

        returns:
            a new GeometricImage contracted by those indices
        """
        assert self.k >= 2
        idx_shift = len(self.image_shape())
        return self.__class__(
            multicontract(self.data, indices, idx_shift),
            self.parity,
            self.D,
            self.is_torus,
        )

    def levi_civita_contract(self: Self, indices: Union[tuple[int, ...], int]) -> Self:
        """
        Perform the Levi-Civita contraction. Outer product with the Levi-Civita Symbol, then perform D-1 contractions.
        Resulting image has k= self.k - self.D + 2

        args:
            indices: indices of tensor to perform contractions on

        returns:
            a new GeometricImage contracted by those indices
        """
        assert self.k >= (
            self.D - 1
        )  # so we have enough indices to work on since we perform D-1 contractions
        if not isinstance(indices, tuple):
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
        Get the rotated keys of this GeometricImage by the rotation gg.
        Slightly messier than with GeometricFilter because self.N-1 / 2 might not be an integer, but should work

        args:
            gg: a DxD matrix that rotates the tensor

        returns:
            the rotated keys
        """
        return get_rotated_keys(self.D, self.data, gg)

    def times_group_element(
        self: Self, gg: np.ndarray, precision: Optional[jax.lax.Precision] = None
    ) -> Self:
        """
        Apply a group element of SO(2) or SO(3) to the geometric image. First apply the action to the location of the
        pixels, then apply the action to the pixels themselves.

        args:
            gg: a DxD matrix that rotates the tensor
            precision: precision level for einsum, for equality tests use Precision.HIGH

        returns:
            a new GeometricImage that has been rotated
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
        ax: Optional[matplotlib.axes.Axes] = None,
        title: str = "",
        boxes: bool = False,
        fill: bool = True,
        symbols: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = False,
        vector_scaling: float = 0.5,
    ) -> None:
        """
        Plot the geometric image.

        args:
            ax: matplotlib.pyplot Axes to plot this geometric image one
            title: title of the plot
            boxes: whether to plot boxes around each pixel
            fill: whether to fill the pixels with an appropriate color
            symbols: whether to fill the pixels with a symbol
            vmin: min value to plot, everything below this is cut off. If none, will use actual min
            vmax: max value to plot, everything above this is cut off. If none, will use actual max
            colorbar: whether to plot a colorbar
            vector_scaling: how much to scale the vectors
        """
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
        key_array_transpose = np.array(self.key_array()).T
        xs = key_array_transpose[0]
        ys = key_array_transpose[1]
        zs = key_array_transpose[2:]
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
    """
    A subclass of GeometricImage that enforces square, odd spatial dimensions.
    """

    def __init__(
        self: Self,
        data: jnp.ndarray,
        parity: int,
        D: int,
        is_torus: Union[bool, tuple[bool, ...]] = True,
    ) -> None:
        """
        Constructor for GeometricFilter.

        args:
            data: the image data of shape (spatial,tensor). Spatial dimensions must be square, odd
            parity: parity of tensor, 0 for vector, 1 for pseudo-vector
            D: dimension of the image
            is_torus: which dimensions are toroidal
        """
        super(GeometricFilter, self).__init__(data, parity, D, is_torus)
        assert (
            self.spatial_dims == (self.spatial_dims[0],) * self.D
        ), "GeometricFilter: Filters must be square."  # I could remove  this requirement in the future

    @classmethod
    def from_image(cls, geometric_image: GeometricImage) -> Self:
        """
        Constructor that copies a GeometricImage and returns a GeometricFilter

        args:
            geometric_image: the GeometricImage to copy

        returns:
            a new GeometricFilter copy
        """
        return cls(
            geometric_image.data,
            geometric_image.parity,
            geometric_image.D,
            geometric_image.is_torus,
        )

    def __str__(self: Self) -> str:
        """
        returns:
            the string representation of the GeometricFilter
        """
        return "<geometric filter object in D={} with spatial_dims={}, k={}, parity={}, and is_torus={}>".format(
            self.D, self.spatial_dims, self.k, self.parity, self.is_torus
        )

    def bigness(self: Self) -> float:
        """
        Gives an idea of size for a filter, sparser filters are smaller while less sparse filters are larger

        returns:
            the bigness value
        """
        norms = self.norm().data
        numerator = 0.0
        for key in self.key_array():
            numerator += jnp.linalg.norm(key * norms[tuple(key)], ord=2)

        denominator = float(jnp.sum(norms))
        return numerator / denominator

    def rectify(self: Self) -> Self:
        """
        Filters form an equivalence class up to multiplication by a scalar, so if its negative we want to flip the sign

        returns:
            a new GeometricImage that has been scaled
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
        ax: Optional[Any] = None,
        title: str = "",
        boxes: bool = True,
        fill: bool = True,
        symbols: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = False,
        vector_scaling: float = 0.33,
    ) -> None:
        """
        Plot the geometric filter. Has different default vmin, vmax, vector_scalings than
        GeometricImage.

        args:
            ax: matplotlib.pyplot Axes to plot this geometric filter one
            title: title of the plot
            boxes: whether to plot boxes around each pixel
            fill: whether to fill the pixels with an appropriate color
            symbols: whether to fill the pixels with a symbol
            vmin: min value to plot, everything below this is cut off. If none, will use -3 for
                scalars and 0 otherwise.
            vmax: max value to plot, everything above this is cut off. If none, will use 3
            colorbar: whether to plot a colorbar
            vector_scaling: how much to scale the vectors
        """
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
    """
    Get an image with a Kronecker Delta in every pixel.

    TODO: This should not be able to create Kronecker Deltas that are not 2-tensors.

    args:
        N: the sidelength of the image
        D: the dimension of the image
        k: the tensor order

    returns:
        a new GeometricImage.
    """
    return GeometricImage(
        jnp.stack([KroneckerDeltaSymbol.get(D, k) for _ in range(N**D)]).reshape(
            ((N,) * D + (D,) * k)
        ),
        0,
        D,
    )
