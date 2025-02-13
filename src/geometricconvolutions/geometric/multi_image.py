import functools
import numpy as np
from typing_extensions import NewType, Optional, Self, Sequence, Union
from collections.abc import ItemsView, KeysView, ValuesView

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import ArrayLike

from geometricconvolutions.geometric.constants import TINY
from geometricconvolutions.geometric.functional_geometric_image import norm, times_group_element
from geometricconvolutions.geometric.geometric_image import GeometricImage

Signature = NewType("Signature", tuple[tuple[tuple[int, int], int], ...])


@register_pytree_node_class
class MultiImage:

    # Constructors

    def __init__(
        self: Self,
        data: dict[tuple[int, int], jax.Array],
        D: int,
        is_torus: Union[bool, tuple[bool, ...]] = True,
    ) -> None:
        """
        Construct a MultiImage
        args:
            data: dictionary by k of jnp.array
            D: dimension of the image, and length of vectors or side length of matrices or tensors.
            is_torus: whether the datablock is a torus, used for convolutions.
        """
        self.D = D
        assert (isinstance(is_torus, tuple) and (len(is_torus) == D)) or isinstance(is_torus, bool)
        if isinstance(is_torus, bool):
            is_torus = (is_torus,) * D

        self.is_torus = is_torus
        # copy dict, but image_block is immutable jnp array
        self.data = {key: image_block for key, image_block in data.items()}

    def copy(self: Self) -> Self:
        return self.__class__(self.data, self.D, self.is_torus)

    def empty(self: Self) -> Self:
        return self.__class__({}, self.D, self.is_torus)

    @classmethod
    def from_images(cls, images: Sequence[GeometricImage]) -> Optional[Self]:
        # We assume that all images have the same D and is_torus
        if len(images) == 0:
            return None

        out = cls({}, images[0].D, images[0].is_torus)
        for image in images:
            out.append(image.k, image.parity, image.data.reshape((1,) + image.data.shape))

        return out

    @classmethod
    def from_vector(cls, vector: jax.Array, multi_image: Self) -> Self:
        """
        Convert a vector to a MultiImage, using the shape and parity of the provided MultiImage.
        args:
            vector: a 1-D array of values
            multi_image: a MultiImage providing the parity and shape for the resulting new one
        """
        idx = 0
        out = multi_image.empty()
        for (k, parity), img in multi_image.items():
            out.append(k, parity, vector[idx : (idx + img.size)].reshape(img.shape))
            idx += img.size

        return out

    def __str__(self: Self) -> str:
        multi_image_repr = f"{self.__class__} D: {self.D}, is_torus: {self.is_torus}\n"
        for k, image_block in self.items():
            multi_image_repr += f"\t{k}: {image_block.shape}\n"

        return multi_image_repr

    def size(self: Self) -> int:
        return functools.reduce(lambda size, img: size + img.size, self.values(), 0)

    def get_spatial_dims(self: Self) -> tuple[int, ...]:
        """
        Get the spatial dimensions.
        """
        if len(self.values()) == 0:
            return ()

        (k, _), image_block = next(iter(self.items()))
        prior_indices = image_block.ndim - (k + self.D)  # handles batch or channels
        return image_block.shape[prior_indices : prior_indices + self.D]

    # Functions that map directly to calling the function on data

    def keys(self: Self) -> KeysView[tuple[int, int]]:
        return self.data.keys()

    def values(self: Self) -> ValuesView[jax.Array]:
        return self.data.values()

    def items(self: Self) -> ItemsView[tuple[int, int], jax.Array]:
        return self.data.items()

    def __getitem__(self: Self, idx: tuple[int, int]) -> jax.Array:
        return self.data[idx]

    def __setitem__(self: Self, idx: tuple[int, int], val: jnp.ndarray) -> jnp.ndarray:
        self.data[idx] = val
        return self.data[idx]

    def __contains__(self: Self, idx: tuple[int, int]) -> bool:
        return idx in self.data

    def __eq__(self: Self, other: object, rtol: float = TINY, atol: float = TINY) -> bool:
        if isinstance(other, MultiImage):
            if (
                (self.D != other.D)
                or (self.is_torus != other.is_torus)
                or (self.keys() != other.keys())
            ):
                return False

            for key in self.keys():
                if not jnp.allclose(self[key], other[key], rtol, atol):
                    return False

            return True
        else:
            return False

    # Other functions

    def append(self: Self, k: int, parity: int, image_block: jax.Array, axis: int = 0) -> Self:
        """
        Append an image block at (k,parity). It will be concatenated along axis=0, so channel for
        MultiImage and vmapped BatchMultiImage, and batch for normal BatchMultiImage
        """
        parity = parity % 2
        # will this work for BatchMultiImage?
        if (
            k > 0
        ):  # very light shape checking, other problematic cases should be caught in concatenate
            assert image_block.shape[-k:] == (self.D,) * k

        if (k, parity) in self:
            self[(k, parity)] = jnp.concatenate((self[(k, parity)], image_block), axis=axis)
        else:
            self[(k, parity)] = image_block

        return self

    def __add__(self: Self, other: Self) -> Self:
        """
        Addition operator for MultiImages, must have the same types of MultiImages, adds them together
        """
        assert type(self) == type(
            other
        ), f"{self.__class__}::__add__: Types of MultiImages being added must match, had {type(self)} and {type(other)}"
        assert (
            self.D == other.D
        ), f"{self.__class__}::__add__: Dimension of MultiImages must match, had {self.D} and {other.D}"
        assert (
            self.is_torus == other.is_torus
        ), f"{self.__class__}::__add__: is_torus of MultiImages must match, had {self.is_torus} and {other.is_torus}"
        assert (
            self.keys() == other.keys()
        ), f"{self.__class__}::__add__: Must have same types of images, had {self.keys()} and {other.keys()}"

        return self.__class__.from_vector(self.to_vector() + other.to_vector(), self)

    def __mul__(self: Self, other: Union[Self, float]) -> Self:
        """
        Multiplication operator for a MultiImage and a scalar
        """
        assert not isinstance(
            other, MultiImage
        ), f"MultiImage multiplication is only implemented for numbers, got {type(other)}."

        return self.__class__.from_vector(self.to_vector() * other, self)

    def __truediv__(self: Self, other: float) -> Self:
        """
        True division (a/b) for a MultiImage and a scalar.
        """
        return self * (1.0 / other)

    def concat(self: Self, other: Self, axis: int = 0) -> Self:
        """
        Concatenate the MultiImages along a specified axis.
        args:
            other: a MultiImage with the same dimension and qualities as this one
            axis: the axis along with the concatenate the other MultiImage
        """
        assert type(self) == type(
            other
        ), f"{self.__class__}::concat: Types of MultiImages being added must match, had {type(self)} and {type(other)}"
        assert (
            self.D == other.D
        ), f"{self.__class__}::concat: Dimension of MultiImages must match, had {self.D} and {other.D}"
        assert (
            self.is_torus == other.is_torus
        ), f"{self.__class__}::concat: is_torus of MultiImages must match, had {self.is_torus} and {other.is_torus}"

        out = self.copy()
        for (k, parity), image_block in other.items():
            out.append(k, parity, image_block, axis)

        return out

    def to_images(self: Self) -> list[GeometricImage]:
        # Should only be used in MultiImage of vmapped BatchMultiImage
        images = []
        for image_block in self.values():
            for image in image_block:
                images.append(
                    GeometricImage(image, 0, self.D, self.is_torus)
                )  # for now, assume 0 parity

        return images

    def to_vector(self: Self) -> jnp.ndarray:
        """
        Vectorize a MultiImage in the natural way
        """
        return functools.reduce(
            lambda x, y: jnp.concatenate([x, y.reshape(-1)]),
            self.values(),
            jnp.zeros(0),
        )

    def to_scalar_multi_image(self: Self) -> Self:
        """
        Convert MultiImage to a MultiImage where all the channels and components are in the scalar
        """
        # convert to channels of a scalar MultiImage
        out = self.empty()
        for (k, _), image in self.items():
            transpose_idxs = (
                (0,) + tuple(range(1 + self.D, 1 + self.D + k)) + tuple(range(1, 1 + self.D))
            )
            out.append(
                0, 0, image.transpose(transpose_idxs).reshape((-1,) + image.shape[1 : 1 + self.D])
            )

        return out

    def from_scalar_multi_image(self: Self, layout: Signature) -> Self:
        """
        Convert a scalar MultiImage back to a MultiImage with the specified layout
        args:
            layout: signature of keys (k,parity) and values num_channels for the output MultiImage
        """
        assert list(self.keys()) == [(0, 0)]
        spatial_dims = self[(0, 0)].shape[1:]

        out = self.empty()
        idx = 0
        for (k, parity), num_channels in layout:
            length = num_channels * (self.D**k)
            # reshape, it is (num_channels*(D**k), spatial_dims) -> (num_channels, (D,)*k, spatial_dims)
            reshaped_data = self[(0, 0)][idx : idx + length].reshape(
                (num_channels,) + (self.D,) * k + spatial_dims
            )
            # tranpose (num_channels, (D,)*k, spatial_dims) -> (num_channels, spatial_dims, (D,)*k)
            transposed_data = reshaped_data.transpose(
                (0,) + tuple(range(1 + k, 1 + k + self.D)) + tuple(range(1, 1 + k))
            )
            out.append(k, parity, transposed_data)
            idx += length

        return out

    def times_group_element(
        self: Self, gg: np.ndarray, precision: Optional[jax.lax.Precision] = None
    ) -> Self:
        """
        Apply a group element of O(2) or O(3) to the MultiImage. First apply the action to the
        location of the pixels, then apply the action to the pixels themselves.
        args:
            gg (group operation matrix): a DxD matrix that rotates the tensor
            precision (jax.lax.Precision): precision level for einsum, for equality tests use Precision.HIGH
        """
        vmap_rotate = jax.vmap(times_group_element, in_axes=(None, 0, None, None, None))
        out = self.empty()
        for (k, parity), image_block in self.items():
            out.append(k, parity, vmap_rotate(self.D, image_block, parity, gg, precision))

        return out

    def norm(self: Self) -> Self:
        out = self.empty()
        for image_block in self.values():
            out.append(0, 0, norm(self.D + 1, image_block))  # norm is even parity

        return out

    def get_component(self: Self, component: Union[int, slice], future_steps: int = 1) -> Self:
        """
        Given a MultiImage with data with shape (channels*future_steps,spatial,tensor), combine all
        fields into a single block of data (future_steps,spatial,channels*tensor) then pick the
        ith channel in the last axis, where i = component. For example, if the MultiImage has
        density (scalar), pressure (scalar), and velocity (vector) then i=0 -> density, i=1 ->
        pressure, i=2 -> velocity 1, and i=3 -> velocity 2. This assumes D=2.

        args:
            component: which component to select
            future_steps: the number of future timesteps of this MultiImage, defaults to 1
        """
        spatial_dims = self.get_spatial_dims()

        data = None
        for (k, _), img in self.items():
            exp_data = img.reshape(
                (-1, future_steps) + spatial_dims + (self.D,) * k
            )  # (c,time,spatial,tensor)
            exp_data = jnp.moveaxis(exp_data, 0, 1 + self.D)  # (time,spatial,c,tensor)
            exp_data = exp_data.reshape(
                (future_steps,) + spatial_dims + (-1,)
            )  # (time,spatial,c*tensor)

            data = exp_data if data is None else jnp.concatenate([data, exp_data], axis=-1)

        assert data is not None, "MultiImage::get_component: Multi Image has no images of any order"
        component_data = data[..., component].reshape((future_steps,) + spatial_dims + (-1,))
        component_data = jnp.moveaxis(component_data, -1, 0).reshape((-1,) + spatial_dims)
        return self.__class__({(0, 0): component_data}, self.D, self.is_torus)

    def get_signature(self: Self) -> Signature:
        """
        Get a tuple of ( ((k,p),channels), ((k,p),channels), ...). This works for MultiImages and
        BatchMultiImages.
        """
        k, p = next(iter(self.data.keys()))
        leading_axes = self[k, p].ndim - self.D - k
        return Signature(
            tuple((k_p, img.shape[leading_axes - 1]) for k_p, img in self.data.items())
        )

    def device_replicate(self: Self, sharding: jax.sharding.PositionalSharding) -> Self:
        """
        Put the MultiImage on particular devices according to the sharding and num_devices
        args:
            sharding (jax sharding): jax positional sharding to be reshaped
            num_devices (int): number of gpus to split the batches over
        """
        return self.__class__(
            jax.device_put(self.data, sharding.replicate()), self.D, self.is_torus
        )

    # JAX helpers
    def tree_flatten(self):
        """
        Helper function to define GeometricImage as a pytree so jax.jit handles it correctly. Children
        and aux_data must contain all the variables that are passed in __init__()
        """
        children = (self.data,)  # arrays / dynamic values
        aux_data = {
            "D": self.D,
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
class BatchMultiImage(MultiImage):
    # I may want to only have MultiImage, and find out a better way of tracking this

    # Constructors

    def __init__(
        self: Self,
        data: dict[tuple[int, int], jax.Array],
        D: int,
        is_torus: Union[bool, tuple[bool, ...]] = True,
    ) -> None:
        """
        Construct a BatchMultiImage
        args:
            data (dictionary of jnp.array): dictionary by k of jnp.array
            parity (int): 0 or 1, 0 is normal vectors, 1 is pseudovectors
            D (int): dimension of the image, and length of vectors or side length of matrices or tensors.
            is_torus (bool): whether the datablock is a torus, used for convolutions. Defaults to true.
        """
        super(BatchMultiImage, self).__init__(data, D, is_torus)

        self.L = None
        for image_block in data.values():  # if empty, this won't get set
            if isinstance(image_block, jnp.ndarray):
                self.L = len(image_block)  # shape (batch, channels, (N,)*D, (D,)*k)
            break

    @classmethod
    def from_images(cls, images: Sequence[GeometricImage]) -> Optional[Self]:
        # We assume that all images have the same D and is_torus
        if len(images) == 0:
            return None

        out = cls({}, images[0].D, images[0].is_torus)
        for image in images:
            out.append(image.k, image.parity, image.data.reshape((1, 1) + image.data.shape))

        batch_image_block = list(out.values())[0]
        out.L = batch_image_block.shape[0]

        return out

    def get_L(self: Self) -> int:
        """
        Get the batch size. This will return the wrong value if the batch is vmapped.
        """
        if len(self.values()) == 0:
            return 0

        return len(next(iter(self.values())))

    def get_subset(self: Self, idxs: jnp.ndarray) -> Self:
        """
        Select a subset of the batch, picking the indices idxs
        args:
            idxs (jnp.array): array of indices to select the subset
        """
        assert isinstance(
            idxs, jnp.ndarray
        ), "BatchMultiImage::get_subset arg idxs must be a jax array"
        assert len(
            idxs.shape
        ), "BatchMultiImage::get_subset arg idxs must be a jax array, e.g. jnp.array([0])"
        return self.__class__(
            {k: image_block[idxs] for k, image_block in self.items()},
            self.D,
            self.is_torus,
        )

    def get_one(self: Self, idx: int = 0) -> Self:
        return self.get_subset(jnp.array([idx]))

    def get_one_batch_multi_image(self: Self, idx: int = 0) -> MultiImage:
        """
        Similar to get_one, but instead get a single index of a BatchMultiImage as a MultiImage
        args:
            idx (int): the index along the batch to get as a MultiImage.
        """
        return MultiImage(
            {k: image_block[idx] for k, image_block in self.items()},
            self.D,
            self.is_torus,
        )

    def times_group_element(
        self: Self, gg: np.ndarray, precision: Optional[jax.lax.Precision] = None
    ) -> Self:
        return self._times_group_element(gg, precision)

    @functools.partial(jax.vmap, in_axes=(0, None, None))
    def _times_group_element(
        self: Self, gg: np.ndarray, precision: Optional[jax.lax.Precision]
    ) -> Self:
        return super(BatchMultiImage, self).times_group_element(gg, precision)

    @jax.vmap
    def to_vector(self: Self) -> jnp.ndarray:
        return super(BatchMultiImage, self).to_vector()

    @jax.vmap
    def to_scalar_multi_image(self: Self) -> Self:
        return super(BatchMultiImage, self).to_scalar_multi_image()

    @functools.partial(jax.vmap, in_axes=(0, None))
    def from_scalar_multi_image(self: Self, layout: Signature) -> Self:
        return super(BatchMultiImage, self).from_scalar_multi_image(layout)

    @classmethod
    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def from_vector(cls, vector: jax.Array, multi_image: Self) -> Self:
        return super().from_vector(vector, multi_image)

    @jax.vmap
    def norm(self: Self) -> Self:
        return super(BatchMultiImage, self).norm()

    def get_component(self: Self, component: Union[int, slice], future_steps: int = 1) -> Self:
        return self._get_component(component, future_steps)

    @functools.partial(jax.vmap, in_axes=(0, None, None))
    def _get_component(self: Self, component: Union[int, slice], future_steps: int) -> Self:
        return super(BatchMultiImage, self).get_component(component, future_steps)

    def device_put(self: Self, sharding: jax.sharding.PositionalSharding) -> Self:
        """
        Put the BatchMultiImage on particular devices according to the sharding
        args:
            sharding (jax sharding): jax positional sharding to be reshaped
        """
        num_devices = sharding.shape[0]
        # number of batches must device evenly into number of devices
        assert (self.get_L() % num_devices) == 0

        new_data = {}
        for key, image_block in self.items():
            sharding_shape = (num_devices,) + (1,) * len(image_block.shape[1:])
            new_data[key] = jax.device_put(image_block, sharding.reshape(sharding_shape))

        return self.__class__(new_data, self.D, self.is_torus)

    def reshape_pmap(self: Self, devices) -> Self:
        """
        Reshape the batch to allow pmap to work. E.g., if shape is (batch,1,N,N) and num_devices=2, then
        reshape to (2,batch/2,1,N,N)
        args:
            devices (list): list of gpus or cpu that we are using
        """
        assert self.get_L() % len(devices) == 0, (
            f"BatchMultiImage::reshape_pmap: length of devices must evenly "
            f"divide the total batch size, but got batch_size: {self.get_L()}, devices: {devices}"
        )

        num_devices = len(devices)

        out = self.empty()
        for (k, parity), image in self.items():
            out.append(
                k,
                parity,
                image.reshape((num_devices, self.get_L() // num_devices) + image.shape[1:]),
            )

        return out

    def merge_pmap(self: Self) -> Self:
        """
        Take the output BatchMultiImage of a pmap and recombine the batch.
        """
        out = self.empty()
        for (k, parity), image_block in self.items():
            out.append(k, parity, image_block.reshape((-1,) + image_block.shape[2:]))

        return out
