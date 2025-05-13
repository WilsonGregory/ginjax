import functools
import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
from typing_extensions import Any, NewType, Optional, Self, Sequence, Union
from collections.abc import ItemsView, KeysView, ValuesView

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import equinox as eqx

from ginjax.geometric.constants import TINY
from ginjax.geometric.functional_geometric_image import (
    average_pool,
    norm,
    raise_lower,
    rotate_is_torus,
    times_group_element,
)
from ginjax.geometric.geometric_image import GeometricImage, get_metric_inverse

Signature = NewType("Signature", tuple[tuple[tuple[tuple[bool, ...], int], int], ...])


def signature_union(signature_a: Signature, signature_b: Signature, num_channels: int) -> Signature:
    key_union = {k_p for k_p, _ in signature_a}.union({k_p for k_p, _ in signature_b})
    return Signature(tuple((k_p, num_channels) for k_p in key_union))


@register_pytree_node_class
class MultiImage:
    """
    The MultiImage holds a collection of geometric images of any tensor orders and parities, with
    possibly multiple channels. This is the primary class used for machine learning because each
    layer maps multiple different tensor orders/parities to multiple different tensor orders/parities.

    The data of a MultiImage is held in a dictionary whose keys are (k,parity) tuples and whose
    values are jax arrays of shape (channels,spatial,tensor). The k is itself a tuple of boolean
    values specifying which tensor axes are covariant (True) or contravariant (False). When the
    metric tensor is the flat Euclidean metric, the axes are equivalent numerically and we just
    assume that every axis is contravariant. However, when there is a non-flat metric tensor, the
    covariant and contravariant axes transform differently under coordinate changes. If those
    coordinate changes are a subgroup of O(d), then they are actually the same again. Additionally,
    contracting axes can only be done when one is covariant and the other is contravariant. This
    is not enforced explicitly at the moment, it is just implemented for ConvContract.

    There could be a variable number of axes of channels, from 0 to whatever. The most common
    options are 1 (channels) or 2 (batch,channels). The number of leading axes must be the same for
    all (k,parity). This setup allows you to easily vmap a MultiImage, which vmaps over those axes
    for each image type.
    """

    D: int
    is_torus: tuple[bool, ...]
    data: dict[tuple[tuple[bool, ...], int], jax.Array]
    metric_tensor: Optional[GeometricImage]
    metric_tensor_inv: Optional[GeometricImage]

    # Constructors

    def __init__(
        self: Self,
        data: Union[
            dict[tuple[tuple[bool, ...], int], jax.Array], dict[tuple[int, int], jax.Array]
        ],
        D: int,
        is_torus: Union[bool, tuple[bool, ...]] = True,
        metric_tensor: Optional[GeometricImage] = None,
        metric_tensor_inv: Optional[GeometricImage] = None,
    ) -> None:
        """
        Construct a MultiImage

        args:
            data: dictionary by (k,parity) of jnp.array
            D: dimension of the image, and length of vectors or side length of matrices or tensors.
            is_torus: whether the datablock is a torus, used for convolutions.
            metric_tensor: metric tensor as an image, should be same spatial dimensions as the data
                If none, assume the metric is the flat Euclidean metric.
        """
        self.D = D
        assert (isinstance(is_torus, tuple) and (len(is_torus) == D)) or isinstance(is_torus, bool)
        if isinstance(is_torus, bool):
            is_torus = (is_torus,) * D

        self.is_torus = is_torus
        if metric_tensor is not None:
            assert metric_tensor.k == 2
            assert metric_tensor.parity == 0
            assert metric_tensor.covariant_axes == (True, True)

        self.metric_tensor = metric_tensor
        self.metric_tensor_inv = metric_tensor_inv
        # copy dict, but image_block is immutable jnp array
        self.data = {}
        for (k, parity), image_block in data.items():
            if isinstance(k, int):
                k = (False,) * k

            self.data[k, parity] = image_block

    def copy(self: Self) -> Self:
        """
        Copy constructor for MultiImage.

        returns:
            a copy of this multi image
        """
        return self.__class__(
            self.data, self.D, self.is_torus, self.metric_tensor, self.metric_tensor_inv
        )

    def empty(self: Self, same_metric: bool = True) -> Self:
        """
        A copy of this MultiImage without the data. In some cases we might want to use the same
        metric tensor in which case use same_metric = True, but in other situations you might want
        a different one (for example, if the spatial dimensions change).

        args:
            same_metric: The new image will have the same metric tensor
            metric_tensor: To use the same metric tensor, call image.empty(image.metric_tensor)

        returns:
            the new empty MultiImage
        """
        return self.__class__(
            {},
            self.D,
            self.is_torus,
            self.metric_tensor if same_metric else None,
            self.metric_tensor_inv if same_metric else None,
        )

    @classmethod
    def from_images(
        cls,
        images: Sequence[GeometricImage],
        n_lead_axes: int = 1,
        axis=0,
        metric_tensor: Optional[GeometricImage] = None,
    ) -> Self:
        """
        Construct a MultiImage from a sequence of GeometricImages.

        args:
            images: the GeometricImages
            n_lead_axes: number of leading axes to append
            axis: what axis to append to
            metric_tensor: the common metric tensor for all the images

        returns:
            a new MultiImage
        """
        # We assume that all images have the same D and is_torus
        assert len(images) != 0, "MultiImage.from_images was passed an empty list of images."
        out = cls({}, images[0].D, images[0].is_torus, metric_tensor)
        for image in images:
            out.append(
                image.k,
                image.parity,
                image.data.reshape((1,) * n_lead_axes + image.data.shape),
                axis=axis,
            )

        return out

    @classmethod
    def from_vector(cls, vector: jax.Array, multi_image: Self) -> Self:
        """
        Convert a vector to a MultiImage, using the shape and parity of the provided MultiImage.

        args:
            vector: a 1-D array of values
            multi_image: a MultiImage providing the parity and shape for the resulting new one

        returns:
            a new MultiImage
        """
        idx = 0
        out = multi_image.empty()
        for (k, parity), img in multi_image.items():
            out.append(k, parity, vector[idx : (idx + img.size)].reshape(img.shape))
            idx += img.size

        return out

    def __str__(self: Self) -> str:
        """
        returns:
            the string representation of the MultiImage
        """
        multi_image_repr = f"{self.__class__} D: {self.D}, is_torus: {self.is_torus}\n"
        for k, image_block in self.items():
            multi_image_repr += f"\t{k}: {image_block.shape}\n"

        if self.metric_tensor is not None:
            multi_image_repr += f"\tmetric tensor: {self.metric_tensor.shape()}\n"

        return multi_image_repr

    def size(self: Self) -> int:
        """
        Get the total image size from all images

        returns:
            the total image size
        """
        return functools.reduce(lambda size, img: size + img.size, self.values(), 0)

    def get_spatial_dims(self: Self) -> tuple[int, ...]:
        """
        Get the spatial dimensions.

        returns:
            the spatial dimensions
        """
        if len(self.values()) == 0:
            return ()

        (k, _), image_block = next(iter(self.items()))
        prior_indices = image_block.ndim - (len(k) + self.D)  # handles batch or channels
        return image_block.shape[prior_indices : prior_indices + self.D]

    # Functions that map directly to calling the function on data

    def keys(self: Self) -> KeysView[tuple[tuple[bool, ...], int]]:
        """
        returns:
            the (k,parity) keys of the MultiImage
        """
        return self.data.keys()

    def values(self: Self) -> ValuesView[jax.Array]:
        """
        returns:
            the image values of the MultiImage (channels,spatial,tensor)
        """
        return self.data.values()

    def items(self: Self) -> ItemsView[tuple[tuple[bool, ...], int], jax.Array]:
        """
        returns:
            the key (k,parity) value (image data array) of the MultiImage
        """
        return self.data.items()

    def __getitem__(self: Self, idx: tuple[tuple[bool, ...], int]) -> jax.Array:
        """
        Get an image block of a particular tensor order and parity

        args:
            idx: the tensor order and parity

        returns:
            an image block (channels,spatial,tensor)
        """
        return self.data[idx]

    def __setitem__(self: Self, idx: tuple[tuple[bool, ...], int], val: jax.Array) -> jax.Array:
        """
        Set an image block for a specific tensor order and parity

        args:
            idx: the tensor order and parity
            val: the image block, shape (channel, spatial, tensor)

        returns:
            the image block that was set, shape (channel, spatial, tensor)
        """
        self.data[idx] = val
        return self.data[idx]

    def __contains__(self: Self, idx: tuple[tuple[bool, ...], int]) -> bool:
        """
        Check whether a particular tensor order and parity image block is in the MultiImage

        args:
            idx: the tensor order and parity

        returns:
            whether that image block is in the MultiImage
        """
        return idx in self.data

    def __eq__(self: Self, other: object, rtol: float = TINY, atol: float = TINY) -> bool:
        """
        Check whether another MultiImage is equal to this one

        args:
            other: other MultiImage to compare to this one
            rtol: relative tolerance, passed to jnp.allclose
            atol: absolute tolerance, passed to jnp.allclose

        returns:
            whether the MultiImages are equal
        """
        if isinstance(other, MultiImage):
            if (
                (self.D != other.D)
                or (self.is_torus != other.is_torus)
                or (self.keys() != other.keys())
                or (self.metric_tensor != other.metric_tensor)
            ):
                return False

            for key in self.keys():
                if not jnp.allclose(self[key], other[key], rtol, atol):
                    return False

            return True
        else:
            return False

    # Other functions

    def append(
        self: Self,
        k: Union[tuple[bool, ...], int],
        parity: int,
        image_block: jax.Array,
        axis: int = 0,
    ) -> Self:
        """
        Append an image block at (k,parity). It will be concatenated along the specified axis which
        must be one of the leading axes.

        args:
            k: the tensor order
            parity: the parity, either 0 or 1 for regular tensors or pseudotensors
            image_block: the image data, shape (channel,spatial,tensor)
            axis: what axis to append along

        returns:
            this MultiImage, now updated
        """
        n_leading_axes = self.get_n_leading()
        assert (self.data == {}) or (
            axis < n_leading_axes
        ), f"axis={axis} must be one of {n_leading_axes} n_leading_axes"
        parity = parity % 2
        if isinstance(k, int):
            k = (False,) * k

        if len(k) > 0:  # light shape checking
            assert image_block.shape[-len(k) :] == (self.D,) * len(k)
        if self.D == 1:
            assert k == (), "MultiImage::append: 1D images can only have scalars and pseudoscalars"

        if (k, parity) in self:
            self[(k, parity)] = jnp.concatenate((self[(k, parity)], image_block), axis=axis)
        else:
            self[(k, parity)] = image_block

        return self

    def __add__(self: Self, other: Self) -> Self:
        """
        Addition operator for MultiImages, must have the same types of MultiImages, adds them together

        args:
            other: other MultiImage to add to this one

        returns:
            a new MultiImage that is the sum of this and other
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
        assert (
            self.metric_tensor == other.metric_tensor
        ), f"{self.__class__}::__add__: Metric tensors do not match"

        return self.__class__.from_vector(self.to_vector() + other.to_vector(), self)

    def __sub__(self: Self, other: Self) -> Self:
        """
        Subtraction operator for MultiImages, must have the same types of MultiImages, adds them together

        args:
            other: other MultiImage to subtract from this one

        returns:
            a new MultiImage that is the difference of this and other
        """
        assert type(self) == type(
            other
        ), f"{self.__class__}::__sub__: Types of MultiImages being added must match, had {type(self)} and {type(other)}"
        assert (
            self.D == other.D
        ), f"{self.__class__}::__sub__: Dimension of MultiImages must match, had {self.D} and {other.D}"
        assert (
            self.is_torus == other.is_torus
        ), f"{self.__class__}::__sub__: is_torus of MultiImages must match, had {self.is_torus} and {other.is_torus}"
        assert (
            self.keys() == other.keys()
        ), f"{self.__class__}::__sub__: Must have same types of images, had {self.keys()} and {other.keys()}"
        assert (
            self.metric_tensor == other.metric_tensor
        ), f"{self.__class__}::__sub__: Metric tensors do not match"

        return self.__class__.from_vector(self.to_vector() - other.to_vector(), self)

    def __mul__(self: Self, other: Union[Self, float]) -> Self:
        """
        Multiplication operator for a MultiImage and a scalar

        args:
            other: other MultiImage or float to multiply this MultiImage by

        returns:
            a new MultiImage that is the product of this and other
        """
        assert not isinstance(
            other, MultiImage
        ), f"MultiImage multiplication is only implemented for numbers, got {type(other)}."

        return self.__class__.from_vector(self.to_vector() * other, self)

    def __truediv__(self: Self, other: float) -> Self:
        """
        True division (a/b) for a MultiImage and a scalar.

        args:
            other: number to divide this MultiImage by

        returns:
            a new MultiImage divided by other
        """
        return self * (1.0 / other)

    def concat(self: Self, other: Self, axis: int = 0) -> Self:
        """
        Concatenate the MultiImages along a specified axis.

        args:
            other: a MultiImage with the same dimension and qualities as this one
            axis: the axis along with the concatenate the other MultiImage

        returns:
            a new MultiImage that has been concatenated
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
        assert (
            self.metric_tensor == other.metric_tensor
        ), f"{self.__class__}::concat: Metric tensors do not match"

        out = self.copy()
        for (k, parity), image_block in other.items():
            out.append(k, parity, image_block, axis)

        return out

    def concat_inverse(
        self: Self,
        signature: Union[Signature, dict[tuple[tuple[bool, ...], int], int]],
        axis: int = 0,
    ) -> tuple[Self, Self]:
        """
        Given a signature, split the multi image into a and b, where b has the signature provided
        and a has the remaining images. Thus a.concat(b, axis).inverse_concat(b.get_signature(), axis)
        will return a and b. Note that get_signature() will use the last leading axis, for this
        function you can also manually construct the signature-like object if you want to use a
        different axis.

        args:
            signature: signature of the second multi image to split off, either signature or dict
                version of the signature
            axis: the axis that we are splitting on

        returns:
            the two multi images from the reverse concat
        """
        if isinstance(signature, tuple):
            signature_dict = {(k, parity): size for (k, parity), size in signature}
        else:
            signature_dict = signature

        a = self.empty()
        b = self.empty()
        for (k, parity), image_block in self.items():
            size = signature_dict[(k, parity)] if (k, parity) in signature_dict else 0
            axis_size = image_block.shape[axis]
            assert 0 <= size <= axis_size

            if size == 0:
                a.append(k, parity, image_block)
            elif size == axis_size:
                b.append(k, parity, image_block)
            else:
                a.append(k, parity, image_block[(slice(None),) * axis + (slice(0, -size),)])
                b.append(k, parity, image_block[(slice(None),) * axis + (slice(-size, axis_size),)])

        return a, b

    def to_images(self: Self) -> list[GeometricImage]:
        """
        Convert this MultiImage to a list of GeometricImages.

        returns:
            the list of new GeometricImages
        """
        images = []
        for (k, parity), image_block in self.items():
            for image in image_block.reshape((-1,) + self.get_spatial_dims() + (self.D,) * len(k)):
                images.append(GeometricImage(image, parity, self.D, self.is_torus))

        return images

    def to_vector(self: Self) -> jax.Array:
        """
        Vectorize a MultiImage in the natural way

        returns:
            the vectorized MultiImage
        """
        return functools.reduce(
            lambda x, y: jnp.concatenate([x, y.reshape(-1)]),
            self.values(),
            jnp.zeros(0),
        )

    def to_scalar_multi_image(self: Self) -> Self:
        """
        Convert MultiImage to a MultiImage where all the channels and components are in the scalar.
        Each component of each geometric image becomes one channel of the scalar image.

        returns:
            a new scalar MultiImage with # of channels equal to D^k1 + D^k2 + ... for all the ki
        """
        out = self.empty()
        n_batch_axes = self.get_n_leading() - 1
        assert (
            n_batch_axes >= 0
        ), "MultiImage::to_scalar_multi_image: assume that there is at least a channels axis"
        for (k, _), image in self.items():
            # (...,c,spatial,tensor) -> (...,spatial,c,tensor)
            image = jnp.moveaxis(image, n_batch_axes, -(1 + len(k)))
            # (...,spatial,c*tensor)
            image = image.reshape(image.shape[: n_batch_axes + self.D] + (-1,))
            # (...,c*tensor,spatial)
            out.append(0, 0, jnp.moveaxis(image, -1, n_batch_axes), axis=n_batch_axes)

        return out

    def from_scalar_multi_image(self: Self, layout: Signature) -> Self:
        """
        Convert a scalar MultiImage back to a MultiImage with the specified layout

        args:
            layout: signature of keys (k,parity) and values num_channels for the output MultiImage

        returns:
            a new MultiImage with the same signature as layout
        """
        assert list(self.keys()) == [((), 0)]
        spatial_dims = self.get_spatial_dims()
        n_batch_axes = self.get_n_leading() - 1

        out = self.empty()
        idx = 0
        # (...,c*tensor,spatial) -> (...,spatial,c*tensor)
        image = jnp.moveaxis(self[((), 0)], n_batch_axes, -1)
        for (k, parity), num_channels in layout:
            length = num_channels * (self.D ** len(k))
            # (...,spatial,num_channels*(D**k)) -> (...,spatial,num_channels,tensor)
            reshaped_data = image[..., idx : idx + length].reshape(
                image.shape[:n_batch_axes] + spatial_dims + (num_channels,) + (self.D,) * len(k)
            )
            # (...,num_channels,spatial,tensor). Can append on any axis cause its always the first
            out.append(k, parity, jnp.moveaxis(reshaped_data, -(1 + len(k)), n_batch_axes))
            idx += length

        return out

    def raise_lower(
        self: Self,
        new_signature: Signature,
        channel_axis: Optional[int] = None,
        precision: Optional[jax.lax.Precision] = None,
    ) -> Self:
        """
        Raise or lower the axes of each image according to the new_signature. This has the potential
        to cause issues if there is a many to many conversion of key types for the same number of
        axes and parity. For example, doing
        [(True,False) (True, True)] -> [(True, True), (False,False)] will, if given in this order
        convert (True,False) -> (True,True) and (True,True) -> (False,False), which may not be what
        you expect. This issue can also arise if they are all converted to something, operated on,
        then later converted back to two separate types.

        Therefore, care must be taken.

        args:
            new_signature: new signature of the resulting multi_image. This must have the same
                total number of channels per (len_k,parity)
            channels_axis: what axis is the channel axis for concatenation. If none, defaults to
                the last axis before the spatial axes.
            precision: the einsum precision

        returns:
            a new MultiImage with the specified signature
        """
        assert self.metric_tensor is not None, "MultiImage::raise_lower: metric tensor is None"
        if channel_axis is None:
            channel_axis = self.get_n_leading() - 1

        assert channel_axis >= 0, f"MultiImage::raise_lower: need at least one channel axis."

        curr_sig_by_k = {}
        for (k, parity), n_channels in self.get_signature():
            if (len(k), parity) in curr_sig_by_k:
                curr_sig_by_k[(len(k), parity)].append(((k, parity), n_channels))
            else:
                curr_sig_by_k[(len(k), parity)] = [((k, parity), n_channels)]

        new_sig_by_k = {}
        for (k, parity), n_channels in new_signature:
            if (len(k), parity) in new_sig_by_k:
                new_sig_by_k[(len(k), parity)].append(((k, parity), n_channels))
            else:
                new_sig_by_k[(len(k), parity)] = [((k, parity), n_channels)]

        assert (
            curr_sig_by_k.keys() == new_sig_by_k.keys()
        ), f"MultiImage::raise_lower: Signatures must have same k, {self.get_signature()} != {new_signature}"

        if self.metric_tensor_inv is None:
            self.metric_tensor_inv = get_metric_inverse(self.metric_tensor)

        out = self.empty()
        for (len_k, parity), curr_key_list in curr_sig_by_k.items():
            # new to convert, and line up the channels
            new_key_list = new_sig_by_k[(len_k, parity)]
            assert sum([n_c for _, n_c in curr_key_list]) == sum(
                [n_c for _, n_c in new_key_list]
            ), f"MultiImage::raise_lower: total number of channels must equal, got {curr_key_list} and {new_key_list}"

            i, j = 0, 0
            start = 0
            while i < len(curr_key_list) and j < len(new_key_list):
                (curr_k, curr_p), curr_channels = curr_key_list[i]
                (new_k, new_p), new_channels = new_key_list[j]

                end = min(curr_channels - start, new_channels) + start
                image_block = self[(curr_k, curr_p)][
                    (slice(None),) * (self.get_n_leading() - 1) + (slice(start, end),)
                ]

                new_image_block = raise_lower(
                    image_block,
                    self.metric_tensor.data,
                    self.metric_tensor_inv.data,
                    curr_k,
                    new_k,
                    precision,
                )
                out.append(new_k, new_p, new_image_block, axis=channel_axis)

                if out[(new_k, new_p)].shape[channel_axis] == new_channels:
                    j += 1  # done filling j

                if end == curr_channels:
                    i += 1  # done using i, onto next one
                    start = 0
                else:
                    start = end

            assert i == len(curr_key_list) and j == len(new_key_list)

        return out

    def raise_all(self: Self, precision: Optional[jax.lax.Precision] = None) -> Self:
        """
        Raise all the tensor axes to contravariant.

        args:
            precision: einsum precision

        returns:
            a MultiImage where every tensor axis is contravariant
        """
        channels_by_k = {}
        for (k, parity), n_channels in self.get_signature():
            if (len(k), parity) in channels_by_k:
                channels_by_k[(len(k), parity)] = channels_by_k[(len(k), parity)] + n_channels
            else:
                channels_by_k[(len(k), parity)] = n_channels

        contra_sig = ()
        for (len_k, parity), n_channels in channels_by_k.items():
            contra_sig += ((((False,) * len_k, parity), n_channels),)

        return self.raise_lower(Signature(contra_sig), precision=precision)

    def lower_all(self: Self, precision: Optional[jax.lax.Precision] = None) -> Self:
        """
        Lower all the tensor axes to covariant.

        args:
            precision: einsum precision

        returns:
            a MultiImage where every tensor axis is covariant
        """
        channels_by_k = {}
        for (k, parity), n_channels in self.get_signature():
            if (len(k), parity) in channels_by_k:
                channels_by_k[(len(k), parity)] = channels_by_k[(len(k), parity)] + n_channels
            else:
                channels_by_k[(len(k), parity)] = n_channels

        covariant_sig = ()
        for (len_k, parity), n_channels in channels_by_k.items():
            covariant_sig += ((((True,) * len_k, parity), n_channels),)

        return self.raise_lower(Signature(covariant_sig), precision=precision)

    def times_group_element(
        self: Self, gg: np.ndarray, precision: Optional[jax.lax.Precision] = None
    ) -> Self:
        """
        Apply a group element of O(2) or O(3) to the MultiImage. First apply the action to the
        location of the pixels, then apply the action to the pixels themselves.

        args:
            gg: a DxD matrix that rotates the tensor
            precision: precision level for einsum, for equality tests use Precision.HIGH

        returns:
            a new MultiImage that has been rotated
        """
        if self.metric_tensor is None:
            out = self.empty()
        else:
            # should this be upper?
            out = self.empty(same_metric=False)
            out.metric_tensor = self.metric_tensor.times_group_element(gg, precision)

        if self.metric_tensor_inv is not None:
            out.metric_tensor_inv = self.metric_tensor_inv.times_group_element(gg, precision)

        for (k, parity), image_block in self.items():
            rotated_img_block = times_group_element(
                self.D,
                image_block,
                parity,
                gg,
                k,
                precision,
            )
            out.append(k, parity, rotated_img_block)

        out.is_torus = rotate_is_torus(out.is_torus, gg)

        return out

    def times_gg_precise(self: Self, gg: np.ndarray) -> Self:
        return self.times_group_element(gg, jax.lax.Precision.HIGHEST)

    def norm(self: Self) -> Self:
        """
        Apply norm to all types of geometric images in this multi image, and make them a channel.
        The channels will be concatenated along the axis immediately prior to the spatial axes.
        If there are no channel axes, this function will break an assert.

        returns:
            a new MultiImage with one channel per input channel per type of input image
        """
        n_lead_axes = self.get_n_leading()
        assert n_lead_axes > 0, "MultiImage::norm: must have at least one channel axis."

        out = self.empty()
        for (k, _), image_block in self.items():
            # norm is even parity
            out.append(0, 0, norm(n_lead_axes + self.D, image_block), axis=n_lead_axes - 1)

        return out

    def average_pool(self: Self, patch_len: int) -> Self:
        out = self.empty(same_metric=False)
        # TODO: what should the metric tensor be here?
        vmap_avg_pool = jax.vmap(average_pool, in_axes=(None, 0, None))
        n_leading_axes = self.get_n_leading()
        for (k, parity), image_block in self.items():
            img_pooled = vmap_avg_pool(
                self.D, image_block.reshape((-1,) + image_block.shape[n_leading_axes:]), patch_len
            )
            out.append(
                k,
                parity,
                img_pooled.reshape((image_block.shape[:n_leading_axes] + img_pooled.shape[1:])),
            )

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
            future_steps: the number of future timesteps of this MultiImage

        returns:
            a new MultiImage with a single scalar geometric image corresponding to the chosen
                component.
        """
        assert (
            self.get_n_leading() == 1
        ), f"MultiImage::get_component: must have exactly 1 leading axis"
        spatial_dims = self.get_spatial_dims()

        data = None
        for (k, _), img in self.items():
            # (c,time,spatial,tensor)
            exp_data = img.reshape((-1, future_steps) + spatial_dims + (self.D,) * len(k))
            exp_data = jnp.moveaxis(exp_data, 0, 1 + self.D)  # (time,spatial,c,tensor)
            exp_data = exp_data.reshape(
                (future_steps,) + spatial_dims + (-1,)
            )  # (time,spatial,c*tensor)

            data = exp_data if data is None else jnp.concatenate([data, exp_data], axis=-1)

        assert data is not None, "MultiImage::get_component: Multi Image has no images of any order"
        component_data = data[..., component].reshape((future_steps,) + spatial_dims + (-1,))
        component_data = jnp.moveaxis(component_data, -1, 0).reshape((-1,) + spatial_dims)
        return self.__class__(
            {(0, 0): component_data},
            self.D,
            self.is_torus,
            self.metric_tensor,
            self.metric_tensor_inv,
        )

    @eqx.filter_vmap
    def batch_get_component(
        self: Self, component: Union[int, slice], future_steps: int = 1
    ) -> Self:
        """
        Batched version of get_component, when the first axis of the image blocks is a batch axis.
        This style of function can be written for any function, but ideally we just write the
        original function to handle leading axes correctly. In this case, it was a pain.

         args:
            component: which component to select
            future_steps: the number of future timesteps of this MultiImage

        returns:
            a new MultiImage with a single scalar geometric image corresponding to the chosen
                component.
        """
        return self.get_component(component, future_steps)

    def expand(self: Self, axis: int, size: int) -> Self:
        """
        A common task is that we have a MultiImage with shape (batch,c*timesteps,spatial,tensor)
        and we want to split the channels*timestep channel into (c,timesteps). To achieve this, we
        can call expand(axis=1,size=timesteps) which will reshape all the images to
        (batch,c,timesteps,spatial,tensor)

        args:
            axis: the axis we are expanding
            size: size of one of the new axes, must divide the current size

        returns:
            a MultiImage that has been reshaped
        """
        out = self.empty()
        for (k, parity), image_block in self.items():
            out.append(
                k,
                parity,
                image_block.reshape(
                    image_block.shape[:axis] + (-1, size) + image_block.shape[axis + 1 :]
                ),
            )

        return out

    def combine_axes(self: Self, axes: Sequence[int]) -> Self:
        """
        Combine multiple adjacent axes into a single one on all images. Useful for recombining
        after splitting for timesteps.

        args:
            axes: the axes to recombine

        returns:
            a MultiImage that has been reshaped
        """
        assert list(axes) == list(range(axes[0], axes[-1] + 1))
        out = self.empty()
        for (k, parity), image_block in self.items():
            out.append(
                k,
                parity,
                image_block.reshape(
                    image_block.shape[: axes[0]] + (-1,) + image_block.shape[axes[-1] + 1 :]
                ),
            )

        return out

    def get_signature(self: Self) -> Signature:
        """
        Get a tuple of ( ((k,p),channels), ((k,p),channels), ...). Channels is the last axis prior
        to the spatial dimensions.

        returns:
            the signature tuple
        """
        leading_axes = self.get_n_leading()
        return Signature(
            tuple((k_p, img.shape[leading_axes - 1]) for k_p, img in self.data.items())
        )

    def get_signature_dict(self: Self) -> dict[tuple[tuple[bool, ...], int], int]:
        """
        Get the signature as a dictionary of keys (k,parity) and values channels. Channels is the
        last axis prior to spatial dimensions.

        returns:
            the signature as a dictionary
        """
        return {key: val for key, val in self.get_signature()}

    def get_n_leading(self: Self) -> int:
        """
        Get the number of leading axes prior to spatial and tensor. This number is guaranteed to
        be the same for all image blocks in the multi image.

        returns:
            the number of leading axes
        """
        for (k, _), image_block in self.items():
            return image_block.ndim - (self.D + len(k))

        return 0

    # The below functions make the most sense when working with batch axes.

    def get_L(self: Self) -> int:
        """
        Get the length of the first axis of the first image block in the MultiImage. If this is a
        batch axis, it should be the same for all image blocks, but if it is a channel it need not
        be.

        returns:
            the batch size
        """
        if len(self.values()) == 0:
            return 0

        return len(next(iter(self.values())))

    def reshape_pmap(self: Self, devices: list[jax.Device], axis: int = 0) -> Self:
        """
        Reshape the batch to allow pmap to work. E.g., if shape is (batch,1,N,N) and num_devices=2, then
        reshape to (2,batch/2,1,N,N). Axis specifies the axis to be split.

        args:
            devices: list of gpus or cpu that we are using
            axis: the axis to reshape, assumed that its the first axis

        returns:
            a new MultiImage that has been shaped appropriately for the pmap
        """
        num_devices = len(devices)
        assert self.get_L() % num_devices == 0, (
            f"MultiImage::reshape_pmap: length of devices must evenly "
            f"divide the total batch size, but got batch_size: {self.get_L()}, devices: {devices}"
        )

        out = self.empty()
        for (k, parity), image in self.items():
            new_shape = (
                image.shape[:axis]
                + (num_devices, self.get_L() // num_devices)
                + image.shape[axis + 1 :]
            )
            out.append(k, parity, image.reshape(new_shape))

        return out

    def merge_axes(self: Self, axes: Sequence[int]) -> Self:
        """
        Given a contiguous sequence of axes, merge them together using reshape(-1).

        args:
            axes: a contiguous sequence of axes to merge together

        returns:
            a new MultiImage that has been merged again
        """
        assert len(axes) > 1
        first = axes[0]
        last = axes[-1]
        out = self.empty()
        for (k, parity), image_block in self.items():
            new_shape = image_block.shape[:first] + (-1,) + image_block.shape[last + 1 :]
            out.append(k, parity, image_block.reshape(new_shape))

        return out

    def get_subset(self: Self, idxs: jax.Array) -> Self:
        """
        Select a subset of the leading axes, picking the indices idxs

        args:
            idxs (jnp.array): array of indices to select the subset

        returns:
            a new MultiImage that only has that subset
        """
        assert isinstance(idxs, jnp.ndarray), "MultiImage::get_subset arg idxs must be a jax array"
        assert len(
            idxs.shape
        ), "MultiImage::get_subset arg idxs must be a jax array, e.g. jnp.array([0])"
        return self.__class__(
            {k: image_block[idxs] for k, image_block in self.items()},
            self.D,
            self.is_torus,
            self.metric_tensor,
            self.metric_tensor_inv,
        )

    def get_one(self: Self, idx: int = 0, keepdims=True) -> Self:
        """
        Get a single MultiImage along the first axis. If keepdims is true, that axis is still there
        as a 1, otherwise it is removed. This makes the most sense when the first axis is the batch
        dimension.

        args:
            idx: index of the single batch we are getting

        returns:
            a new MultiImage that is only that batch
        """
        if keepdims:
            return self.get_subset(jnp.array([idx]))
        else:
            return self.__class__(
                {k: image_block[idx] for k, image_block in self.items()},
                self.D,
                self.is_torus,
                self.metric_tensor,
                self.metric_tensor_inv,
            )

    def plot(
        self: Self,
        rows_axis: int = 0,
        cols_axis: int = 1,
        fig: Optional[matplotlib.figure.Figure] = None,
        axes: Any = None,
        row_titles: list = [],
        col_titles: list = [],
        colorbar: bool = False,
    ) -> tuple[matplotlib.figure.Figure, Any]:
        """
        Plot a MultiImage. This implicitly assumes that the MultiImage has at least two axes that
        aren't the spatial ones.

        Plot all timesteps of a particular component of two MultiImages, and the differences between them.
        args:
            test_multi_image: the predicted MultiImage
            actual_multi_image: the ground truth MultiImage
            save_loc: file location to save the image
            future_steps: the number future time steps in the MultiImage
            component: index of the component to plot, default to 0
            show_power: whether to also plot the power spectrum
            title: additional str to add to title, will be "test {title} {col}"
                "actual {title} {col}"
            minimal: if minimal, no titles, colorbars, or axes labels
        """
        assert ((), 0) in self, "MultiImage::plot: Currently only plots scalar multi images."

        # move the rows axis to the first axis, and the cols axis to the second
        image_block = jnp.moveaxis(self[((), 0)], (rows_axis, cols_axis), (0, 1))
        nrows = image_block.shape[0]
        ncols = image_block.shape[1]
        if fig is None or axes is None:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))

        vmax = float(jnp.max(jnp.abs(image_block)))
        vmin = -1 * vmax

        # figsize is 6 per col, 6 per row, (cols,rows)
        for i, row_title in zip(range(nrows), row_titles):
            for j, col_title in zip(range(ncols), col_titles):
                if row_title and col_title:
                    title = f"{row_title} {col_title}"
                else:
                    title = f"{row_title}{col_title}"
                GeometricImage(image_block[i, j], 0, self.D, self.is_torus).plot(
                    axes[i, j], title, vmin=vmin, vmax=vmax, colorbar=colorbar
                )

        return fig, axes

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
