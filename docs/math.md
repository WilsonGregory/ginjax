This page is intended to be a brief introduction to the mathematics behind this model. For a more in depth explanation, please check out our paper: [[1](https://arxiv.org/abs/2305.12585)].

Let $G$ be a group with an action on vector spaces $X$ and $Y$. 
Let $f$ be a function from $X$ to $Y$. 
Then we say $f$ is $G$-equivariant if for all $g \in G$, $x \in X$, we have $f(g \cdot x) = g \cdot f(x)$. 
We will consider our group $G$ to be the _hyperoctahedral group_ of dimension $d$, denoted $G_{N,d}$. 
This is the group of translations, rotations, and reflections of a $d$-dimensional hypercube. 
This group will act on _geometric images_, that is images where every pixel is a scalar, vector, or higher order tensor. 

Lets consider some examples of geometric images. 
A scalar image is like a regular black and white image where each pixel is a single grayscale value. 
When we rotate a scalar image, the pixels move to their new location according to the rotation. We can have a color image by having multiple channels of a scalar image, for example 3 channels for a RGB image. 
A vector image could be something like an ocean current map where each pixel is a $d$-dimensional vector that shows where the water is flowing. 
When we rotate a vector image, not only do the pixels move to their new location, the vector-valued pixels themselves are also rotated. 
The behavior of the image under rotations and reflections is what distinguishes a vector image or a higher order tensor image from multiple channels of a scalar image.

To properly write functions geometric images, we have to keep track of the different types of images to make sure we respect the rotation properties of each. 
In the Steerable CNNs literature, [[2](https://arxiv.org/abs/1612.08498), [3](https://arxiv.org/abs/1911.08251)], these are known as "steerable" spaces. 
If $A$ is a geometric image of tensor order $k$ and $C$ is a geometric image of tensor order $k'$, then value of $A$ convolved with $C$ at pixel $\bar\imath$ is given by:

$$
(A \ast C)(\bar\imath) = \sum_{\bar a} A(\bar\imath - \bar a) \otimes C(\bar a) ~,
$$

where the sum is over all pixels $\bar a$ of $C$, and $\bar\imath - \bar a$ is the translation of $\bar\imath$ by $\bar a$. 
The result is a geometric image of tensor order $k+k'$. 
To produce geometric images of smaller tensor order, the tensor contraction can be applied to each pixel. 
Convolution and contraction are combined into a single operation to form linear layers. 
By restricting the convolution filters $C$ to rotation and reflection invariant filters, we can create linear layers which are $G_{N,d}$-equivariant.