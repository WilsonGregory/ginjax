def cnn_flops(D: int, N: int, c_in: int, c_out: int, kernel_size: int, max_k: int) -> int:
    # equivariant cnn flop count, with scalar,vector -> scalar,vector with same number of channels
    # return (N**D) * c_in * c_out * (kernel_size**D) * (D**3 + D**2 + D + 1)
    # I believe our efficient implementation of this is what it is
    return (N**D) * c_in * c_out * (kernel_size**D) * ((D**max_k) ** 2)


def vector_neuron_flops(D: int, N: int, c_in: int, tensor_order: int) -> int:
    # this is highly simplified, ignores some stuff
    return (D**tensor_order) * c_in * (N**D)


def max_norm_pooling(D: int, N: int, c_in: int, tensor_order: int, patch_size: int) -> int:
    # calculate the norm of each point of each channel, and then compare within patches.
    return (D**tensor_order) * c_in * (N**D) + patch_size**D


def unet_flops(
    D: int, N: int, num_conv: int, num_downsamples: int, depth: int, kernel_size: int, max_k: int
) -> int:
    total = 0
    for _ in range(num_conv):
        total += cnn_flops(D, N, 2, depth, kernel_size, max_k)
        total += vector_neuron_flops(D, N, depth, max_k)

    N_curr = N
    for downsample in range(1, num_downsamples + 1):
        c_out = depth * (2**downsample)
        total += max_norm_pooling(D, N_curr, c_out, max_k, 2)
        N_curr = N_curr // 2

        for conv_idx in range(num_conv):
            c_in = (depth * (2 ** (downsample - 1))) if conv_idx == 0 else c_out
            total += cnn_flops(D, N_curr, c_in, c_out, kernel_size, max_k)
            total += vector_neuron_flops(D, N_curr, c_out, max_k)

    for upsample in reversed(range(num_downsamples)):
        c_out = depth * (2**upsample)

        # transposed convolution
        total += cnn_flops(D, N_curr, depth * (2 ** (upsample + 1)), c_out, 2, max_k)
        N_curr = N_curr * 2

        for conv_idx in range(num_conv):
            # first one has the skip layer from the other side
            c_in = depth * (2 ** (upsample + 1)) if conv_idx == 0 else c_out
            total += cnn_flops(D, N_curr, c_in, c_out, kernel_size, max_k)
            total += vector_neuron_flops(D, N_curr, c_out, max_k)

    total += cnn_flops(D, N, depth, 2, kernel_size, max_k)

    return total


N = 64
num_conv = 2
num_downsamples = 4
kernel_size = 3
max_k = 1
depth = 48


print("For the Heat Equation")

d2_one_forward = unet_flops(2, N, num_conv, num_downsamples, depth, kernel_size, max_k=0)

print(f"D=2 one forward: {d2_one_forward / 1_000_000_000} G-FLOPS")

d2_pretrain_full = d2_one_forward * 128 * 50

print(f"pre-train: {d2_pretrain_full / 1_000_000_000} G-FLOPS")

d3_one_forward = unet_flops(3, N, num_conv, num_downsamples, depth, kernel_size, max_k=0)

print(f"D=3 {d3_one_forward / 1_000_000_000} G-FLOPS")


print("For the Burgers'/CFD")

d2_one_forward = unet_flops(2, N, num_conv, num_downsamples, depth, kernel_size, max_k=1)

print(f"D=2 one forward: {d2_one_forward / 1_000_000_000} G-FLOPS")

d2_pretrain_full = d2_one_forward * 128 * 50

print(f"pre-train: {d2_pretrain_full / 1_000_000_000} G-FLOPS")

d3_one_forward = unet_flops(3, N, num_conv, num_downsamples, depth, kernel_size, max_k=1)

print(f"D=3 {d3_one_forward / 1_000_000_000} G-FLOPS")
