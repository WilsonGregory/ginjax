from typing_extensions import Self


class SPoly:
    """
    A symbolic polynomial.
    """

    poly: dict[tuple[str, ...], int]

    def __init__(self: Self, base: str | None = None, scale: int = 1) -> None:
        self.poly = {}
        if base is not None:
            self.poly[(base,)] = scale

    @classmethod
    def from_dict(cls, poly: dict[tuple[str, ...], int], remove_zero_scale: bool = True) -> Self:
        new_poly = cls()

        if remove_zero_scale:
            # go through again, and make sure we aren't adding ones with scale 0
            cleaned_poly = {}
            for base, val in poly.items():
                if val != 0:
                    cleaned_poly[base] = val

            new_poly.poly = cleaned_poly
        else:
            new_poly.poly = poly

        return new_poly

    def __add__(self: Self, other: Self) -> Self:
        new_poly = self.poly
        for base, val in other.poly.items():
            if base in new_poly:
                new_poly[base] = new_poly[base] + val
            else:
                new_poly[base] = val

        return self.__class__.from_dict(new_poly)

    def __mul__(self: Self, other: Self) -> Self:
        new_poly = {}
        for base1, val1 in self.poly.items():
            for base2, val2 in other.poly.items():
                if base1 == ("1",) and base2 == ("1",):
                    new_base = ("1",)
                elif base1 == ("1",):
                    new_base = base2
                elif base2 == ("1",):
                    new_base = base1
                else:
                    new_base = tuple(sorted(base1 + base2))

                if new_base not in new_poly:
                    new_poly[new_base] = val1 * val2
                else:
                    new_poly[new_base] += val1 * val2

        return self.__class__.from_dict(new_poly)

    def __str__(self: Self) -> str:
        return str(self.poly)

    def __repr__(self: Self) -> str:
        return str(self)


def conv1d(A, ff):
    N = len(A)
    out = []
    for i in range(N):
        out_pixel = SPoly()
        for j in [-1, 0, 1]:
            idx = (i + j) % N
            out_pixel += A[idx] * ff[j + 1]

        out.append(out_pixel)

    return out


def conv2d(A, ff):
    """
    Convolve 2d, then return a 1d flattened image
    """
    N = len(A)
    out = []
    for i in range(N):
        for j in range(N):
            out_pixel = SPoly()
            for i2 in [-1, 0, 1]:
                for j2 in [-1, 0, 1]:
                    idx_i = (i + i2) % N
                    idx_j = (j + j2) % N
                    out_pixel += A[idx_i][idx_j] * ff[i2][j2]

            out.append(out_pixel)

    return out


def conv3d(A, ff):
    """
    Convolve 2d, then return a 1d flattened image
    """
    N = len(A)
    out = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                out_pixel = SPoly()
                for i2 in [-1, 0, 1]:
                    for j2 in [-1, 0, 1]:
                        for k2 in [-1, 0, 1]:
                            idx_i = (i + i2) % N
                            idx_j = (j + j2) % N
                            idx_k = (k + k2) % N
                            out_pixel += A[idx_i][idx_j][idx_k] * ff[i2][j2][k2]

                out.append(out_pixel)

    return out


def prod1d(A1, A2):
    """
    Pixel-wise prod
    """
    N = len(A1)
    out = []
    for p1, p2 in zip(A1, A2):
        out.append(p1 * p2)

    return out


def convertPoly(poly: SPoly, false_bases: set[str]) -> dict[tuple[str, ...], str]:
    new_dict = {}
    for k, v in poly.poly.items():
        new_base = tuple()
        mult = ""
        for basis_elem in k:
            if basis_elem in false_bases:
                mult += basis_elem
            else:
                new_base += (basis_elem,)

        if len(new_base) == 0:
            new_base = "1"

        if new_base in new_dict:
            new_dict[new_base].append(str(v) + mult)
        else:
            new_dict[new_base] = [str(v) + mult]

    new_new_dict = {}
    for k, v in new_dict.items():
        new_new_dict[k] = " + ".join(sorted(v))

    return new_new_dict


# N = 3
# D = 1
# A = [SPoly("a-1"), SPoly("a0"), SPoly("a1")]
# ff1 = [SPoly("x"), SPoly("x", -2), SPoly("x")]
# ff2 = [SPoly("x"), SPoly.from_dict({("x",): -2, ("1",): 2}), SPoly("x")]

# A_ff1 = conv1d(A, ff1)
# A_ff2 = conv1d(A, ff2)

# A_prod = prod1d(A_ff1, A_ff2)

# out = sum(A_prod, start=SPoly())

# print(convertPoly(out, {"x"}))

# ~~~~~~~~ D=2 ~~~~~~~~
# N = 3
# D = 2
# A = [
#     [SPoly("a-1-1"), SPoly("a-10"), SPoly("a-11")],
#     [SPoly("a0-1"), SPoly("a00"), SPoly("a01")],
#     [SPoly("a1-1"), SPoly("a10"), SPoly("a11")],
# ]
# ff1 = [
#     [SPoly("x"), SPoly("y"), SPoly("x")],
#     [SPoly("y"), SPoly.from_dict({("x",): -4, ("y",): -4}), SPoly("y")],
#     [SPoly("x"), SPoly("y"), SPoly("x")],
# ]
# ff2 = [
#     [SPoly("x"), SPoly("y"), SPoly("x")],
#     [SPoly("y"), SPoly.from_dict({("x",): -4, ("y",): -4, ("1",): 2}), SPoly("y")],
#     [SPoly("x"), SPoly("y"), SPoly("x")],
# ]

# A_ff1 = conv2d(A, ff1)
# A_ff2 = conv2d(A, ff2)

# A_prod = prod1d(A_ff1, A_ff2)

# out = sum(A_prod, start=SPoly())
# # print(out)
# consolidated_out = convertPoly(out, {"x", "y"})
# print(consolidated_out)

# reversed_dict = {}
# for k, v in consolidated_out.items():
#     if v in reversed_dict:
#         reversed_dict[v].append("".join(k))
#     else:
#         reversed_dict[v] = ["".join(k)]

# print("~~~~~~~~~~~~~~~~~~~~~~")
# print("REVERSED DICT")
# print(reversed_dict)

# unique terms
# -8x + -8y + 20xx + 20yy + 32xy
# -2x + -2y + 5xx + 5yy + 8xy

# -14yy + -8xy + 4xx + 4y
# -7yy - 4xy + 2xx + 2y

# -14xx + -8xy + 4x + 4yy
# -7xx -4xy + 2x + 2yy

# ~~~~~~~ D = 3 ~~~~~~~
N = 3
D = 3

A = [
    [
        [SPoly("a-1-1-1"), SPoly("a-1-10"), SPoly("a-1-11")],
        [SPoly("a-10-1"), SPoly("a-100"), SPoly("a-101")],
        [SPoly("a-11-1"), SPoly("a-110"), SPoly("a-111")],
    ],
    [
        [SPoly("a0-1-1"), SPoly("a0-10"), SPoly("a0-11")],
        [SPoly("a00-1"), SPoly("a000"), SPoly("a001")],
        [SPoly("a01-1"), SPoly("a010"), SPoly("a011")],
    ],
    [
        [SPoly("a1-1-1"), SPoly("a1-10"), SPoly("a1-11")],
        [SPoly("a10-1"), SPoly("a100"), SPoly("a101")],
        [SPoly("a11-1"), SPoly("a110"), SPoly("a111")],
    ],
]

ff1 = [
    [
        [SPoly("x"), SPoly("y"), SPoly("x")],
        [SPoly("y"), SPoly("z"), SPoly("y")],
        [SPoly("x"), SPoly("y"), SPoly("x")],
    ],
    [
        [SPoly("y"), SPoly("z"), SPoly("y")],
        [SPoly("z"), SPoly.from_dict({("x",): -8, ("y",): -12, ("z",): -6}), SPoly("z")],
        [SPoly("y"), SPoly("z"), SPoly("y")],
    ],
    [
        [SPoly("x"), SPoly("y"), SPoly("x")],
        [SPoly("y"), SPoly("z"), SPoly("y")],
        [SPoly("x"), SPoly("y"), SPoly("x")],
    ],
]

ff2 = [
    [
        [SPoly("x"), SPoly("y"), SPoly("x")],
        [SPoly("y"), SPoly("z"), SPoly("y")],
        [SPoly("x"), SPoly("y"), SPoly("x")],
    ],
    [
        [SPoly("y"), SPoly("z"), SPoly("y")],
        [SPoly("z"), SPoly.from_dict({("x",): -8, ("y",): -12, ("z",): -6, ("1",): 2}), SPoly("z")],
        [SPoly("y"), SPoly("z"), SPoly("y")],
    ],
    [
        [SPoly("x"), SPoly("y"), SPoly("x")],
        [SPoly("y"), SPoly("z"), SPoly("y")],
        [SPoly("x"), SPoly("y"), SPoly("x")],
    ],
]

A_ff1 = conv3d(A, ff1)
A_ff2 = conv3d(A, ff2)

A_prod = prod1d(A_ff1, A_ff2)

out = sum(A_prod, start=SPoly())
# print(out)
consolidated_out = convertPoly(out, {"x", "y", "z"})
print(consolidated_out)

reversed_dict = {}
for k, v in consolidated_out.items():
    if v in reversed_dict:
        reversed_dict[v].append("".join(k))
    else:
        reversed_dict[v] = ["".join(k)]

print("~~~~~~~~~~~~~~~~~~~~~~")
print("REVERSED DICT")
print({k: len(v) for k, v in reversed_dict.items()})

## unique terms
# -12z + -16x + -24y + 144yz + 156yy + 192xy + 42zz + 72xx + 96xz
# -6z - 8x - 12y + 72yz + 78yy + 96xy + 21zz + 36xx + 48xz

# -22zz + -32xz + -32yz + 16xy + 4z + 8xx + 8yy
# -11zz - 16xz - 16yz + 8xy +2z + 4xx + 4yy

# -16xy + -16yz + -38yy + 4xx + 4y + 4zz + 8xz
# -8xy - 8 yz - 19yy + 2xx + 2y + 2zz + 4xz

# -12xz + -30xx + -36xy + 12yy + 12yz + 4x
# -6xz - 15xx - 18xy + 6yy + 6yz + 2x
