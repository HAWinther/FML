"""Small sparse-Fourier EdS oracle for arbitrary-order LPT validation.

The main solver extracts coefficients of the determinant/cofactor form of the
Lagrangian equations. It does not use the pair/triple source recursion planned
for the FFT implementation. Products are integer-wavevector convolutions with
no wrapping or truncation. This is a small-field reference, not a mesh solver.
"""

from functools import lru_cache
from itertools import permutations, product
from math import factorial, pi


ZERO = (0, 0, 0)
PERMUTATIONS = tuple(permutations(range(3)))


def parity(permutation):
    inversions = sum(permutation[i] > permutation[j] for i in range(3) for j in range(i + 1, 3))
    return (-1) ** inversions


def add(*terms):
    """Sum (coefficient, scalar field) pairs."""
    result = {}
    for coefficient, field in terms:
        for mode, value in field.items():
            result[mode] = result.get(mode, 0j) + coefficient * value
    return {mode: value for mode, value in result.items() if value != 0}


def multiply(left, right):
    result = {}
    for a, x in left.items():
        for b, y in right.items():
            mode = tuple(a[i] + b[i] for i in range(3))
            result[mode] = result.get(mode, 0j) + x * y
    return {mode: value for mode, value in result.items() if value != 0}


def derivative(field, axis):
    return {mode: 2j * pi * mode[axis] * value for mode, value in field.items() if mode[axis]}


def gradient(displacement):
    return [[derivative(displacement[i], j) for j in range(3)] for i in range(3)]


def coefficient(factors, degree):
    """Coefficient of g**degree in a product of sparse time polynomials."""
    result = {}
    for orders in product(*(factor.keys() for factor in factors)):
        if sum(orders) != degree:
            continue
        term = {ZERO: 1}
        for factor, order in zip(factors, orders):
            term = multiply(term, factor[order])
        result = add((1, result), (1, term))
    return result


def deformation(orders):
    """F_ij(g) = delta_ij + sum g**n d_j Psi_n,i."""
    matrix = [[{} for j in range(3)] for i in range(3)]
    for i in range(3):
        matrix[i][i][0] = {ZERO: 1}
    for n, displacement in orders.items():
        tensor = gradient(displacement)
        for i in range(3):
            for j in range(3):
                matrix[i][j][n] = tensor[i][j]
    return matrix


def determinant_coefficient(matrix, degree):
    return add(*[(parity(p), coefficient([matrix[i][p[i]] for i in range(3)], degree))
                 for p in PERMUTATIONS])


def cofactor(matrix, row, column, max_degree):
    rows = [i for i in range(3) if i != row]
    columns = [j for j in range(3) if j != column]
    a, b = rows
    c, d = columns
    return {n: add(((-1) ** (row + column), coefficient([matrix[a][c], matrix[b][d]], n)),
                   (-(-1) ** (row + column), coefficient([matrix[a][d], matrix[b][c]], n)))
            for n in range(max_degree + 1)}


def acceleration_divergence(matrix, max_degree):
    """Coefficient of C_ij (d_t^2+2H d_t) F_ij / H^2 in EdS."""
    terms = []
    for i in range(3):
        for j in range(3):
            acceleration = {n: add((n * (n + 0.5), field))
                            for n, field in matrix[i][j].items() if n}
            terms.append((1, coefficient([cofactor(matrix, i, j, max_degree), acceleration], max_degree)))
    return add(*terms)


def helmholtz(divergence, curl):
    """Psi = (-i k div + i k cross curl)/k^2, mean Psi = 0."""
    modes = set(divergence).union(*(component.keys() for component in curl)) - {ZERO}
    result = [{}, {}, {}]
    for mode in sorted(modes):
        wave = tuple(2 * pi * x for x in mode)
        k2 = dot(wave, wave)
        rotational = cross(wave, tuple(component.get(mode, 0j) for component in curl))
        for i in range(3):
            result[i][mode] = (-1j * wave[i] * divergence.get(mode, 0j) + 1j * rotational[i]) / k2
    return result


def solve(delta, max_order):
    """Return {n: [Psi_n,x, Psi_n,y, Psi_n,z]} in unit-box coordinates."""
    if max_order < 1 or delta.get(ZERO, 0) != 0:
        raise ValueError("Need positive maximum order and zero-mean linear density")
    orders = {1: helmholtz(add((-1, delta)), [{}, {}, {}])}
    gradients = {1: gradient(orders[1])}
    for n in range(2, max_order + 1):
        # Leave Psi_n absent. Its coefficient in the scalar equation is
        # [n(n+1/2)-3/2] div(Psi_n), so the remaining terms determine it.
        matrix = deformation(orders)
        denominator = n * (n + 0.5) - 1.5
        divergence = add((1.5 / denominator, determinant_coefficient(matrix, n)),
                         (-1 / denominator, acceleration_divergence(matrix, n)))
        curl = []
        for axis in range(3):
            j, k = (axis + 1) % 3, (axis + 2) % 3
            terms = []
            # Coefficient of F^T dF/dln(a) being symmetric (Cauchy invariants).
            for r in range(1, n):
                s = n - r
                for i in range(3):
                    terms.append((s / n, multiply(gradients[r][i][j], gradients[s][i][k])))
                    terms.append((-s / n, multiply(gradients[r][i][k], gradients[s][i][j])))
            curl.append(add(*terms))
        orders[n] = helmholtz(divergence, curl)
        gradients[n] = gradient(orders[n])
    return orders


def dot(left, right):
    return sum(x * y for x, y in zip(left, right))


def cross(left, right):
    return tuple(left[(i + 1) % 3] * right[(i + 2) % 3] - left[(i + 2) % 3] * right[(i + 1) % 3]
                 for i in range(3))


def vector_sum(*vectors):
    return tuple(sum(vector[i] for vector in vectors) for i in range(3))


def scaled(vector, factor):
    return tuple(factor * value for value in vector)


def u(left, right):
    denominator = dot(left, left) * dot(right, right)
    return dot(cross(left, right), cross(left, right)) / denominator if denominator else 0.0


def v(first, second, third):
    denominator = dot(first, first) * dot(second, second) * dot(third, third)
    return dot(first, cross(second, third)) ** 2 / denominator if denominator else 0.0


def w(left, right):
    denominator = dot(left, left) * dot(right, right)
    return scaled(cross(left, right), dot(left, right) / denominator) if denominator else ZERO


@lru_cache(maxsize=None)
def published_kernel_parts(waves):
    """Unsymmetrized S_n,T_n from Matsubara appendix B, equations 82-87.

    The k here are integer mode numbers. Kernel displacements require an
    additional i/(2 pi n!) when contracted with ordered density products.
    These explicit low-order formulae are independent of the cofactor solver.
    """
    n = len(waves)
    if n == 1:
        return 1.0, ZERO
    a, b = waves[:2]
    if n == 2:
        return 3 / 7 * u(a, b), ZERO
    rest = vector_sum(*waves[1:])
    s2, _ = published_kernel_parts(waves[-2:])
    if n == 3:
        return 5 / 3 * u(a, rest) * s2 - v(*waves) / 3, scaled(w(a, rest), s2)
    if n != 4:
        raise ValueError("Published explicit comparator supports orders 1-4")
    s3, t3 = published_kernel_parts(waves[1:])
    first_pair, last_pair = vector_sum(a, b), vector_sum(*waves[2:])
    first_s2, _ = published_kernel_parts(waves[:2])
    longitudinal = (28 / 11 * (u(a, rest) * s3 - dot(w(a, rest), t3))
                    + 17 / 11 * u(first_pair, last_pair) * first_s2 * s2
                    - 26 / 11 * v(a, b, last_pair) * s2)
    denominator = dot(a, a) * dot(rest, rest)
    correction = scaled(cross(a, rest), dot(cross(a, rest), t3) / denominator) if denominator else ZERO
    transverse = scaled(vector_sum(scaled(w(a, rest), s3), correction), 2)
    return longitudinal, transverse


def published_field(delta, order):
    """All ordered tuples implicitly symmetrize the kernel, including repeats."""
    result = [{}, {}, {}]
    for waves in product(delta, repeat=order):
        mode = vector_sum(*waves)
        k2 = dot(mode, mode)
        if k2 == 0:
            continue
        s, t = published_kernel_parts(waves)
        kernel = vector_sum(scaled(mode, s), cross(mode, t))
        amplitude = 1j / (2 * pi * factorial(order) * k2)
        for wave in waves:
            amplitude *= delta[wave]
        for i in range(3):
            result[i][mode] = result[i].get(mode, 0j) + amplitude * kernel[i]
    return result


def fixture():
    """Three non-coplanar waves with phases and unequal lengths, plus conjugates."""
    positive = {(1, 0, 0): 0.12 + 0.03j, (0, 1, 1): -0.08 + 0.04j, (1, -1, 2): 0.06 - 0.02j}
    return {**positive, **{tuple(-x for x in mode): value.conjugate() for mode, value in positive.items()}}
