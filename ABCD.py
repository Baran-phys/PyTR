import time

import numba

# from sympy import factorial2 as _factorial2
from sympy.abc import s
from scipy.special import zeta
import itertools
from numpy import linalg as LA
from fractions import Fraction
import numpy as np
import functools
from scipy.special import factorial2 as _factorial2

factorial = np.math.factorial
factorial2 = functools.partial(_factorial2, exact=True)

# def factorial2(x: int) -> int:
#     return _factorial2(x)  # type: ignore


@numba.jit(nopython=True)
def KroneckerDelta(x, y):
    if x == y:
        return 1.0
    return 0.0


# Initial data for Kontsevich volumes
@numba.jit(nopython=True)
def awk(i, j, k):
    if 0 == i == j == k:
        return 1
    return 0


def bwk(k, a, b):
    return (
        KroneckerDelta(k + a, b + 1)
        * factorial2(2 * b + 1)
        / (factorial2(2 * k + 1) * factorial2(2 * a + 1))
        * (2 * a + 1)
    )


def cwk(k, a, b):
    return KroneckerDelta(k, a + b + 2) * factorial2(2 * a + 1) * factorial2(2 * b + 1) / factorial2(2 * k + 1)


@numba.jit(nopython=True)
def dwk(k):
    if k == 1:
        return 1 / 24.0
    return 0.0


# Initial data for multicurve count
@numba.jit()
def amv(i, j, k):
    if 0 == i == j == k:
        return 1
    return 0


@numba.jit(nopython=False)
def bmv(k, a, b):
    return (2 * a + 1) * KroneckerDelta(k + a, b + 1) + KroneckerDelta(k, 0) * KroneckerDelta(a, 0) * zeta(
        2 * b + 2
    ) / s ** (2 * b + 2)


@numba.jit()
def cmv(k, a, b):
    p1 = 0
    if b - k + 1 >= 0:
        p1 = (
            factorial(2 * a + 2 * b - 2 * k + 3)
            / (factorial(2 * a + 1) * factorial(2 * b - 2 * k + 2))
            * zeta(2 * a + 2 * b - 2 * k + 4)
            / s ** (2 * a + 2 * b - 2 * k + 4)
        )

    p2 = 0
    if a - k + 1 >= 0:
        p2 = (
            factorial(2 * a + 2 * b - 2 * k + 3)
            / (factorial(2 * b + 1) * factorial(2 * a - 2 * k + 2))
            * zeta(2 * a + 2 * b - 2 * k + 4)
            / s ** (2 * a + 2 * b - 2 * k + 4)
        )

    return (
        KroneckerDelta(k, a + b + 2)
        + KroneckerDelta(k, 0) * KroneckerDelta(k, 0) * zeta(2 * a + 2) * zeta(2 * b + 2) / s ** (2 * a + 2 * b + 4)
        + p1
        + p2
    )


@numba.jit()
def dmv(k):
    return 1 / (2 * s**2) * zeta(2) * KroneckerDelta(k, 0) + 1 / 8 * KroneckerDelta(k, 1)


# Intermediate functions



# given a natural number n, this function return a list, whose element are the partitions of {2, ... , n}
def partition(lst):
    import more_itertools as mit
    return [part for k in range(1, len(lst) + 1) for part in mit.set_partitions(lst, k)]


# given a natural number n, this function return a list, whose element are the partitions of {2, ... , n}
# of the form {J_1, J_2}, with J_i nonempty
def bipartitions(n):
    collection = list(range(2, n + 1))
    #print(collection)
    return list(sorted(elem) for elem in partition(collection) if len(elem) == 2)


# given natural numbers h and n, this function return a list, whose element are the partitions of {2, ..., n} of
# the form {J_1, J_2}, with J_i nonempty if h=0.
def specialbipartitions(h, n):
    if h > 0:
        return bipartitions(n) + [[[], list(range(2, n))]]
    else:
        return bipartitions(n)


# given natural numbers n and d, this function return a list, whose element are the multiindices \[Mu]  \[Element]  \
# [DoubleStruckCapitalN]^n such that | \[Mu] | =  d
def multiindex(n, d):
    a = [range(0, d + 1)] * n
    return list(vector for vector in list(itertools.product(*a)) if LA.linalg.norm(vector, ord=1) <= d)


def noduplicate(myList):
    return sorted(set(myList))


@numba.jit(nopython=True)
def dim(g, n):
    return 3 * g - 3 + n



#Create ABCD tensors
def AC(g,n, fun_b, fun_c):
    vc=np.vectorize(fun_c)
    vb=np.vectorize(fun_b)
    float_formatter = "{:.5f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    size = dim(g, n)
    c = np.zeros((size,size,size))
    b = np.zeros((size,size,size))
    for i in range(size):
        for j in range(size):
            for k in range(size):
                c[i,j,k] = vc(i, j, k)
                b[i,j,k] = vb(i, j, k)

    return b,c


AC(2, 3, bwk, cwk)
