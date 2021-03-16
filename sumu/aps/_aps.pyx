import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.utility cimport move  # min Cython 0.29.17
from libc.stdint cimport uint32_t as bm32
from .utils.bitmap import bm_to_ints#, ikbit


cdef extern from "../bitmap/bitmap.hpp":
    int ikbit_32(bm32 bitmap, int k, int bit)


cdef extern from "aps-0.9.1/aps/logdouble.h" namespace "aps":

    cdef struct LogDouble:
        double log


cdef extern from "aps-0.9.1/aps/array.h" namespace "aps":

    cdef cppclass Array[T]:
        Array() except +
        Array(int n) except +
        void fill(const T & val)
        T & operator[](int)
        T * data()
        size_t size()


cdef class ArrayArray:

    cdef Array[Array[LogDouble]] * thisptr

    def __init__(self, n):
        self.thisptr = new Array[Array[LogDouble]](n)

    def init_with_double_array(self, i, n):
        self.thisptr.data()[i] = move((new Array[LogDouble](n))[0])

    def set_value(self, i, j, double val):
        cdef Array[LogDouble] * ith = &(self.thisptr.data()[i])
        cdef LogDouble * jth = &(ith.data()[j])
        cdef LogDouble x
        x.log = val
        jth[0] = x

    def get_value(self, i, j):
        return self.thisptr.data()[i][j].log


cdef extern from "aps-0.9.1/aps/simple_modular.cpp" namespace "aps":
    Array[Array[LogDouble]] modularAPS_simple(Array[Array[LogDouble]] & w, bool test)


def aps(weights, as_dict=False, normalize=False):

    # NOTE: Preliminary analysis shows this uses about 2.6x more space
    #       than C++ version.

    V = weights.shape[0]
    J = weights.shape[1]

    w = ArrayArray(V)
    cdef int i, j

    for i in range(V):
        w.init_with_double_array(i, J)
        for j in range(J):
            w.set_value(i, j, weights[i, j])

    del weights

    cdef Array[Array[LogDouble]] probs

    probs = modularAPS_simple(w.thisptr[0], False)
    del w

    if as_dict is True:
        weights = {v: dict() for v in range(V)}
        for i in range(V):
            for j in range(J):
                pset = bm_to_ints(ikbit_32(j, i, 0))
                weights[i][pset] = probs.data()[i][j].log
            if normalize is True:
                normalizer = np.logaddexp.reduce(list(weights[i].values()))
                for pset in weights[i].keys():
                    weights[i][pset] -= normalizer
    else:
        weights = np.empty((V, J))
        for i in range(V):
            for j in range(J):
                weights[i, j] = probs.data()[i][j].log
            if normalize is True:
                weights[i] -= np.logaddexp.reduce(weights[i])

    # The following fails with "Deletion of non-heap C++ object",
    # which is curious as the code runs with large inputs
    # which should not fit into stack.
    # del probs
    return weights
