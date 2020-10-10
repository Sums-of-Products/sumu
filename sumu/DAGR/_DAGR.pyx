from libc.stdint cimport uint64_t as bm64
from libc.stdint cimport uint32_t as bm32

import numpy as np
cimport numpy as np

cdef extern from "DAGR.hpp":

    cdef cppclass CppDAGR "DAGR":

        CppDAGR(double* score_array, int* C, int n, int K, double tolerance)
        void precompute(int v)
        double ** m_f
        double f(bm32 X, bm32 Y)

    unsigned int count_32(bm32 bitmap)


cdef class DAGR:

    cdef CppDAGR * thisptr;
    cdef list f

    def __cinit__(self, score_array, C, K, tolerance=2.0**(-32)):

        cdef double[:, ::1] memview_score_array
        memview_score_array = score_array

        cdef int[:, ::1] memview_C
        memview_C = C

        self.thisptr = new CppDAGR(& memview_score_array[0, 0],
                                   & memview_C[0, 0],
                                   score_array.shape[0],
                                   K,
                                   tolerance)
    def __dealloc__(self):
        del self.thisptr

    def precompute(self, int v, int K):
        self.thisptr.precompute(v)
        cdef list f = [np.int8(0)]*2**K
        cdef bm32 X, Y
        cdef int k_Y
        for X in np.arange(2**K, dtype=np.int32):
            k_Y = 2**(K - count_32(X))
            f[X] = np.empty(k_Y)
            for Y in np.arange(k_Y, dtype=np.int32):
                f[X][Y] = self.thisptr.m_f[X][Y]
        return f

    def f(self, bm32 X, bm32 Y):
        return self.thisptr.f(X, Y)

