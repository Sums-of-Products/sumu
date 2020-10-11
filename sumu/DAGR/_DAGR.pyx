from libc.stdint cimport uint64_t as bm64
from libc.stdint cimport uint32_t as bm32


cdef extern from "DAGR.hpp":

    cdef cppclass CppDAGR "DAGR":

        CppDAGR(double* score_array, int* C, int n, int K, double tolerance)
        void precompute(int v)
        void clear()
        double f(bm32 X, bm32 Y)


cdef class DAGR:

    cdef CppDAGR * thisptr

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

    def precompute(self, int v):
        self.thisptr.precompute(v)

    def clear(self):
        self.thisptr.clear()

    def f(self, bm32 X, bm32 Y):
        return self.thisptr.f(X, Y)

