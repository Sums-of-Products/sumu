# distutils: language=c++

from libc.stdint cimport uint64_t
from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef extern from "weight_sum.h":

    double weight_sum_64(double w, uint64_t * psets, int m, double * weights, int j, int n, uint64_t U, uint64_t T, int t_ub)
    pair[double, vector[int]] weight_sum_contribs_64(double w, uint64_t * psets, int m, double * weights, int j, int n, uint64_t U, uint64_t T, int t_ub)

    double weight_sum_128(double w, uint64_t * psets, int m, double * weights, int j, int n, vector[uint64_t] U, vector[uint64_t] T, int t_ub)
    pair[double, vector[int]] weight_sum_contribs_128(double w, uint64_t * psets, int m, double * weights, int j, int n, vector[uint64_t] U, vector[uint64_t] T, int t_ub)


def weight_sum(w, psets, weights, n, U, T, t_ub):

    cdef uint64_t[::1] psets_memview_64
    cdef uint64_t[:, ::1] psets_memview_128

    cdef double[::1] weights_memview = weights

    if psets.ndim == 1:
        psets_memview_64 = psets
        return weight_sum_64(w,
                             & psets_memview_64[0],
                             psets_memview_64.shape[0],
                             & weights_memview[0],
                             weights_memview.shape[0], n, U, T, t_ub)
    elif psets.ndim == 2:
        psets_memview_128 = psets
        return weight_sum_128(w,
                              & psets_memview_128[0, 0],
                              psets_memview_128.shape[0],
                              & weights_memview[0],
                              weights_memview.shape[0], n, U, T, t_ub)


def weight_sum_contribs(w, psets, weights, n, U, T, t_ub):

    cdef uint64_t[::1] psets_memview_64
    cdef uint64_t[:, ::1] psets_memview_128

    cdef double[::1] weights_memview = weights

    if psets.ndim == 1:
        psets_memview_64 = psets
        return weight_sum_contribs_64(w,
                                      & psets_memview_64[0],
                                      psets_memview_64.shape[0],
                                      & weights_memview[0],
                                      weights_memview.shape[0], n, U, T, t_ub)
    elif psets.ndim == 2:
        psets_memview_128 = psets
        return weight_sum_contribs_128(w,
                                       & psets_memview_128[0, 0],
                                       psets_memview_128.shape[0],
                                       & weights_memview[0],
                                       weights_memview.shape[0], n, U, T, t_ub)
