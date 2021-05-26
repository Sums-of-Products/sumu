# distutils: language=c++

from libc.stdint cimport uint32_t as bm32
from libc.stdint cimport uint64_t as bm64
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from libcpp.string cimport string

import numpy as np
cimport numpy as np

from .utils.bitmap import bm, bm_to_np64, np64_to_bm, bm_to_ints
from .utils.math_utils import comb

ctypedef IntersectSums* isum_ptr


cdef extern from "CandidateRestrictedScore.hpp":

    cdef cppclass CppCandidateRestrictedScore "CandidateRestrictedScore":

        CppCandidateRestrictedScore(double* w, int* C, int n, int K,
                                    int cc_limit, double cc_tol, double isum_tol,
                                    string logfile,
                                    bool silent
                                    )
        double sum(int v, bm32 U, bm32 T, bool isum)
        double sum(int v, bm32 U)
        pair[bm32, double] sample_pset(int v, bm32 U, bm32 T, double wcum)
        pair[bm32, double] sample_pset(int v, bm32 U, double wcum)


cdef extern from "IntersectSums.hpp":

    struct bm128:
        bm64 s1
        bm64 s2

    struct bm192:
        bm64 s1
        bm64 s2
        bm64 s3

    struct bm256:
        bm64 s1
        bm64 s2
        bm64 s3
        bm64 s4

    cdef cppclass IntersectSums:

        IntersectSums(double *w, bm64 * pset, bm64 m, int n, double eps)
        void dummy()
        double scan_sum_64(double w, bm64 U, bm64 T, bm64 t_ub)
        double scan_sum_128(double w, bm128 U, bm128 T, bm64 t_ub)
        double scan_sum_192(double w, bm192 U, bm192 T, bm64 t_ub)
        double scan_sum_256(double w, bm256 U, bm256 T, bm64 t_ub)
        pair[bm64, double] scan_rnd_64(bm64 U, bm64 T, double wcum)
        pair[bm128, double] scan_rnd_128(bm128 U, bm128 T, double wcum)
        pair[bm192, double] scan_rnd_192(bm192 U, bm192 T, double wcum)
        pair[bm256, double] scan_rnd_256(bm256 U, bm256 T, double wcum)


cdef class CandidateRestrictedScore:

    cdef CppCandidateRestrictedScore * thisptr;

    def __cinit__(self, *, score_array, C, K, cc_cache_size, cc_tolerance,
                  pruning_eps, logfile="", silent):

        cdef double[:, ::1] memview_score_array
        memview_score_array = score_array

        cdef int[:, ::1] memview_C
        memview_C = C

        self.thisptr = new CppCandidateRestrictedScore(& memview_score_array[0, 0],
                                                       & memview_C[0, 0],
                                                       score_array.shape[0],
                                                       K, cc_cache_size,
                                                       cc_tolerance, pruning_eps,
                                                       logfile.encode('utf-8'),
                                                       silent
                                                       )

    def __dealloc__(self):
       del self.thisptr

    def sum(self, int v, bm32 U, bm32 T=0, isum=False):
        if T == 0:
            return self.thisptr.sum(v, U)
        return self.thisptr.sum(v, U, T, isum)

    def sample_pset(self, int v, bm32 U, bm32 T, double wcum):
        # wcum needs to be scaled by corresponding sum
        if T > 0:
            return self.thisptr.sample_pset(v, U, T, wcum)
        return self.thisptr.sample_pset(v, U, wcum)


cdef class CandidateComplementScore:

    cdef vector[isum_ptr] * isums
    cdef int k, d
    cdef int[:, :] t_ub
    cdef bm64[:, ::1] psets_memview_n_leq_64
    cdef bm64[:, :, ::1] psets_memview_n_g_64
    cdef double[:, ::1] w_memview
    cdef object localscore

    def __cinit__(self, *, localscore, C, d, eps):

        n = len(C)
        self.d = d
        self.k = (n-1)//64+1
        self.localscore = localscore

        self.t_ub = np.zeros(shape=(n, n), dtype=np.int32)
        for u in range(1, n+1):
            for t in range(1, u+1):
                self.t_ub[u-1][t-1] = self._n_valids_ub(u, t)

        self.isums = new vector[isum_ptr]()

        cdef bm64[:, ::1] psets_memview
        cdef double[::1] scores_memview
        cdef int v
        for v in range(n):
            psets_memview, scores_memview = localscore.complement_psets_and_scores(v, C, d)
            self.isums.push_back(new IntersectSums(& scores_memview[0],
                                                   & psets_memview[0, 0],
                                                   scores_memview.shape[0],
                                                   self.k,
                                                   eps))

    def _n_valids_ub(self, u, t):
        n = 0
        for k in range(self.d+1):
            n += comb(u, k) - comb(u - t, k)
        return n

    def sum(self, v, U, T, w=-float("inf")):

        cdef bm128 U128
        cdef bm128 T128
        cdef bm192 U192
        cdef bm192 T192
        cdef bm256 U256
        cdef bm256 T256

        if self.k == 1:
            return self.isums[0][v].scan_sum_64(w,
                                                bm_to_np64(bm(U))[0],
                                                bm_to_np64(bm(T))[0],
                                                self.t_ub[len(U)-1][len(T)-1])
        if self.k == 2:
            U = bm_to_np64(bm(U), k=self.k)
            T = bm_to_np64(bm(T), k=self.k)
            U128 = [U[0], U[1]]
            T128 = [T[0], T[1]]
            return self.isums[0][v].scan_sum_128(w, U128, T128,
                                                 self.t_ub[len(U)-1][len(T)-1])
        if self.k == 3:
            U_bm = bm_to_np64(bm(U), k=self.k)
            T_bm = bm_to_np64(bm(T), k=self.k)
            U192 = [U_bm[0], U_bm[1], U_bm[2]]
            T192 = [T_bm[0], T_bm[1], T_bm[2]]
            return self.isums[0][v].scan_sum_192(w, U192, T192,
                                                 self.t_ub[len(U)-1][len(T)-1])
        if self.k == 4:
            U = bm_to_np64(bm(U), k=self.k)
            T = bm_to_np64(bm(T), k=self.k)
            U256 = [U[0], U[1], U[2], U[3]]
            T256 = [T[0], T[1], T[2], T[3]]
            return self.isums[0][v].scan_sum_256(w, U256, T256,
                                                 self.t_ub[len(U)-1][len(T)-1])

    def sample_pset(self, v, U, T, wcum):

        cdef bm128 U128
        cdef bm128 T128
        cdef bm192 U192
        cdef bm192 T192
        cdef bm256 U256
        cdef bm256 T256

        U = bm_to_np64(bm(U), k=self.k)
        T = bm_to_np64(bm(T), k=self.k)

        if self.k == 1:
            pset, score = self.isums[0][v].scan_rnd_64(U, T, wcum)
            pset = bm_to_ints(pset)

        if self.k == 2:
            U128 = [U[0], U[1]]
            T128 = [T[0], T[1]]
            pset, score = self.isums[0][v].scan_rnd_128(U128, T128, wcum)
            pset = bm_to_ints(np64_to_bm(np.array([pset["s1"], pset["s2"]],
                                                  dtype=np.uint64)))

        if self.k == 3:
            U192 = [U[0], U[1], U[2]]
            T192 = [T[0], T[1], T[2]]
            pset, score = self.isums[0][v].scan_rnd_192(U192, T192, wcum)
            pset = bm_to_ints(np64_to_bm(np.array([pset["s1"], pset["s2"], pset["s3"]],
                                                  dtype=np.uint64)))

        if self.k == 4:
            U256 = [U[0], U[1], U[2], U[3]]
            T256 = [T[0], T[1], T[2], T[3]]
            pset, score = self.isums[0][v].scan_rnd_256(U256, T256, wcum)
            pset = bm_to_ints(np64_to_bm(np.array([pset["s1"], pset["s2"], pset["s3"], pset["s4"]],
                                                  dtype=np.uint64)))

        return pset, score
