import time

import numpy as np

cimport numpy as np
from libc.stdint cimport uint32_t as bm32
from libc.stdint cimport uint64_t as bm64
from libcpp.vector cimport vector

from .utils.bitmap import bm, bm_to_np64
from .utils.math_utils import comb, subsets

cimport cython

# A bit of a hack to make BGe available at sumu.scorer.BGe
# Unnecessary after BGe is implemented in C++ similarly to BDeu

from .scores.bge import BGe


cdef extern from "../bitmap/bitmap.hpp":
    int count_32(bm32 bitmap)

cdef extern from "Wsets.hpp":
    cdef struct wset:
        bm64 set
        double weight

cdef extern from "BDeu.hpp":

    cdef cppclass CppBDeu "BDeu":
        BDeu()
        int n
        vector[wset] * fscores
        double fscore(int i, int * Y, int lY)
        void read(int * data, int m, int n)
        void set_ess(double val)
        double cliq(int * var, int d)
        double fami(int i, int * par, int d)
        double fami(int i, int * par, int K, int d)
        void clear_fami(int i)
        void clear_cliqc()

cdef class BDeu:

    cdef CppBDeu * thisptr
    cdef int maxid

    cdef float t

    def __cinit__(self, *, data, ess, maxid):
        self.thisptr = new CppBDeu()
        self.maxid = maxid
        self._read(data)
        self.thisptr.set_ess(ess)

    def __dealloc__(self):
        del self.thisptr

    def clear_cache(self):
        self.thisptr.clear_cliqc()

    def _read(self, data):
        # NOTE: BDeu.hpp uses int type for data, which requires data
        # to be of type np.int32. Could use uint8_t perhaps to allow
        # np.int8
        cdef int[:, ::1] memview_data
        memview_data = data
        return self.thisptr.read(& memview_data[0, 0],
                                 memview_data.shape[0],
                                 memview_data.shape[1])

    @cython.boundscheck(False)
    def _cliq(self, nodes):
        cdef int[::1] memview_nodes
        memview_nodes = nodes
        return self.thisptr.cliq(& memview_nodes[0],
                                 memview_nodes.shape[0])

    def local(self, node, pset):
        # NOTE: This does not do any caching
        node = np.int32(node)
        pset = pset.astype(np.int32)
        vset = np.append(pset, node)
        return self._cliq(vset) - self._cliq(pset)


    @cython.boundscheck(False)
    def complement_psets_and_scores(self, int v, C, d):
        # This needs to be replicated in every score class, e.g., BGe
        cdef int[::1] memview_C_all
        cdef int[::1] memview_pset
        cdef int i

        n = self.thisptr.n
        K = len(C[0])
        k = (n - 1) // 64 + 1

        scores = np.empty((sum(comb(n-1, k) - comb(K, k) for k in range(d+1))))

        C_all = np.array([u for u in range(n) if u != v], dtype=np.int32)
        memview_C_all = C_all

        # Compute all scores for all psets with indegree <= d.
        self.thisptr.fami(v, & memview_C_all[0], n-1, d)

        # Then fetch scores for those psets that are not subsets of C[v].
        # For some cython specific reason this needs to be expanded to list:
        pset_tuple = list(filter(lambda ss: not set(ss).issubset(C[v]),
                                 subsets([u for u in C if u != v], 1, d)))
        pset_len = np.array(list(map(len, pset_tuple)), dtype=np.int32)

        t = time.time()
        pset_bm = list(map(lambda pset: bm_to_np64(bm(set(pset)), k), pset_tuple))
        self.t += time.time() - t

        pset = list(map(lambda pset: np.array(pset - 1*(pset > v), dtype=np.int32),
                        map(lambda pset: np.array(pset, dtype=np.int32), pset_tuple)))
        for i in range(scores.shape[0]):
            memview_pset = pset[i]
            scores[i] = self.thisptr.fscore(v, & memview_pset[0], pset_len[i])
        # clear cache
        self.thisptr.clear_fami(v)
        return np.array(pset_bm), scores, pset_len


    @cython.boundscheck(False)
    def candidate_score_array(self, C):
        cdef int[::1] memview_pset
        cdef int[:, ::1] memview_C
        cdef int v, i, n, K
        n = len(C)
        K = len(C[0])
        cdef np.ndarray score_array = np.full((n, int(2**K)), -np.inf)
        memview_C = C
        for v in np.arange(memview_C.shape[0], dtype=np.int32):
            self.thisptr.fami(v, & memview_C[v, 0],
                              memview_C.shape[1],
                              [memview_C.shape[1]
                               if self.maxid == -1
                               else self.maxid][0])

            if self.maxid == -1:
                for i in np.arange(2**K, dtype=np.int32):
                    score_array[v][i] = self.thisptr.fscores[v][i].weight
            else:
                for pset in subsets(range(K), 0, self.maxid):
                    pset_bm = bm(pset)
                    pset = np.array(pset, dtype=np.int32)
                    memview_pset = pset
                    pset_l = len(pset)
                    score_array[v][pset_bm] = self.thisptr.fscore(v,
                                                                  & memview_pset[0],
                                                                  pset_l)

            self.thisptr.clear_fami(v)
        return score_array
