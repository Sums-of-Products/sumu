import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdint cimport uint64_t as bm64
from libc.stdint cimport uint32_t as bm32
from .utils.math_utils import subsets
from .utils.bitmap import bm

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
        int m
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
        double test(int v, int pset)

cdef class BDeu:

    cdef CppBDeu * thisptr
    cdef int maxid

    def __cinit__(self, *, data, ess, maxid):
        self.thisptr = new CppBDeu()
        self.maxid = maxid
        self._read(data)
        self._set_ess(ess)

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

    def _set_ess(self, value):
        self.thisptr.set_ess(value)

    @cython.boundscheck(False)
    def cliq(self, nodes):
        cdef int[::1] memview_nodes
        memview_nodes = nodes
        return self.thisptr.cliq(& memview_nodes[0],
                                 memview_nodes.shape[0])

    def local(self, node, pset):
        node = np.int32(node)
        pset = pset.astype(np.int32)
        vset = np.append(pset, node)
        return self.cliq(vset) - self.cliq(pset)

    @cython.boundscheck(False)
    def fami(self, node, pset):
        cdef int[::1] memview_pset = pset
        return self.thisptr.fami(node,
                                 & memview_pset[0],
                                 memview_pset.shape[0])

    @cython.boundscheck(False)
    def all_candidate_restricted_scores(self, C):
        cdef int[::1] memview_pset
        cdef int[:, ::1] memview_C
        cdef int v, i, n, K
        n = len(C)
        K = len(C[0])
        cdef np.ndarray score_array = np.full((n, 2**K), -np.inf)
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

    def test(self, int v, int pset):
        return self.thisptr.fscores[v][pset].weight
