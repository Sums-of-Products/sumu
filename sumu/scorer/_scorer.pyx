"""
import numpy as np
import scorer
a = scorer.BDeu()
data = np.ones((50,10), dtype=np.int32)
a.read(data)
a.set_ess(10)
a.cliq(np.array([1,2,3], dtype=np.int32))
a.fami(4, np.array([1,2,3,7,9,10], dtype=np.int32))
"""

from libcpp.vector cimport vector


cdef extern from "BDeu.hpp":

    cdef cppclass CppBDeu "BDeu":
        BDeu()
        int m
        int n
        void read(int * data, int m, int n)
        void set_ess(double val)
        double cliq(int * var, int d)
        double fami(int i, int * par, int d)

cdef class BDeu:

    cdef CppBDeu * thisptr

    def __cinit__(self):
        self.thisptr = new CppBDeu()

    def __dealloc__(self):
        del self.thisptr

    def read(self, data):
        cdef int[:, ::1] memview_data
        memview_data = data
        return self.thisptr.read(& memview_data[0, 0],
                                 memview_data.shape[0],
                                 memview_data.shape[1])

    def set_ess(self, value):
        self.thisptr.set_ess(value)

    def cliq(self, nodes):
        cdef int[::1] memview_nodes
        memview_nodes = nodes
        return self.thisptr.cliq(& memview_nodes[0],
                                 memview_nodes.shape[0])

    def fami(self, node, pset):
        cdef int[::1] memview_pset = pset
        return self.thisptr.fami(node,
                                 & memview_pset[0],
                                 memview_pset.shape[0])
