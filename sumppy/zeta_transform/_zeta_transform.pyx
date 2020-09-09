# distutils: language=c++

from libcpp.vector cimport vector

cdef extern from "zeta_transform.h":
    vector[double] from_list(vector[double] arg)

def solve(arg):
    return from_list(arg)
