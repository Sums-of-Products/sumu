from libcpp.vector cimport vector

cdef extern from "zeta_transform.h":
    vector[double] zeta_transform_vector(vector[double] arg)
    void zeta_transform_array_inplace(double * a, int n)
