# distutils: language=c++

cimport cpp_zeta_transform

def zeta_transform_vector(arg):
    return cpp_zeta_transform.zeta_transform_vector(arg)

def zeta_transform_array_inplace(a):
    cdef double[::1] memview_a
    memview_a = a
    return cpp_zeta_transform.zeta_transform_array_inplace(& memview_a[0], len(a))
