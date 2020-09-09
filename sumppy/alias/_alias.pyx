from libcpp.vector cimport vector

cdef extern from "discrete_random_variable.cpp":
    cdef cppclass discrete_random_variable:
        discrete_random_variable(vector[int], vector[double])
        vector[int] vals
        vector[double] probs
        int sample()

cdef class Alias:
    cdef discrete_random_variable *thisptr
    def __cinit__(self, vector[int] vals, vector[double] probs):
        self.thisptr = new discrete_random_variable(vals, probs)
    def __dealloc__(self):
        del self.thisptr
    def sample(self):
        return self.thisptr.sample()
