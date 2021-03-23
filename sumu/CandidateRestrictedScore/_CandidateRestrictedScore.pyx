from libcpp.vector cimport vector
from libc.stdint cimport uint64_t as bm64
from libc.stdint cimport uint32_t as bm32
from libcpp cimport bool
from libcpp.string cimport string


cdef extern from "CandidateRestrictedScore.hpp":

    cdef cppclass CppCandidateRestrictedScore "CandidateRestrictedScore":

        CppCandidateRestrictedScore(double* score_array, int* C, int n, int K,
                                    int cc_limit, double cc_tol, double isum_tol,
                                    string logfile,
                                    bool silent
                                    )
        double sum(int v, bm32 U, bm32 T)
        double sum(int v, bm32 U)
        double test_sum(int v, bm32 U, bm32 T)
        double get_cc(int v, bm64 key)
        double get_tau_simple(int v, bm32 U)
        bm32 sample_pset(int v, bm32 U, bm32 T, double wcum)
        bm32 sample_pset(int v, bm32 U, double wcum)
        int n
        int K
        double** score_array
        int** C
        double m_cc_tol
        # void read(int * data, int m, int n)
        # void set_ess(double val)
        # double cliq(int * var, int d)
        # double fami(int i, int * par, int d)

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

    @property
    def cc_tol(self):
        return self.thisptr.m_cc_tol

    def sum(self, int v, bm32 U, bm32 T=0):
        if T == 0:
            return self.thisptr.sum(v, U)
        return self.thisptr.sum(v, U, T)

    def testi_sum(self, int v, bm32 U, bm32 T):
        return self.thisptr.test_sum(v, U, T)

    def get_cc(self, int v, bm64 key):
        return self.thisptr.get_cc(v, key)

    def get_tau_simple(self, int v, bm32 U):
        return self.thisptr.get_tau_simple(v, U)

    def sample_pset(self, int v, bm32 U, bm32 T, double wcum):
        if T > 0:
            return self.thisptr.sample_pset(v, U, T, self.thisptr.sum(v, U, T) + wcum)
        return self.thisptr.sample_pset(v, U, self.thisptr.sum(v, U) + wcum)

    # def tau(self):
    #     return np.ctypeslib.as_array(self.thisptr.get_tau(), shape=(n, 2**K))
        #return np.asarray(<np.float[:n, :2**K]> self.thisptr.get_tau())
        # return self.thisptr.get_tau()

    # def read(self, data):
    #     cdef int[:, ::1] memview_data
    #     memview_data = data
    #     return self.thisptr.read(& memview_data[0, 0],
    #                              memview_data.shape[0],
    #                              memview_data.shape[1])

    # def set_ess(self, value):
    #     self.thisptr.set_ess(value)

    # def cliq(self, nodes):
    #     cdef int[::1] memview_nodes
    #     memview_nodes = nodes
    #     return self.thisptr.cliq(& memview_nodes[0],
    #                              memview_nodes.shape[0])

    # def fami(self, node, pset):
    #     cdef int[::1] memview_pset = pset
    #     return self.thisptr.fami(node,
    #                              & memview_pset[0],
    #                              memview_pset.shape[0])
