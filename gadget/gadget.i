%module gadget
%{
    /* the resulting C file should be built as a python extension */
    #define SWIG_FILE_WITH_INIT
    /*  Includes the header in the wrapper code */
    #include "gadget.h"
%}
/*  Parse the header file to generate wrappers */

%include "numpy.i"
%include <stdint.i>

%include "std_vector.i"
%include <std_pair.i>
namespace std {
  %template(VecInt) vector<int>;
  %template() pair<double,vector<int>>;
   //%template() vector<uint64_t>;
}

%init %{
import_array();
%}

%apply (uint64_t* INPLACE_ARRAY1, int DIM1) {(uint64_t* ivec, int ni)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* dvec, int nd)}

%apply (uint64_t* INPLACE_ARRAY2, int DIM1, int DIM2) {(uint64_t* ivec, int d1, int d2)}

%include "gadget.h"
