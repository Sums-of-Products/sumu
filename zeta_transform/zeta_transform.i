/*  Example of wrapping cos function from math.h using SWIG. */

%module zeta_transform
%{
    /* the resulting C file should be built as a python extension */
    #define SWIG_FILE_WITH_INIT
    /*  Includes the header in the wrapper code */
    #include "zeta_transform.h"
%}
/*  Parse the header file to generate wrappers */

%include "std_vector.i"
namespace std {
  %template(VecDouble) vector<double>;
}

%include "zeta_transform.h"
