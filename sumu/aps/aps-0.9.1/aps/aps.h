#pragma once

#include "array.h"

#include <vector>

namespace aps {

// Arguments:
//   - Array of weights (outer indexed by vertex, inner indexed by bitvector
//     of the parent set, in packed form so size is 2^(#verts - 1)).
//   - Boolean signifying whether this is a test (if true, the solution may
//     reject the case if solving it would be slow)
// Return value:
//   - The all-parent-set weights, or an empty array if the solution rejects
//     the case (if the size or weight type is not supported)
template <typename T>
using APSFunc = Array<Array<T>> (*)(const Array<Array<T>>&, bool);

// Method name-function pairs in the order of preference (most preferred first).
template <typename T>
using APSFuncList = std::vector<std::pair<std::string, APSFunc<T>>>;

template <typename T>
const APSFuncList<T>& getOrderModularAPSFuncs();

template <typename T>
const APSFuncList<T>& getModularAPSFuncs();

}
