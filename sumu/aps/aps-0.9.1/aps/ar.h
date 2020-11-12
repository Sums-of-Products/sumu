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
//   - In index [i][j], the total weight of the DAGs where j is an ancestor of i
//     or an empty array if the solution rejects the case (if the size or weight
//     type is not supported)
template <typename T>
using ARFunc = Array<Array<T>>(*)(const Array<Array<T>>&, bool);

// Method name-function pairs in the order of preference (most preferred first).
template <typename T>
using ARFuncList = std::vector<std::pair<std::string, ARFunc<T>>>;

template <typename T>
const ARFuncList<T>& getOrderModularARFuncs();

template <typename T>
const ARFuncList<T>& getModularARFuncs();

}
