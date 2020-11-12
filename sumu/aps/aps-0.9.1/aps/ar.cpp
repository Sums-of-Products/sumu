#include "ar.h"
#include "types.h"

namespace aps {

template <typename T>
Array<Array<T>> orderModularAR_bruteforce(const Array<Array<T>>& w, bool test);
template <typename T>
Array<Array<T>> orderModularAR_simple(const Array<Array<T>>& w, bool test);

template <typename T>
const ARFuncList<T>& getOrderModularARFuncs() {
	static ARFuncList<T> funcs;
	if(!funcs.empty()) {
		return funcs;
	}

	funcs.emplace_back("simple", orderModularAR_simple<T>);
	funcs.emplace_back("bruteforce", orderModularAR_bruteforce<T>);

	return funcs;
}

template <typename T>
Array<Array<T>> modularAR_bruteforce(const Array<Array<T>>& w, bool test);
template <typename T>
Array<Array<T>> modularAR_simple(const Array<Array<T>>& w, bool test);

template <typename T>
const ARFuncList<T>& getModularARFuncs() {
	static ARFuncList<T> funcs;
	if(!funcs.empty()) {
		return funcs;
	}

	funcs.emplace_back("simple", modularAR_simple<T>);
	funcs.emplace_back("bruteforce", modularAR_bruteforce<T>);

	return funcs;
}

#undef APS_FOR_EACH_NUMBER_TYPE_TEMPLATE
#define APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(T) \
	template const ARFuncList<T>& getModularARFuncs<T>(); \
	template const ARFuncList<T>& getOrderModularARFuncs<T>();
APS_FOR_EACH_NUMBER_TYPE

}
