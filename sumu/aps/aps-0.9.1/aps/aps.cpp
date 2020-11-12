#include "aps.h"
#include "types.h"

namespace aps {

template <typename T>
Array<Array<T>> orderModularAPS_bruteforce(const Array<Array<T>>& w, bool test);
template <typename T>
Array<Array<T>> orderModularAPS_simple(const Array<Array<T>>& w, bool test);
template <typename T>
Array<Array<T>> orderModularAPS_singlethread(const Array<Array<T>>& w, bool test);

template <typename T>
const APSFuncList<T>& getOrderModularAPSFuncs() {
	static APSFuncList<T> funcs;
	if(!funcs.empty()) {
		return funcs;
	}

	funcs.emplace_back("singlethread", orderModularAPS_singlethread<T>);
	funcs.emplace_back("simple", orderModularAPS_simple<T>);
	funcs.emplace_back("bruteforce", orderModularAPS_bruteforce<T>);

	return funcs;
}

template <typename T>
Array<Array<T>> modularAPS_bruteforce(const Array<Array<T>>& w, bool test);
template <typename T>
Array<Array<T>> modularAPS_simple(const Array<Array<T>>& w, bool test);

template <typename T>
const APSFuncList<T>& getModularAPSFuncs() {
	static APSFuncList<T> funcs;
	if(!funcs.empty()) {
		return funcs;
	}

	funcs.emplace_back("simple", modularAPS_simple<T>);
	funcs.emplace_back("bruteforce", modularAPS_bruteforce<T>);

	return funcs;
}

#undef APS_FOR_EACH_NUMBER_TYPE_TEMPLATE
#define APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(T) \
	template const APSFuncList<T>& getModularAPSFuncs<T>(); \
	template const APSFuncList<T>& getOrderModularAPSFuncs<T>();
APS_FOR_EACH_NUMBER_TYPE

}
