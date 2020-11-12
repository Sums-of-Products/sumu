#pragma once

#include "common.h"

#include <array>
#include <utility>

namespace aps {

template <size_t N, typename F>
struct SelectConstant_ {
	template <size_t X>
	struct ValueWrapper {
		static const size_t Val = X;
	};

	template <size_t X>
	static void run(F& f) {
		f(ValueWrapper<X>());
	}

	typedef void(*RunPtr)(F&);
	typedef std::array<RunPtr, N> RunPtrArray;
	static const RunPtrArray runs;

	template <size_t... I>
	static RunPtrArray initRuns_(std::index_sequence<I...> seq) {
		return {run<I>...};
	}
	static RunPtrArray initRuns() {
		return initRuns_(std::make_index_sequence<N>());
	}

	static void selectRun(size_t x, F& f) {
		runs[x](f);
	}
};
template <size_t N, typename F>
const typename SelectConstant_<N, F>::RunPtrArray SelectConstant_<N, F>::runs = SelectConstant_<N, F>::initRuns();

// Passes to f a struct with constant size_t element Val that is equal to x.
// Assumes that x < N.
template <size_t N, typename F>
void selectConstant(size_t x, F f) {
	SelectConstant_<N, F>::selectRun(x, f);
}

}
