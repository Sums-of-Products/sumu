#pragma once

#include "common.h"
#include "extdouble.h"
#include "logdouble.h"

#include <algorithm>

namespace aps {

// To use, redefine macro APS_FOR_EACH_NUMBER_TYPE_TEMPLATE
#define APS_FOR_EACH_NUMBER_TYPE \
	APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(LogDouble) \
	APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(ExtDouble) \
	APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(double) \
	APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(uint64_t)

inline uint64_t nonnegativeSubtraction(uint64_t a, uint64_t b) {
	return a - b;
}
inline double nonnegativeSubtraction(double a, double b) {
	return std::max(a - b, 0.0);
}

template <typename T>
struct GetConstant_ {};

template <>
struct GetConstant_<LogDouble> {
	static LogDouble zero() {
		return LogDouble::zero();
	}
	static LogDouble one() {
		return LogDouble::one();
	}
};
template <>
struct GetConstant_<ExtDouble> {
	static ExtDouble zero() {
		return ExtDouble::zero();
	}
	static ExtDouble one() {
		return ExtDouble::one();
	}
};
template <>
struct GetConstant_<double> {
	static double zero() {
		return 0.0;
	}
	static double one() {
		return 1.0;
	}
};
template <>
struct GetConstant_<uint64_t> {
	static uint64_t zero() {
		return 0;
	}
	static uint64_t one() {
		return 1;
	}
};

template <typename T>
T getZero() {
	return GetConstant_<T>::zero();
}
template <typename T>
T getOne() {
	return GetConstant_<T>::one();
}

}
