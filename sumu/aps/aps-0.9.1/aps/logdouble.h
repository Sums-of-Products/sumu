#pragma once

#include "common.h"

#include <cmath>
#include <limits>

namespace aps {

// Nonnegative real number represented as its logarithm
struct LogDouble {
	double log;

	LogDouble() {}
	explicit LogDouble(double val) :
		log(std::log(val))
	{ }
	static LogDouble fromLog(double log) {
		LogDouble ret;
		ret.log = log;
		return ret;
	}
	static LogDouble zero() {
		return LogDouble::fromLog(-std::numeric_limits<double>::infinity());
	}
	static LogDouble one() {
		return LogDouble::fromLog(0.0);
	}

	explicit operator double() const {
		return std::exp(log);
	}
};

inline LogDouble operator*(LogDouble a, LogDouble b) {
	return LogDouble::fromLog(a.log + b.log);
}
inline LogDouble operator/(LogDouble a, LogDouble b) {
	return LogDouble::fromLog(a.log - b.log);
}
inline LogDouble operator+(LogDouble a, LogDouble b) {
	if(a.log == -std::numeric_limits<double>::infinity()) {
		return b;
	}
	if(b.log == -std::numeric_limits<double>::infinity()) {
		return a;
	}
	if(a.log > b.log) {
		return LogDouble::fromLog(a.log + std::log(1 + std::exp(b.log - a.log)));
	} else {
		return LogDouble::fromLog(b.log + std::log(1 + std::exp(a.log - b.log)));
	}
}
inline LogDouble nonnegativeSubtraction(LogDouble a, LogDouble b) {
	if(b.log == -std::numeric_limits<double>::infinity()) {
		return a;
	}
	if(a.log <= b.log) {
		return LogDouble::zero();
	}
	return LogDouble::fromLog(a.log + std::log(1 - std::exp(b.log - a.log)));
}

inline LogDouble& operator*=(LogDouble& a, LogDouble b) {
	a = a * b;
	return a;
}
inline LogDouble& operator+=(LogDouble& a, LogDouble b) {
	a = a + b;
	return a;
}

}
