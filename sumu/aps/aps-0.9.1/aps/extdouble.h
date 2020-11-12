#pragma once

#include "common.h"
#include "logdouble.h"

#include <cmath>
#include <limits>
#include <tuple>

namespace aps {

// Nonnegative real number represented by a normalized double-precision number
// and an 64-bit integer exponent (e-base)
struct ExtDouble {
	double coef;
	int64_t exp;
	
	static constexpr int64_t MinExp = std::numeric_limits<int64_t>::lowest() / 4;
	static constexpr int64_t MaxExp = std::numeric_limits<int64_t>::max() / 4;
	static constexpr double MinExpD = (double)MinExp;
	static constexpr double MaxExpD = (double)MaxExp;
	static constexpr double E = 2.718281828459045;
	static constexpr double E2 = E * E;
	static constexpr double E4 = E2 * E2;
	static constexpr double E8 = E4 * E4;
	static constexpr double E16 = E8 * E8;
	static constexpr double E32 = E16 * E16;
	static constexpr double InvE = 1.0 / E;
	static constexpr double InvE2 = InvE * InvE;
	static constexpr double InvE4 = InvE2 * InvE2;
	static constexpr double InvE8 = InvE4 * InvE4;
	static constexpr double InvE16 = InvE8 * InvE8;
	static constexpr double InvE32 = InvE16 * InvE16;
	static constexpr double InvE3 = InvE2 * InvE;
	static constexpr double InvE7 = InvE3 * InvE3 * InvE;
	static constexpr double InvE15 = InvE7 * InvE7 * InvE;
	static constexpr double InvE31 = InvE15 * InvE15 * InvE;

	ExtDouble() {}
	explicit ExtDouble(LogDouble val) {
		double expD = std::floor(val.log);
		if(expD < MinExpD) {
			coef = 0.0;
			exp = 0;
		} else {
			if(expD > MaxExpD) {
				fail("ExtDouble: overflow");
			}
			exp = (int64_t)expD;
			coef = std::exp(val.log - expD);
		}
	}
	explicit ExtDouble(double val) :
		ExtDouble((LogDouble)val)
	{}
	static ExtDouble zero() {
		ExtDouble ret;
		ret.coef = 0.0;
		ret.exp = 0;
		return ret;
	}
	static ExtDouble one() {
		ExtDouble ret;
		ret.coef = 1.0;
		ret.exp = 0;
		return ret;
	}

	explicit operator LogDouble() const {
		if(coef == 0.0) {
			return LogDouble(0.0);
		} else {
			return LogDouble::fromLog((double)exp + std::log(coef));
		}
	}
	explicit operator double() const {
		return (double)(LogDouble)*this;
	}
};

inline ExtDouble operator*(ExtDouble a, ExtDouble b) {
	ExtDouble ret;
	ret.coef = a.coef * b.coef;
	if(ret.coef == 0.0) {
		ret.exp = 0;
	} else {
		ret.exp = a.exp + b.exp;
		if(ret.exp < ExtDouble::MinExp) {
			ret.coef = 0.0;
			ret.exp = 0;
		} else {
			if(ret.coef >= ExtDouble::E) {
				++ret.exp;
				ret.coef *= ExtDouble::InvE;
			}
			if(ret.exp > ExtDouble::MaxExp) {
				fail("ExtDouble: overflow");
			}
		}
	}
	return ret;
}

inline double invExp_(double val, uint64_t exp) {
	exp = std::min(exp, (uint64_t)63);
	if(exp & 1) val *= ExtDouble::InvE;
	if(exp & 2) val *= ExtDouble::InvE2;
	if(exp & 4) val *= ExtDouble::InvE4;
	if(exp & 8) val *= ExtDouble::InvE8;
	if(exp & 16) val *= ExtDouble::InvE16;
	if(exp & 32) val *= ExtDouble::InvE32;
	return val;
}

inline ExtDouble operator+(ExtDouble a, ExtDouble b) {
	if(a.coef == 0.0) {
		return b;
	}
	if(b.coef == 0.0) {
		return a;
	}
	if(a.exp < b.exp) {
		std::swap(a, b);
	}

	a.coef += invExp_(b.coef, a.exp - b.exp);

	if(a.coef >= ExtDouble::E) {
		++a.exp;
		a.coef *= ExtDouble::InvE;
		if(a.exp > ExtDouble::MaxExp) {
			fail("ExtDouble: overflow");
		}
	}

	return a;
}
inline ExtDouble nonnegativeSubtraction(ExtDouble a, ExtDouble b) {
	if(b.coef == 0.0) {
		return a;
	}
	if(a.coef == 0.0 || a.exp < b.exp) {
		return ExtDouble::zero();
	}

	a.coef -= invExp_(b.coef, a.exp - b.exp);
	if(a.coef <= 0.0) {
		return ExtDouble::zero();
	}

	if(a.coef < ExtDouble::InvE31) {
		a.coef *= ExtDouble::E32;
		a.exp -= 32;
	}
	if(a.coef < ExtDouble::InvE15) {
		a.coef *= ExtDouble::E16;
		a.exp -= 16;
	}
	if(a.coef < ExtDouble::InvE7) {
		a.coef *= ExtDouble::E8;
		a.exp -= 8;
	}
	if(a.coef < ExtDouble::InvE3) {
		a.coef *= ExtDouble::E4;
		a.exp -= 4;
	}
	if(a.coef < ExtDouble::InvE) {
		a.coef *= ExtDouble::E2;
		a.exp -= 2;
	}
	if(a.coef < 1.0) {
		a.coef *= ExtDouble::E;
		a.exp -= 1;
	}
	if(a.coef < 1.0) {
		a.coef = 1.0;
	}

	if(a.exp < ExtDouble::MinExp) {
		return ExtDouble::zero();
	} else {
		return a;
	}
}

inline ExtDouble& operator*=(ExtDouble& a, ExtDouble b) {
	a = a * b;
	return a;
}
inline ExtDouble& operator+=(ExtDouble& a, ExtDouble b) {
	a = a + b;
	return a;
}

}
