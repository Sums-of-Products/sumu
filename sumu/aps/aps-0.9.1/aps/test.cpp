#include "aps.h"
#include "ar.h"
#include "array.h"
#include "conf.h"
#include "logdouble.h"
#include "types.h"

#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace aps {

namespace {

template <typename A, typename B>
void check_equal(const A& a, const B& b, const char* file, int line) {
	if(!(a == b)) {
		fail("CHECK_EQUAL(", a, ", ", b, ") failed in ", file, ":", line);
	}
}
#define CHECK_EQUAL(a, b) check_equal((a), (b), __FILE__, __LINE__)

void check_true(bool val, const char* file, int line) {
	if(!val) {
		fail("CHECK_TRUE failed in ", file, ":", line);
	}
}
#define CHECK_TRUE(v) check_true((v), __FILE__, __LINE__)

std::mt19937 rng(std::random_device{}());

template <typename T>
void testNumber() {
	for(int i = 0; i < 1000; ++i) {
		double a = std::uniform_real_distribution<double>(0.0, 10.0)(rng);
		double b = std::uniform_real_distribution<double>(0.0, 10.0)(rng);
		if(!std::uniform_int_distribution<int>(0, 9)(rng)) {
			a = 0.0;
		}
		if(!std::uniform_int_distribution<int>(0, 9)(rng)) {
			b = 0.0;
		}
		if(!std::uniform_int_distribution<int>(0, 9)(rng)) {
			b = a;
		}
		T a2 = (T)a;
		T b2 = (T)b;

		{
			double c = a + b;
			T c2 = a2 + b2;
			double d = std::abs(c - (double)c2);

			CHECK_TRUE(d < 1e-10);
		}

		{
			double c = a * b;
			T c2 = a2 * b2;
			double d = std::abs(c - (double)c2);

			CHECK_TRUE(d < 1e-10);
		}

		{
			double c = nonnegativeSubtraction(a, b);
			T c2 = nonnegativeSubtraction(a2, b2);

			double d = std::abs(c - (double)c2);
			CHECK_TRUE(d < 1e-10);
		}
	}
}

void checkCanonical(ExtDouble x) {
	CHECK_TRUE(
		(x.coef == 0.0 && x.exp == 0) ||
		(
			x.coef >= 1.0 - 1e-6 &&
			x.coef < (1.0 + 1e-6) * ExtDouble::E &&
			x.exp >= ExtDouble::MinExp &&
			x.exp <= ExtDouble::MaxExp
		)
	);
}

void crossTestExtLogDouble() {
	for(int t = 0; t < 100; ++t) {
		LogDouble a = LogDouble::fromLog(
			std::uniform_real_distribution<double>(-1e6, 1e6)(rng)
		);
		LogDouble b = LogDouble::fromLog(
			std::uniform_real_distribution<double>(-1e6, 1e6)(rng)
		);
		ExtDouble a2 = (ExtDouble)a;
		ExtDouble b2 = (ExtDouble)b;

		{
			LogDouble c = a + b;
			ExtDouble c2 = a2 + b2;
			checkCanonical(c2);
			CHECK_TRUE(std::abs(c.log - LogDouble(c2).log) < 1e-6);
		}
		{
			LogDouble c = nonnegativeSubtraction(a, b);
			ExtDouble c2 = nonnegativeSubtraction(a2, b2);
			checkCanonical(c2);
			if(c.log != LogDouble(c2).log) {
				CHECK_TRUE(std::abs(c.log - LogDouble(c2).log) < 1e-6);
			}
		}
		{
			LogDouble c = a * b;
			ExtDouble c2 = a2 * b2;
			checkCanonical(c2);
			CHECK_TRUE(std::abs(c.log - LogDouble(c2).log) < 1e-6);
		}
	}
}

bool isValidWeightValue(uint64_t) {
	return true;
}
bool isValidWeightValue(double x) {
	return std::isfinite(x);
}
bool isValidWeightValue(LogDouble x) {
	return std::isfinite(x.log) || x.log == -std::numeric_limits<double>::infinity();
}
bool isValidWeightValue(ExtDouble x) {
	return
		(x.coef == 0.0 && x.exp == 0) ||
		(
			x.coef >= 1.0 - 1e-6 &&
			x.coef < (1.0 + 1e-6) * ExtDouble::E &&
			x.exp >= ExtDouble::MinExp &&
			x.exp <= ExtDouble::MaxExp
		);
}

void genWeightValue(uint64_t& ret) {
	ret = std::uniform_int_distribution<uint64_t>()(rng);
}
void genWeightValue(double& ret) {
	ret = std::uniform_real_distribution<double>(1.0, 2.0)(rng);
}
void genWeightValue(LogDouble& ret) {
	ret = (LogDouble)std::uniform_real_distribution<double>(1.0, 2.0)(rng);
}
void genWeightValue(ExtDouble& ret) {
	ret = (ExtDouble)std::uniform_real_distribution<double>(1.0, 2.0)(rng);
}

template <typename T>
void genWeight(size_t n, Array<Array<T>>& ret) {
	ret = Array<Array<T>>(n);
	for(size_t v = 0; v < n; ++v) {
		ret[v] = Array<T>(S1 << (n - 1));
		for(size_t i = 0; i < ret[v].size(); ++i) {
			genWeightValue(ret[v][i]);
		}
	}
}

template <typename T>
void checkWeightValuesEqual(const T& a, const T& b) {
	CHECK_EQUAL(a, b);
}
void checkWeightValuesEqual(double a, double b) {
	CHECK_TRUE(std::abs((b - a) / a) < 1e-8);
}
void checkWeightValuesEqual(LogDouble a, LogDouble b) {
	CHECK_TRUE(std::abs(b.log - a.log) < 1e-8);
}
void checkWeightValuesEqual(ExtDouble a, ExtDouble b) {
	return checkWeightValuesEqual((LogDouble)a, (LogDouble)b);
}

template <typename T>
void checkWeightsEqual(const Array<Array<T>>& a, const Array<Array<T>>& b) {
	size_t n = a.size();
	CHECK_EQUAL(b.size(), n);
	for(size_t v = 0; v < n; ++v) {
		CHECK_EQUAL(a[v].size(), S1 << (n - 1));
		CHECK_EQUAL(b[v].size(), S1 << (n - 1));
		for(size_t i = 0; i < a[v].size(); ++i) {
			checkWeightValuesEqual(a[v][i], b[v][i]);
		}
	}
}

template <typename T>
void checkARResultsEqual(const Array<Array<T>>& a, const Array<Array<T>>& b) {
	size_t n = a.size();
	CHECK_EQUAL(b.size(), n);
	for(size_t v = 0; v < n; ++v) {
		CHECK_EQUAL(a[v].size(), n);
		CHECK_EQUAL(b[v].size(), n);
		for(size_t i = 0; i < n; ++i) {
			checkWeightValuesEqual(a[v][i], b[v][i]);
		}
	}
}

template <typename T>
void testAPS(const std::vector<std::pair<std::string, APSFunc<T>>>& funcs) {
	for(size_t n = 0; n <= 16; ++n) {
		if(conf::verbose) {
			std::cerr << "    n = " << n << ":\n";
		}
		Array<Array<T>> w;
		genWeight(n, w);

		Array<Array<T>> cmpResult;
		for(const std::pair<std::string, APSFunc<T>>& func : funcs) {
			if(conf::verbose) {
				std::cerr << "      Method " << func.first << "\n";
			}
			Array<Array<T>> result = func.second(w, true);
			if(n && !result.size()) {
				if(conf::verbose) {
					std::cerr << "        ^ SKIPPED\n";
				}
				continue;
			}
			CHECK_EQUAL(result.size(), n);
			for(size_t v = 0; v < n; ++v) {
				CHECK_EQUAL(result[v].size(), S1 << (n - 1));
				for(size_t i = 0; i < result[v].size(); ++i) {
					CHECK_TRUE(isValidWeightValue(result[v][i]));
				}
			}
			if(cmpResult.size()) {
				checkWeightsEqual(cmpResult, result);
			} else {
				cmpResult = move(result);
			}
		}
	}
}

template <typename T>
void testOrderModularAPS() {
	if(conf::verbose) {
		std::cerr << "  Order-modular APS\n";
	}
	testAPS(getOrderModularAPSFuncs<T>());
}

template <typename T>
void testModularAPS() {
	if(conf::verbose) {
		std::cerr << "  Modular APS\n";
	}
	testAPS(getModularAPSFuncs<T>());
}

template <typename T>
void testAPS() {
	testModularAPS<T>();
	testOrderModularAPS<T>();
}

template <typename T>
void testAR(const std::vector<std::pair<std::string, APSFunc<T>>>& funcs) {
	for(size_t n = 0; n <= 16; ++n) {
		if(conf::verbose) {
			std::cerr << "    n = " << n << ":\n";
		}
		Array<Array<T>> w;
		genWeight(n, w);

		Array<Array<T>> cmpResult;
		for(const std::pair<std::string, APSFunc<T>>& func : funcs) {
			if(conf::verbose) {
				std::cerr << "      Method " << func.first << "\n";
			}
			Array<Array<T>> result = func.second(w, true);
			if(n && !result.size()) {
				if(conf::verbose) {
					std::cerr << "        ^ SKIPPED\n";
				}
				continue;
			}
			CHECK_EQUAL(result.size(), n);
			for(size_t v = 0; v < n; ++v) {
				CHECK_EQUAL(result[v].size(), n);
				for(size_t i = 0; i < result[v].size(); ++i) {
					CHECK_TRUE(isValidWeightValue(result[v][i]));
				}
			}
			if(cmpResult.size()) {
				checkARResultsEqual(cmpResult, result);
			} else {
				cmpResult = move(result);
			}
		}
	}
}

template <typename T>
void testOrderModularAR() {
	if(conf::verbose) {
		std::cerr << "  Order-modular AR\n";
	}
	testAR(getOrderModularARFuncs<T>());
}

template <typename T>
void testModularAR() {
	if(conf::verbose) {
		std::cerr << "  Modular AR\n";
	}
	testAR(getModularARFuncs<T>());
}

template <typename T>
void testAR() {
	testModularAR<T>();
	testOrderModularAR<T>();
}

}

void runTests() {
	testNumber<LogDouble>();
	testNumber<ExtDouble>();
	crossTestExtLogDouble();

#undef APS_FOR_EACH_NUMBER_TYPE_TEMPLATE
#define APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(T) \
	if(conf::verbose) { \
		std::cerr << "T = " << #T << "\n"; \
	} \
	testAPS<T>(); \
	testAR<T>();

	APS_FOR_EACH_NUMBER_TYPE

	if(conf::verbose) {
		std::cerr << "All tests completed successfully\n";
	}
}

}
