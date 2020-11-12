#include "aps.h"
#include "ar.h"
#include "array.h"
#include "conf.h"
#include "logdouble.h"
#include "types.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <limits>
#include <random>
#include <vector>

namespace aps {

namespace {

std::mt19937 rng(std::random_device{}());

void genWeightValue(uint64_t& ret) {
	ret = std::uniform_int_distribution<uint64_t>()(rng);
}
void genWeightValue(double& ret) {
	ret = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}
void genWeightValue(LogDouble& ret) {
	ret = (LogDouble)std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}
void genWeightValue(ExtDouble& ret) {
	ret = (ExtDouble)std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}

template <typename T>
Array<Array<T>> genWeight(size_t n) {
	Array<Array<T>> ret = Array<Array<T>>(n);
	for(size_t v = 0; v < n; ++v) {
		ret[v] = Array<T>(S1 << (n - 1));
		for(size_t i = 0; i < ret[v].size(); ++i) {
			genWeightValue(ret[v][i]);
		}
	}
	return ret;
}

template <typename T>
void perfTest(const APSFuncList<T>& funcs) {
	const double timeLimit = 10.0;
	const double minTime = 0.0001;
	std::vector<double> time1(funcs.size(), minTime);
	std::vector<double> time2(funcs.size(), minTime);

	std::cout << "            ";
	for(size_t fi = 0; fi < funcs.size(); ++fi) {
		std::cout << " " << std::setw(17) << funcs[fi].first;
	}
	std::cout << "\n";

	size_t n = 0;
	while(true) {
		std::cout << "    n = " << std::setw(2) << n << ": ";
		Array<Array<T>> w = genWeight<T>(n);

		bool empty = true;

		for(size_t fi = 0; fi < funcs.size(); ++fi) {
			if(time1[fi] * time1[fi] / time2[fi] < timeLimit) {
				using namespace std::chrono;

				auto start = high_resolution_clock::now();
				std::clock_t start_cpu = std::clock();

				Array<Array<T>> result = funcs[fi].second(w, false);

				auto end = high_resolution_clock::now();
				std::clock_t end_cpu = std::clock();

				if(!n || result.size()) {
					double elapsed = duration<double>(end - start).count();
					double elapsed_cpu = (double)(end_cpu - start_cpu) / (double)CLOCKS_PER_SEC;
					elapsed = std::max(elapsed, minTime);
					elapsed_cpu = std::max(elapsed_cpu, minTime);

					double avg_threads = elapsed_cpu / elapsed;

					std::stringstream ss;
					ss << std::fixed << std::setprecision(4) << elapsed << " x" << std::setprecision(2) << avg_threads;
					std::cout << " " << std::setw(17) << ss.str();

					time2[fi] = time1[fi];
					time1[fi] = elapsed;

					if(time1[fi] * time1[fi] / time2[fi] < timeLimit) {
						empty = false;
					}
				} else {
					std::cout << " " << std::setw(17) << "";
				}
			} else {
				std::cout << " " << std::setw(17) << "";
			}
		}

		std::cout << "\n";

		if(empty) {
			break;
		}

		++n;
	}
}

}

void runPerfTests() {
	std::cout << "\nModular APS\n";
#undef APS_FOR_EACH_NUMBER_TYPE_TEMPLATE
#define APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(T) \
	std::cout << "\n  T = " << #T << "\n"; \
	perfTest<T>(getModularAPSFuncs<T>());

	APS_FOR_EACH_NUMBER_TYPE

	std::cout << "\nOrder-modular APS\n";
#undef APS_FOR_EACH_NUMBER_TYPE_TEMPLATE
#define APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(T) \
	std::cout << "\n  T = " << #T << "\n"; \
	perfTest<T>(getOrderModularAPSFuncs<T>());
	
	APS_FOR_EACH_NUMBER_TYPE

	std::cout << "\nModular AR\n";
#undef APS_FOR_EACH_NUMBER_TYPE_TEMPLATE
#define APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(T) \
	std::cout << "\n  T = " << #T << "\n"; \
	perfTest<T>(getModularARFuncs<T>());

	APS_FOR_EACH_NUMBER_TYPE

	std::cout << "\nOrder-modular AR\n";
#undef APS_FOR_EACH_NUMBER_TYPE_TEMPLATE
#define APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(T) \
	std::cout << "\n  T = " << #T << "\n"; \
	perfTest<T>(getOrderModularARFuncs<T>());

	APS_FOR_EACH_NUMBER_TYPE
}

}
