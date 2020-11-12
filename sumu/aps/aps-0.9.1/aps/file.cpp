#include "file.h"

#include "bits.h"
#include "types.h"

#include <cmath>
#include <fstream>
#include <unordered_map>

namespace aps {

namespace {

template <typename T>
void fromLogDouble(const LogDouble& x, T& y) {
	y = (T)x;
}
void fromLogDouble(const LogDouble& x, uint64_t& y) {
	y = (uint64_t)std::round((double)x);
}

template <typename T>
LogDouble toLogDouble(const T& x) {
	return (LogDouble)x;
}
LogDouble toLogDouble(uint64_t x) {
	return (LogDouble)(double)x;
}

}

template <typename T>
Instance<T> readInstance(const std::string& filename) {
	Instance<T> ret;

	size_t n;
	{
		std::ifstream fp;
		fp.exceptions(fp.eofbit | fp.failbit | fp.badbit);
		fp.open(filename);
		fp >> n;
		ret.names = Array<std::string>(n);
		std::string tmp;
		for(size_t i = 0; i < n; ++i) {
			fp >> ret.names[i];
			size_t parentSetCount;
			fp >> parentSetCount;
			for(size_t j = 0; j < parentSetCount; ++j) {
				double score;
				size_t parentCount;
				fp >> score >> parentCount;
				for(size_t k = 0; k < parentCount; ++k) {
					fp >> tmp;
				}
			}
		}
	}

	ret.weights = Array<Array<T>>(n);
	for(size_t i = 0; i < n; ++i) {
		ret.weights[i] = Array<T>(S1 << (n - 1));
		ret.weights[i].fill(T(0));
	}

	std::unordered_map<std::string, size_t> nameToIdx;
	for(size_t i = 0; i < n; ++i) {
		if(nameToIdx.count(ret.names[i])) {
			fail("Variable name ", ret.names[i], " defined multiple times in the input file");
		}
		nameToIdx[ret.names[i]] = i;
	}

	{
		std::ifstream fp;
		fp.exceptions(fp.eofbit | fp.failbit | fp.badbit);
		fp.open(filename);
		size_t n2;
		fp >> n2;
		if(n2 != n) {
			fail("Input file changed during reading");
		}
		std::string tmp;
		for(size_t i = 0; i < n; ++i) {
			fp >> tmp;
			if(tmp != ret.names[i]) {
				fail("Input file changed during reading");
			}
			size_t parentSetCount;
			fp >> parentSetCount;
			for(size_t j = 0; j < parentSetCount; ++j) {
				double score;
				size_t parentCount;
				fp >> score >> parentCount;
				if(!std::isfinite(score)) {
					fail("Input file contains infinite or NaN score");
				}
				size_t parentMask = 0;
				for(size_t k = 0; k < parentCount; ++k) {
					fp >> tmp;
					auto it = nameToIdx.find(tmp);
					if(it == nameToIdx.end()) {
						fail("Unknown variable ", tmp, " referenced in input file");
					}
					if(it->second == i) {
						fail("Input file contains weight with a self-loop");
					}
					parentMask |= S1 << it->second;
				}
				parentMask = collapseBit(parentMask, i);
				fromLogDouble(LogDouble::fromLog(score), ret.weights[i][parentMask]);
			}
		}
	}

	return ret;
}

template <typename T>
void writeInstance(const std::string& filename, const Instance<T>& instance) {
	size_t n = instance.names.size();
	if(instance.weights.size() != n) {
		fail("Internal error: invalid instance structure");
	}
	for(size_t i = 0; i < n; ++i) {
		if(instance.weights[i].size() != (S1 << (n - 1))) {
			fail("Internal error: invalid instance structure");
		}
	}

	std::ofstream fp;
	fp.exceptions(fp.eofbit | fp.failbit | fp.badbit);
	fp.precision(8);
	fp.open(filename);

	fp << n << '\n';
	for(size_t i = 0; i < n; ++i) {
		size_t parentSetCount = 0;
		for(size_t j = 0; j < instance.weights[i].size(); ++j) {
			double weightLog = toLogDouble(instance.weights[i][j]).log;
			if(std::isnan(weightLog)) {
				fail("NaN weight in output");
			}
			if(weightLog != -std::numeric_limits<double>::infinity()) {
				if(!std::isfinite(weightLog)) {
					fail("Infinite weight in output");
				}
				++parentSetCount;
			}
		}
		
		fp << instance.names[i] << ' ' << parentSetCount << '\n';
		size_t parentSetsWritten = 0;
		for(size_t j = 0; j < instance.weights[i].size(); ++j) {
			double weightLog = toLogDouble(instance.weights[i][j]).log;
			if(weightLog != -std::numeric_limits<double>::infinity()) {
				if(parentSetsWritten >= parentSetCount) {
					fail("Internal error: invalid nonzero-weight parent set count");
				}
				fp << weightLog << ' ' << popCount(j);
				size_t parentMask = expandBit(j, i);
				for(size_t b = 0; b < n; ++b) {
					if(parentMask & (S1 << b)) {
						fp << ' ' << instance.names[b];
					}
				}
				fp << '\n';
				++parentSetsWritten;
			}
		}
		if(parentSetsWritten != parentSetCount) {
			fail("Internal error: invalid nonzero-weight parent set count");
		}
	}

	fp.close();
}

template <typename T>
void writeAROutput(const std::string& filename, const AROutput<T>& output) {
	size_t n = output.names.size();
	if(output.weights.size() != n) {
		fail("Internal error: invalid instance structure");
	}
	for(size_t i = 0; i < n; ++i) {
		if(output.weights[i].size() != n) {
			fail("Internal error: invalid instance structure");
		}
	}

	std::ofstream fp;
	fp.exceptions(fp.eofbit | fp.failbit | fp.badbit);
	fp.precision(8);
	fp.open(filename);

	fp << n << "\n";
	for(size_t i = 0; i < n; ++i) {
		fp << output.names[i] << "\n";
	}
	for(size_t i = 0; i < n; ++i) {
		for(size_t j = 0; j < n; ++j) {
			if(j) {
				fp << " ";
			}
			double weightLog = toLogDouble(output.weights[i][j]).log;
			if(weightLog != -std::numeric_limits<double>::infinity()) {
				fp << weightLog;
			} else {
				fp << "-inf";
			}
		}
		fp << "\n";
	}

	fp.close();
}


#undef APS_FOR_EACH_NUMBER_TYPE_TEMPLATE
#define APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(T) \
	template Instance<T> readInstance(const std::string& filename); \
	template void writeInstance(const std::string& filename, const Instance<T>& instance); \
	template void writeAROutput(const std::string& filename, const AROutput<T>& output);
APS_FOR_EACH_NUMBER_TYPE

}
