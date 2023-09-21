#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstdint>
#include "Breal.hpp"
#include <chrono>

namespace wsum {

	using bm32 = uint32_t;
	using bm64 = uint64_t;
	using Treal = B2real;
	void fzt_inpl(Treal* b, bm32 n);

}

class Timer {
public:
	Timer();
	~Timer();
	int lap();
private:
	std::chrono::time_point<std::chrono::steady_clock> t;
};

#endif
