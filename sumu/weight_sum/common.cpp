#include "common.hpp"

namespace wsum {

	void fzt_inpl(Treal* b, bm32 n){ // 550 adds per microsecond, Treal = Breal.
		for (bm32 i = 0; i < n; ++i) for (bm32 m = 0; m < (1L << n); ++m) if (m & (1L << i)) b[m] += b[m ^ (1L << i)];
	}

}

Timer::Timer() {
	t = std::chrono::steady_clock::now();
}

Timer::~Timer() {}

int Timer::lap() {
	int laptime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t).count();
	t = std::chrono::steady_clock::now();
	return laptime;
}
