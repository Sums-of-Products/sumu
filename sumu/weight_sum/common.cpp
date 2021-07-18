#include "common.hpp"

namespace wsum {

void fzt_inpl(Treal* b, bm32 n){ // 550 adds per microsecond, Treal = Breal.
	for (bm32 i = 0; i < n; ++i) for (bm32 m = 0; m < (1L << n); ++m) if (m & (1L << i)) b[m] += b[m ^ (1L << i)];
}

}
