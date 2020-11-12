#pragma once

#include "common.h"

#include <limits>

#ifdef _MSC_VER // Only for MSVC
#   include <intrin.h>
#	pragma intrinsic(_BitScanReverse64)
#	pragma intrinsic(_BitScanForward64)
#	pragma intrinsic(__popcnt64)
#	pragma warning (disable : 4146) // bottomOneBit
#endif

namespace aps {

inline size_t topOneBitIdx(size_t x) {
#ifdef _MSC_VER
	static_assert(sizeof(uint64_t) >= sizeof(size_t), "The case uint64_t < size_t is not implemented");
	unsigned long idx;
	_BitScanReverse64(&idx, (uint64_t)x);
	return (size_t)idx;
#else
	static_assert(sizeof(unsigned long long) >= sizeof(size_t), "The case unsigned long long < size_t is not implemented");
	return (size_t)(std::numeric_limits<unsigned long long>::digits - 1 - __builtin_clzll((unsigned long long)x));
#endif
}

inline size_t bottomOneBitIdx(size_t x) {
#ifdef _MSC_VER
	static_assert(sizeof(uint64_t) >= sizeof(size_t), "The case uint64_t < size_t is not implemented");
	unsigned long idx;
	_BitScanForward64(&idx, (uint64_t)x);
	return (size_t)idx;
#else
	static_assert(sizeof(unsigned long long) >= sizeof(size_t), "The case unsigned long long < size_t is not implemented");
	return (size_t)__builtin_ctzll((unsigned long long)x);
#endif
}

inline constexpr size_t bottomOneBit(size_t x) {
	return x & -x;
}

inline size_t popCount(size_t x) {
#ifdef _MSC_VER
	static_assert(sizeof(uint64_t) >= sizeof(size_t), "The case uint64_t < size_t is not implemented");
	return (size_t)__popcnt64((uint64_t)x);
#else
	static_assert(sizeof(unsigned long long) >= sizeof(size_t), "The case unsigned long long < size_t is not implemented");
	return (size_t)__builtin_popcountll((unsigned long long)x);
#endif
}

inline size_t collapseBit(size_t mask, size_t bitIdx) {
	uint64_t lowMask = (S1 << bitIdx) - S1;
	uint64_t highMask = (~lowMask) << 1;
	return (mask & lowMask) | ((mask & highMask) >> 1);
}
inline size_t expandBit(size_t mask, size_t bitIdx) {
	uint64_t lowMask = (S1 << bitIdx) - S1;
	uint64_t highMask = ~lowMask;
	return (mask & lowMask) | ((mask & highMask) << 1);
}

}
