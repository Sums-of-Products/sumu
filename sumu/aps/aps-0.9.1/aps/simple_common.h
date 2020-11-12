#pragma once

namespace aps {
namespace {

template <typename T>
Array<T> downZeta(const Array<T>& src) {
	size_t n = bottomOneBitIdx(src.size());
	Array<T> dest(S1 << n);
	if(n) {
		for(size_t i = 0; i < dest.size(); i += 2) {
			dest[i] = src[i];
			dest[i + 1] = src[i] + src[i + 1];
		}
	} else {
		dest[0] = src[0];
	}
	for(size_t b = 1; b < n; ++b) {
		size_t bBit = S1 << b;
		size_t highdiff = S1 << (b + 1);
		for(size_t high = 0; high < dest.size(); high += highdiff) {
			for(size_t low = 0; low < bBit; ++low) {
				size_t x = high | low;
				size_t y = x | bBit;
				dest[y] += dest[x];
			}
		}
	}
	return dest;
}

template <typename T>
void upZeta(Array<T>& w) {
	size_t n = bottomOneBitIdx(w.size());
	for(size_t b = 0; b < n; ++b) {
		size_t bBit = S1 << b;
		size_t highdiff = S1 << (b + 1);
		for(size_t high = 0; high < w.size(); high += highdiff) {
			for(size_t low = 0; low < bBit; ++low) {
				size_t x = high | low;
				size_t y = x | bBit;
				w[x] += w[y];
			}
		}
	}
}

}
}
