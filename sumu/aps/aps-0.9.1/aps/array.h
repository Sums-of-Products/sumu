#pragma once

#include "common.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <type_traits>

#ifdef _MSC_VER
#include <malloc.h>
#else
#include <stdlib.h>
#endif

namespace aps {

// Array of elements with fixed size. For trivially copyable types, initially
// uninitialized and aligned to allow using aligned vectorized loads and avoid
// false sharing.
template <typename T, bool IsTriviallyCopyable = std::is_trivially_copyable<T>::value>
class Array {
public:
	Array() :
		ptr_(nullptr),
		size_(0) {
	}

	Array(size_t size) :
		ptr_(new T[size]),
		size_(size)
	{}

	size_t size() const {
		return size_;
	}

	T* data() {
		return ptr_.get();
	}
	const T* data() const {
		return ptr_.get();
	}

	T& operator[](size_t i) {
		return data()[i];
	}
	const T& operator[](size_t i) const {
		return data()[i];
	}

	void fill(const T& val) {
		std::fill(data(), data() + size(), val);
	}

private:
	std::unique_ptr<T[]> ptr_;
	size_t size_;
};

template <typename T>
class Array<T, true> {
public:
	Array() :
		ptr_(nullptr),
		size_(0)
	{}

	Array(size_t size) :
		size_(size)
	{
		const size_t align = 64;
		size_t bytes = size * sizeof(T);
		void* addr;
		if(bytes) {
#ifdef _MSC_VER
			addr = _aligned_malloc(bytes, align);
			if(!addr) {
				fail("Could not allocate memory (_aligned_malloc(", bytes, ", ", align, ") failed)");
			}
#else
			if(posix_memalign(&addr, align, bytes)) {
				fail("Could not allocate memory (posix_memalign(..., ", align, ", ", bytes, ") failed)");
			}
#endif
		} else {
			addr = nullptr;
		}
		ptr_.reset((T*)addr);
	}

	size_t size() const {
		return size_;
	}

	T* data() {
		return ptr_.get();
	}
	const T* data() const {
		return ptr_.get();
	}

	T& operator[](size_t i) {
		return data()[i];
	}
	const T& operator[](size_t i) const {
		return data()[i];
	}

	void fill(const T& val) {
		std::fill(data(), data() + size(), val);
	}

private:
	struct Free {
		void operator()(T* ptr) {
#ifdef _MSC_VER
			_aligned_free(ptr);
#else
			free(ptr);
#endif
		}
	};

	std::unique_ptr<T, Free> ptr_;
	size_t size_;
};

}
