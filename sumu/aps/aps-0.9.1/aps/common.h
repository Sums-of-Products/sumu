#pragma once

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <sstream>
#include <utility>

namespace aps {

using std::move;
using std::size_t;
using std::int64_t;
using std::uint64_t;

constexpr size_t S1 = 1;

namespace stderr_print_ {
template <typename T>
auto print(const T& val, bool) -> decltype(std::cerr << val, void()) {
	std::cerr << val;
}
template <typename T>
void print(const T& val, int) {
	std::cerr << "?";
}
template <typename T>
void print(const T& val) {
	print(val, true);
}
}

inline void stderrPrint() {
	std::cerr << "\n";
}
template <typename F, typename... T>
void stderrPrint(const F& f, const T&... p) {
	stderr_print_::print(f);
	stderrPrint(p...);
}
template <typename... T>
void fail(const T&... p) {
	stderrPrint("FAIL: ", p...);
	abort();
}

template <typename T>
T fromString(const std::string& str) {
	T ret;
	std::stringstream ss(str);
	ss >> ret;
	if(ss.fail()) fail("fromString: Could not convert string '", str, "' to given type.");
	return ret;
}

}
