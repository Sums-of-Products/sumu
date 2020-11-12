#pragma once

#include "array.h"

namespace aps {

template <typename T>
struct Instance {
	Array<std::string> names;
	Array<Array<T>> weights;
};

template <typename T>
Instance<T> readInstance(const std::string& filename);

template <typename T>
void writeInstance(const std::string& filename, const Instance<T>& instance);

template <typename T>
struct AROutput {
	Array<std::string> names;
	Array<Array<T>> weights;
};

template <typename T>
void writeAROutput(const std::string& filename, const AROutput<T>& output);

}
