// -*- flycheck-clang-language-standard: "c++11"; -*-
#include <cmath>
#include <random>
#include <limits>
#include <functional>
#include <vector>
#include <queue>
#include "assert.h"

//template <typename T>
class discrete_random_variable {
 private:
  const std::vector<int> values_;
  const std::vector<double> probs_;
  const std::vector<std::pair<double, size_t>> alias_;
  //mutable std::random_device   rd_;
  mutable std::mt19937         gen_{std::random_device{}()};
  mutable std::uniform_real_distribution<double> real_dis_{0.0, 1.0};
  mutable std::uniform_int_distribution<size_t>  int_dis_;

 public:
  discrete_random_variable(const std::vector<int>& vals, const std::vector<double>& probs) :
    values_(vals), probs_(probs), alias_(generate_alias_table(probs)), int_dis_(0, probs.size() - 1) {
    assert(vals.size() == probs.size());
    const double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
    assert(std::fabs(1.0 - sum) < std::numeric_limits<double>::epsilon());
  }

  int operator()() const {
    const size_t idx  = int_dis_(gen_);
    if (real_dis_(gen_) >= alias_[idx].first and
          alias_[idx].second != std::numeric_limits<size_t>::max()) {
      return values_[alias_[idx].second];
    } else {
      return values_[idx];
    }
  }

  std::vector<int> get_values() {
    return values_;
  }

  std::vector<double> get_probs() {
    return probs_;
  }

  //not possible to wrap function call operator directly in Cython
  int sample() {
    return (*this)();
  }


 private:
  std::vector<std::pair<double, size_t>> generate_alias_table(const std::vector<double>& probs) {
    const size_t sz = probs.size();
    std::vector<std::pair<double, size_t>> alias(sz, {0.0, std::numeric_limits<size_t>::max()});
    std::queue<size_t>  small, large;

    for (size_t i = 0; i != sz; ++i) {
      alias[i].first = sz * probs[i];
      if (alias[i].first < 1.0) {
        small.push(i);
      } else {
        large.push(i);
      }
    }

    while (not(small.empty()) and not(large.empty())) {
      auto s = small.front(), l = large.front();
      small.pop(), large.pop();
      alias[s].second = l;
      alias[l].first -= (1.0 - alias[s].first);

      if (alias[l].first < 1.0) {
        small.push(l);
      } else {
        large.push(l);
      }
    }

    return alias;
  }
};

