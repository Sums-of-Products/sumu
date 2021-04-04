#ifndef GROUND_SET_INTERSECT_SUMS_HPP
#define GROUND_SET_INTERSECT_SUMS_HPP

#include <vector>
#include <algorithm>
#include "common.hpp"

using namespace wsum;

struct ws32 { bm32 set; Treal weight; };
inline bool intersects(bm32 A, bm32 B){ return A & B; }
inline bool subseteq  (bm32 A, bm32 B){ return (A == (A & B)); }
inline int  popcount  (bm32 S){ int c = 0; while (S) { c += (S & 1L); S >>= 1; } return c; }


// GroundSetIntersectSums -- Given (U, T), computes the weighted sum over the subsets S of U that intersect T.
// Assumpions: the sets are of size at most K <= 32.
//
class GroundSetIntersectSums {

public:
  GroundSetIntersectSums(int K0, double* w, double eps0);
  ~GroundSetIntersectSums();
  double scan_sum(bm32 U, bm32 T);
  double scan_sum(bm32 U);
  bm32   scan_rnd(bm32 U, bm32 T, double wcum);
  bm32   scan_rnd(bm32 U, double wcum);
  std::vector<ws32>	s;	// Scores, pairs (set, weight) sorted by weight.

private:
  int				K;	// Size of the ground set.
  double 			eps;// Tolerated relative error.
  void prune(double *w);
};

#endif
