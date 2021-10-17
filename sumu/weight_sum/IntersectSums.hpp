#ifndef INTERSECT_SUMS_HPP
#define INTERSECT_SUMS_HPP

#include <vector>
#include <utility>
#include "common.hpp"

using namespace wsum;

using bm8  = uint8_t;
using bm16 = uint16_t;

struct bm128 { bm64 s1; bm64 s2; };
struct bm192 { bm64 s1; bm64 s2; bm64 s3; };
struct bm256 { bm64 s1; bm64 s2; bm64 s3; bm64 s4; };
struct bmx   { bm16 v1, v2, v3, v4; };

struct ws64 { bm64 set; Treal weight; };
struct ws128 { bm128 set; Treal weight; };
struct ws192 { bm192 set; Treal weight; };
struct ws256 { bm256 set; Treal weight; };
struct ws { std::vector<bm64> set; Treal weight; };
struct wsx   { bmx set; Treal weight; };

inline bool intersects_64(bm64 A, bm64 B){ return A & B; }
inline bool subseteq_64(bm64 A, bm64 B){ return (A == (A & B)); }

inline bool intersects_128(bm128 A, bm128 B){ return (A.s1 & B.s1) || (A.s2 & B.s2); }
inline bool subseteq_128(bm128 A, bm128 B){ return ( (A.s1 == (A.s1 & B.s1)) && (A.s2 == (A.s2 & B.s2)) ); }

inline bool intersects_192(bm192 A, bm192 B){ return (A.s1 & B.s1) || (A.s2 & B.s2) || (A.s3 & B.s3); }
inline bool subseteq_192(bm192 A, bm192 B){ return ( (A.s1 == (A.s1 & B.s1)) && (A.s2 == (A.s2 & B.s2)) && (A.s3 == (A.s3 & B.s3)) ); }

inline bool intersects_256(bm256 A, bm256 B){ return (A.s1 & B.s1) || (A.s2 & B.s2) || (A.s3 & B.s3) || (A.s4 & B.s4); }
inline bool subseteq_256(bm256 A, bm256 B){ return ( (A.s1 == (A.s1 & B.s1)) && (A.s2 == (A.s2 & B.s2)) && (A.s3 == (A.s3 & B.s3)) && (A.s4 == (A.s4 & B.s4))); }


class IntersectSums {
public:

  IntersectSums(double *w0, bm64 *pset0, bm64 m0, int k, double eps0);
  ~IntersectSums();

  void dummy();
  double scan_sum_64(double w, bm64 U, bm64 T, bm64 t_ub);
  double scan_sum_128(double w, bm128 U, bm128 T, bm64 t_ub);
  double scan_sum_192(double w, bm192 U, bm192 T, bm64 t_ub);
  double scan_sum_256(double w, bm256 U, bm256 T, bm64 t_ub);
  double scan_sum(double w, std::vector<bm64> U, std::vector<bm64> T, bm64 t_ub);
  double scan_sum_x(double w, std::vector<bm64> U, std::vector<bm64> T, bm64 t_ub);

  std::pair<bm64, double> scan_rnd_64(bm64 U, bm64 T, double wcum);
  std::pair<bm128, double> scan_rnd_128(bm128 U, bm128 T, double wcum);
  std::pair<bm192, double> scan_rnd_192(bm192 U, bm192 T, double wcum);
  std::pair<bm256, double> scan_rnd_256(bm256 U, bm256 T, double wcum);
  std::pair<std::vector<bm64>, double> scan_rnd(std::vector<bm64> U, std::vector<bm64> T, double wcum);
  std::pair<std::vector<bm64>, double> scan_rnd_x(std::vector<bm64> U, std::vector<bm64> T, double wcum);

  bool intersects(std::vector<bm64> A, std::vector<bm64> B);
  bool subseteq(std::vector<bm64> A, std::vector<bm64> B);

private:
  double eps;
  int k;                   // No. 64-bit ints needed to encode var sets, e.g., 4 if 192 < n <= 256.
  bm64 m;                  // No. of psets.
  std::vector<ws64> s64;   // Scores, pairs (set, weight) sorted by weight.
  std::vector<ws128> s128;
  std::vector<ws192> s192;
  std::vector<ws256> s256;
  std::vector<ws> s;
  std::vector<wsx> sx;
  bm64 *idx;               // Sorted indices.
};

#endif
