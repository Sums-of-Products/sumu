#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include "IntersectSums.hpp"
#include <iostream>

using std::vector;
using std::cout;
using std::endl;

template <typename T>
bool decr_ws(T x, T y) { return x.weight > y.weight; }

template <typename T>
void sort_ws(vector<T> &c) { sort(c.begin(), c.end(), decr_ws<T>); }

IntersectSums::IntersectSums(double *w0, bm64 *pset0, bm64 m0, int k, double eps0){
  m = m0;
  bm64 i = 0;
  bm64 j = 0;
  while (i < m) {
	Treal w; w.set_log(w0[i++]);
	if (k == 1) {s64.push_back( {pset0[j++], w} );}
	else if (k == 2) {s128.push_back( { {pset0[j++], pset0[j++]}, w} );}
	else if (k == 3) {s192.push_back( { {pset0[j++], pset0[j++], pset0[j++]}, w} );}
	else if (k == 4) {s256.push_back( { {pset0[j++], pset0[j++], pset0[j++], pset0[j++]}, w} );}
  }
  sort_ws(s64); sort_ws(s128); sort_ws(s192); sort_ws(s256);
  eps = eps0;
};

// d=1 tapauksessa konstruktorille ei anneta pset, koska kaikki on valid?

IntersectSums::~IntersectSums(){ };

double IntersectSums::scan_sum_64(double w0, bm64 U, bm64 T, bm64 t_ub) {
  Treal sum; sum = 0.0;
  if (w0 != -std::numeric_limits<double>::infinity()) {sum.set_log(w0);}
  Treal slack; slack = eps/t_ub;
  Treal factor; factor = 0.0;
  bm64 i = 0;
  for (; i < m; ++i) {
    bm64 P = s64[i].set;
    if ( subseteq_64(P, U) && intersects_64(P, T) ) {
      sum += s64[i].weight;
	  factor = sum * slack;
	  ++i; break;
    }
  }
  for (; i < m; ++i){
    bm64 P = s64[i].set;
    if ( subseteq_64(P, U) && intersects_64(P, T) ) {
      Treal score; score = s64[i].weight;
      if (score < factor) { break; }
      sum += score;
    }
  }
  return sum.get_log();
}

bm64 IntersectSums::scan_rnd_64(bm64 U, bm64 T, double wcum) {
  Treal sum; sum = 0.0;
  Treal target; target.set_log(wcum);
  bm64 i = 0; bm64 P = 0;
  for (; i < m; ++i) {
    P = s64[i].set;
    if ( subseteq_64(P, U) && intersects_64(P, T) ) {
      sum = s64[i].weight; ++i;
	  if (sum > target) i = m;
	  break;
    }
  }
  for (; i < m; ++i) {
    P = s64[i].set;
    if ( subseteq_64(P, U) && intersects_64(P, T) ) {
	  Treal score = s64[i].weight; sum += score;
	  if (sum > target) break;
    }
  }
  return P;
}

double IntersectSums::scan_sum_128(double w0, bm128 U, bm128 T, bm64 t_ub) {
  Treal sum; sum = 0.0;
  if (w0 != -std::numeric_limits<double>::infinity()) {sum.set_log(w0); }
  Treal slack; slack = eps/t_ub;
  Treal factor; factor = 0.0;
  bm64 i = 0;
  for (; i < m; ++i) {
    bm128 P = s128[i].set;
    if ( subseteq_128(P, U) && intersects_128(P, T) ) {
      sum += s128[i].weight;
	  factor = sum * slack;
	  ++i; break;
    }
  }
  for (; i < m; ++i){
    bm128 P = s128[i].set;
    if ( subseteq_128(P, U) && intersects_128(P, T) ) {
      Treal score; score = s128[i].weight;
      if (score < factor) { break; }
      sum += score;
    }
  }
  return sum.get_log();
}

bm128 IntersectSums::scan_rnd_128(bm128 U, bm128 T, double wcum) {
  Treal sum; sum = 0.0;
  Treal target; target.set_log(wcum);
  bm64 i = 0; bm128 P = {0, 0};
  for (; i < m; ++i) {
    P = s128[i].set;
    if ( subseteq_128(P, U) && intersects_128(P, T) ) {
      sum = s128[i].weight; ++i;
	  if (sum > target) i = m;
	  break;
    }
  }
  for (; i < m; ++i) {
    P = s128[i].set;
    if ( subseteq_128(P, U) && intersects_128(P, T) ) {
	  Treal score = s128[i].weight; sum += score;
	  if (sum > target) break;
    }
  }
  return P;
}

double IntersectSums::scan_sum_192(double w0, bm192 U, bm192 T, bm64 t_ub) {
  Treal sum; sum = 0.0;
  if (w0 != -std::numeric_limits<double>::infinity()) {sum.set_log(w0); }
  Treal slack; slack = eps/t_ub;
  Treal factor; factor = 0.0;
  bm64 i = 0;
  for (; i < m; ++i) {
    bm192 P = s192[i].set;
    if ( subseteq_192(P, U) && intersects_192(P, T) ) {
      sum += s192[i].weight;
	  factor = sum * slack;
	  ++i; break;
    }
  }
  for (; i < m; ++i){
    bm192 P = s192[i].set;
    if ( subseteq_192(P, U) && intersects_192(P, T) ) {
      Treal score; score = s192[i].weight;
      if (score < factor) { break; }
      sum += score;
    }
  }
  return sum.get_log();
}

bm192 IntersectSums::scan_rnd_192(bm192 U, bm192 T, double wcum) {
  Treal sum; sum = 0.0;
  Treal target; target.set_log(wcum);
  bm64 i = 0; bm192 P = {0, 0, 0};
  for (; i < m; ++i) {
    P = s192[i].set;
    if ( subseteq_192(P, U) && intersects_192(P, T) ) {
      sum = s192[i].weight; ++i;
	  if (sum > target) i = m;
	  break;
    }
  }
  for (; i < m; ++i) {
    P = s192[i].set;
    if ( subseteq_192(P, U) && intersects_192(P, T) ) {
	  Treal score = s192[i].weight; sum += score;
	  if (sum > target) break;
    }
  }
  return P;
}

double IntersectSums::scan_sum_256(double w0, bm256 U, bm256 T, bm64 t_ub) {
  Treal sum; sum = 0.0;
  if (w0 != -std::numeric_limits<double>::infinity()) {sum.set_log(w0); }
  Treal slack; slack = eps/t_ub;
  Treal factor; factor = 0.0;
  bm64 i = 0;
  for (; i < m; ++i) {
    bm256 P = s256[i].set;
    if ( subseteq_256(P, U) && intersects_256(P, T) ) {
      sum += s256[i].weight;
	  factor = sum * slack;
	  ++i; break;
    }
  }
  for (; i < m; ++i){
    bm256 P = s256[i].set;
    if ( subseteq_256(P, U) && intersects_256(P, T) ) {
      Treal score; score = s256[i].weight;
      if (score < factor) { break; }
      sum += score;
    }
  }
  return sum.get_log();
}

bm256 IntersectSums::scan_rnd_256(bm256 U, bm256 T, double wcum) {
  Treal sum; sum = 0.0;
  Treal target; target.set_log(wcum);
  bm64 i = 0; bm256 P = {0, 0, 0, 0};
  for (; i < m; ++i) {
    P = s256[i].set;
    if ( subseteq_256(P, U) && intersects_256(P, T) ) {
      sum = s256[i].weight; ++i;
	  if (sum > target) i = m;
	  break;
    }
  }
  for (; i < m; ++i) {
    P = s256[i].set;
    if ( subseteq_256(P, U) && intersects_256(P, T) ) {
	  Treal score = s256[i].weight; sum += score;
	  if (sum > target) break;
    }
  }
  return P;
}
