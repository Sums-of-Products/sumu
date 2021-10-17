#include "IntersectSums.hpp"
#include <vector>
#include <utility>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;
using std::pair;
using std::make_pair;


template <typename T>
bool decr_ws(T x, T y) { return x.weight > y.weight; }

template <typename T>
void sort_ws(vector<T> &c) { sort(c.begin(), c.end(), decr_ws<T>); }

IntersectSums::IntersectSums(double *w0, bm64 *pset0, bm64 m0, int k0, double eps0){
  m = m0;
  k = k0;
  bm64 i = 0;
  bm64 j = 0;
  while (i < m) {
	Treal w; w.set_log(w0[i++]);
	if (k == 1) {s64.push_back( {pset0[j++], w} );}
	else if (k == 2) {s128.push_back( { {pset0[j++], pset0[j++]}, w} );}
	else if (k == 3) {s192.push_back( { {pset0[j++], pset0[j++], pset0[j++]}, w} );}
	else if (k == 4) {s256.push_back( { {pset0[j++], pset0[j++], pset0[j++], pset0[j++]}, w} );}
	else if (k > 4) {
	  vector<bm64> set;
	  for (int l = 0; l < k; ++l) { set.push_back(pset0[j++]); }
	  s.push_back( {set, w} );
	}
	else if (k < 0) {	// Read at most elements for the (-k) 64-bit vectors and represent them in a bmx.
		k = -k;
		bmx set = {};	
		int p = 0;		// Number of elements (found).
		for (int l = 0; l < k; ++l) {
			bm64 block = pset0[j++]; int v = l * 64;
			while (block > 0){	// Somewhat slow scanning. On the other hand, not many blocks can be nonzero.
				if (block & (bm64)1){	// New element v found; insert to the set.
					++p;
					switch (p){
						case 1: {set.v1 = v; set.v2 = v; set.v3 = v; set.v4 = v;} break;
						case 2:  set.v2 = v; break;
						case 3:  set.v3 = v; break;
						case 4:  set.v4 = v; break;
						default: break;
					}
				}
				++v; block >>= 1;
			}
		}
		sx.push_back( {set, w} );
	}
  }
  sort_ws(s64); sort_ws(s128); sort_ws(s192); sort_ws(s256); sort_ws(s); sort_ws(sx);
  eps = eps0;
};

// d=1 tapauksessa konstruktorille ei anneta pset, koska kaikki on valid?

IntersectSums::~IntersectSums(){ };

bool IntersectSums::intersects(vector<bm64> A, vector<bm64> B) {
  for (int i = 0; i < k; ++i) {
	if (A[i] & B[i]) return true;
  }
  return false;
}

bool IntersectSums::subseteq(vector<bm64> A, vector<bm64> B) {
  bool sse = true;
  for (int i = 0; i < k; ++i) {
	sse = sse && A[i] == (A[i] & B[i]);
  }
  return sse;
}

double IntersectSums::scan_sum(double w0, vector<bm64> U, vector<bm64> T, bm64 t_ub) {
  Treal sum; sum = 0.0;
  if (w0 != -std::numeric_limits<double>::infinity()) {sum.set_log(w0);}
  Treal slack; slack = eps/t_ub;
  Treal factor; factor = 0.0;
  bm64 i = 0;
  for (; i < m; ++i) {
    vector<bm64> P = s[i].set;
    if ( subseteq(P, U) && intersects(P, T) ) {
      sum += s[i].weight;
	  factor = sum * slack;
	  ++i; break;
    }
  }
  for (; i < m; ++i){
    vector<bm64> P = s[i].set;
    if ( subseteq(P, U) && intersects(P, T) ) {
      Treal score; score = s[i].weight;
      if (score < factor) { break; }
      sum += score;
    }
  }
  return sum.get_log();
}

pair<vector<bm64>, double> IntersectSums::scan_rnd(vector<bm64> U, vector<bm64> T, double wcum) {
  Treal sum; sum = 0.0;
  Treal target; target.set_log(wcum);
  bm64 i = 0; vector<bm64> P; bm64 P_i = 0;
  for (; i < m; ++i) {
    P = s[i].set;
    if ( subseteq(P, U) && intersects(P, T) ) {
	  P_i = i;
      sum = s[i].weight;
	  if (sum > target) i = m;
	  ++i;
	  break;
    }
  }
  for (; i < m; ++i) {
    P = s[i].set;
    if ( subseteq(P, U) && intersects(P, T) ) {
	  P_i = i;
	  Treal score = s[i].weight; sum += score;
	  if (sum > target) break;
    }
  }
  return make_pair(s[P_i].set, s[P_i].weight.get_log());
}

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

pair<bm64, double> IntersectSums::scan_rnd_64(bm64 U, bm64 T, double wcum) {
  Treal sum; sum = 0.0;
  Treal target; target.set_log(wcum);
  bm64 i = 0; bm64 P = 0; bm64 P_i = 0;
  for (; i < m; ++i) {
    P = s64[i].set;
    if ( subseteq_64(P, U) && intersects_64(P, T) ) {
	  P_i = i;
      sum = s64[i].weight;
	  if (sum > target) i = m;
	  ++i;
	  break;
    }
  }
  for (; i < m; ++i) {
    P = s64[i].set;
    if ( subseteq_64(P, U) && intersects_64(P, T) ) {
	  P_i = i;
	  Treal score = s64[i].weight; sum += score;
	  if (sum > target) break;
    }
  }
  return make_pair(s64[P_i].set, s64[P_i].weight.get_log());
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

pair<bm128, double> IntersectSums::scan_rnd_128(bm128 U, bm128 T, double wcum) {
  Treal sum; sum = 0.0;
  Treal target; target.set_log(wcum);
  bm64 i = 0; bm128 P = {0, 0}; bm64 P_i = 0;
  for (; i < m; ++i) {
    P = s128[i].set;
    if ( subseteq_128(P, U) && intersects_128(P, T) ) {
	  P_i = i;
      sum = s128[i].weight;
	  if (sum > target) i = m;
	  i++;
	  break;
    }
  }
  for (; i < m; ++i) {
    P = s128[i].set;
    if ( subseteq_128(P, U) && intersects_128(P, T) ) {
	  P_i = i;
	  Treal score = s128[i].weight; sum += score;
	  if (sum > target) break;
    }
  }
  return make_pair(s128[P_i].set, s128[P_i].weight.get_log());
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

pair<bm192, double> IntersectSums::scan_rnd_192(bm192 U, bm192 T, double wcum) {
  Treal sum; sum = 0.0;
  Treal target; target.set_log(wcum);
  bm64 i = 0; bm192 P = {0, 0, 0}; bm64 P_i = 0;
  for (; i < m; ++i) {
    P = s192[i].set;
    if ( subseteq_192(P, U) && intersects_192(P, T) ) {
	  P_i = i;
      sum = s192[i].weight; ++i;
	  if (sum > target) i = m;
	  break;
    }
  }
  for (; i < m; ++i) {
    P = s192[i].set;
    if ( subseteq_192(P, U) && intersects_192(P, T) ) {
	  P_i = i;
	  Treal score = s192[i].weight; sum += score;
	  if (sum > target) break;
    }
  }
  return make_pair(s192[P_i].set, s192[P_i].weight.get_log());
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

pair<bm256, double> IntersectSums::scan_rnd_256(bm256 U, bm256 T, double wcum) {
  Treal sum; sum = 0.0;
  Treal target; target.set_log(wcum);
  bm64 i = 0; bm256 P = {0, 0, 0, 0}; bm64 P_i = 0;
  for (; i < m; ++i) {
    P = s256[i].set;
    if ( subseteq_256(P, U) && intersects_256(P, T) ) {
	  P_i = i;
      sum = s256[i].weight; ++i;
	  if (sum > target) i = m;
	  break;
    }
  }
  for (; i < m; ++i) {
    P = s256[i].set;
    if ( subseteq_256(P, U) && intersects_256(P, T) ) {
	  P_i = i;
	  Treal score = s256[i].weight; sum += score;
	  if (sum > target) break;
    }
  }
  return make_pair(s256[P_i].set, s256[P_i].weight.get_log());
}

double IntersectSums::scan_sum_x(double w0, vector<bm64> U, vector<bm64> T, bm64 t_ub) {
  Treal sum; sum = 0.0;
  if (w0 != -std::numeric_limits<double>::infinity()) {sum.set_log(w0);}
  Treal slack; slack = eps/t_ub;
  Treal factor; factor = 0.0;

  uint8_t *u = new uint8_t[k * 64]; uint8_t *t = new uint8_t[k * 64]; int p = 0;
/*
  for (int l = 0; l < k; ++l){
	bm64 Ul = U[l]; bm64 Tl = T[l];
	for (int j = 0; j < 64; ++j){ u[p] = Ul & (bm64)1; t[p] = Tl & (bm64)1; Ul >>= 1; Tl >>= 1; ++p; }
  }
  bm64 i = 0;
  for (; i < m; ++i) {
    bmx = sx[i].set;
    if ((u[P.v1] & u[P.v2] & u[P.v3] & u[P.v4]) & (t[P.v1] | t[P.v2] | t[P.v3] | t[P.v4])) {
      sum += sx[i].weight;
	  factor = sum * slack;
	  ++i; break;
    }
  }
  for (; i < m; ++i){
    bmx P = sx[i].set;
    if ((u[P.v1] & u[P.v2] & u[P.v3] & u[P.v4]) & (t[P.v1] | t[P.v2] | t[P.v3] | t[P.v4])) {
      Treal score; score = sx[i].weight;
      if (score < factor) { break; }
      sum += score;
    }
  }
*/
  delete[] u; delete[] t;

  return sum.get_log();
}


