// -*- flycheck-clang-language-standard: "c++17"; -*-
#include <cmath>
#include <cstdint>
#include <vector>

using namespace std;
using bm64 = uint64_t;
struct bm128 { bm64 s1; bm64 s2; };
struct bm192 { bm64 s1; bm64 s2; bm64 s3; };
struct bm256 { bm64 s1; bm64 s2; bm64 s3; bm64 s4; };

inline bool intersects_64(bm64 A, bm64 B){ return A & B; }
inline bool subseteq_64(bm64 A, bm64 B){ return (A == (A & B)); }
inline bool intersects_128(bm128 A, bm128 B){ return (A.s1 & B.s1) || (A.s2 & B.s2); }
inline bool subseteq_128(bm128 A, bm128 B){ return ( (A.s1 == (A.s1 & B.s1)) && (A.s2 == (A.s2 & B.s2)) ); }

inline bool intersects_192(bm192 A, bm192 B){ return (A.s1 & B.s1) || (A.s2 & B.s2) || (A.s3 & B.s3); }
inline bool subseteq_192(bm192 A, bm192 B){ return ( (A.s1 == (A.s1 & B.s1)) && (A.s2 == (A.s2 & B.s2)) && (A.s3 == (A.s3 & B.s3)) ); }

inline bool intersects_256(bm256 A, bm256 B){ return (A.s1 & B.s1) || (A.s2 & B.s2) || (A.s3 & B.s3) || (A.s4 & B.s4); }
inline bool subseteq_256(bm256 A, bm256 B){ return ( (A.s1 == (A.s1 & B.s1)) && (A.s2 == (A.s2 & B.s2)) && (A.s3 == (A.s3 & B.s3)) && (A.s4 == (A.s4 & B.s4))); }


double weight_sum_64(double w, bm64 *psets, int m, double *weights, int j, int n, bm64 U, bm64 T, int t_ub) {
  double sum = 0;
  double slack = log(0.1/t_ub);
  double factor = 0;
  j = 0;
  int i = 0;
  int n_parts = 1;
  while (i < n_parts*m) {
    bm64 P = psets[i++];
    if ( intersects_64(P, T) && subseteq_64(P, U) ) {
      double score = weights[i-n_parts];
      if (j == 0) {
		j++;
		if (score > w) {factor = score; score = w;}
		else {factor = w;}
		sum += exp(score - factor) + 1;
		continue;
      }
      else if (score < factor + slack) { break; }
      sum += exp(score - factor);
    }
  }
  if (j == 0) {return w;}
  return log(sum) + factor;
}


double weight_sum_128(double w, bm64 *psets, int m, double *weights, int j, int n, vector<bm64> U_vec, vector<bm64> T_vec, int t_ub) {
  bm128 T = {T_vec[0], T_vec[1]};
  bm128 U = {U_vec[0], U_vec[1]};
  double sum = 0;
  double slack = log(0.1/t_ub);
  double factor = 0;
  j = 0;
  int i = 0;
  int n_parts = 2;
  while (i < n_parts*m) {
    bm128 P = {psets[i++], psets[i++]};
    if ( intersects_128(P, T) && subseteq_128(P, U) ) {
      double score = weights[(i-n_parts)/n_parts];
      if (j == 0){
		j++;
		if (score > w) {factor = score; score = w;}
		else {factor = w;}
		sum += exp(score - factor) + 1;
		continue;
      }
      else if (score < factor + slack) { break; }
      sum += exp(score - factor);
    }
  }
  if (j == 0) {return w;}
  return log(sum) + factor;
}


double weight_sum_192(double w, bm64 *psets, int m, double *weights, int j, int n, vector<bm64> U_vec, vector<bm64> T_vec, int t_ub) {
  bm192 T = {T_vec[0], T_vec[1], T_vec[2]};
  bm192 U = {U_vec[0], U_vec[1], U_vec[2]};
  double sum = 0;
  double slack = log(0.1/t_ub);
  double factor = 0;
  j = 0;
  int i = 0;
  int n_parts = 3;
  while (i < n_parts*m) {
    bm192 P = {psets[i++], psets[i++], psets[i++]};
    if ( intersects_192(P, T) && subseteq_192(P, U) ) {
      double score = weights[(i-n_parts)/n_parts];
      if (j == 0){
		j++;
		if (score > w) {factor = score; score = w;}
		else {factor = w;}
		sum += exp(score - factor) + 1;
		continue;
      }
      else if (score < factor + slack) { break; }
      sum += exp(score - factor);
    }
  }
  if (j == 0) {return w;}
  return log(sum) + factor;
}


double weight_sum_256(double w, bm64 *psets, int m, double *weights, int j, int n, vector<bm64> U_vec, vector<bm64> T_vec, int t_ub) {
  bm256 T = {T_vec[0], T_vec[1], T_vec[2], T_vec[3]};
  bm256 U = {U_vec[0], U_vec[1], U_vec[2], U_vec[3]};
  double sum = 0;
  double slack = log(0.1/t_ub);
  double factor = 0;
  j = 0;
  int i = 0;
  int n_parts = 4;
  while (i < n_parts*m) {
    bm256 P = {psets[i++], psets[i++], psets[i++], psets[i++]};
    if ( intersects_256(P, T) && subseteq_256(P, U) ) {
      double score = weights[(i-n_parts)/n_parts];
      if (j == 0){
		j++;
		if (score > w) {factor = score; score = w;}
		else {factor = w;}
		sum += exp(score - factor) + 1;
		continue;
      }
      else if (score < factor + slack) { break; }
      sum += exp(score - factor);
    }
  }
  if (j == 0) {return w;}
  return log(sum) + factor;
}


pair<double, vector<int>> weight_sum_contribs_64(double w, bm64 *psets, int m, double *weights, int j, int n, bm64 U, bm64 T, int t_ub) {
  pair <double, vector<int>> w_and_contribs;
  vector<int> contribs(0);
  double sum = 0;
  double slack = log(0.1/t_ub);
  double factor = 0;
  j = 0;
  int i = 0;
  int n_parts = 1;
  while (i < n_parts*m) {
    bm64 P = psets[i++];
    if ( intersects_64(P, T) && subseteq_64(P, U) ) {
      double score = weights[i-n_parts];
      if (j == 0){
		j++;
		if (score > w) {factor = score; score = w;}
		else {factor = w;}
		sum += exp(score - factor) + 1;
		contribs.insert(contribs.end(), i-n_parts);
		continue;
      }
      else if (score < factor + slack) { break; }
      sum += exp(score - factor);
      contribs.insert(contribs.end(), i-n_parts);
    }
  }
  w_and_contribs = make_pair(log(sum) + factor, contribs);
  return w_and_contribs;
}


pair<double, vector<int>> weight_sum_contribs_128(double w, bm64 *psets, int m, double *weights, int j, int n, vector<bm64> U_vec, vector<bm64> T_vec, int t_ub) {
  bm128 T = {T_vec[0], T_vec[1]};
  bm128 U = {U_vec[0], U_vec[1]};
  pair <double, vector<int>> w_and_contribs;
  vector<int> contribs(0);
  double sum = 0;
  double slack = log(0.1/t_ub);
  double factor = 0;
  j = 0;
  int i = 0;
  int n_parts = 2;
  while (i < n_parts*m) {
    bm128 P = {psets[i++], psets[i++]};
    if ( intersects_128(P, T) && subseteq_128(P, U) ) {
      double score = weights[(i-n_parts)/n_parts];
      if (j == 0){
		j++;
		if (score > w) {factor = score; score = w;}
		else {factor = w;}
		sum += exp(score - factor) + 1;
		contribs.insert(contribs.end(), (i-n_parts)/n_parts);
		continue;
      }
      else if (score < factor + slack) { break; }
      sum += exp(score - factor);
      contribs.insert(contribs.end(), (i-n_parts)/n_parts);
    }
  }
  w_and_contribs = make_pair(log(sum) + factor, contribs);
  return w_and_contribs;
}


pair<double, vector<int>> weight_sum_contribs_192(double w, bm64 *psets, int m, double *weights, int j, int n, vector<bm64> U_vec, vector<bm64> T_vec, int t_ub) {
  bm192 T = {T_vec[0], T_vec[1], T_vec[2]};
  bm192 U = {U_vec[0], U_vec[1], U_vec[2]};
  pair <double, vector<int>> w_and_contribs;
  vector<int> contribs(0);
  double sum = 0;
  double slack = log(0.1/t_ub);
  double factor = 0;
  j = 0;
  int i = 0;
  int n_parts = 3;
  while (i < n_parts*m) {
    bm192 P = {psets[i++], psets[i++], psets[i++]};
    if ( intersects_192(P, T) && subseteq_192(P, U) ) {
      double score = weights[(i-n_parts)/n_parts];
      if (j == 0){
		j++;
		if (score > w) {factor = score; score = w;}
		else {factor = w;}
		sum += exp(score - factor) + 1;
		contribs.insert(contribs.end(), (i-n_parts)/n_parts);
		continue;
      }
      else if (score < factor + slack) { break; }
      sum += exp(score - factor);
      contribs.insert(contribs.end(), (i-n_parts)/n_parts);
    }
  }
  w_and_contribs = make_pair(log(sum) + factor, contribs);
  return w_and_contribs;
}


pair<double, vector<int>> weight_sum_contribs_256(double w, bm64 *psets, int m, double *weights, int j, int n, vector<bm64> U_vec, vector<bm64> T_vec, int t_ub) {
  bm256 T = {T_vec[0], T_vec[1], T_vec[2], T_vec[3]};
  bm256 U = {U_vec[0], U_vec[1], U_vec[2], U_vec[3]};
  pair <double, vector<int>> w_and_contribs;
  vector<int> contribs(0);
  double sum = 0;
  double slack = log(0.1/t_ub);
  double factor = 0;
  j = 0;
  int i = 0;
  int n_parts = 4;
  while (i < n_parts*m) {
    bm256 P = {psets[i++], psets[i++], psets[i++], psets[i++]};
    if ( intersects_256(P, T) && subseteq_256(P, U) ) {
      double score = weights[(i-n_parts)/n_parts];
      if (j == 0){
		j++;
		if (score > w) {factor = score; score = w;}
		else {factor = w;}
		sum += exp(score - factor) + 1;
		contribs.insert(contribs.end(), (i-n_parts)/n_parts);
		continue;
      }
      else if (score < factor + slack) { break; }
      sum += exp(score - factor);
      contribs.insert(contribs.end(), (i-n_parts)/n_parts);
    }
  }
  w_and_contribs = make_pair(log(sum) + factor, contribs);
  return w_and_contribs;
}
