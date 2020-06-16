// -*- flycheck-clang-language-standard: "c++17"; -*-
#include <cmath>
#include <cstdint>
#include <vector>

using namespace std;
using bitmap = uint64_t;
struct bitmap_2 { bitmap s1; bitmap s2; };

inline bool intersects(bitmap A, bitmap B){ return A & B; }
inline bool subseteq(bitmap A, bitmap B){ return (A == (A & B)); }
inline bool intersects_2(bitmap_2 A, bitmap_2 B){ return (A.s1 & B.s1) || (A.s2 & B.s2); }
inline bool subseteq_2(bitmap_2 A, bitmap_2 B){ return ( (A.s1 == (A.s1 & B.s1)) && (A.s2 == (A.s2 & B.s2)) ); }

/*
double weight_sum_1(double w, bitmap *psets, int m, double *weights, int j, int n, bitmap U, bitmap T, int t_ub);
pair<double, vector<int>> weight_sum_contribs_1(double w, bitmap *psets, int m, double *weights, int j, int n, bitmap U, bitmap T, int t_ub);

double weight_sum_2(double w, bitmap *psets, int m, double *weights, int j, int n, vector<bitmap> U_vec, vector<bitmap> T_vec, int t_ub);
pair<double, vector<int>> weight_sum_contribs_2(double w, bitmap *psets, int m, double *weights, int j, int n, vector<bitmap> U_vec, vector<bitmap> T_vec, int t_ub);
*/

int idx2d(int row, int col, int dim2) {
  return dim2*row + col;
}

/*
double weight_sum(double w, uint64_t *psets, int m, int dim2, double *weights, int j, int n, vector<uint64_t> U, vector<uint64_t> T, int t_ub) {
  if (dim2 == 1) {
    return weight_sum_1(w, psets, m, weights, j, n, U[0], T[0], t_ub);
    }
  else if (dim2 == 2) {
    return weight_sum_2(w, psets, m, weights, j, n, U, T, t_ub);
  }
  return 0;
}

pair<double, vector<int>> weight_sum_contribs(double w, uint64_t *psets, int m, int dim2, double *weights, int j, int n, vector<uint64_t> U, vector<uint64_t> T, int t_ub) {
  if (dim2 == 1) {
    return weight_sum_contribs_1(w, psets, m, weights, j, n, U[0], T[0], t_ub);
    }
  else if (dim2 == 2) {
    return weight_sum_contribs_2(w, psets, m, weights, j, n, U, T, t_ub);
  }
  return weight_sum_contribs_1(w, psets, m, weights, j, n, U[0], T[0], t_ub);
}
*/

int testi(bitmap *psets, int m) {
  return m;
}

double weight_sum_64(double w, bitmap *ivec, int ni, double *dvec, int nd, bitmap U, bitmap T, int t_ub) {
  double sum = 0;
  double slack = log(0.1/t_ub);
  nd = -1;
  double factor = 0;
  for (int i = 0; i < ni; i ++){
    bitmap P = ivec[i];
    if ( intersects(P, T) && subseteq(P, U) ) {
      nd++;
      double score = dvec[i];
      if (nd == 0){
	factor = fmax(score, w);
	sum += exp(w - factor);
      }
      else if (score < factor + slack) { break; }
      sum += exp(score - factor);
    }
  }
  if (nd == -1) {return w;}
  return log(sum) + factor;
}

pair<double, vector<int>> weight_sum_contribs_64(double w, bitmap *psets, int m, double *weights, int j, bitmap U, bitmap T, int t_ub) {
  pair <double, vector<int>> w_and_contribs;
  vector<int> contribs(0);
  double sum = 0;
  double slack = log(0.1/t_ub);
  j = -1;
  double factor = 0;
  for (int i = 0; i < m; i ++){
    bitmap P = psets[i];
    if ( intersects(P, T) && subseteq(P, U) ) {
      j++;
      double score = weights[i];
      if (j == 0){
	factor = fmax(score, w);
	sum += exp(w - factor);
      }
      else if (score < factor + slack) { break; }
      sum += exp(score - factor);
      contribs.insert(contribs.end(), i);
    }
  }
  w_and_contribs = make_pair(log(sum) + factor, contribs);
  return w_and_contribs;
}

double weight_sum_128(double w, bitmap *psets, int m, int dim2, double *weights, int j, bitmap U1, bitmap U2, bitmap T1, bitmap T2, int t_ub) {
  bitmap_2 T;
  T.s1 = T1;
  T.s2 = T2;
  bitmap_2 U;
  U.s1 = U1;
  U.s2 = U2;

  double sum = 0;
  double slack = log(0.1/t_ub);
  j = -1;
  double factor = 0;
  for (int i = 0; i < m; i ++){
    bitmap P1 = psets[idx2d(i, 0, 2)];
    bitmap P2 = psets[idx2d(i, 1, 2)];
    bitmap_2 P;
    P.s1 = P1;
    P.s2 = P2;

    if ( intersects_2(P, T) && subseteq_2(P, U) ) {
      j++;
      double score = weights[i];
      if (j == 0){
	factor = fmax(score, w);
	sum += exp(w - factor);
      }
      else if (score < factor + slack) { break; }
      sum += exp(score - factor);
    }
  }
  if (j == -1) {return w;}
  return log(sum) + factor;
}

pair<double, vector<int>> weight_sum_contribs_128(double w, bitmap *psets, int m, int dim2, double *weights, int j, bitmap U1, bitmap U2, bitmap T1, bitmap T2, int t_ub) {
  bitmap_2 T;
  T.s1 = T1;
  T.s2 = T2;
  bitmap_2 U;
  U.s1 = U1;
  U.s2 = U2;

  pair <double, vector<int>> w_and_contribs;
  vector<int> contribs(0);
  double sum = 0;
  double slack = log(0.1/t_ub);
  j = -1;
  double factor = 0;
  for (int i = 0; i < m; i ++){
    bitmap P1 = psets[idx2d(i, 0, 2)];
    bitmap P2 = psets[idx2d(i, 1, 2)];
    bitmap_2 P;
    P.s1 = P1;
    P.s2 = P2;

    if ( intersects_2(P, T) && subseteq_2(P, U) ) {
      j++;
      double score = weights[i];
      if (j == 0){
	factor = fmax(score, w);
	sum += exp(w - factor);
      }
      else if (score < factor + slack) { break; }
      sum += exp(score - factor);
      contribs.insert(contribs.end(), i);
    }
  }
  w_and_contribs = make_pair(log(sum) + factor, contribs);
  return w_and_contribs;
}

/*
double weight_sum_2(double w, bitmap *psets, int m, double *weights, int j, int n, vector<uint64_t> U_vec, vector<uint64_t> T_vec, int t_ub) {
  bitmap_2 T;
  T.s1 = T_vec[0];
  T.s2 = T_vec[1];
  bitmap_2 U;
  U.s1 = U_vec[0];
  U.s2 = U_vec[1];

  double sum = 0;
  double slack = log(0.1/t_ub);
  j = -1;
  double factor = 0;
  for (int i = 0; i < m; i ++){
    bitmap P1 = psets[idx2d(i, 0, 2)];
    bitmap P2 = psets[idx2d(i, 1, 2)];
    bitmap_2 P;
    P.s1 = P1;
    P.s2 = P2;

    if ( intersects_2(P, T) && subseteq_2(P, U) ) {
      j++;
      double score = weights[i];
      if (j == 0){
	factor = fmax(score, w);
	sum += exp(w - factor);
      }
      else if (score < factor + slack) { break; }
      sum += exp(score - factor);
    }
  }
  if (j == -1) {return w;}
  return log(sum) + factor;
}

pair<double, vector<int>> weight_sum_contribs_2(double w, bitmap *psets, int m, double *weights, int j, int n, vector<uint64_t> U_vec, vector<uint64_t> T_vec, int t_ub) {
  bitmap_2 T;
  T.s1 = T_vec[0];
  T.s2 = T_vec[1];
  bitmap_2 U;
  U.s1 = U_vec[0];
  U.s2 = U_vec[1];

  pair <double, vector<int>> w_and_contribs;
  vector<int> contribs(0);
  double sum = 0;
  double slack = log(0.1/t_ub);
  j = -1;
  double factor = 0;
  for (int i = 0; i < m; i ++){
    bitmap P1 = psets[idx2d(i, 0, 2)];
    bitmap P2 = psets[idx2d(i, 1, 2)];
    bitmap_2 P;
    P.s1 = P1;
    P.s2 = P2;

    if ( intersects_2(P, T) && subseteq_2(P, U) ) {
      j++;
      double score = weights[i];
      if (j == 0){
	factor = fmax(score, w);
	sum += exp(w - factor);
      }
      else if (score < factor + slack) { break; }
      sum += exp(score - factor);
      contribs.insert(contribs.end(), i);
    }
  }
  w_and_contribs = make_pair(log(sum) + factor, contribs);
  return w_and_contribs;
}
*/
