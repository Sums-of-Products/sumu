// -*- flycheck-clang-language-standard: "c++17"; -*-
#include <cmath>
#include <cstdint>
#include <vector>

using namespace std;

using bitmap = uint64_t;

inline bool intersects(bitmap A, bitmap B){ return A & B; }
inline bool subseteq(bitmap A, bitmap B){ return (A == (A & B)); }

//double log_add(double x, double y){ return fmax(x, y) + log1p(exp( -fabs(x - y) )); }

double weight_sum(double w, bitmap *psets, int m, double *weights, int j, int n, bitmap U, bitmap T, int t_ub) {
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
    }
  }
  if (j == -1) {return w;}
  return log(sum) + factor;
}

pair<double, vector<int>> weight_sum_contribs(double w, bitmap *psets, int m, double *weights, int j, int n, bitmap U, bitmap T, int t_ub) {
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


/*
double _weight_sum(double lsum, bitmap *psets, int m, double *weights, int j, int n, bitmap U, bitmap T, int t_ub) {
  //double sum = lsum;
  double slack = log(0.1/t_ub);
  j = 0;
  double factor = 0;
  for (int i = 0; i < m; i ++){
    bitmap P = psets[i];
    if ( intersects(P, T) && subseteq(P, U) ) {
      double score = weights[i];
      if (j == 0){ factor = score; }
      else if (score < factor + slack) { break; }
      lsum = log_add(lsum, score);
      //sum += exp(score - factor);
      j++;
    }
  }
  //sum = log(sum) + factor;
  return lsum;
}
*/
