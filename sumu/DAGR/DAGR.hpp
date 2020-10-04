// -*- flycheck-clang-language-standard: "c++17"; -*-

#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <limits>
#include <iostream>

#include "../zeta_transform/zeta_transform.h"
#include "../bitmap/bitmap.hpp"

#define bm32 uint32_t
#define bm64 uint64_t
using namespace std;


// Is this the way to go?
double inf{std::numeric_limits<double>::infinity()};


// Move these functions somewhere

// Could optimize by knowing tau(U) > tau(U-T)
bool close(double a, double b, double tolerance) {
  return fmax(a, b) - fmin(a, b) < tolerance;
}

double log_minus_exp(double p1, double p2) {
  if (exp(fmin(p1, p2) - fmax(p1, p2)) == 1) {
    return -inf;
  }
  return fmax(p1, p2) + log1p(-exp(fmin(p1, p2) - fmax(p1, p2)));
}

inline double log_add_exp(double p1, double p2) {
  return fmax(p1, p2) + log1p(exp(fmin(p1, p2) - fmax(p1, p2)));
}


class DAGR {
public:

  DAGR(double* score_array, int* C, int n, int K, double tolerance);
  void precompute(int v);
  double f(bm32 X, bm32 Y);

private:

  int m_n; // Number of variables.
  int m_K; // Number of candidate parents.

  double* m_score_array;
  double** m_f; // 2^K arrays of varying lengths
  bm32* m_f_n; // need to store the lenghts?

  int** m_C; // Matrix of size n x K.
  double m_tolerance; // Used as threshold for preparing for cc.

};


DAGR::DAGR(double* score_array, int* C, int n, int K, double tolerance) {

  m_score_array = score_array;
  m_tolerance = tolerance;
  m_n = n;
  m_K = K;

  m_C = new int*[n];
  m_f = new double*[((bm32) 1 << m_K)];

  int i = 0;
  for (int v = 0; v < n; ++v) {
    m_C[v] = new int[K];
    for (int c = 0; c < K; ++c) {
      m_C[v][c] = C[++i];
    }
  }

}


void DAGR::precompute(int v) {

  bm32 c_X;
  bm32 r_X;
  bm32 c_Y;
  bm32 r_Y;

  for (bm32 X = 0; X < ((bm32) 1 << m_K); ++X) {
    int k = count_32(X);
    m_f[X] = new double[(bm32) 1 << (m_K - k)];
    m_f[X][0] = m_score_array[v*((bm32) 1 << m_K) + X];
    for (bm32 i = 1; i < (bm32) 1 << (m_K - k); ++i) {
      m_f[X][i] = -inf;
    }
  }

  for (int k = 1; k < m_K + 1; ++k) {
    for (int k_x = 0; k < m_K - k + 1; ++k) {

      // Loop through subsets X of size k_x of K bits
      bm32 X = ((bm32) 1 << k_x) - 1;
      bm32 limit_X = (bm32) 1 << m_K;
      while (X < limit_X) {

	// Loop through subsets Y of size k of K - k_x bits
	bm32 Y = ((bm32) 1 << k) - 1;
	bm32 limit_Y = (bm32) 1 << (m_K - k_x);
	while (Y < limit_Y) {

	  int i = fbit_32(Y);
	  m_f[X][Y] = log_add_exp(m_f[kzon_32(X, i)][dkbit_32(Y, i)], m_f[X][Y & ~(Y & ~Y)]);

	  // Next subset Y
	  c_Y = Y & - Y;
	  r_Y = Y + c_Y;
	  Y = (((r_Y ^ Y) >> 2) / c_Y) | r_Y;

	}

	// Next subset X
	if (X == 0) {
	  X = limit_X;
	}
	else {
	  c_X = X & - X;
	  r_X = X + c_X;
	  X = (((r_X ^ X) >> 2) / c_X) | r_X;
	}

      }

    }
  }
}


double DAGR::f(bm32 X, bm32 Y) {
  return m_f[X][Y];
}
