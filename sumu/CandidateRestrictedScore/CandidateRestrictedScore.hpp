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


class CandidateRestrictedScore {
public:

  int m_n; // Number of variables.
  int m_K; // Number of candidate parents.

  double* m_score_array;
  double** m_tau_simple; // Matrix of size n x 2^K.
  unordered_map< bm64, double > * m_tau_cc;
  int** m_C; // Matrix of size n x K.
  double m_tolerance; // Used as threshold for preparing for cc.

  CandidateRestrictedScore(double* scores, int* C, int n, int K, double tolerance);
  ~CandidateRestrictedScore();
  double sum(int v, bm32 U, bm32 T);
  double get_cc(int v, bm64 key);
  double get_tau_simple(int v, bm32 U);

  double test_sum(int v, bm32 U, bm32 T);

private:

  void precompute_tau_simple();
  void precompute_tau_cc_basecases();
  void precompute_tau_cc();
  void rec_dfs(int v, bm32 U, bm32 R, bm32 T);

};

CandidateRestrictedScore::CandidateRestrictedScore(double* score_array, int* C, int n, int K, double tolerance) {


  m_score_array = score_array;
  m_tolerance = tolerance;
  m_n = n;
  m_K = K;

  m_tau_simple = new double*[n];
  m_C = new int*[n];
  m_tau_cc = new unordered_map< bm64, double >[n];

  int i = 0;
  int j = 0;
  for (int v = 0; v < n; ++v) {
    m_tau_simple[v] = new double[ (bm32) 1 << K];
    m_C[v] = new int[K];
    m_tau_cc[v] = unordered_map< bm64, double >();

    for (bm32 s = 0; s < (bm32) 1 << K; ++s) {
      m_tau_simple[v][s] = score_array[i++];
    }

    for (int c = 0; c < K; ++c) {
      m_C[v][c] = C[j++];
    }

  }

  precompute_tau_simple();
  precompute_tau_cc_basecases();
  precompute_tau_cc();
}

CandidateRestrictedScore::~CandidateRestrictedScore() {

  delete [] m_tau_cc;
  for (int v = m_n - 1; v > -1; --v) {
    delete[] m_tau_simple[v];
    delete[] m_C[v];
  }

}


void CandidateRestrictedScore::precompute_tau_simple() {
  for (int v = 0; v < m_n; ++v) {
    zeta_transform_array_inplace(& m_tau_simple[v][0], 1 << m_K);
  }
}

void CandidateRestrictedScore::precompute_tau_cc_basecases() {
  for (int v = 0; v < m_n; ++v) {
    for (int k = 0; k < m_K; ++k) {
      bm32 j = 1 << k;
      bm32 U_minus_j = (( (bm32) 1 << m_K) - 1) & ~j;
      double* tmp = new double[1 << (m_K - 1)];
      tmp[0] = m_score_array[v*( (bm32) 1 << m_K) + j]; // Make indexing inline function?

      // Iterate over subsets of U_minus_j.
      for (bm32 S = U_minus_j; S > 0; S = (S-1) & U_minus_j) {
	tmp[dkbit_32(S, k)] = m_score_array[v*( (bm32) 1 << m_K) + (S | j)];
      }
      zeta_transform_array_inplace(& tmp[0], (bm32) 1 << (m_K - 1));

      for (bm32 S = 0; S < (bm32) 1 << (m_K - 1); ++S) {
	bm32 U = ikbit_32(S, k, 1);
	if (close(m_tau_simple[v][U],
		  m_tau_simple[v][U & ~j], m_tolerance)) {
	  m_tau_cc[v].insert({(bm64) U << 32 | j, tmp[S]});
	}
      }

      delete[] tmp;
    }
  }
}

void CandidateRestrictedScore::precompute_tau_cc() {
  for (int v = 0; v < m_n; ++v) {
    for (bm32 U = 0; U < (bm32) 1 << m_K; ++U) {
      rec_dfs(v, U, U, 0);
    }
  }
}

void CandidateRestrictedScore::rec_dfs(int v, bm32 U, bm32 R, bm32 T) {
  if (close(m_tau_simple[v][U], m_tau_simple[v][U & ~T], m_tolerance)) {
    bm32 T1 = T & ~T;
    bm32 U1 = U & ~T1;
    bm32 T2 = T & ~T1;
    m_tau_cc[v][(bm64) U << 32 | T] = log_add_exp(sum(v, U, T1), sum(v, U1, T2));
  }
  else {
    return;
  }
  while (R) {
    bm32 B = lsb_32(R);
    R = R ^ B;
    rec_dfs(v, U, R, T | B);
  }
}


double CandidateRestrictedScore::sum(int v, bm32 U, bm32 T) {

  if (U == 0 && T == 0) {
    return m_score_array[v*(1 << m_K)];
  }

  // Is this needed still?
  if (T == 0) {
    return -inf;
  }

  if (m_tau_cc[v].count((bm64) U << 32 | T)) {
    return m_tau_cc[v][(bm64) U << 32 | T];
  }

  return log_minus_exp(m_tau_simple[v][U], m_tau_simple[v][U & ~T]);
}

double CandidateRestrictedScore::get_cc(int v, bm64 key) {
  return m_tau_cc[v][key];
}

double CandidateRestrictedScore::get_tau_simple(int v, bm32 U) {
  return m_tau_simple[v][U];
}


double CandidateRestrictedScore::test_sum(int v, bm32 U, bm32 T) {

  if (U == 0 && T == 0) {
    cout << "empty pset" << endl;
    return m_score_array[v*(1 << m_K)];
  }

  // Is this needed still?
  if (T == 0) {
    cout << "T == 0, U != 0" << endl;
    return -inf;
  }

  if (m_tau_cc[v].count((bm64) U << 32 | T)) {
    cout << "cc cache: " << m_tau_cc[v][(bm64) U << 32 | T] << endl;
    return m_tau_cc[v][(bm64) U << 32 | T];
  }

  cout << "subtraction" << endl;
  return log_minus_exp(m_tau_simple[v][U], m_tau_simple[v][U & ~T]);
}
