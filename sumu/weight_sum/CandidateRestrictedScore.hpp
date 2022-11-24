#ifndef CANDIDATE_RESTRICTED_SCORE_HPP
#define CANDIDATE_RESTRICTED_SCORE_HPP

#include "GroundSetIntersectSums.hpp"
#include <unordered_map>
#include <utility>
#include <vector>
#include "common.hpp"

using std::unordered_map;
using std::string;
using std::streambuf;
using std::pair;
using namespace wsum;

class CandidateRestrictedScore {
public:

	int m_n; // Number of variables.
	int m_K; // Number of candidate parents.

	Treal* m_score_array;
	Treal** m_tau_simple; // Matrix of size n x 2^K.
	unordered_map< bm64, Treal > * m_tau_cc;

	int** m_C; // Matrix of size n x K.
	Treal m_cc_tol;
	int m_cc_limit;

	CandidateRestrictedScore(double* scores, int* C, int n, int K, int cc_limit, double cc_tol, double pruning_eps, double score_sum_eps,
	                         bool silent, int debug
		);
	~CandidateRestrictedScore();
	Treal sum(int v, bm32 U, bm32 T, bool isum=false);
	Treal sum(int v, bm32 U);
	pair<bm32, double> sample_pset(int v, bm32 U, bm32 T, double wcum);
	pair<bm32, double> sample_pset(int v, bm32 U, double wcum);
	int number_of_scoresums_in_cache(int v);
	void precompute_tau_simple();
	std::vector<double> precompute_tau_cc_basecases();
	std::vector<double> precompute_tau_cc();

private:

	GroundSetIntersectSums **isums;
	Timer timer;
	void rec_it_dfs(int v, bm32 U, bm32 R, bm32 T, int T_size, int T_size_limit, int * count);

};




#endif
