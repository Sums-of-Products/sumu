#include "GroundSetIntersectSums.hpp"
#include <utility>
#include "common.hpp"

using std::vector;
using std::pair;
using std::make_pair;

bool decr_ws32(ws32 x, ws32 y) { return x.weight > y.weight; }
void sort_ws32(vector<ws32> &c){ sort(c.begin(), c.end(), decr_ws32); }

GroundSetIntersectSums::GroundSetIntersectSums(int K0, double *w, double eps_pruning0, double eps_score_sum0){
	K = K0; eps_pruning = eps_pruning0; eps_score_sum = eps_score_sum0;
	if (eps_pruning > 0) {prune(w);} else {init(w);}
	sort_ws32(s);
}

GroundSetIntersectSums::~GroundSetIntersectSums(){ }
void GroundSetIntersectSums::init(double *w){
	bm32 l = 1L << K;
	Treal* c = new Treal[l];
	for (bm32 S = 0; S < l; ++S){ c[S].set_log(w[S]); s.push_back({ S, c[S] });}
	delete[] c;
}

void GroundSetIntersectSums::prune(double *w){
	//	Pruning rule: given a tolerance of relative error eps > 0, prune S if
	//		w(S) < eps \sum_{j in R subset of S} w(R) 2^{|R| - K} for all j in S.
	//	(The RHS can be efficiently computed for all S by fast zeta transform.)
	//	Note: We assume the input argument w actually represents the values  ln w(S).
	bm32 l = 1L << K; int tol = (int) -log(eps_pruning)/log(2.0);
	Treal* a = new Treal[l], * b = new Treal[l], * c = new Treal[l]; bool* keepit = new bool[l];
	for (bm32 S = 1; S < l; ++S){ keepit[S] = false; } keepit[0] = true;
	for (bm32 R = 0; R < l; ++R){ a[R].set_log(w[R]); c[R] = a[R]; a[R] >>= K - popcount(R); c[R] <<= tol; }
	for (int j = 0; j < K; ++j){
		for (bm32 R = 0; R < l; ++R) if (R & (1L << j)) b[R] = a[R]; else b[R] = (int64_t) 0L;
		fzt_inpl(b, K);
		for (bm32 S = 1; S < l; ++S) if (S & (1L << j)) keepit[S] |= (c[S] > b[S]);
	}
	// Recall that c[] contain the original scores, divided by eps; now multiply it back.
	for (bm32 S = 0; S < l; ++S) if (keepit[S]){ c[S] >>= tol; s.push_back({ S, c[S] }); }

	delete[] a; delete[] b; delete[] c; delete[] keepit;
}

Treal GroundSetIntersectSums::scan_sum(bm32 U, bm32 T){
	int m = s.size(); int count = 0;
	Treal sum; sum = 0.0;
	Treal slack; slack = eps_score_sum/m;
	Treal factor; factor = 0.0;
	int i = 0;
	for (; i < m; ++i){
		bm32 P = s[i].set;
		if ( intersects(P, T) && subseteq(P, U) ) {
			sum = s[i].weight; factor = sum * slack;
			++i; ++count; break;
		}
	}
	for (; i < m; ++i){
		bm32 P = s[i].set;
		if ( intersects(P, T) && subseteq(P, U) ) {
			Treal score; score = s[i].weight;
			if (score < factor) break;
			sum += score; ++count;
		}
	}
	return sum;
}

pair<bm32, double> GroundSetIntersectSums::scan_rnd(bm32 U, bm32 T, double wcum){ // Returns the first set when the cumulative weight exceeds wcum.
	int m = s.size();
	Treal sum; sum = 0.0;
	Treal target; target.set_log(wcum);
	int i = 0; int P_i = 0; bm32 P = 0L;
	for (; i < m; ++i){
		P = s[i].set;
		if ( subseteq(P, U) && intersects(P, T) ) {
			P_i = i;
			sum = s[i].weight; ++i;
			if (sum > target) i = m;
			break;
		}
	}
	for (; i < m; ++i){
		P = s[i].set;
		if ( subseteq(P, U) && intersects(P, T) ) {
			P_i = i;
			Treal score = s[i].weight; sum += score;
			if (sum > target) break;
		}
	}
	return make_pair(s[P_i].set, s[P_i].weight.get_log());
}

Treal GroundSetIntersectSums::scan_sum(bm32 U){
	int m = s.size(); int count = 0;
	Treal sum; sum = 0.0;
	Treal slack; slack = eps_score_sum/m;
	Treal factor; factor = 0.0;
	int i = 0;
	for (; i < m; ++i){
		bm32 P = s[i].set;
		if ( subseteq(P, U) ) {
			sum = s[i].weight; factor = sum * slack;
			++i; ++count; break;
		}
	}
	for (; i < m; ++i){
		bm32 P = s[i].set;
		if ( subseteq(P, U) ) {
			Treal score; score = s[i].weight;
			if (score < factor) break;
			sum += score; ++count;
		}
	}
	return sum;
}

pair<bm32, double> GroundSetIntersectSums::scan_rnd(bm32 U, double wcum){
	int m = s.size();
	Treal sum; sum = 0.0;
	Treal target; target.set_log(wcum);
	int i = 0; int P_i = 0; bm32 P = 0L;
	for (; i < m; ++i){
		P = s[i].set;
		if ( subseteq(P, U) ) {
			P_i = i;
			sum = s[i].weight; ++i;
			if (sum > target) i = m;
			break;
		}
	}
	for (; i < m; ++i){
		P = s[i].set;
		if ( subseteq(P, U) ) {
			P_i = i;
			Treal score = s[i].weight; sum += score;
			if (sum > target) break;
		}
	}
	return make_pair(s[P_i].set, s[P_i].weight.get_log());
}
