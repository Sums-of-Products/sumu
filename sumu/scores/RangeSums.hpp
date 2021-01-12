// Compile: g++ -Wall -O3 -ffast-math -funroll-loops
#ifndef RANGESUMS_HPP
#define RANGESUMS_HPP

#include <vector>
#include <algorithm>
#include "headers.hpp"
#include "Breal.hpp"

#define bm32 uint32_t
#define Treal	B2real 

using namespace std;

inline bm32 remove    (bm32 S, bm32 J){ bm32 Z = S & (J - 1L); return ((S - Z) >> 1) + Z; }
inline bm32 insert    (bm32 S, bm32 J){ bm32 Z = S & (J - 1L); return ((S - Z) << 1) + Z; }
inline int  popcount  (bm32 S){ int c = 0; while (S) { c += (S & 1L); S >>= 1; } return c; }
inline bm32 contract  (bm32 X,  bm32 Y){ // Encodes X wrt Y.
	bm32 Xt = 0; int j = 0;
	while (X){ bm32 Z = Y & 1L; Xt |= (X & Z) << j; j += Z; X >>= 1; Y >>= 1; } 
//	while (X){ if (Y & 1L){ Xt |= (X & 1L) << j; j += 1; } X >>= 1; Y >>= 1; }
	return Xt;
}
inline bm32 stretch (bm32 Xt, bm32 Y){ // Assumes Xt is encoded wrt Y, and decodes Xt. Could be faster...
	bm32 X = Y; int j = 0;
	while (Y){
		if (Y & 1L){ if (!(Xt & 1L)){ X ^= 1L << j; } Xt >>= 1; } 
		Y >>= 1; ++j;
	} return X;
}

// RangeSums -- To compute and store weighted sums over subset intervals. Requires O(3^K) space.
//
class RangeSums {
public:
	RangeSums(int K0);
	~RangeSums();
	void precompute(double* w);
	double rangesum(bm32 X, bm32 Y);	// Assumes X is a subset of Y, both in absolute coordinates.
private:
	int K;					// Maximum size of the sets considered.
	Treal** f;				// f[Y][Xt] is the rangesum of [X, Y], with Xt trimmed relative to Y.
};

RangeSums::RangeSums(int K0){
	K = K0; f = new Treal*[ 1L << K ];
	for (bm32 Y = 0; Y < (1L << K); ++Y) f[Y] = new Treal[ 1L << popcount(Y) ];
}

RangeSums::~RangeSums(){ for (bm32 Y = 0; Y < (1L << K); ++Y){ delete[] f[Y]; } delete[] f; }

double RangeSums::rangesum(bm32 X, bm32 Y){ return f[Y][contract(X, Y)].get_log(); }

void RangeSums::precompute(double* w){ // Speed: 158 pairs (X, Y) per microsecond, for K = 18.
	bm32 mypow[32] { 0 }; // Will store powers of 1-bits in Y. Init to zero to avoid compiler warnings.
	for (bm32 Y = 0; Y < (1L << K); ++Y){
		int lY = 0; for (int p = 0; p < K; ++p){ if (Y & (1L << p)) mypow[ lY++ ] = (1L << p); }
		bm32 Yt = (1L << lY) - 1L; // All 1s. Ytrim is Y relative to Y. 		
		f[Y][Yt].set_log(w[Y]); // Base case of the recurrence.
		for (int X = Yt - 1; X >= 0; --X){ // Note that X is relative to Y.
			bm32* P = mypow, J = 1L;
			while (X & J){ J <<= 1; ++P; } // J indicates the first 0-bit of X. Only 1 step on average.
			f[Y][X] = f[Y][ X | J ] + f[ Y ^ (*P) ][ remove(X, J) ];
		}
	}
}

// IntersectSums -- Given (U, T), computes the weighted sum over the subsets S of U that intersect T. 
// Assumpions: the sets are of size at most K <= 32. 
//
using bitmap = unsigned long long int;
#define setbit(S, i) (S |= (1L << i))
inline bool intersects(bm32 A, bm32 B){ return A & B; }
inline bool subseteq  (bm32 A, bm32 B){ return (A == (A & B)); } 
struct ws32 { bm32 set; Treal weight; };
bool incr_ws32(ws32 x, ws32 y) { return x.weight < y.weight; }
bool decr_ws32(ws32 x, ws32 y) { return x.weight > y.weight; }
void sort_ws32(vector<ws32> &c){ std::sort(c.begin(), c.end(), decr_ws32); }
void fzt_inpl(Treal* b, bm32 n){ // 550 adds per microsecond, Treal = Breal.
	for (bm32 i = 0; i < n; ++i) for (bm32 m = 0; m < (1L << n); ++m) if (m & (1L << i)) b[m] += b[m ^ (1L << i)];
}
//
class IntersectSums {
public:
	IntersectSums(int K0, double* w, double eps0);
	~IntersectSums();
	double scan_sum(bm32 U, bm32 T);
	bm32   scan_rnd(bm32 U, bm32 T, double wcum);

private:
	int				K;	// Size of the ground set.
	vector<ws32>	s;	// Scores, pairs (set, weight) sorted by weight.
	double 			eps;// Tolerated relative error. 

	void prune(double *w);	
};

IntersectSums::IntersectSums(int K0, double *w, double eps0){ K = K0; eps = eps0; prune(w); sort_ws32(s); }
IntersectSums::~IntersectSums(){ }
void IntersectSums::prune(double *w){
//	Pruning rule: given a tolerance of relative error eps > 0, prune S if
//		w(S) < eps \sum_{j in R subset of S} w(R) 2^{|R| - K} for all j in S.
//	(The RHS can be efficiently computed for all S by fast zeta transform.)
//	Note: We assume the input argument w actually represents the values  ln w(S).
	bm32 l = 1L << K; int tol = (int) -log(eps)/log(2.0);
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
	cout << " l = " << l << ", after pruning: " << s.size() << ", tol = " << tol << endl;
}

double IntersectSums::scan_sum(bm32 U, bm32 T){
	int m = s.size(); int count = 0;
	Treal sum; sum = 0.0; 
	Treal slack; slack = eps/m; 
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
	cout << " (count = " << count << ") ";
	return sum.get_log();
}
bm32 IntersectSums::scan_rnd(bm32 U, bm32 T, double wcum){ // Returns the first set when the cumulative weight exceeds wcum.
	int m = s.size();
	Treal sum; sum = 0.0; 
	Treal target; target.set_log(wcum);
	int i = 0; bm32 P = 0L;
	for (; i < m; ++i){
		P = s[i].set; 
		if ( subseteq(P, U) && intersects(P, T) ) {
			sum = s[i].weight; ++i; 
			if (sum > target) i = m; 
			break;
		}
	} 
	for (; i < m; ++i){
		P = s[i].set; 
		if ( subseteq(P, U) && intersects(P, T) ) {
			Treal score = s[i].weight; sum += score; 
			if (sum > target) break; 
		}
	}
	return P;
}


//#undef bm32
#undef Treal 
#undef setbit

#endif
