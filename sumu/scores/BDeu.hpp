// Compile with g++ using options -Wall -O3 

#ifndef BDEU_HPP
#define BDEU_HPP

#include <iostream> 
#include <fstream> 
#include <iomanip> 
#include <math.h> 
#include <string.h>
#include <stdlib.h>
#include "Wsets.hpp"
#include "HashCounter.hpp"

using namespace std;

// Param -- a fingerprint of the arguments used in the call for scoring a bunch of families (i, Y). 
struct Param { 
	vector<int>	C;		// The given set of candidates; the labeling of Y is relative to C.
	int 		u;		// The given upper bound for the size |Y|.
	bool 		isdense;	// The truth value of (|C| <= 64); determines the encoding of Y. 
};


#define Tdat 		int //uint8_t
#define BDEU_MAXN 	256
#define BDEU_MAXARITY 	256
//#define BDEU_MAXM	1024*1024	
class BDeu {
    public:
	int    		m;		// Number of data points.
	int    		n;		// Number of variables.
	Tdat** 		dat;		// Data matrix of size n x m.
	int*   		r;		// Number of values per variable.
	int*   		w;		// Number of bits reserved per variable, ceil of log2(r). 
	double 		ess;		// Equivalent sample size parameter for the BDeu score.
	vector<wset>* 	fscores;	// Lists of family scores.
	vector<Param>	fparams;	// Parameters used in score computations.

	BDeu (); ~BDeu ();
	void   test ();					// Shows some test runs. 
	void   query_test ();				// Shows some test runs. 
	int    read (int* datavec, int m0, int n0);	// Reads data matrix of size m0 x n0 given as vector datavec.
	int    dear (int* datavec, int m0, int n0);	// Reads data matrix of size m0 x n0 given as vector datavec.
	int    read (string fname);			// Reads data matrix given as a csv file.
	void   set_ess(double val);			// Set the value of the ess parameter.

	double cliq (int* X, int lX);			// Scores clique X of size lX.
	double cliq (int* S, int lS, int u);		// Scores and stores all subsets of S of size at most u.
	double cliqc(int* X, int lX);			// Scores clique X of size lX, with caching.
	void   clear_cliq ();				// Frees the memory reserved for indexing clique scores.
	void   clear_cliqc();				// Frees the memory reserved for caching  clique scores.

	double fami (int i, int* Y, int lY);		// Scores family (i, Y), with Y of size lY.
	double fami (int* S, int lS, int u);		// Scores and stores for all variables and all parent sets of size at most u.
	double fami (int i, int* C, int lC, int u);	// Scores and stores for all (i, Y) with Y a subset of C of size at most u.
	void   clear_fami();				// Frees the memory reserved for indexing fami scores.
	void   clear_fami(int i);			// Frees the memory reserved for indexing fami scores of node i.

	double fscore(int i, int* Y, int lY);		// Fecthes an already compute score of family (i, Y), with Y of size lY.

	friend ostream& operator<<(ostream& os, const BDeu& x); // Currently just prints out the data matrix.

    private:
	vector<wset>  	cscores;		// List of clique scores.
	Wsets         	ascores;		// Storage for arbitrary (clique, score) pairs.
	int**   	tmp;			// Helper array of size n x m;
	uint32_t*	fre;			// Helper array of size m+1 to store a multiset of counts, indexed by counts. 
	double* 	lng;			// Helper array of size m+1 to store ln(Gamma(j + ess')/Gamma(ess')).
	int**   	prt;			// Helper array of size n x m, yet another;
	int binomsum[BDEU_MAXN][32];		// Tail sums of binomial coefficients. Would be better make this static.
	double 		base_delta_lgamma;	// lgamma(m + ess) - lgamma(ess);
	bool 		initdone;		// Flag: whether mem dynamically allocated (to be released at the end). 	

	double score_dfs (int d, int* X, int a, int b, double q);
	void   score_dfs3(int d, int* X, int a, int b);
	double score_dico(int d, int* X);
	double score_hash(int d, int* X);
	double score_has2(int d, int* X);
	double score_radi(int d, int* X);
	double score_all ();			// Just for testing.

	double cliqa(int* S, int lS, int u);	// Scores and stores all subsets of S of size at most u that contain the last of S.
	double dfo2 (int* X, int lX, int lmax, int* S, int jmin, int jmax, int mleft, double rq, bool isdense); // Bunch of scores rec.
	double score_all1();			// Just for testing.

	int     index(int *X, int lX, int u);	// Index to cscores, relative to the S used when built; X in dec order. 
	int    iindex(int *X, int lX, int u);	// Index to cscores, relative to the S used when built; X in inc order. 
	void preindex(int umax);		// Computes the array binomsum[][].

	int width(int d, int* X);

	void print_tmp();
	void init(int m0, int n0);	// Initializes data structures of size m0 x n0.
	void fini();			// Deletes the dynamically allocated data structures. 
	void fill_rnd(int maxr);	// Fills the data matrix in a random  way, just for testing purposes. 
	void set_r();			// Computes and stores the arities.
	void set_r(const vector<int> &v);
};

void BDeu_demo (){ // Demo of the usage of the class. Might define this as a static function of the class, but not done here.
	const string red("\033[0;31m"); const string green("\033[0;32m"); const string cyan("\033[0;36m"); const string reset("\033[0m");
#define STRV(...) #__VA_ARGS__
#define DEMO(command, comment) cout << green << " " << STRV(command) << cyan << "  // " << comment << reset << endl; command 
#define COMMA ,
	cout<<" === DEMO of class BDeu === "<<endl;
	DEMO(BDeu s;, "Introduce a BDeu object")
	DEMO(int *x = new int[15 * 7];, "The original data matrix.")
	DEMO(for (int i = 0; i < 15 * 7; ++i) x[i] = rand() % 5;, "Now fill with random entries.")
	DEMO(s.read(x COMMA 15 COMMA 7);, "Feed x to s as a matrix with 15 datapoints, each over 7 variables.")
	DEMO(delete[] x;, "Now we may delete x. This emphasizes that s made a copy of x.")
	DEMO(cout << s;, "Let's print out the data matrix.")
	DEMO(s.set_ess(10.0);, "Set a value for the ess parameter.")
	DEMO(int par[4] = {4 COMMA 6 COMMA 0 COMMA 1};, "Create a parent set. Note: the order does not matter.")
	DEMO(double score = s.fami(2 COMMA par COMMA 4);, "Get the score of variable 2 with parent set par.")
	DEMO(cout << " Got score = " << score << endl;, "Print out the score we got.")
	DEMO(int S[7] = {0 COMMA 1 COMMA 2 COMMA 3 COMMA 4 COMMA 5 COMMA 6};, "Define S simply as the set of all variables.")
	DEMO(s.fami(S COMMA 7 COMMA 6);, "Compute the scores for all families (i, Y) with i an element of S and Y a subset, |Y| <= 6.")
	DEMO(int Y[4] = {0 COMMA 1 COMMA 3 COMMA 5};, "Take Y to be the same as par above, but relative to S and in INC order.")
	DEMO(score = s.fscore(2 COMMA Y COMMA 4);, "Get the score of variable 2 with parent set Y; notice the function name.")
	DEMO(cout << " Got score = " << score << endl;, "The score is the same as before, as we expected.")
	DEMO(score = s.fscores[2][1+2+8+32].weight;, "In fact, we may index directly by the bitmap representation of Y.")
	DEMO(cout << " Got score = " << score << endl;, "Again, the score is the same as before, as we expected.")
	DEMO(int C2[5] = {0 COMMA 1 COMMA 3 COMMA 4 COMMA 6};, "Define C2 as a set of candidate parents.")
	DEMO(s.fami(2 COMMA C2 COMMA 5 COMMA 5);, "Compute the scores for all families (2, Y) and Y a subset of C2, |Y| <= 5.")
	DEMO(int Z[4] = {0 COMMA 1 COMMA 3 COMMA 4};, "Take Z to be the same as par above, but relative to C2 and in INC order.")
	DEMO(score = s.fscore(2 COMMA Z COMMA 4);, "Get the score of variable 2 with parent set Z; notice the function name.")
	DEMO(cout << " Got score = " << score << endl;, "The score is the same as before, as we expected.")
	DEMO(score = s.fscores[2][1+2+8+16].weight;, "In fact, we may index directly by the bitmap representation of Y.")
	DEMO(cout << " Got score = " << score << endl;, "Again, the score is the same as before, as we expected.")

	cout << " ======  end of demo  ===== " << endl;
#undef COMMA 
#undef DEMO
#undef STRV
}

//////////////////
// Public methods:

// Read the data matrix. Assumes m0 datapoints over n0 variables, given in the order of datapoints.
int BDeu::read(int* datavec, int m0, int n0){
	init(m0, n0); // Sets m and n among other things.
	int j = 0; for (int t = 0; t < m; ++t){ for (int i = 0; i < n; ++i) dat[i][t] = datavec[j++]; }
	set_r(); return 1;
}
// Read the data matrix. Assumes m0 datapoints over n0 variables, given in the order of variables.
int BDeu::dear(int* datavec, int m0, int n0){
	init(m0, n0); // Sets m and n among other things.
	int j = 0; for (int i = 0; i < n; ++i){ for (int t = 0; t < m; ++t) dat[i][t] = datavec[j++]; }
	set_r(); return 1;
}
int BDeu::read(const string fname){
	ifstream infile; infile.open(fname); if (infile.fail()){ cerr << " Error in opening file: " << fname << endl; return 0; }
	vector<string> lines; for (std::string line; getline(infile, line); ) lines.push_back(line); infile.close();
	int m0 = lines.size() - 2; int n0 = 0; int j = 0; int* values = NULL; vector<int> arities;
	for (int t = 1; t < m0 + 2; ++t){ // Note: Skip the first line.
		stringstream ss(lines.at(t)); int value; vector<int> v;
		while (ss >> value){ v.push_back(value); if (ss.peek() == ',') ss.ignore(); } // Also a comma is a valid separator.
		if (t == 1){ arities = v; n0 = v.size(); values = new int[m0 * n0]; continue; }
		for (int i = 0; i < n0; ++i) values[j++] = v.at(i);
	}
	read(values, m0, n0); delete[] values; set_r(arities); return 1;
}
void BDeu::set_ess(double val){ ess = val; base_delta_lgamma = lgamma(ess) - lgamma(m+ess); }

// Returns the BDeu score of the clique X of size d. 
double BDeu::cliq (int* X, int d){ 
	if (d == 0) return 0; 
	int wc = width(d, X);					// Number of bits needed to encode a data point. 			
	if      (1 < d && wc < 64)  return score_hash(d, X);	// Very fast, implemented only up to 64-bit data records.
	else if (1 < d && wc < 128) return score_has2(d, X);	// Fast, currently implemented up to 128-bit data records.
	else if (d < 2)             return score_radi(d, X);	// Might be fastest for very small d. 		
	else                        return score_dico(d, X);	// Somewhat carefully optimized divide & conquer.		
}
// Returns the BDeu score of the clique X of size d, with caching. Note: currently caching assumes the ground set has size at most 64. 
double BDeu::cliqc(int* X, int d){ 
	double val;   if (ascores.get(X, d, &val)) return val;	// If already stored, then just get and return it.
	val = cliq(X, d); ascores.put(X, d,  val); return val;	// Push the computed score to the storage--requires mem!
}
// Scores ALL subsets of S of size at most u. Returns the sum of the scores. Stores the scores.
double BDeu::cliq (int* S, int d, int u){ 
	int* X = new int[d]; for (int j = 0; j < d; ++j) X[j] = S[j]; // This loop should have no effect at the end.
	for (int t = 0; t < m; ++t){ tmp[0][t] = t; } // Init tmp[0][]. No need to init prt[].
	cscores.clear(); cscores.shrink_to_fit(); // Completely clears and resizes cscores.
	preindex(31);	// Computes binomsum[][]. For some unknown reason not sufficient to do this in the init() function.
	cscores.reserve(binomsum[d][u]); // Note: indexing will be relative to S. 
	double scoresum = dfo2(X, 0, u, S, 0, d, m, 1.0, (d <= 64)); // Visit and score in depth-first order.
	delete[] X; return scoresum;	
}
// Frees the memory allocated by ascores, a cahche for arbitrary clique score queries.
void BDeu::clear_cliqc(){ ascores.clear(); /* Note that ascores is not a vector but a Wsets object.*/ }
// Frees the memory allocated by cscores, an array indexing scores of regural clique collections.
void BDeu::clear_cliq (){ cscores.clear(); cscores.shrink_to_fit(); }


// Returns the BDeu score of the family (i, Y) where Y is of size d. 
double BDeu::fami (int i, int* Y, int d){
	double  sp = cliq(Y, d);
	int*     X = new int[d+1]; X[d] = i; for (int j = 0; j < d; ++j) X[j] = Y[j]; 
	double spi = cliq(X, d+1); 
	delete [] X; return spi - sp; 
}
// Scores ALL families (i, Y) where i is an element of S and Y is a subset of S of size at most u.
double BDeu::fami(int* S, int lS, int u){
	clear_fami();
	int uu = min(u+1, lS); bool isdense = (lS <= 64); 
	double scoresum = cliq(S, lS, uu); // Note: Compute clique scores with u+1. Note: "isdense" not passed, but rediscovered.
	int ll = cscores.size(); int* X = new int[uu]; int lX; int* Y = new int[uu]; int* Z = new int[uu];
	for (int j = 0; j < lS; ++j) fscores[j].reserve(ll+1); // Note: Currently reserving a bit too much.
	for (int k = 0; k < ll; ++k){ // Go through all the cliques X.
		wset Xw = cscores.at(k); get_set(Xw.set, X, lX, isdense); // Note: X is in DEC order. Pass "isdense".
		int lY = lX - 1;
		for (int h = 0; h < lY; ++h){ Y[h] = X[h+1]; Z[h] = Y[h]; } // Init Y and its proxy Z with the smallest elements.
		for (int j = 0; j < lX; ++j){
			int i = X[j]; // Child i. Parents Y := X\{i}. *** NOTE: i IS RELATIVE TO S, WHEREAS S[i] IS THE REAL VAR.
			if (j > 0){ Y[j-1] = X[j-1]; Z[j-1] = Y[j-1]-1; } //  A bit clumsy, but works.
			int indY = index(Y, lY, uu); wset Yw = cscores.at(indY); // Note: computing the index does take O(lY) time.
			// Now, indY and Yw are wrt S. However, we want to represent Y wrt S\{i}, for tight indexing etc...
			wset Yiw = get_wset(Z, lY, Xw.weight - Yw.weight, isdense); fscores[i].push_back(Yiw);
			// Sanity check:
			if ((int) index(Z, lY, u) != (int) fscores[i].size()-1){ cerr << " *** ERROR *** EXIT NOW \n"; exit(1); }
		}
	}
	for (int j = 0; j < lS; ++j){ // Set fparams.
		fparams[j].u = u; fparams[j].isdense = isdense;
		fparams[j].C.clear(); fparams[j].C.shrink_to_fit();
		for (int h = 0; h < 5; ++h){ if (h != j) fparams[j].C.push_back(S[h]); } 
	}
	delete[] X; delete[] Y; delete[] Z; 
	return scoresum;
}
// Scores ALL families (i, Y) where i is NOT an element of C and Y is a subset of C of size at most u.
double BDeu::fami(int i, int* C, int lC, int u){
	clear_fami(i);	
	double scoresum = cliq (C, lC, u);	// Begin by simply computing clique scores over C. The scores are stored in cscores.
	int l1 = cscores.size();		// This many scores were inserted.	
	int lS = lC+1; int* S = new int[lS]; for (int j = 0; j < lC; ++j) S[j] = C[j]; S[lC] = i; // S = C U {i}.
	scoresum       += cliqa(S, lS, u+1);	// Continue by adding scores of cliques over S that contain the last element of S.
	int l2 = cscores.size();		// The total number of computed clique scores. 

	fscores[i].reserve(l2/2); 		// Exactly what is needed, since (i, Y) is obtained from Y and Y U {i}. 
	int* X = new int[u+1]; int lX; int* Y = new int[u]; int lY;	  
	for (int k = l1; k < l2; ++k){ // Go through all the cliques X that contain i. In fact, i is encoded as lS-1.
		wset Xw = cscores.at(k); get_set(Xw.set, X, lX, (lS <= 64));	// Note: X is in DEC order. Pass "isdense".
		lY = lX - 1; for (int h = 0; h < lY; ++h) Y[h] = X[h+1];	// Get Y. Could simplify to "Y = X+1".
		int indY = index(Y, lY, u); wset Yw = cscores.at(indY);		// Y is among the first list of cliques.		
		wset Yiw = get_wset(Y, lY, Xw.weight - Yw.weight, (lC <= 64));	// The encoding of Y is wrt C. 
		fscores[i].push_back(Yiw);
		// Sanity check:
		if (indY != (int) fscores[i].size()-1){ cerr << " *** ERROR, k = "<< k <<" *** EXIT NOW \n"; exit(1); }
	}
	//cout<<" [fami:] (i, fscores[i].size) = "; for (int i = 0; i < n; i ++){ cout<<"("<<i<<", "<<fscores[i].size()<<"); "; } cout<<endl;
	// Set fparams.
	fparams[i].u = u; fparams[i].isdense = (lC <= 64);
	fparams[i].C.clear(); fparams[i].C.shrink_to_fit();
	for (int h = 0; h < lC; ++h) fparams[i].C.push_back(S[h]); 
	delete[] S; delete[] X; delete[] Y; 
	return scoresum;
}
// Frees the memory allocated by cscores, an array indexing scores of regural family collections.
void BDeu::clear_fami (int i){ fscores[i].clear(); fscores[i].shrink_to_fit(); }
void BDeu::clear_fami (     ){ for (int i = 0; i < n; ++i) clear_fami(i); }

double BDeu::fscore(int i, int* Y, int lY){ // We assume Y is in increasing order and relative to the given candidate set. 
	int u = fparams[i].u; int ind = iindex(Y, lY, u); 
	return fscores[i][ind].weight;
}


///////////////////
// Private methods: 

int BDeu::width(int d, int *c){ // How many bits occupied by the variables in c.
	int wc = 0; for (int j = 0; j < d; ++j) wc += w[c[j]]; return wc;
}
void BDeu::fill_rnd(int maxr){ // For testing purposes. Could implement by calling read(vector<int>).
	srand(maxr);
	for (int i = 0; i < n; ++i){
		int mr = 2 + (rand() % (maxr - 1));
		for (int t = 0; t < m; ++t){ int v = rand() % mr; dat[i][t] = v; }
	}
	set_r();
}
void BDeu::set_r(){ // Sets r according to the data.
	for (int i = 0; i < n; ++i){
		r[i] = 0; for (int t = 0; t < m; ++t){ if (dat[i][t] > r[i]) r[i] = dat[i][t]; } ++r[i];
		int v = r[i] - 1; w[i] = 0; while (v) { ++w[i]; v >>= 1; } 
	}
}
void BDeu::set_r(const vector<int> &vr){ // Sets s according to the given vector.
	for (int i = 0; i < (int) vr.size(); ++i){ 
		r[i] = vr.at(i); int v = r[i] - 1; w[i] = 0; while (v) { ++w[i]; v >>= 1; }
	} 
}

// Speed: n = 15, m = 1000, sets per microsecond: 0.0027. Old mildly optmized code.
double BDeu::score_dfs(int d, int* c, int a, int b, double q){
	double s = lgamma(ess/q) - lgamma(b - a + ess/q);
	int i = c[d]; q *= r[i];	
	// Partition tmp[] into sublists. First scan and count; then partition.
	int num[BDEU_MAXARITY]; for (int k = 0; k < r[i]; ++k) num[k] = 0; int pos[BDEU_MAXARITY]; pos[0] = 0;
	for (int t = a; t < b; ++t){ int k = dat[i][tmp[d][t]]; ++num[k]; }
	if (d == 0){ // For the sake of speed, do not recurse, but solve it here.
		for (int k = 0; k < r[i]; ++k) s += lgamma(num[k] + ess/q) - lgamma(ess/q);	 
		return s;
	}
	for (int k = 0; k < r[i]-1; ++k) pos[k+1] = pos[k] + num[k];
	for (int t = a; t < b; ++t){ int k = dat[i][tmp[d][t]]; tmp[d-1][a+pos[k]] = tmp[d][t]; ++pos[k]; }
	for (int k = 0; k < r[i]; ++k){
		int bk = a + pos[k], ak = bk - num[k];
		s += lgamma(bk - ak + ess/q) - lgamma(ess/q);
		if (ak < bk) s += score_dfs(d-1, c, ak, bk, q);	
	}
	return s;
}
// Speed: n = 15, m = 1000, maxr = 5, sets per microsecond: 0.038. Optimized. NO RISKS. THE FASTEST DFS!
void BDeu::score_dfs3(int d, int* c, int a, int b){
	const int i = c[d];	
	// Partition tmp[] into sublists. First scan and count; then partition.
	int num[BDEU_MAXARITY]; for (int k = 0; k < r[i]; ++k) num[k] = 0;
	int pos[BDEU_MAXARITY]; pos[0] = 0;
	for (int t = a; t < b; ++t){ int k = dat[i][tmp[d][t]]; ++num[k]; }
	if (d == 0){ for (int k = 0; k < r[i]; ++k) ++fre[num[k]]; return; } // Solve the base case here.
	for (int k = 0; k < r[i]-1; ++k) pos[k+1] = pos[k] + num[k]; // Could get rid of pos[] and only use num[]; not implemented.
	for (int t = a; t < b; ++t){ int k = dat[i][tmp[d][t]]; tmp[d-1][a+pos[k]] = tmp[d][t]; ++pos[k]; }
	for (int k = 0; k < r[i]; ++k){
		switch (num[k]){
			case 0: break;
			case 1: ++fre[1]; break;
			case 2: { int j = d-1; int p = a+pos[k]-1; int t2 = tmp[j][p]; int t1 = tmp[j][--p];
				while ((j >= 0) && (dat[c[j]][t1] == dat[c[j]][t2])) --j; // Note: Switching to tad won't help. 
				if (j == -1) ++fre[2]; else fre[1] += 2; } break;
			default: int bk = a + pos[k]; score_dfs3(d-1, c, bk - num[k], bk); break;
		}	
	}
	return;
}
double BDeu::score_dico(int d, int* X){ // Divide and conquer.
	for (int t = 0; t < m; ++t) tmp[d-1][t] = t;					// Init tmp[d-1].
	score_dfs3(d-1, X, 0, m);
	int maxc = m; while (!fre[maxc]) --maxc;
	double q = 1; for (int j = 0; j < d; ++j) q *= r[X[j]]; 
	double essq = ess/q; lng[0] = 0; for (int c = 0; c < maxc; ++c) lng[c+1] = lng[c] + log(c + essq);		
	double s = base_delta_lgamma;							// lgamma(ess) - lgamma(m + ess);
	for (int c = 1; c <= maxc; ++c){ if (fre[c]) s += fre[c] * lng[c]; fre[c] = 0; }// Reset fre[].	
	return s;
}

double BDeu::score_radi(int d, int* X){ // Non-recursive radix partitioning.
	if (d == 0) return 0;
	for (int t = 0; t < m; ++t) tmp[0][t] = t;					
	int mleft = m; 
	for (int j = 0; j < d; ++j){
		int i = X[j]; int ri = r[i]; int tot = 0; int loc = 0; int p = 0; int pp = 0;
		int num[BDEU_MAXARITY];		
		while (tot < mleft){
			// Handle part p of the partition.
			int qmax = mleft; if (j) qmax = prt[j-1][p];
			// First, get num[].
			for (int k = 0; k < ri; ++k) num[k] = 0;
			for (int q = 0; q < qmax; ++q){ int t = tmp[j][tot+q]; int k = dat[i][t]; ++num[k]; }
			// Second, get prt[].
			int qact = 0; 
			for (int k = 0; k < ri; ++k){							
				switch (num[k]){ // *** NOTE: Here we fix the order of pp in relation to k !!!
					case 0: break; case 1: num[k] = 0; break; // Ignore singletons.
					default: prt[j][pp] = num[k]; ++pp; qact += num[k]; // Only here we can update qact.
				} 
			} 
			if (j < d-1){ 
				// Third, get tmp[].
				int pos[BDEU_MAXARITY]; pos[0] = num[0]; 
				for (int k = 0; k < ri-1; ++k) pos[k+1] = pos[k] + num[k+1];
				for (int q = 0; q < qmax; ++q){ 
					int t = tmp[j][tot+q]; int k = dat[i][t]; 
					if (num[k]){ --pos[k]; tmp[j+1][loc+pos[k]] = t; } 
				}
			}
			++p; tot += qmax; loc += qact; 
		}
		mleft = loc;
	}	
	// Finally, the case j = d-1.
	int p = 0; int maxc = 1; 
	for (int tot = 0; tot < mleft; ){ int c = prt[d-1][p]; ++p; maxc = max(maxc, c); tot += c; } 
	for (int c = 0; c <= maxc; ++c) fre[c] = 0;
	for (int q = 0; q < p; ++q){ int c = prt[d-1][q]; ++fre[c]; }
	fre[1] = m - mleft; // fre[1] = m - mleft, the "dark material"; other singletons should not exist.
	double rq = 1; for (int j = 0; j < d; ++j) rq *= r[X[j]]; 
	double essrq = ess/rq; double val = base_delta_lgamma; // lgamma(ess) - lgamma(m + ess); // THE SAME FOR ALL.	
	double baslg = lgamma(essrq); for (int c = 1; c <= maxc; ++c) if (fre[c]) val += fre[c] * (lgamma(c + essrq) - baslg); 
	for (int c = 0; c <= maxc; ++c) fre[c] = 0; 
	return val; 
}

double BDeu::score_hash(int d, int* X){ // By simply hashing. (Unordered map does not support proper count queries.)
	HashCounter h(m); 								// Hash. Uses a self-made data structure.
	uint64_t* z = new uint64_t[m]; 
	for (int t = 0; t < m; ++t) z[t] = dat[X[0]][t]; // Form a list of keys. 
	for (int j = 1; j < d; ++j){
		int i = X[j]; int l = w[i]; 
		for (int t = 0; t < m; ++t){ z[t] <<= l; z[t] |= dat[i][t]; } 		// Simply encode the data.
	}
	for (int t = 0; t < m; ++t) h.insert(z[t]);  					// Hash.
	int maxc = h.get_freq_and_reset(fre); 						// Get the count frequencies, member var fre[]. 
	double q = 1; for (int j = 0; j < d; ++j) q *= r[X[j]]; double essq = ess/q;	
	double baslg = lgamma(essq); double s = base_delta_lgamma; 			// lgamma(ess) - lgamma(m + ess);	
	for (int c = 1; c <= maxc; ++c) if (fre[c]){ s += fre[c] * (lgamma(c + essq) - baslg); fre[c] = 0; } // Finalize, reset fre[].	
	delete[] z; return s;
}
double BDeu::score_has2(int d, int* X){ // A bit slow.... By simply hashing, allowing for arbitrarily long keys.
	Has2Counter h(m); 								// Hash. Uses a self-made data structure.
	hc2key* zz = new hc2key[m]; 
	for (int t = 0; t < m; ++t) zz[t] = { (uint64_t) dat[X[0]][t], 0L }; // Form a list of keys. 	
	int ll = w[X[0]]; 
	for (int j = 1; j < d; ++j){
		int i = X[j]; int l = w[i]; ll += l; 
		if (ll <= 64) for (int t = 0; t < m; ++t){ zz[t].k1 <<= l; zz[t].k1 |= dat[i][t]; } 
		else          for (int t = 0; t < m; ++t){ zz[t].k2 <<= l; zz[t].k2 |= dat[i][t]; } 	
	}
//	delete[] zz; return 0;
	for (int t = 0; t < m; ++t) h.insert(zz[t]);  					// Hash.
	int maxc = h.get_freq_and_reset(fre); 						// Get the count frequencies, member var fre[]. 
	double q = 1; for (int j = 0; j < d; ++j) q *= r[X[j]]; double essq = ess/q;	
	double baslg = lgamma(essq); double s = base_delta_lgamma; 			// lgamma(ess) - lgamma(m + ess);	
	for (int c = 1; c <= maxc; ++c) if (fre[c]){ s += fre[c] * (lgamma(c + essq) - baslg); fre[c] = 0; } // Finalize, reset fre[].	 
	delete[] zz; return s;
}

double BDeu::score_all(){ // Scoring all nonempty subsets of the n variables.
	int* X = new int[n]; // Set of variables.
	for (int i = 0; i < n; ++i) X[i] =  i;
	double s = 0; int count = 0;
	for (int x = 1; x < (1 << n); ++x){
		int d = 0; for (int i = 0; i < n; ++i){ if (x & (1 << i)){ X[d++] = i; } } // Set X.
		s += cliq(X, d); ++count;
	}
	delete[] X; return s;
}
// Scores ALL subsets of S of size at most u that contain the last element of S. Adds the scores to the storage.
// This is quite a special function, and therefore defined as private. In correct calls: lS >= 1 and u >= 1.
double BDeu::cliqa(int* S, int lS, int u){ 
	int* X = new int[lS];  
	for (int t = 0; t < m; ++t){ tmp[0][t] = t; } // Init tmp[0][]. No need to init prt[].
	int ll = cscores.size(); cscores.reserve(ll + binomsum[lS-1][u-1]); // Note: indexing will be relative to S. 
	double scoresum = dfo2(X, 0, u, S, lS-1, lS, m, 1.0, (lS <= 64)); // Argument lS-1 makes it put X[0] = lS-1.
	delete[] X; return scoresum;	
}
// Subsets X of S in depth-first order. Gathers singleton into "dark material". This is quite a hack.  
double BDeu::dfo2(int* X, int lX, int lmax, int* S, int jmin, int jmax, int mleft, double rq, bool isdense){
//	if (lX == 1) cerr << " " << X[0];
	double val = 0;
	if (lX){ // Score X based on the counts that can be read from prt[lX-1][]. 		
		int p = 0; int maxc = 1; 
		for (int tot = 0; tot < mleft; ){ int c = prt[lX-1][p]; ++p; maxc = max(maxc, c); tot += c; } 
		for (int c = 0; c <= maxc; ++c) fre[c] = 0;
		for (int q = 0; q < p; ++q){ int c = prt[lX-1][q]; ++fre[c]; }
		fre[1] = m - mleft; // fre[1] = m - mleft, the "dark material"; other singletons should not exist.
		double essrq = ess/rq; val = base_delta_lgamma; // lgamma(ess) - lgamma(m + ess); // THE SAME FOR ALL.	
		double baslg = lgamma(essrq); for (int c = 1; c <= maxc; ++c) if (fre[c]) val += fre[c] * (lgamma(c + essrq) - baslg); 
	}
	if (jmin == 0){ wset xv = get_wset(X, lX, val, isdense); cscores.push_back(xv); } // When forcing X[0], don't store the empty set.
	if (lX == lmax) return val; 
	// Branch on "lower variables".
	for (int j = jmin; j < jmax; ++j){ // Alternatively, could consider the opposite order, from depth-1 downto 0.
		int i = S[j]; X[lX] = j; // *** NOTE: WE USE RELATIVE INDEXING WRT S, NOT "X[lX] = i".
		// Find the counts based on tmp[] and prt[]; When X is empty, these are simply (0..m-1) and (m).
		int p = 0; int tot = 0; int loc = 0; int pp = 0; // Here pp is the index of the next partition.
		int ri = r[i]; bool nextnotleaf = (lX+1 != lmax && j > 0);
		int num[BDEU_MAXARITY];
		if (ri < 4){ // If the arity is low, then sparse stepping though the occuring valus k would not pay off.
			while (tot < mleft){
				// Handle part p of the partition.
				int qmax = mleft; if (lX) qmax = prt[lX-1][p];
				// First, get num[].
				for (int k = 0; k < ri; ++k) num[k] = 0;
				for (int q = 0; q < qmax; ++q){ int t = tmp[lX][tot+q]; int k = dat[i][t]; ++num[k]; }
				// Second, get prt[].
				int qact = 0; 
				for (int k = 0; k < ri; ++k){							
					switch (num[k]){ // *** NOTE: Here we fix the order of pp in relation to k !!!
						case 0: break; case 1: num[k] = 0; break; // Ignore singletons.
						default: prt[lX][pp] = num[k]; ++pp; qact += num[k]; // Only here we can update qact.
					} 
				} 
				if (nextnotleaf){ // If the child node is not a leaf of the search tree.
					// Third, get tmp[].
					int pos[BDEU_MAXARITY]; pos[0] = num[0]; 
					for (int k = 0; k < ri-1; ++k) pos[k+1] = pos[k] + num[k+1];
					for (int q = 0; q < qmax; ++q){ 
						int t = tmp[lX][tot+q]; int k = dat[i][t]; 
						if (num[k]){ --pos[k]; tmp[lX+1][loc+pos[k]] = t; } // Note: may have num[k] == 0.  
					}
				}
				++p; tot += qmax; loc += qact; 
			}
		} else { // Now the arity may be high, and so we resort to a more complicated indexing using hit[].
			for (int k = 0; k < ri; ++k) num[k] = 0; // Expensive !?
			int hit[BDEU_MAXARITY]; // Collects the active values k.
			while (tot < mleft){
				int qmax = mleft; if (lX) qmax = prt[lX-1][p]; int lhit = 0;
				for (int q = 0; q < qmax; ++q){ 
					int t = tmp[lX][tot+q]; int k = dat[i][t]; if (!num[k]) hit[lhit++] = k; ++num[k]; 
				}
				int qact = 0; 
				for (int h = 0; h < lhit; ++h){
					int k = hit[h];								
					switch (num[k]){ // *** NOTE: Here we fix the order of pp in relation to k !!!
						case 0: break; case 1: num[k] = 0; break; // Ignore singletons.
						default: prt[lX][pp] = num[k]; ++pp; qact += num[k]; // Only here we can update qact.
					} 
				} 
				if (nextnotleaf){ // If the child node is not a leaf of the search tree.
					int pos[BDEU_MAXARITY]; pos[hit[0]] = num[hit[0]]; 
					for (int h = 0; h < lhit-1; ++h) pos[hit[h+1]] = pos[hit[h]] + num[hit[h+1]];
					for (int q = 0; q < qmax; ++q){ 
						int t = tmp[lX][tot+q]; int k = dat[i][t]; 
						if (num[k]){ --pos[k]; tmp[lX+1][loc+pos[k]] = t; } // Note: may have num[k] == 0.  
					}
				}
				++p; tot += qmax; loc += qact; for (int h = 0; h < lhit; ++h) num[hit[h]] = 0; 
			}
		}
		val += dfo2(X, lX+1, lmax, S, 0, j, loc, rq * ri, isdense); // Updated mleft to loc. Forced jmin := 0. 
	} 
	return val;
}
int BDeu::index(int* X, int l, int u){ // We assume X[0] > X[1] > ...  
	// Note: could implement O(1) time update for 1-element, order-preserving replacements X -> X'. 
	int ix = 0; for (int j = 0; j < l; ++j) ix += binomsum[ X[j] ][ u-j ]; return ix;
}
int BDeu::iindex(int* X, int l, int u){ // We assume X[0] < X[1] < ...
	int ix = 0; for (int j = 0; j < l; ++j) ix += binomsum[ X[l-1-j] ][ u-j ]; return ix;
}
void BDeu::preindex(int umax){
//	cerr << " [preindex:] umax = " << umax << endl;
	for (int j = 0; j <= n; ++j){
		for (int u = 0; u <= umax; ++u){ 
			binomsum[j][u] = 0; 
		} 
	}
	binomsum[0][0] = 1; 	// First just bin coefficients.
	for (int j = 1; j <= n; ++j){
		binomsum[j][0] = 1; 
		for (int k = 1; k <= j && k <= umax; ++k){ 
			binomsum[j][k] = binomsum[j-1][k] + binomsum[j-1][k-1]; 
//			cout << "binom["<<j<<"]["<<k<<"] = "<<binomsum[j][k]<<endl;
		}
	}
	for (int j = 0; j <= n; ++j){ // Then cumulative sums. 
		int u = 1; for (; u <= j && u <= umax; ++u){ binomsum[j][u] += binomsum[j][u-1]; }
		for (; u <= n && u < 32; ++u){ binomsum[j][u] = (1L << j); }
	}
}

double BDeu::score_all1(){ // Another approach to score all nonempty subsets of the n variables.
	int* S = new int[n]; for (int i = 0; i < n; ++i) S[i] = i; // The set of variables.
	int u = n; 
	double scoresum = cliq(S, n, u);
//	double scoresum = fami(n-1, S, n-1, u);	// Now i should scores all possible subsets of S as its parent set.
	delete[] S; return scoresum;
}
ostream& operator<<(ostream& os, const BDeu& x){
	for (int t = 0; t < x.m; ++t){ for (int i = 0; i < x.n; ++i){ os << " " << x.dat[i][t]; } os << endl; }
	return os;
}
void sizeunif(int* X, int& lX, int a, int b){
	lX = 0; int e = rand() % (b - a);
	for (int i = a; i < b; ++i){ int w = rand() % (b - a); if (w <= e) X[lX++] = i; }
}
void BDeu::query_test(){
	double q; int arity = 9;
	cout << " BDeu Speed Test: random query sets X whose size |X| is uniformly distributed between 1 and n\n";
	cout << " Testing cliq():\n";
	for (int n0 = 20; n0 <= 60; n0 += 10){
		int* X = new int[n0]; int lX;
		for (int m0 = 1000; m0 <= 100000; m0 *= 10){
			init(m0, n0); fill_rnd(arity); set_ess(10.0); q = 50000*1000/m0;
			clock_t t1 = clock();
			for (int rr = 0; rr < q; ++rr){ sizeunif(X, lX, 0, n0); cliq(X, lX); }
			clock_t t2 = clock(); double micros = 1000000.0 * (t2 - t1)/CLOCKS_PER_SEC;
			double count = q; 	
			cout << fixed << " n = " << n0 << ", m = " << setw(7) << m0 << ", arity = " << arity;
			cout << scientific << ", microsec per query = " << micros/count;
			cout << scientific << ", queries per microsec = " << count/micros << endl;
			fini();
		}
		delete[] X;
	}
}
void dd_test(){
	BDeu s;
	s.read("dd.csv"); s.set_ess(10.0);
	int C0[4] = {1, 2, 3, 4};
	int C1[4] = {0, 2, 3, 4};
	int C2[4] = {0, 1, 3, 4};
	int C3[4] = {0, 1, 2, 4};
	int C4[4] = {0, 1, 2, 3};
	int  E[0] = { };
	s.fami(0, C0, 4, 4);
	s.fami(1, C1, 4, 4);
	s.fami(2, C2, 4, 4);
	s.fami(3, C3, 4, 4);
	s.fami(4, C4, 4, 4);
	double val0 = s.fami  (0, E, 0);
	double wal0 = s.fscore(0, E, 0);
	cout << fixed << " Score of (0, E) = " << val0 << " = " << s.fscores[0][0].weight << " = " << wal0 << endl;

}
void BDeu::test(void){
	dd_test();
	//query_test();
	read("child1000.csv"); set_ess(10.0); 
	//int m0 = m; int n0 = n; // If we change the key parameters m or n, we need to put them back before the very end.
	for (int round = 37; round <= 37; round += 2){
		//init(3, 2); fill_rnd(9); set_ess(10.0);
		//m = m0 & ((1 << round) - 1); // Artificially trunctating the data matrix for testing purposes.
		clock_t t1 = clock();
		double s = score_all1(); int count = (1 << n) - 1; // Testing...
		clock_t t2 = clock(); double micros = 1000000.0 * (t2 - t1)/CLOCKS_PER_SEC; 	
		cout << " [test:] n = " << n << ", m = " << m << ", count = " << count;
		cout << fixed << ", per microsec = " << (double) count/micros <<", s = " << scientific << s;
		cout << ", #ascores = " << ascores.size() << ", #cscores = " << cscores.size() << endl;

		//fini();	
	} //m = m0; n = n0;
}
BDeu::BDeu(void){ initdone = false; }
BDeu::~BDeu(void){ if (initdone) fini(); }
void BDeu::init(int m0, int n0){
	m = m0; n = n0; //set_ess(1.0); 
	dat = new Tdat*[n]; tmp = new int*[n]; prt = new int*[n];
	r = new int[n]; w = new int[n]; lng = new double[m+1]; fre = new uint32_t[m+1]();
	for (int i = 0; i < n; ++i){ 
		dat[i] = new Tdat[m]; tmp[i] = new int[m]; prt[i] = new int[m]; 
		for (int t = 0; t < m; ++t) tmp[i][t] = 0; 
	}
	fscores = new vector<wset>[n]; fparams.resize(n); ascores.init(n); //ascores.demo();
	preindex(31); initdone = true;
}
void BDeu::fini(){
	for (int i = 0; i < n; ++i){ delete[] dat[i]; delete[] tmp[i]; delete[] prt[i]; }  
	delete[] dat; delete[] tmp; delete[] r; delete[] w; delete[] lng; delete[] fre; delete[] prt;
	delete[] fscores; initdone = false; 
}
void BDeu::print_tmp(){
	cout << "tmp:" << endl;
	for (int t = 0; t < m; ++t){
		for (int i = 0; i < n; ++i){ cout << " \t" << tmp[i][t]; }
		cout << endl;
	}
}

#endif
