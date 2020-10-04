// Compile with g++ using options -Wall -O3 

#include <iostream> 
#include <fstream> 
#include <iomanip> 
#include <math.h> 
//#include <chrono> 
#include <string.h>
#include <stdlib.h>
#include "Wsets.hpp"

using namespace std;

#define Tdat 		uint8_t
#define BDEU_MAXN 	256
#define BDEU_MAXARITY 	256
#define BDEU_MAXM	65536	
class BDeu {
    public:
	int    m;	// Number of data points.
	int    n;	// Number of variables.
	Tdat** dat;	// Data matrix of size n x m.
	int*   r;	// Number of values per variable.
	int*   w;	// Number of bits reserved per variable, floor of log2(r). 
	double ess;	// Equivalent sample size parameter for the BDeu score.

	BDeu ();
	~BDeu ();
	void   test ();					// Shows some test runs. 
	int    read (int* datavec, int m0, int n0);	// Reads data matrix of size m0 x n0 given as vector datavec.
	int    dear (int* datavec, int m0, int n0);	// Reads data matrix of size m0 x n0 given as vector datavec.
	int    read (string fname);			// Reads data matrix given as a csv file.
	void   set_ess(double val);			// Set the value of the ess parameter and base_delta_lgamma.
	double cliq (int* var, int d);			// Scores clique var of size d.
	double cliqc(int* var, int d);			// Scores clique var of size d, with caching.
	double fami (int i, int* par, int d);		// Scores family (i, par), with par of size d.

	double cliq (int* S, int d, int u);		// Stores clique scores for all subsets of size at most u.
	double fami (int* S, int d, int u);		// Stores family scores for all variables and all parent sets of size at most u.

	friend ostream& operator<<(ostream& os, const BDeu& x); // Currently just prints out the data matrix.

    private:
	vector<wset>  	cscores;		// List of clique scores.
	vector<wset>* 	lscores;		// Array of size n of local scores.
	Wsets         	sscores;		// Storage of set scores.
	int**   	tmp;			// Helper array of size n x m;
	uint16_t*	fre;			// Helper array of size m+1 to store a multiset of counts, indexed by counts. 
	double* 	lng;			// Helper array of size m+1 to store ln(Gamma(j + ess')/Gamma(ess')).
	int**   	prt;			// Helper array of size n x m, yet another;
	int binomsum[BDEU_MAXN][BDEU_MAXN];	// Tail sums of binomial coefficients. Would be better make this static.
	double 		base_delta_lgamma;	// lgamma(m+ess)-lgamma(ess);
	bool 		initdone;		// Flag: whether mem dynamically allocated (to be released at the end). 	

	double score_dfs (int d, int* X, int a, int b, double q);
	void   score_dfs3(int d, int* X, int a, int b);
	double score_dico(int d, int* X);
	double score_hash(int d, int* X);
	double score_all ();

	double dfo1(int* X, int sizeX, int sizemax, int* S, int nleft, int mleft, double rq);	// Bunch of scores rec.
	double score_all1();

	int     index(int *X, int len, int u);	// Index to cscores, relative to the S used when built; X in dec order. 
	int    iindex(int *X, int len, int u);	// Index to cscores, relative to the S used when built; X in inc order. 
	void preindex(int umax);		// Computes the array binomsum[][].

	int width(int d, int* X);

	void print_tmp();
	void init(int m0, int n0);	// Initializes data structures of size m0 x n0.
	void fini();			// Deletes the dynamically allocated data structures. 
	void fill_rnd(int maxr);	// Fills the data matrix in a random  way, just for testing purposes. 
	void set_r();			// Computes and stores the arities.
	void set_r(const vector<int> &v);
};

const string red("\033[0;31m"); const string green("\033[0;32m"); const string cyan("\033[0;36m"); const string reset("\033[0m");
#define STRV(...) #__VA_ARGS__
#define DEMO(command, comment) cout << green << STRV(command) << cyan << "  // " << comment << reset << endl; command 
#define COMMA ,
void BDeu_demo (){ // Demo of the usage of the class. Might define this as a static function of the class, but not done here.
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
	//s.read("child1000.csv"); s.set_ess(10.0);
	//cout << s;
	//int par2[2] = {3, 2};
	//cout << " Score of (0, {3, 2}) = " << setprecision(17) << s.fami(0, par2, 2) << endl;
	cout << " ======= end of demo  ===== " << endl;
}
#undef COMMA 
#undef DEMO
#undef STRV

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
// Returns the BDeu score of the clique X of size d. 
double BDeu::cliq (int* X, int d){ 
	int wc = width(d, X); 					 
	if (0 < wc && wc < 64) return score_hash(d, X); 	// Very fast, but currently implemented only up to 64-bit data records.
	else                   return score_dico(d, X);		// Somewhat carefully optimized divide & conquer algorithm.		
}
// Returns the BDeu score of the clique X of size d, with caching. Note: currently caching assumes the ground set has size at most 64. 
double BDeu::cliqc(int* X, int d){ 
	double val;   if (sscores.get(X, d, &val)) return val;	// If already stored, then just get and return it.
	val = cliq(X, d); sscores.put(X, d,  val); return val;	// Push the computed score to the storage--requires mem!
}
// Returns the BDeu score of the family (i, Y) where Y is of size d. 
double BDeu::fami (int i, int* Y, int d){
	double  sp = cliq(Y, d);
	int*     X = new int[d+1]; X[d] = i; for (int j = 0; j < d; ++j) X[j] = Y[j]; 
	double spi = cliq(X, d+1); 
	delete [] X; return spi - sp; 
}
// Scores ALL subsets of S of size at most u. Returns the sum of the scores. Stores the scores.
double BDeu::cliq(int* S, int d, int u){ 
	int* X = new int[d];  
	for (int t = 0; t < m; ++t){ tmp[0][t] = t; } // Init tmp[0][]. No need to init prt[].
	cscores.shrink_to_fit(); // Completely clears and resizes cscores.
	preindex(u); cscores.reserve(iindex(S, d, u)+1); // We assume S is in INCreasing order. 
	double scoresum = dfo1(X, 0, u, S, d, m, 1.0); // Visit and score in depth-first order.
	delete[] X; return scoresum;	
}
// Scores ALL families (i, Y) where Y is a subset of S of size at most u.
// *** *** NOTE: We currently assume that |S| = d = n <= 64. This is a strong, temporary assumption.
double BDeu::fami(int* S, int d, int u){
	int uu = u+1; if (uu > n) uu = n; 
	double scoresum = cliq(S, d, uu); // Note: Need to compute clique scores with u+1.
	int l = cscores.size(); int* X = new int[n]; int lenx; int* Y = new int[n];
	for (int j = 0; j < d; ++j) lscores[j].reserve(l+1); // Note: Currently reserving a bit too much.
	for (int c = 0; c < l; ++c){ // Go through all the cliques X.
		wset xv = cscores.at(c); get_set(xv.set, X, lenx); // Note: X is in increasing order. Assumes dense mode.
		//cout << " [fami:] c = " << c << ", len = " << len << endl;
		int leny = lenx - 1;
		for (int j = 0; j < lenx; ++j){
			int i = X[j]; // Child i. Parents Y := X - {i}. *** NOTE: HERE i IS RELATIVE TO THE SET S, WHEREAS S[i] IS THE REAL VAR.
			for (int h = 0; h < leny-j; ++h)    Y[h] = X[leny-h];	// Cannot do this incrementally as we need to preserve the order.
			for (int h = leny-j; h < leny; ++h) Y[h] = X[leny-h-1];
			int iy = index(Y, leny, uu);
			wset yv = cscores.at(iy); 
			//cout << " [fami:]  i = " << i << endl; cout << " X :: "; show(xv); cout << " Y :: "; show(yv); cout << endl; 
			// Add (i, Y) to the list of i. 
			wset iyv = { yv.set, xv.weight - yv.weight };
			lscores[i].push_back(iyv);
		}
	}
//	cout<<" [fami:] (i, lscores[i].size) = "; for (int i = 0; i < n; i ++){ cout<<"("<<i<<", "<<lscores[i].size()<<"); "; } cout<<endl;
	delete[] X; delete[] Y; return scoresum;
}


void BDeu::set_ess(double val){ ess = val; base_delta_lgamma = lgamma(ess) - lgamma(m+ess); }


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
		r[i] = 1; for (int t = 0; t < m; ++t){ if (dat[i][t] + 1 > r[i]) r[i] = dat[i][t] + 1; }
		int v = r[i] - 1; w[i] = 0; while (v) { ++w[i]; v >>= 1; } 
	}
}
void BDeu::set_r(const vector<int> &v){ // Sets s according to the given vector.
	for (int i = 0; i < (int) v.size(); ++i){ 
		r[i] = v.at(i); int v = r[i] - 1; w[i] = 0; while (v) { ++w[i]; v >>= 1; }
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
	int pos[BDEU_MAXARITY]; pos[0] = 0; // Do this already here; free in practice, due to cpu-level parallelism.
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
	for (int t = 0; t <  m; ++t) tmp[d-1][t] = t;	// Init tmp[d-1].
	for (int c = 0; c <= m; ++c) fre[c] = 0;	// Init fre.
	score_dfs3(d-1, X, 0, m);
	int maxc = m; while (!fre[maxc]) --maxc;
	double q = 1; for (int j = 0; j < d; ++j) q *= r[X[j]]; 
	double essq = ess/q; lng[0] = 0; for (int c = 0; c < maxc; ++c) lng[c+1] = lng[c] + log(c + essq);		
	double s = base_delta_lgamma;			// lgamma(ess) - lgamma(m + ess);
	for (int c = 1; c <= maxc; ++c){ if (fre[c]) s += fre[c] * lng[c]; }	
	return s;
}

struct keycount { uint64_t key; int num; int nxt; }; // The int fields could be replaced by uint16_t for most practical purposes.

class HashCounter { // Relatively huge hash range; active bins iterable. XOR hashing. Currently the count functions not used at all.
    public:
	void insert(uint64_t key){ int p = hash(key); insert(key, p); }
	int   count(uint64_t key){ int p = hash(key); return count(key, p); }
	int    hash(uint64_t k  ){ uint32_t p = k & mask; k >>= lmask; p ^= k & mask; k >>= lmask; return (p ^ k) & mask; } 
	void insert(uint64_t key, int p){
		uint16_t q = ptrs[p]; if (q == 0){ buck[nex] = { key, 1, 0 }; ptrs[p] = nex; act[lact] = p; ++lact; ++nex; return; }
		do { keycount kc = buck[q]; if (kc.key == key){ ++buck[q].num; return; } q = kc.nxt; } while (q); 
		buck[nex] = { key, 1, ptrs[p] }; ptrs[p] = nex; ++nex; // Not found. Becomes the *head*.
	}
	int   count(uint64_t key, int p){
	 	int q = ptrs[p]; while (q){ keycount kc = buck[q]; if (kc.key == key){ return kc.num; } q = kc.nxt; } return 0; 
	}
	int get_freq_and_reset(uint16_t* fre){ // We assume fre[] has been initialized to zero. Returns the largest count encountered.
		int maxc = 0;
		for (int j = 0; j < lact; ++j){
			int p = act[j]; int q = ptrs[p]; ptrs[p] = 0; // Zero the static ptrs[]; we are about to delete the data structure.
			while (q){ keycount kc = buck[q]; int c = buck[q].num; ++fre[c]; if (c > maxc) maxc = c; q = kc.nxt; }
		} return maxc; 
	}
	int maxload(){ // Mainly for testing. Note: a slow routine; we don't want to slow down insert by additional bookkeeping.
		int maxl = 0; 
		for (int j = 0; j < lact; ++j){
			int l = 0; int p = act[j]; int q = ptrs[p]; 
			while (q){ ++l; keycount kc = buck[q]; q = kc.nxt; } if (l > maxl) maxl = l;
		} return maxl; 
	}
	HashCounter (int m){ buck = new keycount[m+1]; nex = 1; act = new int[m]; lact = 0; } 
	~HashCounter(){ delete[] buck; delete[] act; }
    private:
	int* act;		int lact;	// Active bins.
	keycount* buck;		int nex;	// The count of *unique* keys + 1; also the index of the next free slot in buck.
	static uint16_t ptrs[]; static const uint32_t mask, lmask;
};
//uint16_t HashCounter::ptrs[512*1024] { 0 }; const uint32_t HashCounter::mask = 0x7FFFF, HashCounter::lmask = 19; 
uint16_t HashCounter::ptrs[64*1024] { 0 }; const uint32_t HashCounter::mask = 0xFFFF, HashCounter::lmask = 16; 

double BDeu::score_hash(int d, int* X){ // By simply hashing. Does not work: unordered map does not support proper count queries.
	HashCounter h(m); 								// Hash. Uses a self-made data structure.
	uint64_t* z = new uint64_t[m]; for (int t = 0; t < m; ++t) z[t] = dat[X[0]][t]; // Form a list of keys. 
	for (int j = 1; j < d; ++j){
		int i = X[j]; int l = w[i]; 
		for (int t = 0; t < m; ++t){ z[t] <<= l; z[t] |= dat[i][t]; } 		// Simply encode the data.
	}
	for (int t = 0; t <  m; ++t) h.insert(z[t]);  					// Hash.
	for (int c = 0; c <= m; ++c) fre[c] = 0;  					// Init frequencies of counts.
	int maxc = h.get_freq_and_reset(fre); 						// Get the spectrum, i.e., the count frequency array. 
	double q = 1; for (int j = 0; j < d; ++j) q *= r[X[j]]; double essq = ess/q;	// Compute lng[].
	lng[0] = 0; for (int c = 0; c < maxc; ++c) lng[c+1] = lng[c] + log(c + essq); 	
	double s = base_delta_lgamma; 							// lgamma(ess) - lgamma(m + ess);
	for (int c = 1; c <= maxc; ++c){ if (fre[c]) s += fre[c] * lng[c]; }		// Finalize.	
	delete[] z; return s;
}// // // // // //


double BDeu::score_all(){ // Scoring all nonempty subsets of the n variables.
	int* X = new int[n]; // Set of variables.
	for (int i = 0; i < n; ++i) X[i] =  i;
	double s = 0; int count = 0;
	for (int x = 1; x < (1 << n); ++x){
		int d = 0; // Size of x.
		for (int i = 0; i < n; ++i){ if (x & (1 << i)){ X[d++] = i; } } // Set X.
		if (x) s += cliqc(X, d);		
		++count;
	}
	delete[] X; return s;
}

// Subsets X of S in depth-first order. Gathers singleton into "dark material". This is quite a hack.  
double BDeu::dfo1(int* X, int sizeX, int sizemax, int* S, int nleft, int mleft, double rq){ // Under construction...
	double val = 0;
	if (sizeX){ // This is not the time critical part at the moment. 		
		int p = 0; int maxc = 1; 
		for (int tot = 0; tot < mleft; ){ int c = prt[sizeX-1][p]; ++p; if (c > maxc) maxc = c; tot += c; }
		for (int c = 0; c <= maxc; ++c) fre[c] = 0;
		for (int q = 0; q < p; ++q){ int c = prt[sizeX-1][q]; ++fre[c]; }
		fre[1] = m - mleft; // fre[1] = m - mleft, the "dark material"; other singletons should not exist.
		double essrq = ess/rq; lng[0] = 0; for (int c = 0; c < maxc; ++c) lng[c+1] = lng[c] + log(c + essrq);
		val = base_delta_lgamma; // lgamma(ess) - lgamma(m + ess); // NOTE: THE SAME FOR ALL, EACH AND EVERY !	
		for (int c = 1; c <= maxc; ++c){ if (fre[c]) val += fre[c] * lng[c]; }
	}
//	wset xv = get_wset(X, sizeX, val); cscores.push_back(xv); 
	sscores.put(X, sizeX, val); // Store the pair (X, val).
	if (sizeX == sizemax) return val; 
	// Branch on "lower variables".
	for (int j = 0; j < nleft; ++j){ // Alternatively, could consider the opposite order, from depth-1 downto 0.
		int i = S[j]; X[sizeX] = j; // *** NOTE: WE USE RELATIVE INDEXING WRT S, NOT "X[sizeX] = i".
		// Find the counts based on tmp[] and prt[]; When X is empty, these are simply (0..m-1) and (m).
		int p = 0; int tot = 0; int loc = 0; int pp = 0; // Here pp is the index of the next partition.
		while (tot < mleft){
			// Handle part p of the partition.
			int qmax = mleft; if (sizeX) qmax = prt[sizeX-1][p];
			// First, get num[].
			int num[BDEU_MAXARITY]; for (int k = 0; k < r[i]; ++k) num[k] = 0; // Expensive !?
			for (int q = 0; q < qmax; ++q){ int t = tmp[sizeX][tot+q]; int k = dat[i][t]; ++num[k]; }
			// Second, get prt[].
			int qact = 0; 
			for (int k = 0; k < r[i]; ++k){ // Expensive if r[i] large !
				switch (num[k]){
					case 0: break; case 1: num[k] = 0; break; // Ignore singletons. Zero the num[] for a later use.
					default: prt[sizeX][pp] = num[k]; ++pp; qact += num[k]; // Only here we can update qact.
				}
			}
			++p;  
			if (sizeX+1 == sizemax){ tot += qmax; loc += qact; continue; }
			// Third, get tmp[].
			int pos[BDEU_MAXARITY]; pos[0] = num[0]; 
			for (int k = 0; k < r[i]-1; ++k) pos[k+1] = pos[k] + num[k+1]; // Expensive if r[i] large !
			for (int q = 0; q < qmax; ++q){ 
				int t = tmp[sizeX][tot+q]; int k = dat[i][t]; 
				if (num[k]){ --pos[k]; tmp[sizeX+1][loc+pos[k]] = t; } // Note: may have num[k] == 0.  
			}
			tot += qmax; loc += qact;
		}
		val += dfo1(X, sizeX+1, sizemax, S, j, loc, rq * r[i]); // Updated mleft to loc. 
	} 
	return val;
}
int BDeu::index(int* X, int len, int u){ // We assume X[0] > X[1] > ...
	int ix = 0; for (int j = 0; j < len; ++j) ix += binomsum[ X[j] ][ u-j ]; return ix;
}
int BDeu::iindex(int* X, int len, int u){ // We assume X[0] < X[1] < ...
	int ix = 0; for (int j = 0; j < len; ++j) ix += binomsum[ X[len-1-j] ][ u-j ]; return ix;
}
void BDeu::preindex(int umax){
	binomsum[0][0] = 1; 	// First just bin coefficients.
	for (int j = 1; j <= n; ++j){
		binomsum[j][0] = 1; 
		for (int k = 1; k <= j && k <= umax; ++k){ binomsum[j][k] = binomsum[j-1][k] + binomsum[j-1][k-1]; }
	}
	for (int j = 0; j <= n; ++j){ // Then cumulative sums. 
		int u = 1; for (; u <= j && u <= umax; ++u){ binomsum[j][u] += binomsum[j][u-1]; }
		for (; u <= n; ++u){ binomsum[j][u] = (1L << j); }
	}
}

double BDeu::score_all1(){ // Another approach to score all nonempty subsets of the n variables.
	int* S = new int[n]; for (int i = 0; i < n; ++i) S[i] = i; // The set of variables.
	int u = n; 
	double scoresum = cliq(S, n, u);
//	double scoresum = fami(S, n, u);
/*
	int *Y = new int[n];  for (int i = 0; i < n; ++i) Y[i] = n-1-i;
	show(cscores.at(index(Y, n, u))); delete[] Y;
*/
	delete[] S; return scoresum;
}
ostream& operator<<(ostream& os, const BDeu& x){
	for (int t = 0; t < x.m; ++t){ for (int i = 0; i < x.n; ++i){ os << " " << x.dat[i][t]; } os << endl; }
	return os;
}
void BDeu::test(void){
	//cout << "[test:] " << "rand() = " << rand() << " " << rand() << endl;
	read("child5000.csv"); set_ess(10.0); 
	//int m0 = m; int n0 = n; // If we change the key parameters m or n, we need to put them back before the very end.
	for (int round = 10; round <= 10; round += 1){
		//init(100*round, 10+round); fill_rnd(4); set_ess(10.0);
		//m = m0 & ((1 << round) - 1); // Artificially trunctating the data matrix for testing purposes.
		clock_t t1 = clock();
		double s = score_all1(); int count = (1 << n) - 1; // Testing...
		clock_t t2 = clock(); double micros = 1000000.0 * (t2 - t1)/CLOCKS_PER_SEC; 	
		cout << " [test:] done, n = " << n << ", m = " << m << ", count = " << count;
		cout << fixed << ", per microsecond = " << (double) count/micros <<", s = " << scientific << s;
		cout << ", sscores.size = " << sscores.size() << endl;
		//fini();	
	}
	//m = m0; n = n0;
}
BDeu::BDeu(void){ initdone = false; }
BDeu::~BDeu(void){ if (initdone) fini(); }
void BDeu::init(int m0, int n0){
	set_ess(1.0); m = m0; n = n0;
	dat = new Tdat*[n]; tmp = new int*[n]; prt = new int*[n];
	r = new int[n]; w = new int[n]; lng = new double[m+1]; fre = new uint16_t[m+1];
	for (int i = 0; i < n; ++i){ 
		dat[i] = new Tdat[m]; tmp[i] = new int[m]; prt[i] = new int[m]; 
		for (int t = 0; t < m; ++t) tmp[i][t] = 0; 
	}
	lscores = new vector<wset>[n]; sscores.init(n); //sscores.demo();
	initdone = true;
}
void BDeu::fini(){
	for (int i = 0; i < n; ++i){ delete[] dat[i]; delete[] tmp[i]; delete[] prt[i]; }  
	delete[] dat; delete[] tmp; delete[] r; delete[] w; delete[] lng; delete[] fre; delete[] prt;
	delete[] lscores; 
	initdone = false; 
}
void BDeu::print_tmp(){
	cout << "tmp:" << endl;
	for (int t = 0; t < m; ++t){
		for (int i = 0; i < n; ++i){ cout << " \t" << tmp[i][t]; }
		cout << endl;
	}
}



