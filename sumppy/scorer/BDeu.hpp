
// Compile with g++ using options -Wall -O3 

#include <algorithm>
#include <iostream> 
#include <iomanip> 
#include <math.h> 
#include <chrono> 
#include <string.h>
#include <stdlib.h>

using namespace std;
using namespace std::chrono;

#define Tdat int
#define BDEU_MAXARITY 256
class BDeu {
    public:
	int    m;	// Number of data points.
	int    n;	// Number of variables.

	Tdat** dat;	// Data matrix of size n x m.
	Tdat** tad;	// Data matrix of size m x n. 
	int*   r;	// Number of values per variable.
	int*   w;	// Number of bits reserved per variable, floor of log2(r). 

	double ess;	// Equivalent sample size parameter for the BDeu score.

	BDeu ();
	~BDeu ();
	void   test ();					// Shows some test runs. 

	int    read (int* datavec, int m0, int n0);	// Reads data matrix of size m0 x n0 given as vector datavec.
	int    dear (int* datavec, int m0, int n0);	// Reads data matrix of size m0 x n0 given as vector datavec.

	void   set_ess(double val){ ess = val; }	// Set the value of the ess parameter.

	double cliq (int* var, int d);			// Scores clique var of size d.
	double fami (int i, int* par, int d);		// Scores family (i, par), with par of size d.

	//void cliq (int* var, int d, int u);		// Stores clique scores for all subsets of size at most u.
	//void fami (int i, int* var, int d, int u);	// Stores family scores for all subsets of size at most u.
	
	friend ostream& operator<<(ostream& os, const BDeu& x); // Currently just prints out the data matrix.

    private:
	int**   tmp;	// Helper array of size n x m;
	int*    fre;	// Helper array of size m+1 to store a multiset of counts, indexed by counts. 
	double* lng;	// Helper array of size m+1 to store normalized lgamma-values ln(Gamma(j + ess')/Gamma(ess')).

	bool initdone;	// Flag that tells whether mem dynamically allocated (to be released at the end). 	

	double score_dfs (int depth, int* var, int a, int b, double q);
	double score_dfs1(int depth, int* var, int a, int b);
	void   score_dfs2(int depth, int* var, int a, int b);
	void   score_dfs3(int depth, int* var, int a, int b);
	double score_dico(int d, int* var);
	double score_sort(int d, int* var);
	double score_all();
	int    width(int d, int* var);

	void print_tmp();
	void init(int m0, int n0);	// Initializes data structures of size m0 x n0.
	void fini();			// Deletes the dynamically allocated data structures. 
	void fill_mod(int maxr);	// Fills the data matrix in a certain way, just for testing purposes. 
	void fill_rnd(int maxr);	// Fills the data matrix in a random  way, just for testing purposes. 
	void set_r();			// Computes and stores the arities.

};

const std::string   red("\033[0;31m");  
const std::string green("\033[0;32m");  
const std::string  cyan("\033[0;36m");  
const std::string reset("\033[0m");
void BDeu_demo (){ // Gives a demonstration of the usage of the class. 
	cout << " === DEMO of class BDeu === " << endl;

	cout << green << " BDeu s; " << cyan << "// Introduce a BDeu object. " << reset << endl;
	BDeu s;
	
	cout << green << " int *x = new int[15 * 7]; " << cyan << "// The original data matrix. " << endl;
	int *x = new int[15 * 7];  

	cout << green << " for (int i = 0; i < 15 * 7; ++i) x[i] = rand() % 5; " << cyan << "// Now fill with random entries. " << endl;
	for (int i = 0; i < 15 * 7; ++i) x[i] = rand() % 5;

	cout << green << " s.read(x, 15, 7); " << cyan << "// Feed x to s as a matrix with 15 datapoints, each over 7 variables. " << endl;
	s.read(x, 15, 7);

	cout << green << " delete[] x; " << cyan << "// Now we may delete x. This emphasizes that s made a copy of x. " << endl;
	delete[] x;

	cout << green << " cout << s; " << cyan << "// Let's print out the data matrix." << reset << endl;
	cout << s;

	cout << green << " s.set_ess(1.0); " << cyan << "// Set a value for the ess parameter. " << endl;
	s.set_ess(1.0);
 
	cout << green << " int par[4] = {4, 6, 0, 1}; " << cyan << "// Create a parent set. Note: the order does not matter. " << endl;
	int par[4] = {4, 6, 0, 1};

	cout << green << " double score = s.fami(2, par, 4); " << cyan << "// Get the score of variable 2 with parent set par. " << reset << endl;
	double score = s.fami(2, par, 4);
 
	cout << green << " cout << \" Got score = \" << score << endl; " << cyan << "// Print out the score we got. " << reset << endl;
	cout << " Got score = " << score << endl;

	cout << " ======= end of demo  ===== " << endl;
}



//////////////////
// Public methods:

// Read the data matrix. Assumes m0 datapoints over n0 variables, given in the order of datapoints.
int BDeu::read(int* datavec, int m0, int n0){
	init(m0, n0); // Sets m and n among other things.
	int j = 0;
	for (int t = 0; t < m; ++t){
		for (int i = 0; i < n; ++i){
			dat[i][t] = datavec[j++]; // j = t * n + i.
			tad[t][i] = dat[i][t];
		}
	}
	set_r();
	return 1;
}
// Read the data matrix. Assumes m0 datapoints over n0 variables, given in the order of variables.
int BDeu::dear(int* datavec, int m0, int n0){
	init(m0, n0); // Sets m and n among other things.
	int j = 0;
	for (int i = 0; i < n; ++i){
		for (int t = 0; t < m; ++t){
			dat[i][t] = datavec[j++]; // j = i * m + t.
			tad[t][i] = dat[i][t];
		}
	}
	set_r();
	return 1;
}

// Returns the BDeu score of the clique c of size d. 
double BDeu::cliq(int* c, int d){
	if (d > 2 && width(d, c) < 64) return score_sort(d, c); 
	else                           return score_dico(d, c);	
}
// Returns the BDeu score of the family (i, p) where p is of size d. 
double BDeu::fami(int i, int* p, int d){
	double  sp = cliq(p, d);
	int*     c = new int[d+1]; c[d] = i; for (int j = 0; j < d; j ++) c[j] = p[j]; 
	double spi = cliq(c, d+1); 
	delete [] c; 
	return spi - sp; 
}




///////////////////
// Private methods: 

int BDeu::width(int d, int *c){ // How many bits occupied by the variables in c.
	int wc = 0; for (int j = 0; j < d; ++j) wc += w[c[j]];
	return wc;
}

void BDeu::fill_mod(int maxr){ // For testing purposes.
	for (int i = 0; i < n; ++i){
		for (int t = 0; t < m; ++t){
			dat[i][t] = ((unsigned)(1 + t + i + i*t + i*i*t + i*t*t + i*i*t*t) % (i + maxr)) % maxr;
			tad[t][i] = dat[i][t];
		}
	}
	set_r();
}
void BDeu::fill_rnd(int maxr){ // For testing purposes.
	srand(maxr);
	for (int i = 0; i < n; ++i){
		int mr = 2 + (rand() % (maxr - 1));

		for (int t = 0; t < m; ++t){
			int v = rand() % mr;
			dat[i][t] = v;
			tad[t][i] = dat[i][t];
		}
	}
	set_r();
}
void BDeu::set_r(){
	for (int i = 0; i < n; ++i){
		r[i] = 1; for (int t = 0; t < m; ++t){ if (dat[i][t] > r[i] - 1) r[i] = dat[i][t] + 1; }
		int v = r[i] - 1; w[i] = 0; while (v) { ++w[i]; v >>= 1; }  
		//w[i] = (int)(log2(r[i]) + 1);
	}
}
// Speed: n = 15, m = 1000, sets per microsecond: 0.0027.
double BDeu::score_dfs(int d, int* c, int a, int b, double q){
	double s = lgamma(ess/q) - lgamma(b - a + ess/q);
	int i = c[d]; q *= r[i];	
	// Partition tmp[] into sublists. First scan and count; then partition.
	int num[8] = {0, 0, 0, 0, 0, 0, 0, 0}; 
	int pos[8] = {0, 0, 0, 0, 0, 0, 0, 0};
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
double my_lgamma(double z){ // Currently just a place holder.
	return lgamma(z);
}
// Speed: n = 15, m = 1000, sets per microsecond: 0.026, if risky lgamma-tables.
double BDeu::score_dfs1(int d, int* c, int a, int b){
	double s = 0;
	int i = c[d];	
	// Partition tmp[] into sublists. First scan and count; then partition.
	//int num[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	int num[32]; for (int k = 0; k < r[i]; ++k) num[k] = 0;
	for (int t = a; t < b; ++t){ int k = dat[i][tmp[d][t]]; ++num[k]; }
	if (d == 0){ // For the sake of speed, do not recurse, but solve it here.		
		for (int k = 0; k < r[i]; ++k) s += lng[num[k]];	 
		return s;
	}
	//int pos[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	int pos[32]; for (int k = 0; k < r[i]; ++k) pos[k] = 0;
	for (int k = 0; k < r[i]-1; ++k) pos[k+1] = pos[k] + num[k];
	for (int t = a; t < b; ++t){ int k = dat[i][tmp[d][t]]; tmp[d-1][a+pos[k]] = tmp[d][t]; ++pos[k]; }
	
	for (int k = 0; k < r[i]; ++k){
		switch (num[k]){
			case 0: break;
			case 1: s += lng[1]; break;
			default:
				int bk = a + pos[k];
				s += score_dfs1(d-1, c, bk - num[k], bk);
		}	
	}
	return s;
}
// Speed: n = 15, m = 1000, maxr = 5, sets per microsecond: 0.027. NO RISKS.
void BDeu::score_dfs2(int d, int* c, int a, int b){
	const int i = c[d];	
	// Partition tmp[] into sublists. First scan and count; then partition.
	int num[BDEU_MAXARITY]; for (int k = 0; k < r[i]; ++k) num[k] = 0;
	for (int t = a; t < b; ++t){ int k = dat[i][tmp[d][t]]; ++num[k]; }
	if (d == 0){ // For the sake of speed, do not recurse, but solve it here.		
		for (int k = 0; k < r[i]; ++k) ++fre[num[k]];	 
		return;
	}
	int pos[BDEU_MAXARITY]; for (int k = 0; k < r[i]; ++k) pos[k] = 0;
	for (int k = 0; k < r[i]-1; ++k) pos[k+1] = pos[k] + num[k];
	for (int t = a; t < b; ++t){ int k = dat[i][tmp[d][t]]; tmp[d-1][a+pos[k]] = tmp[d][t]; ++pos[k]; }
	
	for (int k = 0; k < r[i]; ++k){
		switch (num[k]){
			case 0: break;
			case 1: ++fre[1]; break;
			default: int bk = a + pos[k]; score_dfs2(d-1, c, bk - num[k], bk); break;
		}	
	}
	return;
}
// Speed: n = 15, m = 1000, maxr = 5, sets per microsecond: 0.038. NO RISKS. THE FASTEST DFS!
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
double BDeu::score_dico(int d, int* c){ // Divide and conquer.
	for (int t = 0; t < m; ++t){ tmp[d-1][t] = t; } // Init tmp[d-1].
	//return score_dfs(d-1, c, 0, m, 1); // Old for testing purposes.
	// Note: dfs1 below requires precomputation of lng.
	//s = score_dfs1(d-1, c, 0, m) + my_lgamma(ess) - my_lgamma(m + ess);
	for (int t = 0; t <= m; ++t){ fre[t] = 0; } // Init fre.
	score_dfs3(d-1, c, 0, m);
	int maxcount = m; while (!fre[maxcount]) --maxcount;
	double q = 1; for (int j = 0; j < d; ++j) q *= r[c[j]]; 
	lng[0] = 0; for (int t = 0; t < maxcount; ++t) lng[t+1] = lng[t] + log(t + ess/q);		
	double s = my_lgamma(ess) - my_lgamma(m + ess);
	for (int t = 1; t <= maxcount; ++t){ if (fre[t]) s += fre[t] * lng[t]; }	
	return s;
}

bool equal(long *x, int len){ // Are all equal?
	int i = 1;
	while (i < len && x[i] == x[0]) ++i;
	return (i == len);
}

void isort(long* x, int len){ // Insertion sort. Note: Only guarantees that equal keys will be adjacent.   
    	for (int i = 1; i < len; ++i){  
        	int j = i; long key = x[i]; 
        	while ((j) && (x[j-1] > key)){ x[j] = x[j-1]; --j; } 
        	x[j] = key;  
    	}  
}  
 
void bsort256(long* x, int len){ // Bucket sort.
	int num[256]; for (int k = 0; k < 256; ++k) num[k] = 0;
	int pos[256]; pos[0] = 0;
	uint8_t* bin = new uint8_t[len]; long* y = new long[len];
	for (int t = 0; t < len; ++t){ 
		uint64_t v = x[t]; 
		uint8_t k = v & 0xFF; for (int p = 0; p < 7; ++p){ v >>= 8; k ^= (v & 0xFF); }
		bin[t] = k; ++num[k]; 
	}
	for (int k = 0; k < 255; ++k) pos[k+1] = pos[k] + num[k];
	for (int t = 0; t < len; ++t){ uint8_t k = bin[t]; int q = pos[k]; ++pos[k]; y[q] = x[t]; }
	for (int k = 0; k < 256; ++k){ 
		switch (num[k]){
			case 0: break; case 1: break; case 2: break;
			default: isort(y + pos[k] - num[k], num[k]); //sort(y + pos[k] - num[k], y + pos[k]);
		} 
	}
	for (int t = 0; t < len; ++t) x[t] = y[t];
	delete[] bin; delete[] y;
}
void bsort1024(long *x, int len){ // Bucket sort.
	int num[1024]; for (int k = 0; k < 1024; ++k) num[k] = 0;
	int pos[1024]; pos[0] = 0;
	uint16_t* bin = new uint16_t[len]; long* y = new long[len];
	for (int t = 0; t < len; ++t){ 
		uint64_t v = x[t]; 
		//uint32_t k = v & 0x3FF; for (int p = 0; p < 5; ++p){ v >>= 10; k ^= (v & 0x3FF); }
		//v >>= 10; k^= (v & 0xF);
		//uint32_t k = v & 0x3FF; v >>= 10; k ^= (v & 0x3FF); v >>= 10; k ^= (v & 0x3FF); 
		uint32_t k1 = v & 0x3FF, k2 = (v & 0xFFC00) >> 10, k3 = (v & 0x3FF00000) >> 20; 
		uint32_t k = k1 ^ k2 ^ k3; 	
		bin[t] = k; ++num[k]; 
	}
	for (int k = 0; k < 1023; ++k) pos[k+1] = pos[k] + num[k];
	for (int t = 0; t < len; ++t){ uint16_t k = bin[t]; int q = pos[k]; ++pos[k]; y[q] = x[t]; }
	for (int k = 0; k < 1024; ++k){ 
		switch (num[k]){
			case 0: break; case 1: break; case 2: break;
			default:
				if (equal(y + pos[k] - num[k], num[k])) break;
				else if (num[k] < 24) isort(y + pos[k] - num[k], num[k]);
				else                  sort (y + pos[k] - num[k], y + pos[k]);
		} 
	}
	for (int t = 0; t < len; ++t) x[t] = y[t];
	delete[] bin; delete[] y;
}
void bsort4096(long *x, int len){ // Bucket sort.
	int num[4096]; for (int k = 0; k < 4096; ++k) num[k] = 0;
	int pos[4096]; pos[0] = 0;
	uint16_t* bin = new uint16_t[len]; long* y = new long[len];
	for (int t = 0; t < len; ++t){ 
		uint64_t v = x[t]; // + t*t; 
		//uint32_t k = v & 0xFFF; for (int p = 0; p < 4; ++p){ v >>= 12; k ^= (v & 0xFFF); }
		//v >>= 12; k^= (v & 0xF); 
		uint64_t k = v & 0xFFF; v >>= 12; k ^= (v & 0xFFF);
		bin[t] = k; ++num[k]; 
	}
	for (int k = 0; k < 4095; ++k) pos[k+1] = pos[k] + num[k];
	for (int t = 0; t < len; ++t){ uint16_t k = bin[t]; int q = pos[k]; ++pos[k]; y[q] = x[t]; }
	for (int k = 0; k < 4096; ++k){ 
		switch (num[k]){
			case 0: break; case 1: break; case 2: break;
			default: 
				if (equal(y + pos[k] - num[k], num[k])) break;
				else if (num[k] < 24) isort(y + pos[k] - num[k], num[k]);
				else                  sort (y + pos[k] - num[k], y + pos[k]);

		}  
	}
	for (int t = 0; t < len; ++t) x[t] = y[t];
	delete[] bin; delete[] y;
}


// Speed: n = 15, m = 1000, maxr = 5, sets per microsecond: 0.068 [40.6 s in tot], using bsort1024. THE FASTEST !!
double BDeu::score_sort(int d, int* c){ // By (any) sorting algorithm.
	// Form a list of keys.
	long* x = new long[m+1];
	for (int t = 0; t < m; ++t) x[t] = dat[c[0]][t]; // Note: Switching to tad won't help, only harm.
	for (int j = 1; j < d; ++j){
		int i = c[j];
		for (int t = 0; t < m; ++t){ uint64_t y = x[t]; y <<= w[i]; y |= dat[i][t]; x[t] = y; }
	}
	x[m] = -1;
	// Sort the list. This is the bottleneck.
	if      (m <  128) sort     (x, x+m);
	else if (m <  512) bsort256 (x, m);
	else if (m < 2048) bsort1024(x, m);
	else               bsort4096(x, m);
	// Find multiplicities.
	int t = 0; int maxcount = 0;
	while (t < m){
		int count = 1; ++t;
		while (x[t-1] == x[t]){ ++count; ++t; }
		if (count > maxcount) maxcount = count;
	}
	double q = 1; for (int j = 0; j < d; ++j) q *= r[c[j]]; 
	lng[0] = 0; for (int t = 0; t < maxcount; ++t) lng[t+1] = lng[t] + log(t + ess/q);	
	// BDeu.
	t = 0; double s = my_lgamma(ess) - my_lgamma(m + ess);
	while (t < m){
		int count = 1; ++t; 
		while (x[t-1] == x[t]){ ++count; ++t; } 
		s += lng[count];
	}
	delete[] x;
	return s;
}

double BDeu::score_all(){ // Scoring all nonempty subsets the n variables.
	int* c = new int[n]; // Set of variables.
	for (int i = 0; i < n; ++i) c[i] =  i;
	
	ess = 0.1; // Set ess. Better do this elsewhere.
	//cout << " [score_all:] Got his far... " << endl;
	double s = 0; int count = 0;
	for (int X = 1; X < (1 << n); ++X){
		int d = 0; // Size of X.
		for (int i = 0; i < n; ++i){ if (X & (1 << i)){ c[d++] = i; } } // Set c.
		if (X) s += cliq(c, d);		
		++count;
	}
	delete[] c;
	return s;
}

ostream& operator<<(ostream& os, const BDeu& x){
	for (int t = 0; t < x.m; ++t){
		for (int i = 0; i < x.n; ++i){
			os << " " << x.dat[i][t];
		}
		os << endl;			
	}
	return os;
}
void BDeu::test(void){
	cout << "[test:] " << "rand() = " << rand() << " " << rand() << endl;
	for (int round = 5; round <= 15; ++round){
		init(100*round, 5 + round); fill_rnd(5);
		//cout << " [test:] Fill done " << endl;
		auto start = high_resolution_clock::now();
		double s = score_all(); int count = (1 << n) - 1;
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start); 	
		cout << setw(6);
		cout << "[score_all] done, n = " << n << ", m  = " << m << ", count = " << count;
		cout << ", per microsecond = " << (double) count/duration.count()<< ", s = " << s << endl;
		fini();	
	}
}
BDeu::BDeu(void){ initdone = false; }
BDeu::~BDeu(void){ if (initdone) fini(); }
void BDeu::init(int m0, int n0){
	ess = 1;
	m = m0; n = n0;
	dat = new Tdat*[n]; tad = new Tdat*[m]; tmp = new int*[n]; 
	r = new int[n]; w = new int[n]; lng = new double[m+1]; fre = new int[m+1];
	for (int i = 0; i < n; ++i){ dat[i] = new Tdat[m]; tmp[i] = new int[m]; for (int t = 0; t < m; ++t) tmp[i][t] = 0; }
	for (int t = 0; t < m; ++t){ tad[t] = new Tdat[n]; }
	initdone = true;
}
void BDeu::fini(){
	for (int i = 0; i < n; ++i){ delete[] dat[i]; delete[] tmp[i]; } 
	for (int t = 0; t < m; ++t){ delete[] tad[t]; } 
	delete[] dat; delete[] tad; delete[] tmp; delete[] r; delete[] w; delete[] lng; delete[] fre;
	initdone = false;
}

void BDeu::print_tmp(){
	cout << "tmp:" << endl;
	for (int t = 0; t < m; ++t){
		for (int i = 0; i < n; ++i){ cout << " \t" << tmp[i][t]; }
		cout << endl;
	}
}



