//////////////////////////////////////////////////////////////////////////////
// Wsets.hpp
//////////////////////////////////////////////////////////////////////////////
// *** Note: Should make separate .h/.hpp and .c/.cpp files. 
// *** Currently all in this file. So, don't include multiple times elsewhere! 
//////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <unordered_map>

using namespace std; 

//============
using bmap = uint64_t;
//using bmap = unsigned __int128;

inline void allzero   (bmap *S) {*S = 0L;}
inline bmap setbit    (bmap S, int i){ return (S | (1L << i)); }
inline bmap setbits   (bmap S, uint8_t block, int p){ return (S | (block << p)); }
inline bool intersects(bmap A, bmap B){ return A & B; }
inline bool subseteq  (bmap A, bmap B){ return (A == (A & B)); } 

void show(bmap S){ for (bmap i = 0; i < 64; ++i){ cout << (S & 1) << ""; S = S >> 1; }}
void show(bmap* a, int m){ for(int i = 0; i < m; ++i){ show(a[i]); }}

bmap get_bmap_sparse(int* X, int l){ bmap S = 0L; for (int j = 0; j < l; ++j) S = setbits(S, (uint8_t)(X[j]+1), 8 * j); return S; } 
bmap get_bmap_dense (int* X, int l){ bmap S = 0L; for (int j = 0; j < l; ++j) S = setbit (S, X[j]); return S; }
bmap get_bmap       (int* X, int l){ return get_bmap_dense(X, l); } // Default is dense.

void get_set_dense_inc(bmap S, int* X, int &len){
	//len = 0; bmap mask = 1L; for (int i = 0; i < n; ++i; mask <<= 1){ if (mask & S){ X[len] = i; ++len; } 
	len = 0; int i = 0; while (S){ if (1L & S){ X[len] = i; ++len; } ++i; S >>= 1; } // Elements are in increasing order.
}
void get_set(bmap S, int* X, int &len){ get_set_dense_inc(S, X, len); }

//============
struct wset { bmap set; double weight; }; // Either "dense" for any subset of [64], or "sparse" for any subset of [256] of size at most 8. 

bool incr(wset x, wset y){ return x.weight < y.weight; }
bool decr(wset x, wset y){ return x.weight > y.weight; }
void sort(wset *c, int m){ std::sort(c, c + m, decr); }

bool incr_set(wset x, wset y){ return x.set < y.set; }
bool decr_set(wset x, wset y){ return x.set > y.set; }
void sort_set(wset *c, int m){ std::sort(c, c + m, incr_set); } // Increasing order by the bmap of the set, the empty set first.

void show(wset s){ show(s.set); cout << " : " << s.weight << endl; }
void show(wset* c, int m){ for(int i = 0; i < m; ++i){ show(c[i]); }}

wset get_wset_sparse(int* X, int l, double val){ return { get_bmap_sparse(X, l), val }; }
wset get_wset_dense (int* X, int l, double val){ return { get_bmap_dense (X, l), val }; }
wset get_wset       (int* X, int l, double val){ return   get_wset_dense (X, l, val); } // Default is dense.

//=== bma4 ========= Can represent a subset of {0, 1, 2, ..., 255 }.
struct bma4 {bmap s[4]; bool operator==(const bma4& S) const { return s[0]==S.s[0] && s[1]==S.s[1] && s[2]==S.s[2] && s[3]==S.s[3]; } }; 
struct has4 { size_t operator()(const bma4& S) const { return S.s[0] ^ S.s[1] ^ S.s[2] ^ S.s[3]; } }; 
inline bma4 setbit(bma4 S, int i){ S.s[i >> 6] |= (1L << (i & 0x3F)); return S; } // a = i / 64; b = i % 64.
bma4 get_bma4(int* X, int l){ bma4 S = { 0L }; for (int j = 0; j < l; ++j) S = setbit (S, X[j]); return S; }

//=== wse4 ========= Paired with a weight.
struct wse4 {bma4 set; double weight; };
wse4 get_wse4(int* X, int l, double val){ return { get_bma4(X, l), val }; }

//=== Wsets ======== Storage for collections indexed by bmap or bma4.
class Wsets { // Chooses how to implement a put/get-storage for wsets.
    public:
	Wsets(){};
	~Wsets(){};
	void init(int n0){ set_n(n0); }
	void set_n(int n0){ n = n0; small = (n <= 64); } 
	void put (int* X, int len, double val){ 
		if (small){ wset x = get_wset(X, len, val); M1.insert({ x.set, x.weight }); }
		else      { wse4 x = get_wse4(X, len, val); M4.insert({ x.set, x.weight }); }
	}
	bool get (int* X, int len, double* val){
		if (small){ 
			bmap S = get_bmap(X, len); it1 = M1.find(S);
			if (it1 == M1.end()){ return false; } *val = it1->second; return true;
		} else { 
			bma4 S = get_bma4(X, len); it4 = M4.find(S);
			if (it4 == M4.end()){ return false; } *val = it4->second; return true;
		}
	}
	int size(){ if (small) return M1.size(); return M4.size(); }
	void demo(){
		int X[3] = {6, 4, 1}; int Y[4] = {1, 183, 4, 3}; double vx = 6.41; double vy = 1.18343; double val; bool b;  
		put(X, 3, vx); b = get(X, 3, &val); 
		cout<<" [Wsets::demo:] weight of {6, 4, 1}      = "<< val <<", success = "<< b <<endl;
		set_n(200); 
		put(Y, 4, vy); b = get(Y, 4, &val); 
		cout<<" [Wsets::demo:] weight of {1, 183, 4, 3} = "<< val <<", success = "<< b << "; "<< get(X, 3, &val) <<endl;
	}   
    private:
	int						n;	// Assumes the stored sets are subsets of {0, 1, 2, ..., n-1}.
	bool						small;	// True if n <= 64 and false otherwise.
	unordered_map < bmap, double >  		M1;	// Currently only supports small ground sets, using the 64-bit bmap.
	unordered_map < bmap, double >::iterator	it1;	// Iterator. Yes, STL containers force us to use one, unfortunately.
	unordered_map < bma4, double, has4 >  		M4;	// Supports ground sets up to 256 elements.
	unordered_map < bma4, double, has4 >::iterator	it4;	// Iterator. Yes, STL containers force us to use one, unfortunately.

};


