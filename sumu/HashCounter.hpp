#ifndef HASHCOUNTER_HPP
#define HASHCOUNTER_HPP

using namespace std;

using Tptrs = int; //uint32_t;

class HashCounterBase { // Reserves a bit less than MiB of space for a *static* hash table. 
    public:
	static Tptrs ptrs[]; static const uint32_t mask, lmask;
};
//Tptrs HashCounterBase::ptrs[512*1024] { 0 }; const uint32_t HashCounterBase::mask = 0x7FFFF, HashCounterBase::lmask = 19; 
  Tptrs HashCounterBase::ptrs[256*1024] { 0 }; const uint32_t HashCounterBase::mask = 0x3FFFF, HashCounterBase::lmask = 18; 
//Tptrs HashCounterBase::ptrs[ 64*1024] { 0 }; const uint32_t HashCounterBase::mask = 0x0FFFF, HashCounterBase::lmask = 16; 


///////////////////////////////////////////////////
// Class HashCounter:

struct keycount { uint64_t key; int num; Tptrs nxt; }; // The int fields could be replaced by Tptrs for most practical purposes.

class HashCounter : public HashCounterBase 
{ // Relatively huge hash range; active bins iterable. XOR hashing. Currently the count functions not used at all.
    public:
	void insert(uint64_t key){ int p = hash(key); insert(key, p); }
	int   count(uint64_t key){ int p = hash(key); return count(key, p); }
	int    hash(uint64_t k  ){ uint32_t p = k & mask; k >>= lmask; p ^= k & mask; k >>= lmask; return (p ^ k) & mask; } 
	void insert(uint64_t key, int p){
		Tptrs q = ptrs[p]; if (q == 0){ buck[nex] = { key, 1, 0 }; ptrs[p] = nex; act[lact] = p; ++lact; ++nex; return; }
		do { keycount kc = buck[q]; if (kc.key == key){ ++buck[q].num; return; } q = kc.nxt; } while (q); 
		buck[nex] = { key, 1, ptrs[p] }; ptrs[p] = nex; ++nex; // Not found. Becomes the *head*.
	}
	int   count(uint64_t key, int p){
	 	int q = ptrs[p]; while (q){ keycount kc = buck[q]; if (kc.key == key){ return kc.num; } q = kc.nxt; } return 0; 
	}
	int get_freq_and_reset(uint32_t* fre){ // We assume fre[] has been initialized to zero. Returns the largest count encountered.
		int maxc = 0;
		for (int j = 0; j < lact; ++j){
			int p = act[j]; int q = ptrs[p]; ptrs[p] = 0; // Zero the static ptrs[]; we are about to delete the data structure.
			while (q){ keycount kc = buck[q]; int c = buck[q].num; ++fre[c]; maxc = max(maxc, c); q = kc.nxt; }
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
//	static Tptrs ptrs[]; static const uint32_t mask, lmask;
};
//Tptrs HashCounter::ptrs[512*1024] { 0 }; const uint32_t HashCounter::mask = 0x7FFFF, HashCounter::lmask = 19; 
//Tptrs HashCounter::ptrs[256*1024] { 0 }; const uint32_t HashCounter::mask = 0x3FFFF, HashCounter::lmask = 18; 
//Tptrs HashCounter::ptrs[ 64*1024] { 0 }; const uint32_t HashCounter::mask = 0x0FFFF, HashCounter::lmask = 16; 


///////////////////////////////////////////////////
// Class Has2Counter:

struct hc2key   { uint64_t k1; uint64_t k2; inline bool operator==(const hc2key& x) const { return k1 == x.k1 && k2 == x.k2; }};
struct ke2count { hc2key key; int num; Tptrs nxt; }; // The int fields could be replaced by Tptrs for most practical purposes.

class Has2Counter : public HashCounterBase 
{ // Relatively huge hash range; active bins iterable. XOR hashing. Currently the count functions not used at all.
    public:
	void insert(hc2key key){ int p = hash(key.k1); insert(key, p); }
	int   count(hc2key key){ int p = hash(key.k1); return count(key, p); }
	int    hash(uint64_t k){ uint32_t p = k & mask; k >>= lmask; p ^= k & mask; k >>= lmask; return (p ^ k) & mask; } 
	void insert(hc2key key, int p){
		Tptrs q = ptrs[p]; if (q == 0){ buck[nex] = { key, 1, 0 }; ptrs[p] = nex; act[lact] = p; ++lact; ++nex; return; }
		do { ke2count kc = buck[q]; if (kc.key == key){ ++buck[q].num; return; } q = kc.nxt; } while (q); 
		buck[nex] = { key, 1, ptrs[p] }; ptrs[p] = nex; ++nex; // Not found. Becomes the *head*.
	}
	int   count(hc2key key, int p){
	 	int q = ptrs[p]; while (q){ ke2count kc = buck[q]; if (kc.key == key){ return kc.num; } q = kc.nxt; } return 0; 
	}
	int get_freq_and_reset(uint32_t* fre){ // We assume fre[] has been initialized to zero. Returns the largest count encountered.
		int maxc = 0;
		for (int j = 0; j < lact; ++j){
			int p = act[j]; int q = ptrs[p]; ptrs[p] = 0; // Zero the static ptrs[]; we are about to delete the data structure.
			while (q){ ke2count kc = buck[q]; int c = buck[q].num; ++fre[c]; maxc = max(maxc, c); q = kc.nxt; }
		} return maxc; 
	}
	int maxload(){ // Mainly for testing. Note: a slow routine; we don't want to slow down insert by additional bookkeeping.
		int maxl = 0; 
		for (int j = 0; j < lact; ++j){
			int l = 0; int p = act[j]; int q = ptrs[p]; 
			while (q){ ++l; ke2count kc = buck[q]; q = kc.nxt; } if (l > maxl) maxl = l;
		} return maxl; 
	}
	Has2Counter (int m){ buck = new ke2count[m+1]; nex = 1; act = new int[m]; lact = 0; } 
	~Has2Counter(     ){ delete[] buck; delete[] act; }
    private:
	int* act;		int lact;	// Active bins.
	ke2count* buck;		int nex;	// The count of *unique* keys + 1; also the index of the next free slot in buck.
//	static Tptrs ptrs[]; static const uint32_t mask, lmask;
};
//Tptrs Has2Counter::ptrs[512*1024] { 0 }; const uint32_t Has2Counter::mask = 0x7FFFF, Has2Counter::lmask = 19; 
//Tptrs Has2Counter::ptrs[256*1024] { 0 }; const uint32_t Has2Counter::mask = 0x3FFFF, Has2Counter::lmask = 18; 
//Tptrs Has2Counter::ptrs[64*1024] { 0 }; const uint32_t HasCounter::mask = 0xFFFF, Has2Counter::lmask = 16; 

///////////////////////////////////////////////////
// Class HasXCounter:

/* Slow, and thus not included. Apparently std::vector<> does not support fast enough primitive operations. 

using hckey_t = vector<uint64_t>;
struct keycount_t { hckey_t key; int num; int nxt; }; // The int fields could be replaced by Tptrs for most practical purposes.

class HasXCounter { // Relatively huge hash range; active bins iterable. XOR hashing. Currently the count functions not used at all.
    public:
	void insert(hckey_t& key){ int p = hash(key[0]); insert(key, p); }
	int   count(hckey_t& key){ int p = hash(key[0]); return count(key, p); }
	int    hash(uint64_t k){ uint32_t p = k & mask; k >>= lmask; p ^= k & mask; k >>= lmask; return (p ^ k) & mask; } 
	void insert(hckey_t& key, int p){
		Tptrs q = ptrs[p]; if (q == 0){ buck[nex] = { key, 1, 0 }; ptrs[p] = nex; act[lact] = p; ++lact; ++nex; return; }
		do { keycount_t kc = buck[q]; if (kc.key == key){ ++buck[q].num; return; } q = kc.nxt; } while (q); 
		buck[nex] = { key, 1, ptrs[p] }; ptrs[p] = nex; ++nex; // Not found. Becomes the *head*.
	}
	int   count(hckey_t& key, int p){
	 	int q = ptrs[p]; while (q){ keycount_t kc = buck[q]; if (kc.key == key){ return kc.num; } q = kc.nxt; } return 0; 
	}
	int get_freq_and_reset(Tptrs* fre){ // We assume fre[] has been initialized to zero. Returns the largest count encountered.
		int maxc = 0;
		for (int j = 0; j < lact; ++j){
			int p = act[j]; int q = ptrs[p]; ptrs[p] = 0; // Zero the static ptrs[]; we are about to delete the data structure.
			while (q){ keycount_t kc = buck[q]; int c = buck[q].num; ++fre[c]; maxc = max(maxc, c); q = kc.nxt; }
		} return maxc; 
	}
	int maxload(){ // Mainly for testing. Note: a slow routine; we don't want to slow down insert by additional bookkeeping.
		int maxl = 0; 
		for (int j = 0; j < lact; ++j){
			int l = 0; int p = act[j]; int q = ptrs[p]; 
			while (q){ ++l; keycount_t kc = buck[q]; q = kc.nxt; } if (l > maxl) maxl = l;
		} return maxl; 
	}
	HasXCounter (int m){ buck = new keycount_t[m+1]; nex = 1; act = new int[m]; lact = 0; } 
	~HasXCounter(     ){ delete[] buck; delete[] act; }
    private:
	int* act;		int lact;	// Active bins.
	keycount_t* buck;	int nex;	// The count of *unique* keys + 1; also the index of the next free slot in buck.
	static Tptrs ptrs[]; static const uint32_t mask, lmask;
};
//Tptrs HasXCounter::ptrs[512*1024] { 0 }; const uint32_t HasXCounter::mask = 0x7FFFF, HasXCounter::lmask = 19; 
//Tptrs HasXCounter::ptrs[256*1024] { 0 }; const uint32_t HasXCounter::mask = 0x3FFFF, HasXCounter::lmask = 18; 
Tptrs HasXCounter::ptrs[64*1024] { 0 }; const uint32_t HasXCounter::mask = 0xFFFF, HasXCounter::lmask = 16; 
*/

#endif

