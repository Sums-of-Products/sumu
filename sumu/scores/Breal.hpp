// Compile with g++ -Wall -O3 -ffast-math
// If used within a tight loop, also try out -funroll-loops
#ifndef BREAL_HPP
#define BREAL_HPP

#include <iostream>
#include <iomanip>
#include <cmath>
#include "headers.hpp"

using namespace std;

// Class Breal

class Breal { 
    public:
	uint32_t a; int32_t b; 
	
	void set_log(double z);
	void set(int64_t z);
	void set(unsigned z){ set((int64_t)z); };
	void set(double z);
	double get_log();
	double get_double();
	long double get_ldouble();
	int64_t get_lint();

	Breal& operator=(const int64_t z){ set(z); return *this; }
	Breal& operator=(const double z){ set(z); return *this; }

	/*inline Breal& operator+=(const Breal y){
		int64_t d = b - y.b; if (d > 31) return *this;
		if (d >= 0){ a += (y.a >>  d); } 
		else if (-d > 31){ *this = y; return *this; } else { a = y.a + (a >> -d); b = y.b; }
		while (a & 0x80000000){ a >>= 4; b += 4; } return *this;
	}*/
	inline Breal& operator+=(const Breal y){
		int64_t d = b - y.b;
		if ( d > 31 || y.a == 0){ return *this; }
		if (-d > 31 ||   a == 0){ *this = y; return *this; }
		if ( d >= 0){ a += (y.a >> d); } else { a = y.a + (a >> -d); b = y.b; }
		while (a & 0x80000000){ a >>= 1; b += 1; } return *this;
	}
	inline Breal& operator|=(const Breal y){ // Computes x := x + y by x |= y, assuming x > y. 
		int64_t d = b - y.b; if (d > 31) return *this;
		a += (y.a >>  d); while (a & 0x80000000){ a >>= 8; b += 8; } return *this;
	}
	inline Breal& operator<<=(int n){ b += n; return *this; }
	inline Breal& operator>>=(int n){ b -= n; return *this; }
};

void Breal::set_log(double z){ // Assumes z is the natural log of the number to be represented.
	b = (int)(z * log2(exp(1.0))) - 30; a = (uint32_t) exp(z - b * log(2.0));
	while (  a & 0x80000000 ) { a >>= 1; b++; }
	while (!(a & 0x40000000)) { a <<= 1; b--; }
}
void Breal::set(int64_t z){
	if (z == 0){ a = 0; b = -(1L << 30); return; }	
	b = 0;
	while (  z & 0x80000000 ) { z >>= 1; b++; } // Truncates the low bits of z if needed.
	while (!(z & 0x40000000)) { z <<= 1; b--; }
	a= (uint32_t)(z);
}
void Breal::set(double z){ set_log(log(z)); }
double Breal::get_log(){ return (double)(log(a) + b*log(2.0)); }
double Breal::get_double(){ return (double)(a * pow(2, b)); }
long double Breal::get_ldouble(){ return (long double)((long double)(a) * pow(2, (long double)(b))); }
int64_t Breal::get_lint(){ int64_t aL = a; if (b >= 0) return aL << b; return aL >> -b; }


inline Breal operator+(Breal x, Breal y){ // Could speed up this by 20 % using certain assumptions.
	int d = x.b - y.b; uint32_t z; int p;
	if (d >= 0){ if ( d > 31){ return x; } z = x.a + (y.a >>  d); p = x.b; } 
	else       { if (-d > 31){ return y; } z = y.a + (x.a >> -d); p = y.b; }
	if (z & 0x80000000){ z >>= 1; p += 1; } return { z, p };	
}
inline Breal operator-(Breal x, Breal y){
	int d = x.b - y.b; uint32_t z; int p;
	if (d >= 0){ if ( d > 31){ return x; } z = x.a - (y.a >>  d); p = x.b; } 
	else       { if (-d > 31){ return y; } z = (x.a >> -d) - y.a; p = y.b; }
	if (z & 0x40000000){ z <<= 1; p -= 1; } return { z, p };	
}
inline uint32_t operator^(Breal x, Breal y){ // Usage: "if (x ^ y) { then x is not close to y  }".
	int d = x.b - y.b;  uint32_t z; 
	if (d >= 0){ if ( d > 31){ return x.a; } z = x.a ^ (y.a >>  d); } 
	else       { if (-d > 31){ return y.a; } z = y.a ^ (x.a >> -d); }
	return z & 0xFFFFFF00; // Return the difference in the matched most significant 24 bits.	
}
inline uint32_t diff(uint32_t m, Breal x, Breal y){
	int d = x.b - y.b;  uint32_t z; 
	if (d >= 0){ if ( d > 31){ return x.a; } z = x.a ^ (y.a >>  d); } 
	else       { if (-d > 31){ return y.a; } z = y.a ^ (x.a >> -d); }
	return z & (((1 << m) - 1) << (31 - m)); // Return the difference in the matched most significant m bits.	
}
inline bool operator==(Breal x, Breal y){ // Usage: "if (x == y) { then x is equal to y  }".
	int d = x.b - y.b;  uint32_t z; 
	if (d >= 0){ if ( d > 31){ return (x.a == 0 && y.a == 0); } z = x.a ^ (y.a >>  d); } 
	else       { if (-d > 31){ return (y.a == 0 && x.a == 0); } z = y.a ^ (x.a >> -d); }
	return (z == 0); // Return true if no difference after matching the exponents.	
}
inline Breal operator*(Breal x, Breal y){
	uint64_t z = (uint64_t) x.a * (uint64_t) y.a;  int p = x.b + y.b;
	if (z & 0x4000000000000000){ z >>= 32; p += 32; } 
	else                       { z >>= 31; p += 31; }
	return { (uint32_t) z, p };	
}

ostream& operator<<(ostream& os, Breal x){
	if (x.b >= 0){ os << x.a << "b+" << x.b; }
	else         { os << x.a << "b"  << x.b; }
	return os;
}
inline bool operator< (const Breal x, const Breal y){ return x.b < y.b || (x.b == y.b && x.a < y.a); }
inline bool operator> (const Breal x, const Breal y){ return   y < x;  }
inline bool operator<=(const Breal x, const Breal y){ return !(y < x); }
inline bool operator>=(const Breal x, const Breal y){ return !(x < y); }

inline Breal operator+(Breal x, int64_t w){ Breal y; y = w; return x + y; }
inline Breal operator+(int64_t w, Breal y){ Breal x; x = w; return x + y; }
inline Breal operator-(Breal x, int64_t w){ Breal y; y = w; return x - y; }
inline Breal operator-(int64_t w, Breal y){ Breal x; x = w; return x - y; }
inline Breal operator*(Breal x, int64_t w){ Breal y; y = w; return x * y; }
inline Breal operator*(int64_t w, Breal y){ Breal x; x = w; return x * y; }

inline bool operator< (const Breal x, const int64_t w){ Breal y; y = w; return x <  y; }
inline bool operator< (const int64_t w, const Breal y){ Breal x; x = w; return x <  y; }
inline bool operator> (const Breal x, const int64_t w){ Breal y; y = w; return x >  y; }
inline bool operator> (const int64_t w, const Breal y){ Breal x; x = w; return x >  y; }
inline bool operator<=(const Breal x, const int64_t w){ Breal y; y = w; return x <= y; }
inline bool operator<=(const int64_t w, const Breal y){ Breal x; x = w; return x <= y; }
inline bool operator>=(const Breal x, const int64_t w){ Breal y; y = w; return x >= y; }
inline bool operator>=(const int64_t w, const Breal y){ Breal x; x = w; return x >= y; }

inline Breal operator+(Breal x, double w){ Breal y; y = w; return x + y; }
inline Breal operator+(double w, Breal y){ Breal x; x = w; return x + y; }
inline Breal operator-(Breal x, double w){ Breal y; y = w; return x - y; }
inline Breal operator-(double w, Breal y){ Breal x; x = w; return x - y; }
inline Breal operator*(Breal x, double w){ Breal y; y = w; return x * y; }
inline Breal operator*(double w, Breal y){ Breal x; x = w; return x * y; }

inline bool operator< (const Breal x, const double w){ Breal y; y = w; return x <  y; }
inline bool operator< (const double w, const Breal y){ Breal x; x = w; return x <  y; }
inline bool operator> (const Breal x, const double w){ Breal y; y = w; return x >  y; }
inline bool operator> (const double w, const Breal y){ Breal x; x = w; return x >  y; }
inline bool operator<=(const Breal x, const double w){ Breal y; y = w; return x <= y; }
inline bool operator<=(const double w, const Breal y){ Breal x; x = w; return x <= y; }
inline bool operator>=(const Breal x, const double w){ Breal y; y = w; return x >= y; }
inline bool operator>=(const double w, const Breal y){ Breal x; x = w; return x >= y; }

 
// Class B2real
class B2real { 
    public:
	uint64_t a; int64_t b; 
	
	void		set_log (double z);
	void		set_logl(long double z);
	void		set(int64_t z);
	void		set(unsigned z){ set((int64_t)z); };
	void		set(double z);
	void		set(long double z);
	double		get_log();
	double		get_double();
	long double	get_ldouble();
	int64_t		get_lint();

	B2real& operator=(const int64_t z){ set(z); return *this; }
	B2real& operator=(const double z){ set(z); return *this; }
	B2real& operator=(const long double z){ set(z); return *this; }
	
/*	inline B2real& operator+=(const B2real y){
		int64_t d = b - y.b;
		if (d >= 0){ if ( d > 63){ return *this; } a += (y.a >> d); } 
		else       { if (-d > 63){ *this = y; return *this; } a = y.a + (a >> -d); b = y.b; }
		while (a & 0x8000000000000000){ a >>= 1; b += 1; } return *this;
	}*/
	inline B2real& operator+=(const B2real y){
		int64_t d = b - y.b;
		if ( d > 63 || y.a == 0){ return *this; }
		if (-d > 63 ||   a == 0){ *this = y; return *this; }
		if ( d >= 0){ a += (y.a >> d); } else { a = y.a + (a >> -d); b = y.b; }
		while (a & 0x8000000000000000){ a >>= 1; b += 1; } return *this;
	}
	inline B2real& operator|=(const B2real y){
		int64_t d = b - y.b; if (d > 63) return *this;
		if (d >= 0){ a += (y.a >> d); } 
		else if (-d > 63){ *this = y; return *this; } else { a = y.a + (a >> -d); b = y.b; }
		while (a & 0x8000000000000000){ a >>= 8; b += 8; } return *this;
	}	
	inline B2real& operator<<=(int n){ b += n; return *this; }
	inline B2real& operator>>=(int n){ b -= n; return *this; }
};

inline void B2real::set_log (double z)     { // Assumes z is the natural log of the number to be represented.
	b = (int64_t)(z * log2 (exp (1.0))) - 62; a = (int64_t) exp (z - ((double)b) * log (2.0));
	while (a & 0x8000000000000000){ a >>= 1; b += 1; }
}
inline void B2real::set_logl(long double z){ // Assumes z is the natural log of the number to be represented.
	b = (int64_t)(z * log2l(expl(1.0))) - 62; a = (int64_t) expl(z - b * logl(2.0));
	while (a & 0x8000000000000000){ a >>= 1; b += 1; }
}
inline void B2real::set(int64_t z){
	if (z == 0){ a = 0; b = -(1LL << 62); return; }	
	b = 0;
	while (  z & 0x8000000000000000) { z >>= 1; b++; } // Truncates the low bits of z if needed.
	while (!(z & 0x4000000000000000)){ z <<= 1; b--; }
	a = (uint64_t)(z);
}
inline void   B2real::set(double z)     { set_log (log (z)); }
inline void   B2real::set(long double z){ set_logl(logl(z)); }
inline double B2real::get_log()         { return (double)(log(a) + b*log(2.0)); }
inline double B2real::get_double()      { return (double)(a * pow(2, b)); }
inline long double B2real::get_ldouble(){ return (long double)((long double)(a) * powl(2, (long double)(b))); }
inline int64_t B2real::get_lint()       { int64_t aL = a; if (b >= 0) return aL << b; return aL >> -b; }


inline B2real operator+(B2real x, B2real y){ // Could speed up this by 20 % using certain assumptions.
	int64_t d = x.b - y.b; uint64_t z; int p;
	if (d >= 0){ if ( d > 63){ return x; } z = x.a + (y.a >>  d); p = x.b; } 
	else       { if (-d > 63){ return y; } z = y.a + (x.a >> -d); p = y.b; }
	if (z & 0x8000000000000000){ z >>= 1; p += 1; } return { z, p };	
}
inline B2real operator-(B2real x, B2real y){
	int64_t d = x.b - y.b; uint64_t z; int p;
	if (d >= 0){ if ( d > 63){ return x; } z = x.a - (y.a >>  d); p = x.b; } 
	else       { if (-d > 63){ return y; } z = (x.a >> -d) - y.a; p = y.b; }
	if (z & 0x4000000000000000){ z <<= 1; p -= 1; } return { z, p };	
}
inline uint64_t operator^(B2real x, B2real y){ // Usage: "if (x ^ y) { then x is not close to y  }".
	int64_t d = x.b - y.b;  uint64_t z; 
	if (d >= 0){ if ( d > 63){ return x.a; } z = x.a ^ (y.a >>  d); } 
	else       { if (-d > 63){ return y.a; } z = y.a ^ (x.a >> -d); }
	return z & 0xFFFFFF0000000000; // Return the difference in the matched most significant 24 bits.	
}
inline uint64_t diff(uint64_t m, B2real x, B2real y){
	int d = x.b - y.b;  uint32_t z; 
	if (d >= 0){ if ( d > 63){ return x.a; } z = x.a ^ (y.a >>  d); } 
	else       { if (-d > 63){ return y.a; } z = y.a ^ (x.a >> -d); }
	return z & (((1LL << m) - 1LL) << (63 - m)); // Return the difference in the matched most significant m bits.	
}
inline bool operator==(B2real x, B2real y){ // Usage: "if (x == y) { then x is equal to y  }".
	int64_t d = x.b - y.b;  uint64_t z; 
	if (d >= 0){ if ( d > 63){ return (x.a == 0 && y.a == 0); } z = x.a ^ (y.a >>  d); } 
	else       { if (-d > 63){ return (y.a == 0 && x.a == 0); } z = y.a ^ (x.a >> -d); }
	return (z == 0); // Return true if no difference after matching the exponents.	
}
inline B2real operator*(B2real x, B2real y){ 
	uint64_t x0 = x.a & 0x7fffffff, y0 = y.a & 0x7fffffff; x.a >>= 31; y.a >>= 31; // Ignore the 31 lsb, 32 msb left.
	x0 *= y.a; y0 *= x.a; uint64_t z = x.a * y.a + (x0 >> 31) + (y0 >> 31); int64_t p = x.b + y.b + 62;
	while (z & 0x8000000000000000){ z >>= 1; ++p; } return { z, p };	
}
ostream& operator<<(ostream& os, B2real x){
	if (x.b >= 0){ os << x.a; os << "B+" << x.b; }
	else         { os << x.a; os << "B"  << x.b; }
	return os;
}
inline bool operator< (const B2real x, const B2real y){ return x.b < y.b || (x.b == y.b && x.a < y.a); }
inline bool operator> (const B2real x, const B2real y){ return   y < x;  }
inline bool operator<=(const B2real x, const B2real y){ return !(y < x); }
inline bool operator>=(const B2real x, const B2real y){ return !(x < y); }

inline B2real operator+(B2real x, int64_t w){ B2real y; y = w; return x + y; }
inline B2real operator+(int64_t w, B2real y){ B2real x; x = w; return x + y; }
inline B2real operator-(B2real x, int64_t w){ B2real y; y = w; return x - y; }
inline B2real operator-(int64_t w, B2real y){ B2real x; x = w; return x - y; }
inline B2real operator*(B2real x, int64_t w){ B2real y; y = w; return x * y; }
inline B2real operator*(int64_t w, B2real y){ B2real x; x = w; return x * y; }

inline bool operator< (const B2real x, const int64_t w){ B2real y; y = w; return x <  y; }
inline bool operator< (const int64_t w, const B2real y){ B2real x; x = w; return x <  y; }
inline bool operator> (const B2real x, const int64_t w){ B2real y; y = w; return x >  y; }
inline bool operator> (const int64_t w, const B2real y){ B2real x; x = w; return x >  y; }
inline bool operator<=(const B2real x, const int64_t w){ B2real y; y = w; return x <= y; }
inline bool operator<=(const int64_t w, const B2real y){ B2real x; x = w; return x <= y; }
inline bool operator>=(const B2real x, const int64_t w){ B2real y; y = w; return x >= y; }
inline bool operator>=(const int64_t w, const B2real y){ B2real x; x = w; return x >= y; }
 
inline B2real operator+(B2real x, double w){ B2real y; y = w; return x + y; }
inline B2real operator+(double w, B2real y){ B2real x; x = w; return x + y; }
inline B2real operator-(B2real x, double w){ B2real y; y = w; return x - y; }
inline B2real operator-(double w, B2real y){ B2real x; x = w; return x - y; }
inline B2real operator*(B2real x, double w){ B2real y; y = w; return x * y; }
inline B2real operator*(double w, B2real y){ B2real x; x = w; return x * y; }

inline bool operator< (const B2real x, const double w){ B2real y; y = w; return x <  y; }
inline bool operator< (const double w, const B2real y){ B2real x; x = w; return x <  y; }
inline bool operator> (const B2real x, const double w){ B2real y; y = w; return x >  y; }
inline bool operator> (const double w, const B2real y){ B2real x; x = w; return x >  y; }
inline bool operator<=(const B2real x, const double w){ B2real y; y = w; return x <= y; }
inline bool operator<=(const double w, const B2real y){ B2real x; x = w; return x <= y; }
inline bool operator>=(const B2real x, const double w){ B2real y; y = w; return x >= y; }
inline bool operator>=(const double w, const B2real y){ B2real x; x = w; return x >= y; }

#endif
