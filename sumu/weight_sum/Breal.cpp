#include "Breal.hpp"

std::ostream& operator<<(std::ostream& os, B2real x){
	if (x.b >= 0){ os << x.a; os << "B+" << x.b; }
	else         { os << x.a; os << "B"  << x.b; }
	return os;
}
