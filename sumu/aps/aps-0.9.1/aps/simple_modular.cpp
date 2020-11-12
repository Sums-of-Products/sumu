#include "array.h"
#include "bits.h"
#include "types.h"

#include "simple_common.h"

#include <algorithm>

namespace aps {

namespace {

template <typename T>
Array<T> alphaSum(const Array<Array<T>>& z) {
	size_t n = z.size();
	Array<T> alpha(S1 << n);

	alpha[0] = getOne<T>();
	for(size_t mask = 1; mask < alpha.size(); ++mask) {
		T plusVal = getZero<T>();
		T minusVal = getZero<T>();

		size_t maskPopCount = popCount(mask);

		for(size_t sub = 0; sub != mask; sub = (sub - mask) & mask) {
			T term = alpha[sub];
			size_t left = mask ^ sub;
			while(left) {
				size_t v = bottomOneBitIdx(left);
				left ^= S1 << v;
				term *= z[v][collapseBit(sub, v)];
			}
			if((popCount(sub) ^ maskPopCount) & 1) {
				plusVal += term;
			} else {
				minusVal += term;
			}
		}

		alpha[mask] = nonnegativeSubtraction(plusVal, minusVal);
	}

	return alpha;
}

template <typename T>
Array<T> betaSum(const Array<Array<T>>& z) {
	size_t n = z.size();
	Array<T> beta(S1 << n);

	size_t full = beta.size() - S1;

	beta[0] = getOne<T>();
	for(size_t mask = 1; mask < beta.size(); ++mask) {
		T plusVal = getZero<T>();
		T minusVal = getZero<T>();

		size_t comp = full ^ mask;
		size_t maskPopCount = popCount(mask);

		for(size_t sub = 0; sub != mask; sub = (sub - mask) & mask) {
			T term = beta[sub];
			size_t left = mask ^ sub;
			while(left) {
				size_t v = bottomOneBitIdx(left);
				left ^= S1 << v;
				term *= z[v][collapseBit(comp, v)];
			}
			if((popCount(sub) ^ maskPopCount) & 1) {
				plusVal += term;
			} else {
				minusVal += term;
			}
		}

		beta[mask] = nonnegativeSubtraction(plusVal, minusVal);
	}

	return beta;
}

template <typename T>
Array<Array<T>> gammaSum(const Array<T>& beta, const Array<Array<T>>& z) {
	size_t n = z.size();
	Array<Array<T>> gamma(n);

	size_t full = (S1 << n) - S1;

	for(size_t v = 0; v < n; ++v) {
		gamma[v] = Array<T>(S1 << (n - 1));

		for(size_t mask = 0; mask < gamma[v].size(); ++mask) {
			size_t expMask = expandBit(mask, v);
			size_t dom = full ^ (S1 << v) ^ expMask;

			T plusVal = getZero<T>();
			T minusVal = getZero<T>();

			size_t sub = 0;
			do {
				T term = beta[dom ^ sub];
				
				size_t left = sub;
				while(left) {
					size_t x = bottomOneBitIdx(left);
					left ^= S1 << x;
					term *= z[x][collapseBit(expMask, x)];
				}

				if(popCount(sub) & 1) {
					minusVal += term;
				} else {
					plusVal += term;
				}

				sub = (sub - dom) & dom;
			} while(sub);

			gamma[v][mask] = nonnegativeSubtraction(plusVal, minusVal);
		}
	}

	return gamma;
}

}

template <typename T>
Array<Array<T>> modularAPS_simple(const Array<Array<T>>& w, bool test) {
	size_t n = w.size();
	if(test && n > 11) {
		return Array<Array<T>>();
	}

	for(size_t v = 0; v < n; ++v) {
		if(w[v].size() != (S1 << (n - 1))) {
			fail("Internal error: weight array with invalid size");
		}
	}

	Array<Array<T>> z(n);
	for(size_t v = 0; v < n; ++v) {
		z[v] = downZeta(w[v]);
	}

	Array<T> alpha = alphaSum(z);
	Array<T> beta = betaSum(z);
	Array<Array<T>> ret = gammaSum(beta, z);

	for(size_t v = 0; v < n; ++v) {
		for(size_t i = 0; i < ret[v].size(); ++i) {
			ret[v][i] *= alpha[expandBit(i, v)];
		}
		upZeta(ret[v]);
		for(size_t i = 0; i < ret[v].size(); ++i) {
			ret[v][i] *= w[v][i];
		}
	}

	return ret;
}

template <typename T>
Array<Array<T>> modularAR_simple(const Array<Array<T>>& w, bool test) {
	size_t n = w.size();
	if(test && n > 11) {
		return Array<Array<T>>();
	}

	for(size_t v = 0; v < n; ++v) {
		if(w[v].size() != (S1 << (n - 1))) {
			fail("Internal error: weight array with invalid size");
		}
	}

	Array<Array<T>> z(n);
	for(size_t v = 0; v < n; ++v) {
		z[v] = downZeta(w[v]);
	}

	Array<T> alpha = alphaSum(z);
	Array<T> beta = betaSum(z);
	Array<Array<T>> gamma = gammaSum(beta, z);

	Array<Array<T>> ret(n);
	for(size_t i = 0; i < n; ++i) {
		ret[i] = Array<T>(n);
		ret[i].fill(getZero<T>());
	}
	for(size_t j = 0; j < n; ++j) {
		for(size_t u = 0; u < gamma[j].size(); ++u) {
			size_t ue = expandBit(u, j);
			T val = gamma[j][u] * z[j][u] * alpha[ue];
			size_t left = ((S1 << n) - S1) ^ ue;
			while(left) {
				size_t i = bottomOneBitIdx(left);
				left ^= S1 << i;
				ret[i][j] += val;
			}
		}
	}

	return ret;
}

#undef APS_FOR_EACH_NUMBER_TYPE_TEMPLATE
#define APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(T) \
	template Array<Array<T>> modularAPS_simple(const Array<Array<T>>&, bool); \
	template Array<Array<T>> modularAR_simple(const Array<Array<T>>&, bool);
APS_FOR_EACH_NUMBER_TYPE

}
