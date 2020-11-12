#include "array.h"
#include "bits.h"
#include "types.h"

#include "simple_common.h"

namespace aps {

namespace {

template <typename T>
Array<T> alphaSum(const Array<Array<T>>& z) {
	size_t n = z.size();
	Array<T> alpha(S1 << n);

	alpha[0] = getOne<T>();
	for(size_t mask = 1; mask < alpha.size(); ++mask) {
		T val = getZero<T>();

		size_t left = mask;
		while(left) {
			size_t v = bottomOneBitIdx(left);
			size_t vBit = S1 << v;
			left ^= vBit;

			val += z[v][collapseBit(mask, v)] * alpha[mask ^ vBit];
		}

		alpha[mask] = val;
	}

	return alpha;
}

template <typename T>
Array<T> betaSum(const Array<Array<T>>& z) {
	size_t n = z.size();
	Array<T> beta(S1 << n);

	size_t full = beta.size() - 1;

	beta[0] = getOne<T>();
	for(size_t mask = 1; mask < beta.size(); ++mask) {
		T val = getZero<T>();

		size_t left = mask;
		while(left) {
			size_t v = bottomOneBitIdx(left);
			size_t vBit = S1 << v;
			left ^= vBit;

			val += z[v][collapseBit(full ^ mask, v)] * beta[mask ^ vBit];
		}

		beta[mask] = val;
	}

	return beta;
}

}

template <typename T>
Array<Array<T>> orderModularAPS_simple(const Array<Array<T>>& w, bool test) {
	size_t n = w.size();
	for(size_t v = 0; v < n; ++v) {
		if(w[v].size() != (S1 << (n - 1))) {
			fail("Internal error: weight array with invalid size");
		}
	}

	Array<Array<T>> ret(n);
	for(size_t v = 0; v < n; ++v) {
		ret[v] = downZeta(w[v]);
	}

	Array<T> alpha = alphaSum(ret);
	Array<T> beta = betaSum(ret);

	for(size_t v = 0; v < n; ++v) {
		size_t full = (S1 << (n - 1)) - S1;

		for(size_t i = 0; i < ret[v].size(); ++i) {
			ret[v][i] = alpha[expandBit(i, v)] * beta[expandBit(full ^ i, v)];
		}
		upZeta(ret[v]);
		for(size_t i = 0; i < ret[v].size(); ++i) {
			ret[v][i] *= w[v][i];
		}
	}

	return ret;
}

template <typename T>
Array<Array<T>> orderModularAR_simple(const Array<Array<T>>& w, bool test) {
	size_t n = w.size();
	if(test && n > 11) {
		return Array<Array<T>>();
	}

	Array<Array<T>> hatBeta(n);
	for(size_t v = 0; v < n; ++v) {
		hatBeta[v] = downZeta(w[v]);
	}

	Array<Array<T>> ret(n);
	for(size_t i = 0; i < n; ++i) {
		ret[i] = Array<T>(n);
		ret[i].fill(getZero<T>());
	}

	for(size_t s = 0; s < n; ++s) {
		auto betaBar = [&](size_t v, size_t S, size_t U) {
			size_t vb = S1 << v;
			if(v != s && (U & vb)) {
				return nonnegativeSubtraction(hatBeta[v][collapseBit(S & ~vb, v)], hatBeta[v][collapseBit((S & ~vb) & ~U, v)]);
			} else {
				return hatBeta[v][collapseBit((S & ~vb) & ~U, v)];
			}
		};

		size_t sb = S1 << s;
		Array<Array<T>> g(S1 << n);

		g[0] = Array<T>(1);
		g[0][0] = getOne<T>();

		for(size_t S = 1; S < g.size(); ++S) {
			size_t cnt = popCount(S);
			g[S] = Array<T>(S1 << cnt);
			
			// Iterate subsets U of S
			size_t U = 0;
			for(size_t US = 0; US < g[S].size(); ++US) {
				if(!(U & sb) && ((S & sb) || U)) {
					g[S][US] = getZero<T>();
				} else {
					T val = getZero<T>();

					size_t left = S;
					size_t vi = 0;
					while(left) {
						size_t v = bottomOneBitIdx(left);
						size_t vb = S1 << v;
						left ^= vb;

						val += g[S & ~vb][collapseBit(US & ~(S1 << vi), vi)] * betaBar(v, S, U);

						++vi;
					}

					g[S][US] = val;
				}

				U = (U - S) & S;
			}
		}

		size_t full = (S1 << n) - S1;
		for(size_t U = 1; U < g[full].size(); ++U) {
			if((U & sb)) {
				T val = g[full][U];

				size_t left = U;
				while(left) {
					size_t t = bottomOneBitIdx(left);
					left ^= S1 << t;

					ret[t][s] += val;
				}
			}
		}
	}

	return ret;
}

#undef APS_FOR_EACH_NUMBER_TYPE_TEMPLATE
#define APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(T) \
	template Array<Array<T>> orderModularAPS_simple<T>(const Array<Array<T>>&, bool); \
	template Array<Array<T>> orderModularAR_simple(const Array<Array<T>>&, bool);
APS_FOR_EACH_NUMBER_TYPE

}
