#include "array.h"
#include "bits.h"
#include "selectconstant.h"
#include "types.h"

namespace aps {

namespace {

static const size_t TplLevels = 8;

template <typename T, size_t L>
struct DownZetaTpl_ {
	static void run(const T* src, T* z) {
		const size_t level = L - 1;
		size_t d = S1 << level;
		DownZetaTpl_<T, level>::run(src, z);
		DownZetaTpl_<T, level>::run(src + d, z + d);
		for(size_t i = 0; i < d; ++i) {
			z[i | d] += z[i];
		}
	}
};

template <typename T>
struct DownZetaTpl_<T, 0> {
	static void run(const T* src, T* z) {
		z[0] = src[0];
	}
};

template <typename T>
void downZeta_(const T* src, T* z, size_t level) {
	if(level < TplLevels) {
		selectConstant<TplLevels>(level, [&](auto sel) {
			DownZetaTpl_<T, sel.Val>::run(src, z);
		});
		return;
	}

	if(level) {
		--level;
		size_t d = S1 << level;
		downZeta_(src, z, level);
		downZeta_(src + d, z + d, level);
		for(size_t i = 0; i < d; ++i) {
			z[i | d] += z[i];
		}
	} else {
		z[0] = src[0];
	}
}

template <typename T>
Array<T> downZeta(const Array<T>& src) {
	size_t n = bottomOneBitIdx(src.size());
	Array<T> z(src.size());
	downZeta_(src.data(), z.data(), n);
	return z;
}

template <typename T, size_t L>
struct UpZetaTpl_ {
	static void run(T* z) {
		const size_t level = L - 1;
		size_t d = S1 << level;
		UpZetaTpl_<T, level>::run(z);
		UpZetaTpl_<T, level>::run(z + d);
		for(size_t i = 0; i < d; ++i) {
			z[i] += z[i | d];
		}
	}
};

template <typename T>
struct UpZetaTpl_<T, 0> {
	static void run(T* z) {}
};

template <typename T>
void upZeta_(T* z, size_t level) {
	if(level < TplLevels) {
		selectConstant<TplLevels>(level, [&](auto sel) {
			UpZetaTpl_<T, sel.Val>::run(z);
		});
		return;
	}

	if(level) {
		--level;
		size_t d = S1 << level;
		upZeta_(z, level);
		upZeta_(z + d, level);
		for(size_t i = 0; i < d; ++i) {
			z[i] += z[i | d];
		}
	}
}

template <typename T>
void upZeta(Array<T>& w) {
	size_t n = bottomOneBitIdx(w.size());
	upZeta_(w.data(), n);
}

template <typename T, size_t L>
struct AlphaSumTpl_ {
	static void run(
		Array<T>& alpha,
		const Array<Array<T>>& z,
		size_t pos,
		bool beenRight
	) {
		const size_t level = L - 1;
		size_t d = S1 << level;
		AlphaSumTpl_<T, level>::run(alpha, z, pos, beenRight);
		for(size_t i = 0; i < d; ++i) {
			size_t a = pos | i;
			size_t b = a | d;
			T add = z[level][collapseBit(b, level)] * alpha[a];
			if(beenRight) {
				alpha[b] += add;
			} else {
				alpha[b] = add;
			}
		}
		AlphaSumTpl_<T, level>::run(alpha, z, pos | d, true);
	}
};

template <typename T>
struct AlphaSumTpl_<T, 0> {
	static void run(
		Array<T>& alpha,
		const Array<Array<T>>& z,
		size_t pos,
		bool beenRight
	) {}
};

template <typename T>
void alphaSum_(
	Array<T>& alpha,
	const Array<Array<T>>& z,
	size_t pos,
	size_t level,
	bool beenRight
) {
	if(level < TplLevels) {
		selectConstant<TplLevels>(level, [&](auto sel) {
			AlphaSumTpl_<T, sel.Val>::run(alpha, z, pos, beenRight);
		});
		return;
	}

	if(level) {
		--level;
		size_t d = S1 << level;
		alphaSum_(alpha, z, pos, level, beenRight);
		for(size_t i = 0; i < d; ++i) {
			size_t a = pos | i;
			size_t b = a | d;
			T add = z[level][collapseBit(b, level)] * alpha[a];
			if(beenRight) {
				alpha[b] += add;
			} else {
				alpha[b] = add;
			}
		}
		alphaSum_(alpha, z, pos | d, level, true);
	}
}

template <typename T>
Array<T> alphaSum(const Array<Array<T>>& z) {
	size_t n = z.size();
	Array<T> alpha(S1 << n);
	alpha[0] = getOne<T>();
	alphaSum_(alpha, z, 0, n, false);
	return alpha;
}

template <typename T, size_t L>
struct BetaSumTpl_ {
	static void run(
		Array<T>& beta,
		const Array<Array<T>>& z,
		size_t pos,
		bool beenRight
	) {
		const size_t level = L - 1;
		size_t d = S1 << level;
		size_t full = beta.size() - S1;
		BetaSumTpl_<T, level>::run(beta, z, pos, beenRight);
		for(size_t i = 0; i < d; ++i) {
			size_t a = pos | i;
			size_t b = a | d;
			T add = z[level][collapseBit(full ^ b, level)] * beta[a];
			if(beenRight) {
				beta[b] += add;
			} else {
				beta[b] = add;
			}
		}
		BetaSumTpl_<T, level>::run(beta, z, pos | d, true);
	}
};

template <typename T>
struct BetaSumTpl_<T, 0> {
	static void run(
		Array<T>& beta,
		const Array<Array<T>>& z,
		size_t pos,
		bool beenRight
	) {}
};

template <typename T>
void betaSum_(
	Array<T>& beta,
	const Array<Array<T>>& z,
	size_t pos,
	size_t level,
	bool beenRight
) {
	if(level < TplLevels) {
		selectConstant<TplLevels>(level, [&](auto sel) {
			BetaSumTpl_<T, sel.Val>::run(beta, z, pos, beenRight);
		});
		return;
	}

	if(level) {
		--level;
		size_t d = S1 << level;
		size_t full = beta.size() - S1;
		betaSum_(beta, z, pos, level, beenRight);
		for(size_t i = 0; i < d; ++i) {
			size_t a = pos | i;
			size_t b = a | d;
			T add = z[level][collapseBit(full ^ b, level)] * beta[a];
			if(beenRight) {
				beta[b] += add;
			} else {
				beta[b] = add;
			}
		}
		betaSum_(beta, z, pos | d, level, true);
	}
}

template <typename T>
Array<T> betaSum(const Array<Array<T>>& z) {
	size_t n = z.size();
	Array<T> beta(S1 << n);
	beta[0] = getOne<T>();
	betaSum_(beta, z, 0, n, false);
	return beta;
}

}

template <typename T>
Array<Array<T>> orderModularAPS_singlethread(const Array<Array<T>>& w, bool test) {
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

#undef APS_FOR_EACH_NUMBER_TYPE_TEMPLATE
#define APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(T) \
	template Array<Array<T>> orderModularAPS_singlethread<T>(const Array<Array<T>>&, bool);
APS_FOR_EACH_NUMBER_TYPE

}
