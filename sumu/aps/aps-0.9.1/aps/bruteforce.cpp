#include "array.h"
#include "bits.h"
#include "types.h"

#include <vector>

namespace aps {

namespace {

template <typename F>
void dags_(std::vector<size_t>& graph, size_t i, size_t j, F& f) {
	if(j == i) {
		dags_(graph, i, j + 1, f);
		return;
	}
	if(j == graph.size()) {
		dags_(graph, i + 1, 0, f);
		return;
	}
	if(i == graph.size()) {
		f((const std::vector<size_t>&)graph);
		return;
	}

	dags_(graph, i, j + 1, f);

	size_t iBit = S1 << i;
	size_t jBit = S1 << j;

	size_t queue = iBit;
	size_t seen = queue;
	size_t target = jBit;

	while(!(seen & target) && queue) {
		size_t v = bottomOneBitIdx(queue);
		queue ^= S1 << v;
		size_t add = graph[v] & ~seen;
		queue |= add;
		seen |= add;
	}

	if(!(seen & target)) {
		graph[j] ^= iBit;
		dags_(graph, i, j + 1, f);
		graph[j] ^= iBit;
	}
}

template <typename F>
void dags(size_t n, F f) {
	std::vector<size_t> graph(n);
	dags_(graph, 0, 0, f);
}

template <typename F>
void topoOrders_(const std::vector<size_t>& graph, std::vector<size_t>& order, size_t done, size_t todo, F& f) {
	if(!todo) {
		f((const std::vector<size_t>&)order);
		return;
	}
	for(size_t v = 0; v < graph.size(); ++v) {
		if((todo & (S1 << v)) && !(graph[v] & todo)) {
			size_t vBit = (S1 << v);
			order[v] = done;
			topoOrders_(graph, order, done | vBit, todo ^ vBit, f);
		}
	}
}

template <typename F>
void topoOrders(const std::vector<size_t>& graph, F f) {
	std::vector<size_t> order(graph.size());
	topoOrders_(graph, order, 0, (S1 << graph.size()) - S1, f);
}

void topoSort_(const std::vector<size_t>& graph, size_t& seen, size_t*& out, size_t v) {
	if(seen & (S1 << v)) {
		return;
	}
	seen |= S1 << v;

	size_t left = graph[v];
	while(left) {
		size_t x = bottomOneBitIdx(left);
		left ^= S1 << x;
		topoSort_(graph, seen, out, x);
	}
	
	*out++ = v;
}

void topoSort(const std::vector<size_t>& graph, size_t* buf) {
	size_t* asd = buf;
	size_t seen = 0;
	for(size_t v = 0; v < graph.size(); ++v) {
		topoSort_(graph, seen, buf, v);
	}

	if(buf - asd != graph.size()) throw 0;
}

}

template <typename T>
Array<Array<T>> orderModularAPS_bruteforce(const Array<Array<T>>& w, bool test) {
	size_t n = w.size();
	if(test && n > 5) {
		return Array<Array<T>>();
	}
	for(size_t v = 0; v < n; ++v) {
		if(w[v].size() != (S1 << (n - 1))) {
			fail("Internal error: weight array with invalid size");
		}
	}

	Array<Array<T>> ret(n);
	for(size_t v = 0; v < n; ++v) {
		ret[v] = Array<T>(S1 << (n - 1));
		ret[v].fill(getZero<T>());
	}
	dags(n, [&](const std::vector<size_t>& dag) {
		T val = getZero<T>();
		const T linWeight = getOne<T>();
		topoOrders(dag, [&](const std::vector<size_t>& lin) {
			val += linWeight;
		});
		for(size_t v = 0; v < n; ++v) {
			val *= w[v][collapseBit(dag[v], v)];
		}
		for(size_t v = 0; v < n; ++v) {
			ret[v][collapseBit(dag[v], v)] += val;
		}
	});

	return ret;
}

template <typename T>
Array<Array<T>> modularAPS_bruteforce(const Array<Array<T>>& w, bool test) {
	size_t n = w.size();
	if(test && n > 5) {
		return Array<Array<T>>();
	}
	for(size_t v = 0; v < n; ++v) {
		if(w[v].size() != (S1 << (n - 1))) {
			fail("Internal error: weight array with invalid size");
		}
	}

	Array<Array<T>> ret(n);
	for(size_t v = 0; v < n; ++v) {
		ret[v] = Array<T>(S1 << (n - 1));
		ret[v].fill(getZero<T>());
	}
	dags(n, [&](const std::vector<size_t>& dag) {
		T val = getOne<T>();
		for(size_t v = 0; v < n; ++v) {
			val *= w[v][collapseBit(dag[v], v)];
		}

		for(size_t v = 0; v < n; ++v) {
			ret[v][collapseBit(dag[v], v)] += val;
		}
	});

	return ret;
}

template <typename T>
Array<Array<T>> orderModularAR_bruteforce(const Array<Array<T>>& w, bool test) {
	size_t n = w.size();
	if(test && n > 5) {
		return Array<Array<T>>();
	}
	for(size_t v = 0; v < n; ++v) {
		if(w[v].size() != (S1 << (n - 1))) {
			fail("Internal error: weight array with invalid size");
		}
	}

	Array<Array<T>> ret(n);
	for(size_t v = 0; v < n; ++v) {
		ret[v] = Array<T>(n);
		ret[v].fill(getZero<T>());
	}

	std::vector<size_t> topo(n);
	std::vector<size_t> ancestors(n);

	dags(n, [&](const std::vector<size_t>& dag) {
		T val = getZero<T>();
		const T linWeight = getOne<T>();
		topoOrders(dag, [&](const std::vector<size_t>& lin) {
			val += linWeight;
		});
		for(size_t v = 0; v < n; ++v) {
			val *= w[v][collapseBit(dag[v], v)];
		}

		topoSort(dag, topo.data());

		for(size_t v : topo) {
			ancestors[v] = S1 << v;
			while(dag[v] & ~ancestors[v]) {
				size_t x = bottomOneBitIdx(dag[v] & ~ancestors[v]);
				ancestors[v] |= ancestors[x];
			}
			size_t left = ancestors[v];
			while(left) {
				size_t x = bottomOneBitIdx(left);
				left ^= S1 << x;
				ret[v][x] += val;
			}
		}
	});

	return ret;
}

template <typename T>
Array<Array<T>> modularAR_bruteforce(const Array<Array<T>>& w, bool test) {
	size_t n = w.size();
	if(test && n > 5) {
		return Array<Array<T>>();
	}
	for(size_t v = 0; v < n; ++v) {
		if(w[v].size() != (S1 << (n - 1))) {
			fail("Internal error: weight array with invalid size");
		}
	}

	Array<Array<T>> ret(n);
	for(size_t v = 0; v < n; ++v) {
		ret[v] = Array<T>(n);
		ret[v].fill(getZero<T>());
	}

	std::vector<size_t> topo(n);
	std::vector<size_t> ancestors(n);

	dags(n, [&](const std::vector<size_t>& dag) {
		T val = getOne<T>();
		for(size_t v = 0; v < n; ++v) {
			val *= w[v][collapseBit(dag[v], v)];
		}

		topoSort(dag, topo.data());

		for(size_t v : topo) {
			ancestors[v] = S1 << v;
			while(dag[v] & ~ancestors[v]) {
				size_t x = bottomOneBitIdx(dag[v] & ~ancestors[v]);
				ancestors[v] |= ancestors[x];
			}
			size_t left = ancestors[v];
			while(left) {
				size_t x = bottomOneBitIdx(left);
				left ^= S1 << x;
				ret[v][x] += val;
			}
		}
	});

	return ret;
}

#undef APS_FOR_EACH_NUMBER_TYPE_TEMPLATE
#define APS_FOR_EACH_NUMBER_TYPE_TEMPLATE(T) \
	template Array<Array<T>> orderModularAPS_bruteforce<T>(const Array<Array<T>>&, bool); \
	template Array<Array<T>> modularAPS_bruteforce(const Array<Array<T>>&, bool); \
	template Array<Array<T>> orderModularAR_bruteforce<T>(const Array<Array<T>>&, bool); \
	template Array<Array<T>> modularAR_bruteforce(const Array<Array<T>>&, bool);
APS_FOR_EACH_NUMBER_TYPE

}
