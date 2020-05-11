#include <vector>
#include <cmath>

using namespace std;

vector<double> from_list(vector<double> arg) {

  int N = log2(arg.size());
  vector<double> a(arg.size());

  for(int i = 0; i<(1<<N); ++i)
	a[i] = arg[i];


  for(int i = 0;i < N; ++i) {
    for(int mask = 0; mask < (1<<N); ++mask) {
	if(mask & (1<<i))
	  //a[mask] += a[mask^(1<<i)];
	  if (!(isinf(a[mask]) && isinf(a[mask^(1<<i)]))) {
	    a[mask] = fmax(a[mask], a[mask^(1<<i)]) + log1p(exp(fmin(a[mask], a[mask^(1<<i)]) - fmax(a[mask], a[mask^(1<<i)])));
	  }
    }
  }

  return a;

}

double testi(vector<double> arg) {

  //return arg[0] + arg[1];
}
