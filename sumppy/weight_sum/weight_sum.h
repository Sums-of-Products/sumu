#include <cstdint>
#include <vector>

using namespace std;
using bitmap = uint64_t;

double weight_sum_64(double w, bitmap *psets, int m, double *weights, int j, int n, bitmap U, bitmap T, int t_ub);
pair<double, vector<int> > weight_sum_contribs_64(double w, bitmap *psets, int m, double *weights, int j, int n, bitmap U, bitmap T, int t_ub);


double weight_sum_128(double w, bitmap *psets, int m, double *weights, int j, int n, vector<bitmap> U_vec, vector<bitmap> T_vec, int t_ub);
pair<double, vector<int> > weight_sum_contribs_128(double w, bitmap *psets, int m, double *weights, int j, int n, vector<bitmap> U_vec, vector<bitmap> T_vec, int t_ub);
