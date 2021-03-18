#include <cstdint>
#include <vector>

using namespace std;
using bm64 = uint64_t;

double weight_sum_64(double w, bm64 *psets, int m, double *weights, int j, int n, bm64 U, bm64 T, int t_ub);
pair<double, vector<int> > weight_sum_contribs_64(double w, bm64 *psets, int m, double *weights, int j, int n, bm64 U, bm64 T, int t_ub);


double weight_sum_128(double w, bm64 *psets, int m, double *weights, int j, int n, vector<bm64> U_vec, vector<bm64> T_vec, int t_ub);
pair<double, vector<int> > weight_sum_contribs_128(double w, bm64 *psets, int m, double *weights, int j, int n, vector<bm64> U_vec, vector<bm64> T_vec, int t_ub);


double weight_sum_192(double w, bm64 *psets, int m, double *weights, int j, int n, vector<bm64> U_vec, vector<bm64> T_vec, int t_ub);
pair<double, vector<int> > weight_sum_contribs_192(double w, bm64 *psets, int m, double *weights, int j, int n, vector<bm64> U_vec, vector<bm64> T_vec, int t_ub);


double weight_sum_256(double w, bm64 *psets, int m, double *weights, int j, int n, vector<bm64> U_vec, vector<bm64> T_vec, int t_ub);
pair<double, vector<int> > weight_sum_contribs_256(double w, bm64 *psets, int m, double *weights, int j, int n, vector<bm64> U_vec, vector<bm64> T_vec, int t_ub);
