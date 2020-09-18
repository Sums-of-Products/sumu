import os
import numpy as np
import sumppy


def test_Gadget_empirical_edge_prob_error_decreases():

    test_path = os.path.dirname(os.path.realpath(__file__))
    bn_path = test_path + "/sachs.dsc"
    data_path = test_path + "/test_data.csv"

    # NOTE: Computed with aps
    edge_probs = np.array([[0.00000000e+00, 3.48408525e-01, 4.02665559e-06,
                            4.60015481e-02, 6.82990031e-05, 5.49232176e-03,
                            4.07196802e-03, 8.73558402e-02, 7.05453454e-06,
                            4.44499709e-04, 5.36262417e-07],
                           [6.51577948e-01, 0.00000000e+00, 1.73957099e-08,
                            2.00043963e-01, 5.29844299e-04, 3.12971895e-03,
                            5.13872894e-04, 1.54594968e-01, 6.21630707e-09,
                            8.96693344e-04, 1.92201025e-07],
                           [3.28116716e-04, 1.75230515e-06, 0.00000000e+00,
                            7.51989643e-06, 1.94580760e-05, 1.43442054e-02,
                            1.19410750e-03, 1.20415450e-01, 1.21985802e-01,
                            1.69322228e-04, 1.24952913e-06],
                           [2.22584154e-01, 5.31890737e-01, 1.87477116e-05,
                            0.00000000e+00, 4.83121134e-04, 2.56591121e-03,
                            3.38379644e-02, 3.84930295e-01, 4.81851663e-01,
                            1.59243940e-04, 8.75376921e-01],
                           [9.31837184e-04, 6.25514727e-05, 3.43063264e-06,
                            8.50151263e-05, 0.00000000e+00, 1.99152478e-02,
                            2.61986311e-02, 1.24873428e-01, 1.18018776e-01,
                            1.05339905e-02, 8.74359093e-08],
                           [7.59047208e-04, 3.23444309e-04, 1.20029023e-03,
                            4.32083765e-04, 2.72874651e-03, 0.00000000e+00,
                            5.51732497e-03, 9.93208735e-04, 7.79544804e-04,
                            9.58342751e-01, 1.74489856e-04],
                           [7.35548104e-04, 4.76431040e-05, 9.98733446e-05,
                            5.89778656e-03, 2.23030688e-03, 5.70881094e-03,
                            0.00000000e+00, 1.66582833e-03, 1.22862497e-04,
                            9.57681012e-01, 5.43897841e-05],
                           [6.57086264e-01, 8.45394899e-01, 8.79584592e-01,
                            6.15082051e-01, 8.75128553e-01, 3.80966861e-03,
                            7.29102654e-03, 0.00000000e+00, 5.39722765e-01,
                            1.34303277e-04, 6.11108784e-03],
                           [4.29352705e-05, 1.46806502e-08, 8.77860938e-01,
                            5.17148603e-01, 8.38227522e-01, 3.71116324e-03,
                            5.89385202e-04, 4.60279669e-01, 0.00000000e+00,
                            1.99850079e-04, 8.67814603e-01],
                           [1.72094283e-03, 3.25447334e-03, 2.90517978e-04,
                            1.00359281e-03, 4.37976683e-04, 4.16550287e-02,
                            4.10644502e-02, 1.00567079e-03, 1.85857544e-03,
                            0.00000000e+00, 9.58963934e-05],
                           [3.22714396e-06, 4.42969756e-08, 4.40022052e-06,
                            1.18218102e-01, 2.11607765e-05, 2.06536522e-03,
                            6.49362145e-04, 8.91742946e-04, 1.32032527e-01,
                            2.81291471e-05, 0.00000000e+00]])

    bn = sumppy.utils.io.read_dsc(bn_path)

    # NOTE: If the bn.sample() implementation is changed, the correct
    # edge probs will have to be recalculated as the data from which
    # they were calculated will not be identical to the data given to
    # Gadget anymore.
    data = bn.sample(200, seed=0)
    sumppy.utils.io.write_data(data, data_path, bn)

    params = {"datapath": data_path,
              "scoref": "bdeu",
              "ess": 10,
              "max_id": -1,
              "K": 8,
              "d": 3,
              "cp_algo": "greedy-lite",
              "mc3_chains": 10,
              "burn_in": 20000,
              "iterations": 20000,
              "thinning": 10,
              "tolerance": 2**(-32)}

    # To set the seed for MCMC
    # np.random.seed(1)
    g = sumppy.Gadget(**params)
    dags, scores = g.sample()

    max_errors = sumppy.utils.utils.edge_empirical_prob_max_error(dags,
                                                                  edge_probs)

    # NOTE: This is more a sanity check rather than performance
    # test. The idea is to run the test frequently, and detect if
    # things go awfully wrong. The error threshold can be set much
    # lower if we can afford more MCMC iterations, e.g., with improved
    # speed performance of the sampler.
    #
    # The resulting error depends somewhat heavily on the random seed
    # for the MCMC run. At the moment only the last error rate is
    # used, but the whole history can be used for debugging in case
    # things seem to break.
    assert max_errors[-1] < 0.5

    os.remove(data_path)


def test_Gadget_runs():

    test_path = os.path.dirname(os.path.realpath(__file__))
    bn_path = test_path + "/insurance.dsc"
    data_path = test_path + "/test_data.csv"

    bn = sumppy.utils.io.read_dsc(bn_path)

    data = bn.sample(100)
    sumppy.utils.io.write_data(data, data_path, bn)

    params = {"datapath": data_path,
              "scoref": "bdeu",
              "ess": 10,
              "max_id": -1,
              "K": 5,
              "d": 2,
              "cp_algo": "greedy-lite",
              "mc3_chains": 8,
              "burn_in": 1000,
              "iterations": 1000,
              "thinning": 100,
              "tolerance": 2**(-32)}

    g = sumppy.Gadget(**params)
    dags, scores = g.sample()

    os.remove(data_path)


def test_gadget_weight_sum_leq_64():

    W_prime = -13.47607732972311
    ordered_psets = np.array([40, 72, 160, 192, 9, 36, 68, 48, 80, 5, 8, 129,
                              17, 12, 136, 24, 4, 128, 16, 132, 20, 144],
                             dtype=np.uint64)
    ordered_scores = np.array([-11.2986829, -11.2986829, -15.72455243,
                               -15.72455243, -16.01751594, -16.64299758,
                               -16.64299758, -16.72539147, -16.72539147,
                               -16.86506699, -17.02176516, -17.23448564,
                               -17.28241371, -18.32064374, -18.5060301,
                               -18.77951521, -19.05906256, -19.13730775,
                               -19.13816039, -20.58275506, -20.59030283,
                               -20.77911241])
    n = 8
    U_bm = 81
    T_bm = 80
    t_ub = 9

    result = sumppy.weight_sum.weight_sum(W_prime, ordered_psets, ordered_scores,
                                          n, U_bm, T_bm, t_ub)

    # NOTE: "correct answer" is the result returned by the version of
    # the code used in the NeurIPS publication, and its correctness is
    # only assumed since the overall results in the pipeline that the
    # weight sum function was part of seemed reasonable.
    correct_answer = -13.416836930533735

    assert abs(result - correct_answer) < 1e-8
