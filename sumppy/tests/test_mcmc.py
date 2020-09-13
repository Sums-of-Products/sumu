import os
import numpy as np
import sumppy


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
