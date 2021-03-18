import os
import pathlib
import numpy as np
import sumu


def test_Gadget_empirical_edge_prob_error_decreases():

    params = {
              # score to use and its parameters
              "score": {"name": "bdeu", "ess": 10},

              # modular structure prior and its parameters
              "prior": {"name": "fair"},

              # constraints on the DAG space
              "max_id": -1,
              "K": 8,
              "d": 3,

              # algorithm to use for finding candidate parents
              "cp_algo": "greedy-lite",

              # generic MCMC parameters
              "mc3": 2,
              "burn_in": 50000,
              "iterations": 50000,
              "thinning": 5,

              # preparing for catastrofic cancellations
              "cc_tolerance": 2**-32,
              "cc_cache_size": 10**7,

              # pruning candidate parent sets
              "pruning_eps": 0.001
    }

    data_path = pathlib.Path(__file__).resolve().parents[2] / "data"
    bn_path = data_path / "sachs.dsc"
    bn = sumu.utils.io.read_dsc(bn_path)
    data = bn.sample(200, seed=0)
    ls = sumu.gadget.LocalScore(data=data, maxid=-1, score=params["score"])
    pset_probs = sumu.aps(ls.all_candidate_restricted_scores(), as_dict=True)
    edge_probs = sumu.utils.edge_probs_from_pset_probs(pset_probs)

    # To set the seed for MCMC
    g = sumu.Gadget(data=data, **params)
    sumu.utils.io.pretty_dict(g.params)
    dags, scores = g.sample()
    max_errors = sumu.utils.utils.edge_empirical_prob_max_error(dags,
                                                                edge_probs)
    print(max_errors[-1])
    assert max_errors[-1] < 0.05


def test_Gadget_runs_n_between_2_and_64():
    # NOTE: This does not test all numbers of variables up to 64
    data_path = pathlib.Path(__file__).resolve().parents[2] / "data"
    bn_path = data_path / "sachs.dsc"
    bn = sumu.utils.io.read_dsc(bn_path)
    data = bn.sample(200, seed=0)
    g = sumu.Gadget(data=data, cp_algo="top", K=10, d=2, mc3=2, burn_in=100, iterations=100, thinning=2)
    sumu.utils.io.pretty_dict(g.params)
    g.sample()
    assert True


def test_Gadget_runs_n_between_65_and_128():
    # NOTE: This does not test all numbers of variables between 65 and 128
    data_path = pathlib.Path(__file__).resolve().parents[2] / "data"
    bn_path = data_path / "hepar2.dsc"
    bn = sumu.utils.io.read_dsc(bn_path)
    data = bn.sample(1000, seed=0)
    g = sumu.Gadget(data=data, cp_algo="top", K=10, d=2, mc3=2, burn_in=100, iterations=100, thinning=2)
    sumu.utils.io.pretty_dict(g.params)
    g.sample()
    assert True


def test_Gadget_runs_n_between_129_and_192():
    # NOTE: This does not test all numbers of variables between 129 and 192
    data_path = pathlib.Path(__file__).resolve().parents[2] / "data"
    bn_path = data_path / "munin1.dsc"
    bn = sumu.utils.io.read_dsc(bn_path)
    data = bn.sample(200, seed=0)
    g = sumu.Gadget(data=data, cp_algo="top", K=10, d=2, mc3=2, burn_in=100, iterations=100, thinning=2)
    sumu.utils.io.pretty_dict(g.params)
    g.sample()
    assert True


def test_Gadget_runs_n_between_193_and_256():
    # NOTE: This does not test all numbers of variables between 193 and 256
    data_path = pathlib.Path(__file__).resolve().parents[2] / "data"
    bn_path = data_path / "andes.dsc"
    bn = sumu.utils.io.read_dsc(bn_path)
    data = bn.sample(200, seed=0)
    g = sumu.Gadget(data=data, cp_algo="top", K=10, d=2, mc3=2, burn_in=100, iterations=100, thinning=2)
    sumu.utils.io.pretty_dict(g.params)
    g.sample()
    assert True


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

    result = sumu.weight_sum.weight_sum(w=W_prime,
                                        psets=ordered_psets,
                                        weights=ordered_scores,
                                        n=n, U=U_bm, T=T_bm, t_ub=t_ub)

    # NOTE: "correct answer" is the result returned by the version of
    # the code used in the NeurIPS publication, and its correctness is
    # only assumed since the overall results in the pipeline that the
    # weight sum function was part of seemed reasonable.
    correct_answer = -13.416836930533735

    assert abs(result - correct_answer) < 1e-8


if __name__ == '__main__':
    test_Gadget_runs_n_between_2_and_64()
    #test_Gadget_runs_n_between_65_and_128()
    #test_Gadget_runs_n_between_129_and_192()
    #test_Gadget_runs_n_between_193_and_256()
    #test_Gadget_empirical_edge_prob_error_decreases()
