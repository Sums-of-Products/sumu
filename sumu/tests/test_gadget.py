import subprocess
import time

import numpy as np
import psutil
import pytest

import sumu

np.random.seed(0)


def test_Gadget_empirical_edge_prob_error_decreases(discrete_bn):

    params = {
        # generic MCMC parameters
        "mcmc": {
            "n_indep": 1,
            "n_target_chain_iters": 50000,
            "burn_in": 0.5,
            "n_dags": 10000,
            "move_weights": [1, 1, 16],
        },
        # Metropolis coupling
        "metropolis_coupling_scheme": {
            "name": "adaptive",
            "params": {
                "M": 2,
                "p_target": 0.234,
                "delta_t_init": 0.5,
                "local_accept_history_size": 1000,
                "update_freq": 100,
            },
        },
        # score to use and its parameters
        "score": {"name": "bdeu", "params": {"ess": 10}},
        # modular structure prior and its parameters
        "structure_prior": {"name": "fair"},
        # constraints on the DAG space
        "constraints": {"max_id": -1, "K": 8, "d": 3, "pruning_eps": 0.001},
        # algorithm to use for finding candidate parents
        "candidate_parent_algorithm": {"name": "greedy", "params": {"k": 6}},
        # preparing for catastrofic cancellations
        "catastrophic_cancellation": {
            "tolerance": 2 ** -32,
            "cache_size": 10 ** 7,
        },
        # Logging
        "logging": {
            "stats_period": 15,
        },
    }

    data = discrete_bn["sachs"].sample(100)
    ls = sumu.gadget.LocalScore(data=data, maxid=-1, score=params["score"])
    pset_probs = sumu.aps(ls.candidate_scores(), as_dict=True)
    edge_probs = sumu.utils.edge_probs_from_pset_probs(pset_probs)

    g = sumu.Gadget(data=data, **params)
    dags, meta = g.sample()
    max_errors = sumu.utils.utils.edge_empirical_prob_max_error(
        dags, edge_probs
    )
    print(max_errors[-1])

    # TODO: Some better test for the validity of returned DAG scores
    assert -float("inf") not in meta["scores"]

    assert max_errors[-1] < 0.10


def test_Gadget_runs_n_between_2_and_64(discrete_bn):
    # NOTE: This does not test all numbers of variables up to 64
    g = sumu.Gadget(
        data=discrete_bn["sachs"].sample(200),
        mcmc={
            "n_target_chain_iters": 200,
            "burn_in": 0.5,
            "n_dags": 50,
        },
        metropolis_coupling_scheme={"name": "linear", "params": {"M": 2}},
        candidate_parent_algorithm={"name": "rnd"},
        constraints={"K": 10, "d": 2},
    )
    g.sample()
    assert True


def test_Gadget_runs_n_between_65_and_128(discrete_bn):
    # NOTE: This does not test all numbers of variables between 65 and 128
    g = sumu.Gadget(
        data=discrete_bn["hepar2"].sample(1000),
        mcmc={
            "n_target_chain_iters": 200,
            "burn_in": 0.5,
            "n_dags": 50,
        },
        metropolis_coupling_scheme={"name": "linear", "params": {"M": 2}},
        candidate_parent_algorithm={"name": "rnd"},
        constraints={"K": 10, "d": 2},
    )
    g.sample()
    assert True


def test_Gadget_runs_n_between_129_and_192(discrete_bn):
    # NOTE: This does not test all numbers of variables between 129 and 192
    g = sumu.Gadget(
        data=discrete_bn["munin1"].sample(200),
        mcmc={
            "n_target_chain_iters": 200,
            "burn_in": 0.5,
            "n_dags": 50,
        },
        metropolis_coupling_scheme={"name": "linear", "params": {"M": 2}},
        candidate_parent_algorithm={"name": "rnd"},
        constraints={"K": 10, "d": 2},
    )
    g.sample()
    assert True


def test_Gadget_runs_n_between_193_and_256(discrete_bn):
    # NOTE: This does not test all numbers of variables between 193 and 256
    g = sumu.Gadget(
        data=discrete_bn["andes"].sample(200),
        mcmc={
            "n_target_chain_iters": 200,
            "burn_in": 0.5,
            "n_dags": 50,
        },
        metropolis_coupling_scheme={"name": "linear", "params": {"M": 2}},
        candidate_parent_algorithm={"name": "rnd"},
        constraints={"K": 10, "d": 2},
    )
    g.sample()
    assert True


def test_Gadget_runs_continuous_data():
    data = np.random.rand(200, 10)
    sumu.Gadget(
        data=data, mcmc={"n_target_chain_iters": 200}, constraints={"K": 8}
    ).sample()
    assert True


def test_Gadget_runs_n_greater_than_256_continuous():
    data = np.random.rand(600, 300)
    sumu.Gadget(
        data=data,
        mcmc={
            "n_target_chain_iters": 200,
            "burn_in": 0.5,
            "n_dags": 50,
        },
        metropolis_coupling_scheme={"name": "linear", "params": {"M": 2}},
        candidate_parent_algorithm={"name": "rnd"},
        constraints={"K": 8, "d": 1},
    ).sample()
    assert True


def test_Gadget_runs_n_greater_than_256_discrete(discrete_bn):
    sumu.Gadget(
        data=discrete_bn["pigs"].sample(1000),
        mcmc={
            "n_target_chain_iters": 200,
            "burn_in": 0.5,
            "n_dags": 50,
        },
        metropolis_coupling_scheme={"name": "linear", "params": {"M": 2}},
        candidate_parent_algorithm={"name": "rnd"},
        constraints={"K": 8, "d": 1},
    ).sample()
    assert True


def test_Gadget_runs_empty_data_continuous():
    data = np.array([], dtype=np.float64).reshape(0, 14)
    sumu.Gadget(
        data=data,
        mcmc={
            "n_target_chain_iters": 200,
            "burn_in": 0.5,
            "n_dags": 50,
        },
        metropolis_coupling_scheme={"name": "linear", "params": {"M": 2}},
        candidate_parent_algorithm={"name": "rnd"},
        constraints={"K": 8, "d": 1},
    ).sample()
    assert True


def test_Gadget_runs_empty_data_discrete():
    data = np.array([], dtype=np.int32).reshape(0, 14)
    sumu.Gadget(
        data=data,
        mcmc={
            "n_target_chain_iters": 200,
            "burn_in": 0.5,
            "n_dags": 50,
        },
        metropolis_coupling_scheme={"name": "linear", "params": {"M": 2}},
        candidate_parent_algorithm={"name": "rnd"},
        constraints={"K": 8, "d": 1},
    ).sample()
    assert True


def test_Gadget_runs_with_anytime_mode(discrete_bn):

    import os
    import signal

    # skip this test on Windows
    if os.name == "nt":
        assert True
        return

    def gadget_anytime():
        g = sumu.Gadget(
            data=discrete_bn["sachs"].sample(200),
            run_mode={"name": "anytime"},
            mcmc={"n_dags": 50},
            metropolis_coupling_scheme={"name": "linear", "params": {"M": 2}},
            candidate_parent_algorithm={"name": "rnd"},
            constraints={"K": 6, "d": 2},
        )
        return g.sample()

    def handler(signum, frame):
        print("Simulating user sent CTRL-C", signum)
        signal.alarm(5)
        raise KeyboardInterrupt("CTRL-C")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(5)
    dags, meta = gadget_anytime()
    signal.alarm(0)
    assert True


def test_Gadget_stays_in_budget(discrete_bn):
    budget = 30
    t = time.time()
    params = {
        "run_mode": {"name": "budget", "params": {"t": budget}},
        "candidate_parent_algorithm": {"name": "greedy"},
    }

    sumu.Gadget(data=discrete_bn["sachs"].sample(100), **params).sample()
    t = time.time() - t
    print(t)
    assert abs(t - budget) < 1


def test_Gadget_stays_in_mem_budget():

    mem_budget = 1000

    cmd = """import numpy as np; import sumu
data = np.random.randint(4, size=(200, 40), dtype=np.int32)
sumu.Gadget(
data=data,
run_mode={"name": "budget", "params": {"mem": 1000}},
constraints={"K": 20, "d": 2},
mcmc={"n_target_chain_iters": 100},
metropolis_coupling_scheme={"name": "linear", "params": {"M": 1}},
candidate_parent_algorithm={"name": "rnd"},
).sample()"""

    # TODO: should be subprocess.run(..., check=True)
    #       but then process.poll() fails.
    process = subprocess.Popen(["python", "-c", cmd])
    maxmem = 0
    while process.poll() is None:
        p = psutil.Process(process.pid)
        memuse = p.memory_info().rss / 1024 ** 2
        if memuse > maxmem:
            maxmem = memuse
        time.sleep(1)
    assert maxmem < mem_budget


def test_adaptive_tempering(discrete_bn):
    n = 100
    t = 60
    p_target = 0.234
    slack = 0.06
    g = sumu.Gadget(
        data=discrete_bn["sachs"].sample(n),
        run_mode={"name": "budget", "params": {"t": t}},
        # mcmc={"iters": 30000},
        metropolis_coupling_scheme={
            "name": "adaptive",
            "params": {
                "M": 2,
                "p_target": p_target,
                "delta_t_init": 0.5,
                "local_accept_history_size": 1000,
                "update_freq": 100,
                "smoothing": 2.0,
            },
        },
    )
    dags, meta = g.sample()

    acc_probs = meta["mcmc"]["accept_prob"]["mc3"][:-1]
    assert all(acc_probs > p_target - slack)


def test_Gadget_runs_without_Metropolis():
    data = np.random.rand(200, 10)
    sumu.Gadget(
        data=data,
        constraints={"K": 8},
        mcmc={"n_target_chain_iters": 100000},
        metropolis_coupling_scheme={"params": {"M": 1}},
        logging={"verbose_prefix": "M1/M1", "overwrite": True},
    ).sample()
    assert True


def test_Gadget_runs_with_preset_candidate_parents(discrete_bn):
    data = discrete_bn["sachs"].sample(100)
    C = sumu.candidates.candidate_parent_algorithm["rnd"](5, data=data)[0]
    sumu.Gadget(data=data, candidate_parents=C)


def test_Gadget_reads_candidate_parents_from_file(discrete_bn, tmp_path):
    K = 5
    data = discrete_bn["sachs"].sample(100)
    C = sumu.candidates.candidate_parent_algorithm["rnd"](K, data=data)[0]
    C_array = np.empty((data.n, K), dtype=np.int32)
    for v in C:
        C_array[v] = C[v]
    log = sumu.gadget.Logger(logfile=tmp_path / "C")
    log.numpy(C_array)
    sumu.Gadget(data=data, candidate_parents_path=str(tmp_path / "C"))


def test_Gadget_runs_initial_rootpartition(discrete_bn):
    g = sumu.Gadget(
        initial_rootpartition=sumu.bnet.partition(discrete_bn["sachs"].dag),
        data=discrete_bn["sachs"].sample(200),
        mcmc={
            "n_target_chain_iters": 200,
            "burn_in": 0.5,
            "n_dags": 50,
        },
        metropolis_coupling_scheme={"name": "linear", "params": {"M": 2}},
        candidate_parent_algorithm={"name": "rnd"},
        constraints={"K": 10, "d": 2},
    )
    g.sample()
    assert True


@pytest.mark.select
def test_Gadget_utility_for_rootpartition_score(discrete_bn):
    g = sumu.Gadget(
        data=discrete_bn["sachs"].sample(200),
        candidate_parent_algorithm={"name": "rnd"},
        constraints={"K": 1, "d": 1},
    )
    g.precompute()
    g.score.score_rootpartition(sumu.bnet.partition(discrete_bn["sachs"].dag))
    assert True
