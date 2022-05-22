import pathlib
import subprocess
import time

import numpy as np
import psutil

import sumu

np.random.seed(0)


def test_Gadget_empirical_edge_prob_error_decreases():

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
        # "mc3": {"name": "linear", "params": {"M": 6}},
        "mc3": {
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
        "prior": {"name": "fair"},
        # constraints on the DAG space
        "cons": {"max_id": -1, "K": 8, "d": 3, "pruning_eps": 0.001},
        # algorithm to use for finding candidate parents
        "candp": {"name": "greedy", "params": {"k": 6}},
        # preparing for catastrofic cancellations
        "catc": {"tolerance": 2 ** -32, "cache_size": 10 ** 7},
        # Logging
        "logging": {
            "stats_period": 15,
        },
    }

    data_path = pathlib.Path(__file__).resolve().parents[2] / "data"
    bn_path = data_path / "sachs.dsc"
    bn = sumu.DiscreteBNet.read_file(bn_path)
    data = bn.sample(100)
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


def test_Gadget_runs_n_between_2_and_64():
    # NOTE: This does not test all numbers of variables up to 64
    data_path = pathlib.Path(__file__).resolve().parents[2] / "data"
    bn_path = data_path / "sachs.dsc"
    bn = sumu.DiscreteBNet.read_file(bn_path)
    data = bn.sample(200)
    g = sumu.Gadget(
        data=data,
        mcmc={
            "n_target_chain_iters": 200,
            "burn_in": 0.5,
            "n_dags": 50,
        },
        mc3={"name": "linear", "M": 2},
        candp={"name": "rnd"},
        cons={"K": 10, "d": 2},
    )
    g.sample()
    assert True


def test_Gadget_runs_n_between_65_and_128():
    # NOTE: This does not test all numbers of variables between 65 and 128
    data_path = pathlib.Path(__file__).resolve().parents[2] / "data"
    bn_path = data_path / "hepar2.dsc"
    bn = sumu.DiscreteBNet.read_file(bn_path)
    data = bn.sample(1000)
    g = sumu.Gadget(
        data=data,
        mcmc={
            "n_target_chain_iters": 200,
            "burn_in": 0.5,
            "n_dags": 50,
        },
        mc3={"name": "linear", "M": 2},
        candp={"name": "rnd"},
        cons={"K": 10, "d": 2},
    )
    g.sample()
    assert True


def test_Gadget_runs_n_between_129_and_192():
    # NOTE: This does not test all numbers of variables between 129 and 192
    data_path = pathlib.Path(__file__).resolve().parents[2] / "data"
    bn_path = data_path / "munin1.dsc"
    bn = sumu.DiscreteBNet.read_file(bn_path)
    data = bn.sample(200)
    g = sumu.Gadget(
        data=data,
        mcmc={
            "n_target_chain_iters": 200,
            "burn_in": 0.5,
            "n_dags": 50,
        },
        mc3={"name": "linear", "M": 2},
        candp={"name": "rnd"},
        cons={"K": 10, "d": 2},
    )
    g.sample()
    assert True


def test_Gadget_runs_n_between_193_and_256():
    # NOTE: This does not test all numbers of variables between 193 and 256
    data_path = pathlib.Path(__file__).resolve().parents[2] / "data"
    bn_path = data_path / "andes.dsc"
    bn = sumu.DiscreteBNet.read_file(bn_path)
    data = bn.sample(200)
    g = sumu.Gadget(
        data=data,
        mcmc={
            "n_target_chain_iters": 200,
            "burn_in": 0.5,
            "n_dags": 50,
        },
        mc3={"name": "linear", "M": 2},
        candp={"name": "rnd"},
        cons={"K": 10, "d": 2},
    )
    g.sample()
    assert True


def test_Gadget_runs_continuous_data():
    data = np.random.rand(200, 10)
    sumu.Gadget(data=data, mcmc={"iters": 200}, cons={"K": 8}).sample()
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
        mc3={"name": "linear", "M": 2},
        candp={"name": "rnd"},
        cons={"K": 8, "d": 1},
    ).sample()
    assert True


def test_Gadget_runs_n_greater_than_256_discrete():
    data_path = pathlib.Path(__file__).resolve().parents[2] / "data"
    bn_path = data_path / "pigs.dsc"
    bn = sumu.DiscreteBNet.read_file(bn_path)
    data = bn.sample(1000)
    sumu.Gadget(
        data=data,
        mcmc={
            "n_target_chain_iters": 200,
            "burn_in": 0.5,
            "n_dags": 50,
        },
        mc3={"name": "linear", "M": 2},
        candp={"name": "rnd"},
        cons={"K": 8, "d": 1},
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
        mc3={"name": "linear", "M": 2},
        candp={"name": "rnd"},
        cons={"K": 8, "d": 1},
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
        mc3={"name": "linear", "M": 2},
        candp={"name": "rnd"},
        cons={"K": 8, "d": 1},
    ).sample()
    assert True


def test_Gadget_runs_with_anytime_mode():

    import os
    import signal

    # skip this test on Windows
    if os.name == "nt":
        assert True
        return

    def gadget_anytime():
        data_path = pathlib.Path(__file__).resolve().parents[2] / "data"
        bn_path = data_path / "sachs.dsc"
        bn = sumu.DiscreteBNet.read_file(bn_path)
        data = bn.sample(200)
        g = sumu.Gadget(
            data=data,
            run_mode={"name": "anytime"},
            mcmc={"n_dags": 50},
            mc3={"name": "linear", "M": 2},
            candp={"name": "rnd"},
            cons={"K": 6, "d": 2},
        )
        return g.sample()

    def handler(signum, frame):
        print("Simulating user sent CTRL-C", signum)
        signal.alarm(5)
        raise KeyboardInterrupt("CTRL-C")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(5)
    dags, meta = gadget_anytime()
    assert True


def test_Gadget_stays_in_budget():
    budget = 30
    t = time.time()
    params = {
        "run_mode": {"name": "budget", "params": {"t": budget}},
        "candp": {"name": "greedy"},
    }

    data_path = pathlib.Path(__file__).resolve().parents[2] / "data"
    bn_path = data_path / "sachs.dsc"
    bn = sumu.DiscreteBNet.read_file(bn_path)
    data = bn.sample(100)
    sumu.Gadget(data=data, **params).sample()
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
cons={"K": 20, "d": 2},
mcmc={"n_target_chain_iters": 1},
mc3={"name": "linear", "M": 1},
candp={"name": "rnd"},
).sample()"""

    process = subprocess.Popen(["python", "-c", cmd])
    maxmem = 0
    while process.poll() is None:
        p = psutil.Process(process.pid)
        memuse = p.memory_info().rss / 1024 ** 2
        if memuse > maxmem:
            maxmem = memuse
        time.sleep(1)
    assert maxmem < mem_budget


def test_adaptive_tempering():

    bnet = "sachs"
    n = 100
    t = 60

    p_target = 0.234
    slack = 0.06
    data_path = pathlib.Path(__file__).resolve().parents[2] / "data"
    bn_path = data_path / f"{bnet}.dsc"
    bn = sumu.DiscreteBNet.read_file(bn_path)
    data = bn.sample(n)
    g = sumu.Gadget(
        data=data,
        run_mode={"name": "budget", "params": {"t": t}},
        # mcmc={"iters": 30000},
        mc3={
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
        cons={"K": 8},
        mcmc={"iters": 1000},
        mc3={"params": {"M": 1}},
    ).sample()
    assert True


if __name__ == "__main__":
    # test_Gadget_runs_n_between_2_and_64()
    # test_Gadget_runs_n_between_65_and_128()
    # test_Gadget_runs_n_between_129_and_192()
    # test_Gadget_runs_n_between_193_and_256()
    # test_Gadget_empirical_edge_prob_error_decreases()
    test_Gadget_runs_n_greater_than_256_discrete()
    # test_Gadget_runs_with_anytime_mode()
    # test_Gadget_stays_in_budget()
    # test_adaptive_tempering()
    # test_Gadget_runs_without_Metropolis()
