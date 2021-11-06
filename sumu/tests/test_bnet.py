import pathlib
import numpy as np
import sumu


def test_bnet_samples_from_correct_distribution():

    np.random.seed(0)

    data_path = pathlib.Path(__file__).resolve().parents[2] / "data"
    bn_path = data_path / "sachs.dsc"
    bn = sumu.BNet.read_file(bn_path)
    data = bn.sample(5000)

    i_u = bn.nodes.index(bn["Mek"])
    i_pset = [bn.nodes.index(p) for p in bn["Mek"].parents]
    pset_config = (0, 1, 2)

    emp_probs = np.array(
        [
            np.all(data[:, [i_u] + i_pset] == (uval,) + pset_config, axis=1).sum()
            for uval in range(3)
        ]
    ).astype(np.float32)
    emp_probs /= emp_probs.sum()

    kl = 0
    for q, p in zip(bn["Mek"].cpt[(0, 1, 2)], emp_probs):
        kl += p * np.log(p / q)

    assert kl < 0.05


def test_bnet_from_dag_produces_bnet():
    dag = sumu.validate.dag([(0, set()), (1, {0, 2}), (2, set())])
    bn = sumu.BNet.from_dag(dag, arity=3, ess=10, params="random")
    bn.sample(20)
    assert True

