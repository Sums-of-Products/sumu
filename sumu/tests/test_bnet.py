import numpy as np

import sumu


def test_discrete_bnet_node_samples_from_correct_distribution(discrete_bn):

    # TODO: Multivariate test

    np.random.seed(0)

    bn = discrete_bn["sachs"]
    data = bn.sample(5000)

    i_u = bn.nodes.index(bn["Mek"])
    i_pset = [bn.nodes.index(p) for p in bn["Mek"].parents]
    pset_config = (0, 1, 2)

    emp_probs = np.array(
        [
            np.all(
                data[:, [i_u] + i_pset] == (uval,) + pset_config, axis=1
            ).sum()
            for uval in range(3)
        ]
    ).astype(np.float32)
    emp_probs /= emp_probs.sum()

    kl = 0
    for q, p in zip(bn["Mek"].cpt[(0, 1, 2)], emp_probs):
        kl += p * np.log(p / q)

    assert kl < 0.05


def test_discrete_bnet_from_dag_produces_bnet():
    dag = sumu.validate.dag([(0, set()), (1, {0, 2}), (2, set())])
    bn = sumu.DiscreteBNet.from_dag(dag, arity=3, ess=10, params="random")
    sumu.DiscreteBNet.from_dag(
        dag, data=bn.sample(20), arity=3, ess=10, params="random"
    )
    assert True


def test_gaussian_bnet_from_dag_produces_bnet():
    dag = sumu.validate.dag([(0, set()), (1, {0, 2}), (2, set())])
    bn = sumu.GaussianBNet(dag=dag)
    sumu.GaussianBNet(dag=dag, data=bn.sample(20))
    assert True


def test_conversion_adj_matrix_family_sequence():
    dag = [(0, set()), (1, {0, 2}), (2, set())]
    assert sumu.validate.dag(dag) == sumu.validate.dag(
        sumu.bnet.adj_mat_to_family_sequence(
            sumu.bnet.family_sequence_to_adj_mat(dag)
        )
    )


def test_random_dag_with_expected_neighbourhood_size():
    n_dags = 1000
    n = 10
    enb = 4  # not counting the node itself

    tolerance = 1e-1

    def compute_error():
        dags = np.array(
            [
                sumu.bnet.family_sequence_to_adj_mat(
                    sumu.bnet.random_dag_with_expected_neighbourhood_size(
                        n, enb=enb
                    )
                )
                for i in range(n_dags)
            ]
        )
        mean_nbsize = (
            dags[:, 0, :].sum(axis=1).mean() + dags[:, :, 0].sum(axis=1).mean()
        ).mean()
        return abs(enb - mean_nbsize)

    error = compute_error()
    if error > tolerance:
        error = compute_error()

    assert error < tolerance
