from sumppy import MCMC_moves


def test_R_basic_move():

    # TODO: test nodes to rescore

    # Test that the (inverse) proposal probabilities are correct.
    # To avoid relying on random seeds this proposes a move
    # 100 times to have high probability that all possible
    # moves are considered.
    R = [{0}, {1, 2}]

    passed = list()
    for i in range(100):
        R_prime, q, q_prime, rescore = MCMC_moves.R_basic_move(R=R)
        if len(R_prime) == 1 and q == 1/3 and q_prime == 1/6:
            passed.append(True)
        elif len(R_prime) == 3 and q == 1/3 and q_prime == 1/2:
            passed.append(True)
        else:
            passed.append(False)

    assert all(passed)
