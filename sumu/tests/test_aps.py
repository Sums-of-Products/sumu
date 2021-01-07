import numpy as np
import sumu


def test_aps_runs():

    w = np.array([
    [-289.523, -261.6225, -166.2411, -167.5349, -193.1454, 19.45066, -71.52742, 29.97363],
    [-289.523, -261.6225, -231.6976, -232.9914, -292.8235, -80.22735, -179.7392, -78.23816],
    [-289.523, -166.2411, -231.6976, -137.6101, -262.9354, -141.3174, -149.8511, -139.3282],
    [-289.523, -193.1454, -292.8235, -11.75032, -262.9354, -168.2217, -210.977, -13.46841]
    ])

    probs = sumu.aps(w, as_dict=True, normalize=True)
    # Dummy test to see if line coverage analysis in CI works for Cython
    assert True


if __name__ == '__main__':
    test_aps_runs()
