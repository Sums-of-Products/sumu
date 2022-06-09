import numpy as np

import sumu


def test_data_info_is_correct():
    info = sumu.Data(np.random.rand(10, 3)).info
    assert info["no. variables"] == 3
    assert info["no. data points"] == 10
    assert info["type of data"] == "continuous"

    info = sumu.Data(np.random.randint(3, size=(1000, 5))).info
    assert info["no. variables"] == 5
    assert info["no. data points"] == 1000
    assert info["type of data"] == "discrete"
    assert info["arities [min, max]"] == "[3, 3]"


if __name__ == "__main__":
    test_data_info_is_correct()
