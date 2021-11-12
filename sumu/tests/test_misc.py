import tempfile
from pathlib import Path
import sumu


def test_str_to_dag_to_str():
    dag = sumu.validate.dag([(0, set()), (1, {0, 2}), (2, set())])
    assert dag == sumu.utils.io.str_to_dag(sumu.utils.io.dag_to_str(dag))


def test_write_read_jkl():
    bn = sumu.GaussianBNet.random(4)
    data = bn.sample(200)
    scores = sumu.LocalScore(data=data).all_scores_dict()
    fpath = tempfile.mkstemp(dir=".")[1]
    sumu.utils.io.write_jkl(scores, fpath)
    assert scores == sumu.utils.io.read_jkl(fpath)
    try:
        Path(fpath).unlink()
    except PermissionError as e:
        print(e)


if __name__ == "__main__":
    test_write_read_jkl()
