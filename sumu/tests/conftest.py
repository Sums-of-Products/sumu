import glob
from pathlib import Path

import pytest

import sumu


@pytest.fixture(scope="session")
def discrete_bn():
    bn = dict()
    data_path = Path(__file__).resolve().parents[2] / "data"
    for bn_path in glob.glob(str(data_path) + "/*.dsc"):
        bn[Path(bn_path).stem] = sumu.DiscreteBNet.read_file(bn_path)
    return bn
