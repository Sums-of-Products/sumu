import sumu
from sumu.validate import ValidationError


def test_candidates_validation():

    test_params = [
        ({0: (1, 3)}, False),
        ({"0": (1, 3)}, False),
        ({0: (1,), 1: (2,)}, False),
        ({0: (1,), 1: (0,)}, True),
        ({0: (1,), 1: (0, 2), 2: (0, 1)}, False),
        ({0: (1,), 1: (0,), 2: (0,)}, True),
    ]

    for p in test_params:
        try:
            sumu.validate.candidates(p[0])
            assert p[1]
        except ValidationError as e:
            assert not p[1]
            print(f"{p[0]} correctly identified as invalid argument: {e}")
