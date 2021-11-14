import sumu


def test_candidates_validation():
    assert sumu.validate.candidates_is_valid({0: (1, 3)}) is False
    assert (
        sumu.validate.candidates_is_valid(
            {
                "0": (1,),
            }
        )
        is False
    )
    assert sumu.validate.candidates_is_valid({0: (1,), 1: (2,)}) is False
    assert sumu.validate.candidates_is_valid({0: (1,), 1: (0,)}) is True
    assert (
        sumu.validate.candidates_is_valid({0: (1,), 1: (0, 2), 2: (0, 1)})
        is False
    )
    assert (
        sumu.validate.candidates_is_valid({0: (1,), 1: (0,), 2: (0,)}) is True
    )
