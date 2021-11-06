"""Module for validating arbitrary objects.

Each validator should be constructed as a dict where the keys are error messages
and the values are functions that take the validated object as a parameter. The
functions should return True if the object passes the test defined in the
function and False otherwise.

The dict names should start with a single underscore.

Upon import of this module, for each validator two functions are created:
- <name of the validator without the initial underscore>:
  - if object is valid: returns the object
  - if object is invalid: throws a ValidationError with the message defined in
    the validator
- <name of the validator without the initial underscore>_is_valid:
  - if object is valid: returns True
  - if object is invalid: returns False
"""

import sys
import numpy as np


_dag = {
    "should be in the format [(int, set()), ...] where int is a node label and the set contains its parents' labels": lambda dag: all(
        [
            type(dag) == list,
            all([type(f) == tuple for f in dag]),
            all([len(f) == 2 for f in dag]),
            all([isinstance(f[0], (np.integer, int))] for f in dag),
            all([type(f[1]) == set for f in dag]),
            all([isinstance(p, (np.integer, int)) for f in dag for p in f[1]]),
        ]
    )
}

_candidates = {
    "should be given as tuples of ints in a dict": lambda C: all(
        [
            type(C) == dict,
            all(type(v) == tuple for v in C.values()),
            all(
                isinstance(vi, (np.integer, int))
                for v in C.values()
                for vi in v
            ),
        ]
    ),
    "the candidates dict should have keys (node labels) from 0 to n": lambda C: sorted(
        C.keys()
    )
    == list(range(max(C) + 1)),
    "there should be from 1 to n-1 candidate parents for each node": lambda C: all(
        len(v) > 0 and len(v) < len(C) for v in C.values()
    ),
    "nodes should be given equal number of candidate parents": lambda C: all(
        len(v) == len(C[0]) for v in C.values()
    ),
    "candidate parents for a node should not contain duplicates": lambda C: all(
        len(set(v)) == len(v) for v in C.values()
    ),
    "candidate parents for each node should be a subset of the other nodes": lambda C: all(
        set(v).issubset(set(C).difference({k})) for k, v in C.items()
    ),
}


class ValidationError(Exception):
    pass


def _make_validator(validator, validator_name, only_check_is_valid=False):
    def validate(item):
        for f in validator:
            try:
                if not validator[f](item):
                    if only_check_is_valid:
                        return False
                    raise ValidationError(f"{validator_name}: {f}")
            except:
                if only_check_is_valid:
                    return False
                raise ValidationError(f"{validator_name}: {f}")
        if only_check_is_valid:
            return True
        return item

    return validate


[
    (
        setattr(
            sys.modules[__name__],
            k[1:],
            _make_validator(globals()[k], k[1:].capitalize()),
        ),
        setattr(
            sys.modules[__name__],
            k[1:] + "_is_valid",
            _make_validator(
                globals()[k], k[1:].capitalize(), only_check_is_valid=True
            ),
        ),
    )
    for k in list(globals())
    if type(globals()[k]) == dict and k[:2] != "__"
]

del _make_validator
