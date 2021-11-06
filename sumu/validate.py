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


def _validate(validator, validator_name, item, only_check_is_valid=False):
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


_validators = [
    x for x in globals() if type(globals()[x]) == dict and x[:2] != "__"
]

[
    (
        setattr(
            sys.modules[__name__],
            k[1:],
            lambda item: _validate(globals()[k], k[1:].capitalize(), item),
        ),
        setattr(
            sys.modules[__name__],
            k[1:] + "_is_valid",
            lambda item: _validate(
                globals()[k],
                k[1:].capitalize(),
                item,
                only_check_is_valid=True,
            ),
        ),
    )
    for k in _validators
]

del _validators
