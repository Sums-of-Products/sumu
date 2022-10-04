
"""Module for validating arbitrary objects.

The module can be used in two ways:
1. To create boolean returning functions that optionally raise ValidationError
with a custom message instead of returning False.
2. To create more complicated validator functions that check an input object
against multiple test functions.

Both are created upon import of this module.

For (1) any function defined in this file without an underscore prefix in the
name will be converted to an identically named function with an additional
optional keyword argument 'msg'. If 'msg' is set, the function will raise
ValidationError with the message if the original function had returned False.
If 'msg' is not set the function will behave like the original.

For (2) each validator should be constructed as a dict where the keys are error
messages and the values are functions that take the object to be validated as a
parameter. The functions should return True if the object passes the test
defined in the function and False otherwise.

The dict names should start with a single underscore.

For each validator two functions are created:
- <name of the validator without the initial underscore>:
  - if object is valid: returns the object
  - if object is invalid: raises a ValidationError with the message defined in
    the validator
- <name of the validator without the initial underscore>_is_valid:
  - if object is valid: returns True
  - if object is invalid: returns False
"""

# Black does not do this file justice:
# https://github.com/psf/black/issues/620
# https://github.com/psf/black/issues/808
# etc.
# The formatting has to be handcrafted, for now.

import sys
import types

import numpy as np


def is_int(val):
    return type(val) == int or np.issubdtype(type(val), np.integer)


def is_float(val):
    return type(val) == float or np.issubdtype(type(val), np.floating)


def is_num(val):
    return is_int(val) or is_float(val)


def is_pos_int(val):
    return is_int(val) and val > 0


def is_nonneg_int(val):
    return is_int(val) and val >= 0


def is_nonneg_num(val):
    return is_num(val) and val >= 0


def is_pos_num(val):
    return is_num(val) and val > 0


def is_boolean(val):
    return type(val) == bool


def is_string(val):
    return type(val) == str


def in_range(val, min_val, max_val, min_incl=True, max_incl=True):
    result = True
    if min_incl:
        result &= val >= min_val
    else:
        result &= val > min_val
    if max_incl:
        result &= val <= max_val
    else:
        result &= val < max_val
    return result


def nested_in_dict(d, *keys):
    try:
        for k in keys:
            d = d[k]
    except (KeyError, TypeError):
        return False
    return True


def keys_in_list_error_if_not(lst, *keys):
    return all(type(lst.index(k)) == int for k in keys)


def max_n_truthy(n, items):
    if sum(map(bool, items)) <= n:
        return True
    return False


_run_mode_args = {
    "_arg_names": ["name", "params"],
    "_names": ["normal", "budget", "anytime"],
    "_params_keys": {"normal": [], "anytime": [], "budget": ["t", "mem"]},
}
_run_mode_args.update(
    {

        f"parameters should be in {_run_mode_args['_arg_names']}":
        lambda p: keys_in_list_error_if_not(
            _run_mode_args["_arg_names"], *p
        ),

        f"run_mode name should be one of {_run_mode_args['_names']}":
        lambda p: keys_in_list_error_if_not(
            _run_mode_args["_names"], p["name"]
        )
        if "name" in p
        else True,

        (
            f"valid 'params' keys for different run modes are "
            f"{_run_mode_args['_params_keys']}"
        ): lambda p: keys_in_list_error_if_not(
            _run_mode_args["_params_keys"][p["name"]], *p["params"]
        )
        if "name" in p and "params" in p
        else True,

    }
)


_mcmc_args = {
    "_arg_names": [
        "n_indep",
        "n_target_chain_iters",
        "burn_in",
        "n_dags",
        "move_weights",
    ]
}
_mcmc_args.update(
    {

        f"parameters should be in {_mcmc_args['_arg_names']}":
        lambda p: keys_in_list_error_if_not(
            _mcmc_args["_arg_names"], *p
        ),

        "n_indep should be non-negative integer":
        lambda p: is_nonneg_int(p["n_indep"])
        if "n_indep" in p
        else True,

        "n_target_chain_iters should be non-negative integer":
        lambda p: is_nonneg_int(
            p["n_target_chain_iters"]
        )
        if "n_target_chain_iters" in p
        else True,

        "burn_in should be a float in range [0, 1]":
        lambda p: is_float(p["burn_in"])
        and in_range(p["burn_in"], 0, 1)
        if "burn_in" in p
        else True,

        "n_dags should be non-negative integer":
        lambda p: is_nonneg_int(p["n_dags"])
        if "n_dags" in p
        else True,

        "move_weights should be a list of 3 non-negative integers":
        lambda p: all(
            [
                type(p["move_weights"]) == list,
                len(p["move_weights"]) == 3,
                all([is_nonneg_int(w) for w in p["move_weights"]]),
            ]
        )
        if "move_weights" in p
        else True,

    }
)


_metropolis_coupling_scheme_args = {
    "_arg_names": {
        "name": ["adaptive", "linear", "inv_linear", "quadratic", "sigmoid"],
        "params": [
            "M",
            "p_target",
            "delta_t_init",
            "local_accept_history_size",
            "update_freq",
            "smoothing",
            "slowdown",
        ],
    }
}
_metropolis_coupling_scheme_args.update(
    {

        (
            f"parameters should be in "
            f"{list(_metropolis_coupling_scheme_args['_arg_names'].keys())}"):
        lambda p: keys_in_list_error_if_not(
            list(_metropolis_coupling_scheme_args["_arg_names"]), *p
        ),

        (
            f"name should be in "
            f"{_metropolis_coupling_scheme_args['_arg_names']['name']}"
        ):
        lambda p: keys_in_list_error_if_not(
            _metropolis_coupling_scheme_args["_arg_names"]["name"], p["name"]
        )
        if "name" in p
        else True,

        (
            f"'params' keys should be in "
            f"{list(_metropolis_coupling_scheme_args['_arg_names']['params'])}"
        ):
        lambda p: keys_in_list_error_if_not(
            _metropolis_coupling_scheme_args["_arg_names"]["params"],
            *p["params"]
        )
        if "params" in p
        else True,

        "M should be a non-negative integer":
        lambda p: is_nonneg_int(p["params"]["M"])
        if nested_in_dict(p, "params", "M")
        else True,

        "p_target should be a float in range [0, 1]":
        lambda p: is_float(
            p["params"]["p_target"]
        )
        and in_range(p["params"]["p_target"], 0, 1)
        if nested_in_dict(p, "params", "p_target")
        else True,

        "delta_t_init should be a positive number":
        lambda p: is_pos_num(
            p["params"]["delta_t_init"]
        )
        if nested_in_dict(p, "params", "delta_t_init")
        else True,

        "local_accept_history_size should be a positive integer":
        lambda p: is_pos_int(
            p["params"]["local_accept_history_size"]
        )
        if nested_in_dict(p, "params", "local_accept_history_size")
        else True,

        "update_freq should be a positive integer":
        lambda p: is_pos_int(
            p["params"]["update_freq"]
        )
        if nested_in_dict(p, "params", "update_freq")
        else True,

        "smoothing should be a non-negative number":
        lambda p: is_nonneg_num(
            p["params"]["smoothing"]
        )
        if nested_in_dict(p, "params", "smoothing")
        else True,

        "slowdown should be a non-negative number":
        lambda p: is_nonneg_num(
            p["params"]["slowdown"]
        )
        if nested_in_dict(p, "params", "slowdown")
        else True,

    }
)


_score_args = {
    "_arg_names": ["name", "params"],
    "_names": ["bdeu", "bge"],
    "_params_keys": {"bdeu": ["ess"], "bge": []},
}
_score_args.update(
    {

        f"parameters should be in {_score_args['_arg_names']}":
        lambda p: keys_in_list_error_if_not(
            _score_args["_arg_names"], *p
        ),

        f"name should be in {_score_args['_names']}":
        lambda p: keys_in_list_error_if_not(
            _score_args["_names"], p["name"]
        )
        if "name" in p
        else True,

        (
            f"valid 'params' keys for different scores are "
            f"{_score_args['_params_keys']}"
        ):
        lambda p: keys_in_list_error_if_not(
            _score_args["_params_keys"][p["name"]], *p["params"]
        )
        if "name" in p and "params" in p
        else True,

        "ess should be a non-negative number":
        lambda p: is_nonneg_num(
            p["params"]["ess"]
        )
        if nested_in_dict(p, "params", "ess")
        else True,

    }
)


_structure_prior_args = {
    "_arg_names": ["name", "params"],
    "_names": ["fair", "unif"],
    "_params_keys": {"fair": [], "unif": []},
}
_structure_prior_args.update(
    {

        f"parameters should be in {_structure_prior_args['_arg_names']}":
        lambda p: keys_in_list_error_if_not(
            _structure_prior_args["_arg_names"], *p
        ),

        f"name should be in {_structure_prior_args['_names']}":
        lambda p: keys_in_list_error_if_not(
            _structure_prior_args["_names"], p["name"]
        )
        if "name" in p
        else True,

        (
            f"valid 'params' keys for structure priors are "
            f"{_structure_prior_args['_params_keys']}"
        ):
        lambda p: keys_in_list_error_if_not(
            _structure_prior_args["_params_keys"][p["name"]], *p["params"]
        )
        if "name" in p and "params" in p
        else True,

    }
)


_constraints_args = {"_arg_names": ["max_id", "K", "d", "pruning_eps"]}
_constraints_args.update(
    {

        f"parameters should be in {_constraints_args['_arg_names']}":
        lambda p: all(
            [type(_constraints_args["_arg_names"].index(k)) == int for k in p]
        ),

        "max_id should be a positive integer or -1 (for no limit on indegree)":
        lambda p: is_int(
            p["max_id"]
        )
        and (p["max_id"] == -1 or p["max_id"] > 0)
        if "max_id" in p
        else True,

        "K should be a positive integer":
        lambda p: is_pos_int(p["K"])
        if "K" in p
        else True,

        "d should be a positive integer":
        lambda p: is_pos_int(p["d"])
        if "d" in p
        else True,

        "pruning_eps should be a positive number":
        lambda p: is_pos_num(
            p["pruning_eps"]
        )
        if "pruning_eps" in p
        else True,

    }
)


_candidate_parent_algorithm_args = {
    "_arg_names": ["name", "params"],
    "_names": ["greedy", "opt", "rnd"],
    "_params_keys": {"greedy": ["k","criterion","opt_criterion","discount","d","var"], "opt": [], "rnd": []},
}
_candidate_parent_algorithm_args.update(
    {

        (
            f"parameters should be in "
            f"{_candidate_parent_algorithm_args['_arg_names']}"
        ):
        lambda p: keys_in_list_error_if_not(
            _candidate_parent_algorithm_args["_arg_names"], *p
        ),

        f"name should be in {_candidate_parent_algorithm_args['_names']}":
        lambda p: keys_in_list_error_if_not(
            _candidate_parent_algorithm_args["_names"], p["name"]
        )
        if "name" in p
        else True,

        (
            f"valid 'params' keys for candidate parent algorithms are "
            f"{_candidate_parent_algorithm_args['_params_keys']}"
        ):
        lambda p: keys_in_list_error_if_not(
            _candidate_parent_algorithm_args["_params_keys"][p["name"]],
            *p["params"]
        )
        if "name" in p and "params" in p
        else True,

        "k should be a positive integer":
        lambda p: is_pos_int(p["params"]["k"])
        if nested_in_dict(p, "params", "k")
        else True,

    }
)


_catastrophic_cancellation_args = {"_arg_names": ["tolerance", "cache_size"]}
_catastrophic_cancellation_args.update(
    {

        (
            f"parameters should be in"
            f"{_catastrophic_cancellation_args['_arg_names']}"
        ):
        lambda p: keys_in_list_error_if_not(
            _catastrophic_cancellation_args["_arg_names"], *p
        ),

        "tolerance should be a positive number":
        lambda p: is_pos_num(p["tolerance"])
        if "tolerance" in p
        else True,

        "cache_size should be a positive integer":
        lambda p: is_pos_int(p["cache_size"])
        if "cache_size" in p
        else True,

    }
)


_logging_args = {
    "_arg_names": ["silent", "verbose_prefix", "stats_period", "overwrite"]
}
_logging_args.update(
    {

        f"parameters should be in {_logging_args['_arg_names']}":
        lambda p: keys_in_list_error_if_not(
            _logging_args["_arg_names"], *p
        ),

        "silent should be a boolean":
        lambda p: type(p["silent"]) == bool
        if "silent" in p
        else True,

        "verbose_prefix should be a string":
        lambda p: type(p["verbose_prefix"]) == str
        if "verbose_prefix" in p
        else True,

        "overwrite should be a boolean":
        lambda p: type(p["overwrite"]) == bool
        if "overwrite" in p
        else True,

        "stats_period should be a positive number":
        lambda p: is_pos_num(
            p["stats_period"]
        )
        if "stats_period" in p
        else True,

    }
)


_rootpartition = {
    "should be a list partitioning integers 0..n to sets":
    lambda R: all([
        type(R) == list,
        all([type(R_i) == set for R_i in R]),
        all([is_int(u) for R_i in R for u in R_i]),
        sorted([u for R_i in R for u in R_i])
        == list(range(max([max(R_i) for R_i in R]) + 1))
    ])
}


_dag = {

    (
        "should be in the format [(int, set()), ...] "
        "where int is a node label and the set contains its parents' labels"
    ):
    lambda dag: all(
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

    "should be given as tuples of ints in a dict":
    lambda C: all(
        [
            type(C) == dict,
            all(type(v) == tuple for v in C.values()),
            all(is_int(vi)
                for v in C.values() for vi in v),
        ]
    ),

    "the candidates dict should have keys (node labels) from 0 to n":
    lambda C: sorted(
        C.keys()
    )
    == list(range(max(C) + 1)),

    "there should be from 1 to n-1 candidate parents for each node":
    lambda C: all(
        len(v) > 0 and len(v) < len(C) for v in C.values()
    ),

    "nodes should be given equal number of candidate parents":
    lambda C: all(
        len(v) == len(C[0]) for v in C.values()
    ),

    "candidate parents for a node should not contain duplicates":
    lambda C: all(
        len(set(v)) == len(v) for v in C.values()
    ),

    "candidate parents for each node should be a subset of the other nodes":
    lambda C: all(
        set(v).issubset(set(C).difference({k})) for k, v in C.items()
    ),

}

# NOTE: Write your validators above this line.


class ValidationError(Exception):
    pass


def _make_validator(validator, validator_name, only_check_is_valid=False):
    def validate(item):
        for f in validator:
            if type(f) == str and f[0] == "_":
                continue
            try:
                if not validator[f](item):
                    if only_check_is_valid:
                        return False
                    raise ValidationError(f"{validator_name}: {f}")
            except ValidationError:
                raise
            except Exception as e:  # noqa
                if only_check_is_valid:
                    return False
                raise ValidationError(f"{validator_name}: {f}: {e}")
        if only_check_is_valid:
            return True
        return item

    return validate


def _make_func(f):
    def func(*args, msg=None, **kwargs):
        result = f(*args, **kwargs)
        if result is False and msg is not None:
            raise ValidationError(msg)
        else:
            return result

    return func


[
    (setattr(sys.modules[__name__], k, _make_func(globals()[k])))
    for k in list(globals())
    if type(globals()[k]) == types.FunctionType and k[0] != "_" # noqa
]


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
            _make_validator(globals()[k], k[1:].capitalize(),
                            only_check_is_valid=True),
        ),
    )
    for k in list(globals())
    if type(globals()[k]) == dict and k[:2] != "__"
]

del _make_validator
del _make_func
