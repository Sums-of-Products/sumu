import os
import sys
import cProfile
import subprocess
import io
from contextlib import redirect_stdout
from functools import wraps
import linecache
import glob

import memory_profiler
import numpy as np

import sumu

from config import params, keys
from functions_to_profile import time_gadget, mem_gadget


def my_profiler(func=None, stream=None, precision=1, backend='psutil'):
    backend = memory_profiler.choose_backend(backend)

    @wraps(func)
    def wrapper(*args, **kwargs):
        prof = memory_profiler.LineProfiler(backend=backend)
        prof(func)(*args, **kwargs)
        return prof

    return wrapper


def create_test_data():
    bn = sumu.utils.io.read_dsc(params["bn_path"])
    data = bn.sample(1000)
    sumu.utils.io.write_data(data, params["datapath"], bn)


def delete_test_data():
    os.remove(params["datapath"])


def mem_profile(target, print_all=False):
    # Profile line by line memory increment of target function
    prof = my_profiler(target)
    prof = prof()
    lines = list(list(prof.code_map.items())[0][1])
    i_last_line = lines[-1][0]
    mem = np.zeros(i_last_line)
    for line, mem_use in lines:
        if mem_use is not None:
            mem[line - 1] = mem_use[0]

    # Find max memory increment starting from line matching key
    # until next matching key
    all_lines = linecache.getlines("functions_to_profile.py")
    i_keys = list()
    for key in keys:
        for i in range(len(all_lines)):
            if key in all_lines[i]:
                i_keys.append(i)
    max_mem = list()
    for i in range(len(i_keys) - 1):
        max_mem.append(mem[i_keys[i]:i_keys[i+1]].max())
    max_mem.append(mem[i_keys[-1]:].max())

    if print_all:
        memory_profiler.show_results(prof, precision=3, stream=sys.stdout)

    return max_mem


def time_profile(target, print_all=False):
    f = io.StringIO()
    with redirect_stdout(f):
        cProfile.run(target.__name__ + "()")
    vals = list()
    for key in keys:
        f.seek(0)
        val = list(filter(lambda line: "(" + key + ")" in line, f.readlines()))[0].split()[-3]
        vals.append(float(val))

    if print_all:
        f.seek(0)
        print("".join(f.readlines()))

    return vals


def at_git_commit():
    return subprocess.check_output(["git", "rev-parse", "--short",
                                    "HEAD"]).decode("utf-8").strip()


def format_results(profile, precision=3):
    profile = list(map(lambda item: str(round(item, precision)), profile))
    return "|" + "|".join([at_git_commit()] + profile) + "|"


def combine_results():
    for i, filepath in enumerate(glob.glob("profile.*")):
        if i == 0:
            data = np.loadtxt(filepath)
        else:
            data = np.vstack((data, np.loadtxt(filepath)))

    return data.mean(axis=0)


if __name__ == '__main__':
    args = sys.argv[1:]
    if args[0] == "profile":
        # Creating and deleting test data creates concurrency problems
        # -- better to just create up front and delete later if
        # necessary.
        # create_test_data()
        mem_use = mem_profile(mem_gadget, print_all=False)
        time_use = time_profile(time_gadget, print_all=False)
        # delete_test_data()

        print(" ".join(map(str, time_use + mem_use)))

    if args[0] == "combine":
        print(format_results(combine_results()))
