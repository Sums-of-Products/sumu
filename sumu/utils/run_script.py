"""Functions used in run scripts for printing stuff nicely etc.
"""


def pretty_dict(d, n=1):
    for k in d:
        if type(d[k]) == dict:
            print("{}{}".format(" "*n, k))
        else:
            print("{}{}: {}".format(" "*n, k, d[k]))
        if type(d[k]) == dict:
            pretty_dict(d[k], n=n+4)


def DAG_to_str(DAG):
    return "|".join([str(f[0]) if len(f) == 1 else " ".join([str(f[0]), *[str(v) for v in sorted(f[1])]]) for f in DAG])
