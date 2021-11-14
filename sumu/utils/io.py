"""Miscellaneous IO related functions.."""


def dag_to_str(dag):
    return "|".join(
        [
            str(f[0])
            if len(f) == 1
            else " ".join([str(f[0]), *[str(v) for v in sorted(f[1])]])
            for f in dag
        ]
    )


def str_to_dag(dag_str):
    def parse_family_str(fstr):
        fstr = fstr.split()
        if len(fstr) > 1:
            return (int(fstr[0]), set(map(int, fstr[1:])))
        else:
            return (int(fstr[0]), set())
    return list(map(parse_family_str, dag_str.split("|")))


def read_candidates(candidate_path):
    """Read parent candidates from file.

    Row number identifies the node and space separated numbers on each row
    identify the candidate parents.
    """
    C = dict()
    with open(candidate_path, "r") as f:
        f = f.readlines()
        for v, row in enumerate(f):
            C[v] = tuple([int(x) for x in row.split()])
    return C


def read_jkl(scorepath):
    scores = dict()
    with open(scorepath, "r") as jkl_file:
        rows = jkl_file.readlines()
        scores = dict()
        n_scores = 0
        for row in rows[1:]:

            if not n_scores:
                n_scores = int(row.strip().split()[1])
                current_var = int(row.strip().split()[0])
                scores[current_var] = dict()
                continue

            row_list = row.strip().split()
            score = float(row_list[0])
            n_parents = int(row_list[1])

            parents = frozenset()
            if n_parents > 0:
                parents = frozenset([int(x) for x in row_list[2:]])
            scores[current_var][frozenset(parents)] = score
            n_scores -= 1

    return scores


def write_jkl(scores, fpath):
    """Assumes the psets are iterables, not bitmaps"""

    with open(fpath, "w") as f:
        lines = list()
        n = len(scores)
        lines.append(str(n) + "\n")
        for v in sorted(scores):
            lines.append(f"{v} {len(scores[v])}\n")
            for pset in sorted(scores[v], key=lambda pset: len(pset)):
                lines.append(
                    f"{scores[v][pset]} {len(pset)} "
                    f"{' '.join([str(p) for p in pset])}\n"
                )
        f.writelines(lines)
