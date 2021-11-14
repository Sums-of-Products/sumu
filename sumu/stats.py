"""Module for collecting various statistics across all the other modules."""

from collections import defaultdict


class Stats(defaultdict):
    def __init__(self, *args, **kwargs):
        super(Stats, self).__init__(Stats, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))


stats = Stats()
