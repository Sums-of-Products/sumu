"""Module for collecting various statistics across all the other modules."""

from abc import ABCMeta, abstractmethod
from collections import defaultdict


class Describable(metaclass=ABCMeta):
    @abstractmethod
    def describe(self):
        """Return a dict of stats as simple key:value pairs.

        E.g., 'p_local': sum(self.local_history) / len(self.local_history)"""
        pass


class Stats(defaultdict):
    def __init__(self, *args, **kwargs):
        super(Stats, self).__init__(Stats, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))


stats = Stats()
