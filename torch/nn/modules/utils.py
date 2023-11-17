import collections
from itertools import repeat


def _ntuple(n):
    def parse(x):
        return x if isinstance(x, collections.Iterable) else tuple(repeat(x, n))

    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
