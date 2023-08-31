class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def flatten(x: list) -> list:
    return sum(map(flatten, x), []) if isinstance(x, list) else [x]
