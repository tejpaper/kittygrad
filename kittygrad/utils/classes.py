class DotDict(dict):  # TODO: dict = DotDict ?
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
