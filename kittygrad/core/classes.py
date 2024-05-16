class DotDict(dict):  # TODO: rethink this part with SimpleNamespace
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
