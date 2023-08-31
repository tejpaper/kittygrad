if __name__ != '__main__':
    from .activation import *
    from .ops import *
else:
    raise ImportError("Package startup error")
