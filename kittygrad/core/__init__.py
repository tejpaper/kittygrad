# native imports
from kittygrad.core.classes import *
from kittygrad.core.constants import *
from kittygrad.core.exceptions import *
from kittygrad.core.namespace import *

# computing core
import numpy as np

# creating strict namespace
from functools import partial
from inspect import getmembers
from types import SimpleNamespace

if NP_STRICT_CONFIG:
    strict = SimpleNamespace(**{
        ufunc_name: type(f'strict.{ufunc_name}', tuple(), dict(
            __call__=partial(ufunc, **NP_STRICT_CONFIG),
            __getattr__=ufunc.__getattribute__,
        ))()
        for ufunc_name, ufunc in getmembers(np, lambda x: isinstance(x, np.ufunc))
    })
else:
    strict = np
