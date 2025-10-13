from .fink_fat import *

try:
    from . import fink_fat as _mod
    __doc__ = getattr(_mod, "__doc__", __doc__)
    if hasattr(_mod, "__all__"):
        __all__ = _mod.__all__
except Exception:
    pass
