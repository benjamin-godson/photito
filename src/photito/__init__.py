from importlib.metadata import version as _version, PackageNotFoundError
import logging
import sys

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass
loggers = logging.getLogger(__name__)
loggers.addHandler(logging.NullHandler())

