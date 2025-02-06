from importlib.metadata import version as _version, PackageNotFoundError
import logging
import sys

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
