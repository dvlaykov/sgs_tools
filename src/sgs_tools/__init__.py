from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sgs_tools")
except PackageNotFoundError:
    # package is not installed
    pass