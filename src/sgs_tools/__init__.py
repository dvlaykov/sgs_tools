from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sgs_tools")
except PackageNotFoundError:
    # package is not installed
    pass
