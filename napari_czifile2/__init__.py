try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._reader import napari_get_reader, reader_function_with_args

__all__ = ["napari_get_reader", "reader_function_with_args"]
