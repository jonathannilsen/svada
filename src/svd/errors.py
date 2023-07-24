
from typing import Any

from .path import AnySPath, SPath, SPathType


class SvdTypeError(TypeError):
    def __init__(self, source: Any, explanation: str = "") -> None:
        formatted_explanation = "" if not explanation else f" ({explanation})"
        message = f"{source!s} is what????{formatted_explanation}"

        super().__init__(message)


class SvdFlatArrayError:
    """Error raised when trying to access an element in a flat array"""

    def __init__(self, path: SPath, source: Any) -> None:
        extra_info = "flat arrays do not have elements"

        super.__init__(path, source, extra_info=extra_info)


class SvdMemoryError(BufferError):
    ...


class SvdIndexError(IndexError):
    ...


class SvdKeyError(KeyError):
    ...


class SvdPathError(IndexError, KeyError):
    """Error raised when trying to access a nonexistent/invalid SVD path."""

    def __init__(self, path: AnySPath, source: Any, explanation: str = "") -> None:
        formatted_explanation = "" if not explanation else f" ({explanation})"
        message = (
            f"{source!s} does not contain an element '{path}'{formatted_explanation}"
        )

        super().__init__(message)
