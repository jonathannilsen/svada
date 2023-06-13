#
# Copyright (c) 2023 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from abc import ABC
from typing import Any, Union

from .path import SPathUnion, SPath, SPathType


class SvdError(Exception):
    """Base class for errors raised by the library"""

    ...


class SvdParseError(SvdError):
    """Raised when an error occurs during SVD parsing."""

    ...


class SvdDefinitionError(SvdError, ValueError):
    """Raised when unrecoverable errors occur due to an invalid definition in the SVD file"""

    def __init__(self, binding: Any, explanation: str):
        super().__init__(f"Invalid SVD file elements ({binding}): {explanation}")


class SvdMemoryError(SvdError, BufferError):
    ...



class SvdPathError(SvdError):
    """Raised when trying to access a nonexistent/invalid SVD path."""

    def __init__(self, path: Union[str, SPathUnion], source: Any, explanation: str = "") -> None:
        formatted_explanation = "" if not explanation else f" ({explanation})"
        message = (
            f"{source!s} does not contain an element '{path}'{formatted_explanation}"
        )

        super().__init__(message)


class SvdIndexError(SvdPathError, IndexError):
    ...


class SvdKeyError(SvdPathError, KeyError):
    ...

