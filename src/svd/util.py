#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import enum
from typing import Any, List


def strip_prefixes_suffixes(word: str, prefixes: List[str], suffixes: List[str]) -> str:
    """
    Emulates the functionality provided by chaining `removeprefix` and `removesuffix`
    to a str object.

    :param word: String to strip prefixes and suffixes from.
    :param prefixes: List of prefixes to strip.
    :param suffixes: List of suffixes to strip.

    :return: String where prefixes and suffixes have been sequentially removed.
    """

    for prefix in prefixes:
        if word.startswith(prefix):
            word = word[len(prefix) :]

    for suffix in suffixes:
        if word.endswith(suffix):
            word = word[: -len(suffix)]

    return word.strip("_")


def to_int(number: str) -> int:
    """
    Convert a string representation of an integer following the SVD format to its corresponding
    integer representation.

    :param number: String representation of the integer.

    :return: Decoded integer.
    """

    if number.startswith("0x"):
        return int(number, base=16)

    if number.startswith("#"):
        return int(number[1:], base=2)

    return int(number)


def to_bool(value: str) -> bool:
    """
    Convert a string representation of a boolean following the SVD format to its corresponding
    boolean representation.

    :param value: String representation of the boolean.

    :return: Decoded boolean.
    """
    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False
    raise ValueError(f"Invalid boolean value: {value}")


class CaseInsensitiveStrEnum(enum.Enum):
    @classmethod
    def from_str(cls, value: str) -> CaseInsensitiveStrEnum:
        value_lower = value.lower()
        for member in cls:
            if member.value.lower() == value_lower:
                return member
        raise ValueError(
            f"Class {cls.__qualname__} has no member corresponding to '{value}'"
        )


class BindingWrapper:
    def __init__(self, binding):
        self._binding = binding
