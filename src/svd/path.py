#
# Copyright (c) 2023 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union


class SPath(Sequence[Union[str, int]]):
    """Path to a SVD element"""

    __slots__ = "_parts"

    # TODO: docstrings

    def __init__(self, *parts: Union[SPath, str, int]) -> None:
        if not parts:
            raise ValueError(f"Empty {self.__class__.__name__} not allowed")

        split_parts: List[Union[str, int]] = []

        # FIXME: ensure no two consecutive ints

        for part in parts:
            if isinstance(part, SPath):
                split_parts.extend(part.parts)
            elif isinstance(part, str):
                if not part.isalpha():
                    split_parts.extend(self._parse_path_str(part))
                else:
                    split_parts.append(part)
            elif isinstance(part, int):
                split_parts.append(part)
            else:
                raise TypeError(
                    f"Invalid {self.__class__.__name__} part {part} of type '{type(part)}'"
                )

        self._parts = tuple(split_parts)

    @property
    def parts(self) -> Tuple[Union[str, int], ...]:
        return self._parts

    @property
    def name(self) -> Optional[str]:
        for i in reversed(range(len(self._parts))):
            if isinstance(self._parts[i], str):
                return self._format_parts(self._parts[i:])
        return None

    @property
    def stem(self) -> Optional[str]:
        for part in reversed(range(len(self._parts))):
            if isinstance(part, str):
                return part
        return None

    @property
    def parent(self) -> Optional[SPath]:
        if len(self._parts) == 1:
            return None
        return SPath(*self._parts[:-1])

    @property
    def index(self) -> Optional[int]:
        """Index of the register in the parent array, if applicable."""
        if not isinstance(self[-1], int):
            return None
        return self[-1]

    def join(self, *other: Union[SPath, str, int]) -> SPath:
        return SPath(*self.parts, *other)

    def __getitem__(self, item: Union[int, slice]) -> Union[int, str, SPath]:
        if isinstance(item, slice):
            return SPath(*self.parts[item])
        else:
            return self.parts[item]

    def __len__(self) -> int:
        return len(self.parts)

    def __repr__(self) -> str:
        return self._format_parts(self.parts)

    def __hash__(self) -> int:
        return hash(self.parts)

    def __eq__(self, other: Any) -> bool: # FIXME: is this valid?
        return self.parts == other

    @staticmethod
    def _format_parts(parts: Iterable[Union[str, int]]) -> str:
        formatted_parts: List[str] = []

        for part in parts:
            if isinstance(part, int):
                formatted_parts.append(f"[{part}]")
            else:
                if not formatted_parts:
                    formatted_parts.append(part)
                else:
                    formatted_parts.append(f".{part}")

        return "".join(formatted_parts)

    # FIXME: this doesn't permit leading int, should it?
    def _parse_path_str(self, part: str) -> Iterable[Union[str, int]]:
        parsed_parts: List[Union[str, int]] = []

        remaining = part
        subpart_match = re.match(
            r"(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)", remaining
        )
        if subpart_match is None:
            raise ValueError(
                f"Invalid {self.__class__.__name__} part '{part}'"
            )

        parsed_parts.append(subpart_match["name"])
        remaining = remaining[subpart_match.end() :]

        while remaining:
            subpart_match = re.match(
                r"(?:(?:\[(?P<index>[0-9]+)\])|(?:\.(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)))",
                remaining
            )
            if subpart_match is None:
                raise ValueError(
                    f"Invalid {self.__class__.__name__} part '{part}'"
                )

            remaining = remaining[subpart_match.end():]

            if (index := subpart_match["index"]) is not None:
                parsed_parts.append(int(index, 10))
            else:
                parsed_parts.append(subpart_match["name"])

        return parsed_parts
