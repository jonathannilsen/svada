from __future__ import annotations

import re
from functools import cached_property
from typing import Iterable, List, Optional, Tuple, Union


class SvdPath:
    """Path to a SVD element"""

    __slots__ = "_parts"

    # TODO: docstrings
    # TODO: can parts reuse a slice of a SvdPart?

    def __init__(self, *parts: Union[SvdPath, str, int]) -> None:
        if not parts:
            raise ValueError(f"Empty {self.__class__.__name__} not allowed")

        split_parts: List[Union[str, int]] = []

        for part in parts:
            if isinstance(part, SvdPath):
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
    def parent(self) -> SvdPath:
        if len(self._parts) == 1:
            return self
        return SvdPath(*self._parts[:-1])

    def join(self, *other: Union[SvdPath, str, int]) -> SvdPath:
        return SvdPath(*self.parts, *other)

    def __repr__(self) -> str:
        return self._format_parts(self.parts)

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


def main():
    # print(SvdPath("a", "b", "c"))
    # print(SvdPath("d", 0, "f"))
    # print(SvdPath("REG", 0).join(SvdPath("FIELD")))
    # print(SvdPath("REG", 0).join("REG2").join("FIELD"))
    # print(SvdPath(0).join(1).join(2).join("R[3]"))
    print(SvdPath("A[0].B[2].C[1].FIELD"))


if __name__ == "__main__":
    main()
