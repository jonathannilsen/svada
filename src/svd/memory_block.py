#
# Copyright (c) 2023 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from functools import partial
from typing import Dict, Callable, Iterator, List, Optional, Tuple, Type, Union
from typing_extensions import Self

import numpy as np
import numpy.ma as ma


SIZE_TO_DTYPE = {
    1: np.uint8,
    2: np.dtype((np.dtype("<u2"), (np.uint8, 2))),
    4: np.dtype((np.dtype("<u4"), (np.uint8, 4))),
}


class MemoryBlock:
    """
    A block of memory.
    """

    class Builder:
        """
        Builder that can be used to construct a MemoryBlock in several steps.
        """

        def __init__(self) -> None:
            self._lazy_base_block: Optional[Callable[[], MemoryBlock]] = None
            self._offset: Optional[int] = None
            self._length: Optional[int] = None
            self._default_value: Optional[int] = None
            self._ops: List[Callable[[MemoryBlock], None]] = []

        def build(self) -> MemoryBlock:
            from_block: Optional[MemoryBlock] = (
                self._lazy_base_block() if self._lazy_base_block is not None else None
            )

            block = MemoryBlock(
                default_value=self._default_value,
                length=self._length,
                offset=self._offset,
                from_block=from_block,
            )

            for op in self._ops:
                op(block)

            return block

        def lazy_copy_from(self, lazy_block: Callable[[], MemoryBlock]) -> Self:
            self._lazy_base_block = lazy_block
            return self

        def set_extent(self, offset: int, length: int) -> Self:
            self._offset = offset
            self._length = length
            return self

        def set_default_value(self, default_value: int) -> Self:
            self._default_value = default_value
            return self

        def fill(self, start: int, end: int, value: int, item_size: int = 1) -> Self:
            """Fill the memory block range [start, end) with a value"""
            if start < end:
                self._ops.append(
                    partial(
                        MemoryBlock._fill,
                        start=start,
                        end=end,
                        value=value,
                        item_size=item_size,
                    )
                )
            return self

        def tile(self, start: int, end: int, times: int) -> Self:
            """Duplicate the values at range [start, end) a number of times."""
            if times > 1 and start < end:
                self._ops.append(
                    partial(MemoryBlock._tile, start=start, end=end, times=times)
                )
            return self

    def __init__(
        self,
        default_value: int,
        offset: Optional[int] = None,
        length: Optional[int] = None,
        from_block: Optional[MemoryBlock] = None,
    ):
        self._offset: int
        self._length: int

        if from_block is not None:
            if offset is not None:
                self._offset = min(offset, from_block._offset)
            else:
                self._offset = from_block._offset

            if length is not None:
                self._length = (
                    max(self._offset + length, from_block._offset + from_block._length)
                    - self._offset
                )
            else:
                self._length = from_block._length

            data = numpy_full(self._length, default_value, dtype=np.uint8)
            address_mask = np.ones_like(data, dtype=bool)
            self._array = ma.MaskedArray(data=data, mask=address_mask, dtype=np.uint8)

            dst_start = from_block._offset - offset if offset is not None else 0
            dst_end = dst_start + from_block._length
            self._array.mask[dst_start:dst_end] &= from_block.array.mask
            np.copyto(dst=self._array[dst_start:dst_end], src=from_block.array)

        else:
            if offset is None or length is None:
                raise ValueError("length is required when no from_block is given")

            self._offset = offset
            self._length = length
            data = numpy_full(length, default_value, dtype=np.uint8)
            address_mask = np.ones_like(length, dtype=bool)
            self._array = ma.MaskedArray(data=data, mask=address_mask, dtype=np.uint8)

    def at(self, idx: Union[int, slice], item_size: int = 4):
        translated_idx, dtype = self._translate_access(idx, item_size)
        return self.array.data.view(dtype=dtype)[translated_idx]

    def set_at(self, idx: Union[int, slice], value, item_size: int = 4):
        translated_idx, dtype = self._translate_access(idx, item_size)
        self.array.data.view(dtype=dtype)[translated_idx] = value

    def __getitem__(self, idx: Union[int, slice]):
        return self.at(idx)

    def __setitem__(self, idx: Union[int, slice], value) -> None:
        self.set_at(idx, value)

    def memory_iter(
        self, item_size: int = 4, with_offset: int = 0
    ) -> Iterator[Tuple[int, int]]:
        if self._length % item_size != 0:
            raise ValueError(
                f"Memory block length {self._length} is not divisible by {item_size}"
            )

        dtype = SIZE_TO_DTYPE[item_size]
        inverse_mask = ~self.array.mask
        address_start = self._offset + with_offset
        addresses = np.linspace(
            address_start,
            address_start + self._length,
            num=self._length,
            endpoint=False,
            dtype=int,
        )[inverse_mask][::item_size]
        values = self.array.compressed().view(dtype)

        for address, value in zip(addresses, values):
            yield int(address), int(value)

    @property
    def array(self) -> ma.MaskedArray:
        return self._array

    def __len__(self) -> int:
        return len(self.array)

    def _translate_access(
        self, idx: Union[int, slice], item_size: int
    ) -> Tuple[Union[int, slice], Type[np.dtype]]:
        dtype = SIZE_TO_DTYPE[item_size]

        if isinstance(idx, int):
            translated_idx = (idx - self._offset) // item_size
        elif isinstance(idx, slice):
            translated_idx = slice(
                (idx.start - self._offset) // item_size
                if idx.start is not None
                else None,
                (idx.stop - self._offset) // item_size
                if idx.stop is not None
                else None,
                (idx.step // item_size) if idx.step else None,
            )
        else:
            raise ValueError(f"Unsupported index: {idx}")

        return translated_idx, dtype

    def _tile(self, /, *, start: int, end: int, times: int) -> None:
        length = end - start
        src_offset_start = start - self._offset
        src_offset_end = end - self._offset
        dst_offset_start = src_offset_start + length
        dst_offset_end = src_offset_start + length * times

        for array in (self.array.data, self.array.mask):
            array_src = array[src_offset_start:src_offset_end]
            array_dst = array[dst_offset_start:dst_offset_end].reshape(
                (times - 1, length)
            )
            array_dst[:] = array_src

    def _fill(self, /, *, start: int, end: int, value: int, item_size: int) -> None:
        dtype = SIZE_TO_DTYPE[item_size]
        offset_start = start - self._offset
        offset_end = end - self._offset
        self.array.mask[offset_start:offset_end] = False
        data_dst = self.array.data[offset_start:offset_end].view(dtype)
        data_dst[:] = value


def numpy_full(length: int, value: int, dtype: np.dtype) -> np.ndarray:
    # numpy doesn't seem to handle numpy.full(..., filL_value=0) in any special way.
    # Using np.zeros() for that case here provides a significant speedup for large arrays.
    if value == 0:
        return np.zeros(length, dtype=dtype)
    else:
        return np.full(
            length,
            value,
            dtype=dtype,
        )
