#
# Copyright (c) 2023 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from functools import partial
from typing import (
    Any,
    Dict,
    Callable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
    overload,
)
from typing_extensions import Self

import numpy as np
import numpy.ma as ma
from numpy.typing import ArrayLike

from .errors import SvdMemoryError


SIZE_TO_DTYPE: Mapping[int, np.dtype] = {
    1: np.uint8,
    2: np.dtype((np.dtype("<u2"), (np.uint8, 2))),
    4: np.dtype((np.dtype("<u4"), (np.uint8, 4))),
}


def _get_dtype_for_size(item_size: int) -> np.dtype:
    try:
        return SIZE_TO_DTYPE[item_size]
    except KeyError:
        raise SvdMemoryError(f"Unsupported item size: {item_size}")


class MemoryBlock:
    """
    A contiguous memory region at a given (offset, length).
    """

    class Builder:
        """
        Builder that can be used to construct a MemoryBlock in several steps.
        """

        def __init__(self) -> None:
            self._lazy_base_block: Optional[Callable[[], MemoryBlock]] = None
            self._offset: Optional[int] = None
            self._length: Optional[int] = None
            self._default_content: Optional[int] = None
            self._default_item_size: Optional[int] = None
            self._ops: List[Callable[[MemoryBlock], None]] = []

        def build(self) -> MemoryBlock:
            """
            Build the memory block based on the parameters set.

            :return: The built memory block.
            """
            if self._default_content is None or self._default_item_size is None:
                raise ValueError("Missing ")

            from_block: Optional[MemoryBlock] = (
                self._lazy_base_block() if self._lazy_base_block is not None else None
            )

            block = MemoryBlock(
                default_content=self._default_content,
                length=self._length,
                offset=self._offset,
                from_block=from_block,
            )

            for op in self._ops:
                op(block)

            return block

        def lazy_copy_from(self, lazy_block: Callable[[], MemoryBlock]) -> Self:
            """
            Use a different memory block as the base for this memory block.
            The lazy_block should be a callable that can be called in build() to get the base
            block.

            :param lazy_block: Callable that returns the base block.
            :return: The builder instance.
            """
            self._lazy_base_block = lazy_block
            return self

        def set_extent(self, offset: int, length: int) -> Self:
            """
            Set the offset and length of the memory block.
            This is required unless lazy_copy_from() is used.

            :param offset: Starting offset of the memory block.
            :param length: Length of the memory block, starting at the given offset.
            :return: The builder instance.
            """
            self._offset = offset
            self._length = length
            return self

        def set_default_content(self, default_content: int, item_size: int = 4) -> Self:
            """
            Set the default content (initial value) of the memory block.
            This is required.

            :param default_content: Default value.
            :param item_size: Size in bytes of default_content.
            :return: The builder instance.
            """
            self._default_content = default_content
            self._default_item_size = item_size
            return self

        def fill(self, start: int, end: int, content: int, item_size: int = 4) -> Self:
            """
            Fill the memory block address range [start, end) with a value.

            :param start: Start offset of the range to be filled.
            :param end: Exclusive end offset of the range to be filled.
            :param content: Value to fill with.
            :param item_size: Size in bytes of content.
            :return: The builder instance.
            """
            if start < end:
                self._ops.append(
                    partial(
                        MemoryBlock._fill,
                        start=start,
                        end=end,
                        value=content,
                        item_size=item_size,
                    )
                )
            return self

        def tile(self, start: int, end: int, times: int) -> Self:
            """
            Duplicate the values at the memory block address range [start, end) a number of times.
            The range is duplicated at the *times* positions following the range.

            :param start: Start offset of the range to be duplicated.
            :param end: Exclusive end offset of the range to be duplicated.
            :param times: Number of times to duplicate the range.
            :return: The builder instance.
            """
            if times > 1 and start < end:
                self._ops.append(
                    partial(MemoryBlock._tile, start=start, end=end, times=times)
                )
            return self

    def __init__(
        self,
        default_content: int,
        default_item_size: int = 4,
        offset: Optional[int] = None,
        length: Optional[int] = None,
        from_block: Optional[MemoryBlock] = None,
    ) -> None:
        """
        :param default_content:
        :param offset: Starting offset of the memory block. Required unless from_block is passed.
        :param length: Length in bytes of the memory block. Required unless from_block is passed.
        :param from_block: Memory block to use as the base for this memory block.
        """
        self._offset: int
        self._length: int

        default_dtype = SIZE_TO_DTYPE[default_item_size]

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

            data = _numpy_full(
                self._length // default_item_size, default_content, dtype=default_dtype
            ).view(np.uint8)
            address_mask = np.ones_like(data, dtype=bool)
            self._array: ma.MaskedArray = ma.MaskedArray(
                data=data, mask=address_mask, dtype=np.uint8
            )
            self._written = np.zeros_like(data, dtype=np.uint8)

            dst_start = from_block._offset - offset if offset is not None else 0
            dst_end = dst_start + from_block._length
            self._array.mask[dst_start:dst_end] &= from_block.array.mask
            np.copyto(dst=self._array[dst_start:dst_end], src=from_block.array)
            np.copyto(dst=self._written[dst_start:dst_end], src=from_block._written)

        else:
            if offset is None or length is None:
                raise ValueError("offset and length are required when no from_block is given")

            self._offset = offset
            self._length = length
            data = _numpy_full(
                length // default_item_size, default_content, dtype=default_dtype
            ).view(np.uint8)
            address_mask = np.ones_like(data, dtype=bool)
            self._array = ma.MaskedArray(data=data, mask=address_mask, dtype=np.uint8)
            self._written = np.zeros_like(data, dtype=np.uint8)

    @overload
    def at(self, idx: int, item_size: int) -> int:
        ...

    @overload
    def at(self, idx: slice, item_size: int) -> ArrayLike:
        ...

    def at(self, idx: Union[int, slice], item_size: int = 4) -> Union[int, ArrayLike]:
        translated_idx, dtype = self._translate_access(idx, item_size)
        return self.array.data.view(dtype=dtype)[translated_idx]

    def set_at(
        self, idx: Union[int, slice], value: Union[int, ArrayLike], item_size: int = 4
    ) -> None:
        translated_idx, dtype = self._translate_access(idx, item_size)
        self.array.data.view(dtype=dtype)[translated_idx] = value
        self._written.view(dtype=dtype)[translated_idx] = 2**(8 * item_size) - 1 # pls fix

    @overload
    def __getitem__(self, idx: int) -> int:
        ...

    @overload
    def __getitem__(self, idx: slice) -> ArrayLike:
        ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[int, ArrayLike]:
        return self.at(idx)

    def __setitem__(self, idx: Union[int, slice], value: Union[int, ArrayLike]) -> None:
        self.set_at(idx, value)

    def memory_iter(
        self, item_size: int = 4, with_offset: int = 0, written_only: bool = False
    ) -> Iterator[Tuple[int, int]]:
        """
        
        """
        if self._length % item_size != 0:
            raise ValueError(
                f"Memory block length {self._length} is not divisible by {item_size}"
            )

        dtype = SIZE_TO_DTYPE[item_size]

        inverse_mask = ~self.array.mask
        if written_only:
            inverse_mask = inverse_mask[self._written.view(dtype=bool)]

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
        dtype: np.dtype = SIZE_TO_DTYPE[item_size]
        translated_idx: Union[int, slice]

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
        """Inline tile operation. See MemoryBlock.Builder.tile() for details"""
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
        """Inline fill operation. See MemoryBlock.Builder.fill() for details"""
        dtype = SIZE_TO_DTYPE[item_size]
        offset_start = start - self._offset
        offset_end = end - self._offset
        self.array.mask[offset_start:offset_end] = False
        data_dst = self.array.data[offset_start:offset_end].view(dtype)
        data_dst[:] = value


def _numpy_full(length: int, value: int, dtype: np.dtype) -> np.ndarray:
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
