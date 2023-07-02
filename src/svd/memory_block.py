from __future__ import annotations

from typing import Dict

import numpy as np
import numpy.ma as ma


class MemoryBlock:
    def __init__(self, length: int, default_value: int, **kwargs):
        if (from_array := kwargs.get("array")) is not None:
            self._array = ma.copy(from_array)
        else:
            content = numpy_full(length, default_value, dtype=np.uint8)
            address_mask = np.ones_like(content, dtype=bool)
            self._array = ma.MaskedArray(data=content, mask=address_mask, dtype=np.uint8)

    def as_dict(valid_only: bool = True) -> Dict[int, int]:
        # TODO
        ...

    @property
    def array(self) -> ma.MaskedArray:
        return self._array

    def __copy__(self) -> MemoryBlock:
        return MemoryBlock(0, 0, from_array=self.array)

    # Contain maskedarray
    # as_dict()
    #


def numpy_full(length: int, value: int, dtype: np.dtype):
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



def main():
    ...


if __name__ == "__main__":
    main()
