from __future__ import annotations

import enum
from functools import cached_property
from typing import List, Optional

from .util import CaseInsensitiveStrEnum, BindingWrapper


@enum.unique
class Access(CaseInsensitiveStrEnum):
    READ_ONLY = "read-only"
    WRITE_ONLY = "write-only"
    READ_WRITE = "read-write"
    WRITE_ONCE = "writeOnce"
    READ_WRITE_ONCE = "read-writeOnce"


@enum.unique
class ReadAction(CaseInsensitiveStrEnum):
    CLEAR = "clear"
    SET = "set"
    MODIFY = "modify"
    MODIFY_EXTERNAL = "modifyExternal"


@enum.unique
class Endian(CaseInsensitiveStrEnum):
    LITTLE = "little"
    BIG = "big"
    SELECTABLE = "selectable"
    OTHER = "other"


@enum.unique
class SauAccess(CaseInsensitiveStrEnum):
    NON_SECURE = "n"
    SECURE_CALLABLE = "c"


@enum.unique
class AddressBlockUsage(CaseInsensitiveStrEnum):
    REGISTER = "registers"
    BUFFERS = "buffers"
    RESERVED = "reserved"


@enum.unique
class Protection(CaseInsensitiveStrEnum):
    SECURE = "s"
    NON_SECURE = "n"
    PRIVILEGED = "p"


@enum.unique
class EnumUsage(CaseInsensitiveStrEnum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read-write"


@enum.unique
class ModifiedWriteValues(CaseInsensitiveStrEnum):
    ONE_TO_CLEAR = "oneToClear"
    ONE_TO_SET = "oneToSet"
    ONE_TO_TOGGLE = "oneToToggle"
    ZERO_TO_CLEAR = "zeroToClear"
    ZERO_TO_SET = "zeroToSet"
    ZERO_TO_TOGGLE = "zeroToToggle"
    CLEAR = "clear"
    SET = "set"
    MODIFY = "modify"


@enum.unique
class DataType(CaseInsensitiveStrEnum):
    UINT8_T ="uint8_t"
    UINT16_T = "uint16_t"
    UINT32_T = "uint32_t"
    UINT64_T = "uint64_t"
    INT8_T = "int8_t"
    INT16_T = "int16_t"
    INT32_T = "int32_t"
    INT64_T = "int64_t"
    UINT8_PTR_T = "uint8_t *"
    UINT16_PTR_T = "uint16_t *"
    UINT32_PTR_T = "uint32_t *"
    UINT64_PTR_T = "uint64_t *"
    INT8_PTR_T = "int8_t *"
    INT16_PTR_T = "int16_t *"
    INT32_PTR_T = "int32_t *"
    INT64_PTR_T = "int64_t *"


@enum.unique
class CpuName(CaseInsensitiveStrEnum):
    CM0 = "CM0"
    CM0_PLUS_ = "CM0PLUS"
    CM0_PLUS = "CM0+"
    CM1 = "CM1"
    CM3 = "CM3"
    CM4 = "CM4"
    CM7 = "CM7"
    CM23 = "CM23"
    CM33 = "CM33"
    CM35P = "CM35P"
    CM55 = "CM55"
    CM85 = "CM85"
    SC000 = "SC000"
    SC300 = "SC300"
    ARMV8MML = "ARMV8MML"
    ARMV8MBL = "ARMV8MBL"
    ARMV81MML = "ARMV81MML"
    CA5 = "CA5"
    CA7 = "CA7"
    CA8 = "CA8"
    CA9 = "CA9"
    CA15 = "CA15"
    CA17 = "CA17"
    CA53 = "CA53"
    CA57 = "CA57"
    CA72 = "CA72"
    SMC1 = "SMC1"
    OTHER = "other"


class Cpu(BindingWrapper):
    @property
    def name(self) -> CpuName:
        return self._binding.name.pyval

    @property
    def revision(self) -> str:
        return self._binding.revision.pyval

    @property
    def endianness(self) -> Endian:
        return self._binding.endian.pyval

    @property
    def has_mpu(self) -> bool:
        return self._binding.mpuPresent.pyval

    @property
    def has_fpu(self) -> bool:
        return self._binding.fpuPresent.pyval

    @property
    def num_nvic_priority_bits(self) -> int:
        return self._binding.nvicPrioBits.pyval

    @property
    def has_vendor_systick(self) -> bool:
        return self._binding.vendorSystickConfig.pyval

    @property
    def num_interrupts(self) -> int:
        return self._binding.deviceNumInterrupts.pyval

    @property
    def num_sau_regions(self) -> int:
        if (prop := self._binding.sauNumRegions) is not None:
            return prop.pyval
        return 0

    # TODO
    # @property
    # def sau_region_configs(self) -> List[]


class RegisterProperties:
    def __init__(self, binding, base: Optional[RegisterProperties] = None):
        # TODO: DRY
        if (size := binding.size) is not None:
            self._size = size.pyval
        elif base is not None:
            self._size = base.size
        else:
            self._size = None

        if (access := binding.access) is not None:
            self._access = access.pyval
        elif base is not None:
            self._access = base.access
        else:
            self._access = None

        if (protection := binding.protection) is not None:
            self._protection = protection.pyval
        elif base is not None:
            self._protection = base.protection
        else:
            self._protection = None

        if (reset_value := binding.resetValue) is not None:
            self._reset_value = reset_value.pyval
        elif base is not None:
            self._reset_value = base.reset_value
        else:
            self._reset_value = None

        if (reset_mask := binding.resetMask) is not None:
            self._reset_mask = reset_mask.pyval
        elif base is not None:
            self._reset_mask = base.reset_mask
        else:
            self._reset_mask = None

    @property
    def size(self) -> Optional[int]:
        return self._size

    @property
    def access(self) -> Optional[Access]:
        return self._access

    @property
    def protection(self) -> Optional[Protection]:
        return self._protection

    @property
    def reset_value(self) -> Optional[int]:
        return self._reset_value

    @property
    def reset_mask(self) -> Optional[int]:
        return self._reset_mask

    def __eq__(self, other: RegisterProperties) -> bool:
        return (self.size == other.size and
                self.access == other.access and
                self.protection == other.protection and
                self.reset_value == other.reset_value and
                self.reset_mask == other.reset_mask)


class DimensionProperties(BindingWrapper):
    @property
    def length(self) -> int:
        if (dim_data := self._binding.dim) is not None:
            return dim_data.pyval
        return 1

    @property
    def step(self) -> int:
        if (dim_increment_data := self._binding.dim) is not None:
            return dim_increment_data.pyval
        return 0
