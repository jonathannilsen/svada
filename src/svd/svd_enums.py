from __future__ import annotations

import enum


class _CaseInsensitiveStrEnum(enum.Enum):
    @classmethod
    def from_str(cls, value: str) -> _CaseInsensitiveStrEnum:
        value_lower = value.lower()
        for member in cls:
            if member.value.lower() == value_lower:
                return member
        raise ValueError(
            f"Class {cls.__qualname__} has no member corresponding to '{value}'"
        )


@enum.unique
class Access(_CaseInsensitiveStrEnum):
    READ_ONLY = "read-only"
    WRITE_ONLY = "write-only"
    READ_WRITE = "read-write"
    WRITE_ONCE = "writeOnce"
    READ_WRITE_ONCE = "read-writeOnce"


@enum.unique
class ReadAction(_CaseInsensitiveStrEnum):
    CLEAR = "clear"
    SET = "set"
    MODIFY = "modify"
    MODIFY_EXTERNAL = "modifyExternal"


@enum.unique
class EndianType(_CaseInsensitiveStrEnum):
    LITTLE = "little"
    BIG = "big"
    SELECTABLE = "selectable"
    OTHER = "other"


@enum.unique
class SauAccess(_CaseInsensitiveStrEnum):
    NON_SECURE = "n"
    SECURE_CALLABLE = "c"


@enum.unique
class AddressBlockUsage(_CaseInsensitiveStrEnum):
    REGISTER = "registers"
    BUFFERS = "buffers"
    RESERVED = "reserved"


@enum.unique
class Protection(_CaseInsensitiveStrEnum):
    SECURE = "s"
    NON_SECURE = "n"
    PRIVILEGED = "p"


@enum.unique
class EnumUsage(_CaseInsensitiveStrEnum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read-write"


@enum.unique
class ModifiedWriteValues(_CaseInsensitiveStrEnum):
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
class DataType(_CaseInsensitiveStrEnum):
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
class CpuName(_CaseInsensitiveStrEnum):
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
