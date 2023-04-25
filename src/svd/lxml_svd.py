from __future__ import annotations

import enum

import lxml.etree as ET
from lxml import objectify

# Use tag based namespace lookup for as much as possible
# use python element class lookup for the rest

_ELEMENT_TAGS = {}


class ParentChildTagLookup(ET.PythonElementClassLookup):
    def lookup(self, _document, element: ET.Element):
        if (parent := element.getparent()) is not None:
            parent_tag = parent.tag
        else:
            parent_tag = None
        print("lookup", (parent_tag, element.tag))
        return _ELEMENT_TAGS.get((parent_tag, element.tag))


_namespace_lookup = ET.ElementNamespaceClassLookup()
ns_elements = _namespace_lookup.get_namespace(None)


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


class AccessElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> Access:
        return Access.from_str(self.text)


@enum.unique
class ReadAction(_CaseInsensitiveStrEnum):
    CLEAR = "clear"
    SET = "set"
    MODIFY = "modify"
    MODIFY_EXTERNAL = "modifyExternal"


class ReadActionElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> ReadAction:
        return ReadAction.from_str(self.text)


@enum.unique
class EndianType(_CaseInsensitiveStrEnum):
    LITTLE = "little"
    BIG = "big"
    SELECTABLE = "selectable"
    OTHER = "other"


class EndianTypeElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> EndianType:
        return EndianType.from_str(self.text)


@enum.unique
class SauAccess(_CaseInsensitiveStrEnum):
    NON_SECURE = "n"
    SECURE_CALLABLE = "c"


class SauAccessElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> SauAccess:
        return SauAccess.from_str(self.text)


@enum.unique
class AddressBlockUsage(_CaseInsensitiveStrEnum):
    REGISTER = "registers"
    BUFFERS = "buffers"
    RESERVED = "reserved"


class AddressBlockUsageElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> AddressBlockUsage:
        return AddressBlockUsage.from_str(self.text)


@enum.unique
class Protection(_CaseInsensitiveStrEnum):
    SECURE = "s"
    NON_SECURE = "n"
    PRIVILEGED = "p"


class ProtectionElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> Protection:
        return Protection.from_str(self.text)


@enum.unique
class EnumUsage(_CaseInsensitiveStrEnum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read-write"


class EnumUsageElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> EnumUsage:
        return EnumUsage.from_str(self.text)


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


@ns_elements("modifiedWriteValues")
class ModifiedWriteValuesElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> ModifiedWriteValues:
        return ModifiedWriteValues.from_str(self.text)


class SvdIntElement(objectify.IntElement):
    def _init(self):
        self._setValueParser(self.to_int)

    @staticmethod
    def to_int(value: str) -> int:
        """Convert an SVD integer string to an int"""
        if value.startswith("0x"):
            return int(value, base=16)
        if value.startswith("#"):
            return int(value[1:], base=2)
        return int(value)


@ns_elements("region")
class SauRegion(ET.ElementBase):
    ...


@ns_elements("sauRegionsConfig")
class SauRegionsConfig(ET.ElementBase):
    ...


@ns_elements("cpu")
class Cpu(ET.ElementBase):
    ...


@ns_elements("peripherals")
class Peripherals(ET.ElementBase):
    ...


@ns_elements("device")
class DeviceElement(ET.ElementBase):
    ...



