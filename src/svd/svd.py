#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import dataclasses as dc
import enum
import re
import typing
import xml.etree.ElementTree as ET
from itertools import chain
from typing import Dict, Generic, List, Optional, NamedTuple, NewType, Set, Tuple, TypeVar, Union

from deserialize import sdataclass, attr, elem


class CaseInsensitiveStrEnum(enum.Enum):
    @classmethod
    def from_str(cls, value: str) -> CaseInsensitiveStrEnum:
        value_lower = value.lower()
        for member in cls:
            if member.value.lower() == value_lower:
                return member
        raise ValueError(
            f"Class {cls.__qualname__} has no member corresponding to '{value}'")


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


@sdataclass(frozen=True)
class RegisterPropertiesGroup:
    """Container for SVD registerPropertiesGroup properties"""

    size: Optional[int] = elem("size", default=None)
    access: Optional[Access] = elem("access", default=None)
    reset_value: Optional[int] = elem("resetValue", default=None)
    reset_mask: Optional[int] = elem("resetMask", default=None)

    def is_valid(self):
        return all(
            v is not None
            for v in (self.size, self.access, self.reset_value, self.reset_mask)
        )


@sdataclass(frozen=True)
class BitRangeLsbMsbStyle:
    lsb: int = elem("lsb")
    msb: int = elem("msb")


@sdataclass(frozen=True)
class BitRangeOffsetWidthStyle:
    bit_offset: int = elem("bitOffset")
    bit_width: int = elem("bitWidth")

    def to_lsb_msb(self) -> BitRangeLsbMsbStyle:
        return BitRangeLsbMsbStyle(lsb=self.bit_offset, msb=self.bit_offset + self.bit_width - 1)


@sdataclass(frozen=True)
class BitRangePattern:
    bit_range: str = elem("bitRange")

    def to_lsb_msb(self) -> BitRangeLsbMsbStyle:
        match = re.fullmatch(r"\[(?P<lsb>[0-9]+):(?P<msb>[0-9]+)[\]")
        assert match is not None
        return BitRangeLsbMsbStyle(lsb=int(match["lsb"]), msb=int(match["msb"]))


@sdataclass(frozen=True)
class BitRange:
    """Container for SVD bitRange properties"""

    offset_width: Optional[BitRangeOffsetWidthStyle] = elem(".", default=None)
    lsb_msb: Optional[BitRangeLsbMsbStyle] = elem(".", default=None)
    pattern: Optional[BitRangePattern] = elem(".", default=None)

    # TODO: how to have a uniform interface
    # define a custom from_xml?

    def __post_init__(self):
        assert sum(1 for v in dc.astuple(self) if v is not None) == 1


@sdataclass(frozen=True)
class DimElementGroup:
    """Container for SVD dimElementGroup properties"""

    dim: Optional[int] = elem("dim", default=None)
    dim_increment: Optional[int] = elem("dimIncrement", default=None)

    def is_valid(self):
        return self.dim_increment is None or self.dim is not None

    def is_specified(self):
        return self.dim_increment is not None


@sdataclass(frozen=True)
class RangeWriteConstraint:
    minimum: int = elem("minimum")
    maximum: int = elem("maximum")


@sdataclass(frozen=True)
class WriteConstraint:
    """Write constraint for a register"""

    write_as_read: Optional[bool] = elem("writeAsRead", default=None)
    use_enumerated_values: Optional[bool] = elem(
        "useEnumeratedValues", default=None)
    range_write_constraint: Optional[RangeWriteConstraint] = elem(
        "range", default=None)

    def __post_init__(self):
        if sum(1 for v in dc.astuple(self) if v is not None) != 1:
            raise ValueError(f"Invalid {self.__class__.__name__}: {self}")


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


@sdataclass(frozen=True)
class Field:
    """Container for relevant fields in a SVD field node"""

    name: str = elem("name")
    bit_range: BitRange = elem(".")
    derived_from: Optional[Field] = attr("derivedFrom", default=None)
    description: Optional[str] = elem("description", default=None)
    access: Optional[Access] = elem("access", default=None)
    modified_write_values: Optional[str] = elem(
        "modifiedWriteValues", default=None)
    write_constraint: Optional[WriteConstraint] = elem(
        "writeConstraint", default=None)
    read_action: Optional[ReadAction] = elem("readAction", default=None)


@enum.unique
class EnumUsage(CaseInsensitiveStrEnum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read-write"


@sdataclass(frozen=True)
class EnumeratedValue:
    """Container for relevant fields in a SVD enumeratedValue node"""

    name: Optional[str] = elem("name", default=None)
    description: Optional[str] = elem("description", default=None)
    value: Optional[int] = elem("value", default=None)
    is_default: bool = elem("isDefault", default=False)

    def __post_init__(self):
        assert self.value is not None or self.is_default


@sdataclass(frozen=True)
class EnumeratedValues:
    """Container for relevant fields in a SVD enumeratedValues node"""

    name: str = elem("name")
    derived_from: Optional[EnumeratedValues] = attr(
        "derivedFrom", default=None)
    enumerated_value: List[EnumeratedValue] = elem(
        "enumeratedValue", default_factory=list)
    header_enum_name: Optional[str] = elem("headerEnumName", default=None)
    usage: EnumUsage = elem("usage", deafult=EnumUsage.READ_WRITE)


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


@sdataclass(frozen=True)
class AddressBlock:
    """Container for relevant fields in a SVD addressBlock node"""

    size: int = elem("size")
    offset: int = elem("offset", default=0)
    usage: Optional[AddressBlockUsage] = elem("usage", default=None)
    protection: Optional[Protection] = elem("protection", default=None)


@sdataclass(frozen=True)
class Interrupt:
    """Container for relevant fields in a SVD interrupt node"""

    name: str = elem("name")
    value: int = elem("value")
    description: Optional[str] = elem("description", default=None)


@sdataclass(frozen=True)
class Register:
    """Container for relevant fields in a SVD register node"""

    name: str = elem("name")
    dim: DimElementGroup = elem(".")
    reg: RegisterPropertiesGroup = elem(".")
    derived_from: Optional[Register] = attr("derivedFrom", default=None)
    display_name: Optional[str] = elem("displayName", default=None)
    description: Optional[str] = elem("description", default=None)
    alternate_group: Optional[str] = elem("alternateGroup", default=None)
    alternate_register: Optional[Register] = elem(
        "alternateRegister", default=None)
    header_struct_name: Optional[str] = elem("headerStructName", default=None)
    address_offset: int = elem("addressOffset", default=0)
    data_type: Optional[str] = elem("dataType", default=None)
    modified_write_values: Optional[str] = elem(
        "modifiedWriteValues", default=None)
    write_constraint: Optional[WriteConstraint] = elem(
        "writeConstraint", default=None)
    fields: List[Field] = elem("fields/field", default_factory=list)

    def __post_init__(self):
        pass
        #assert self.dim.is_valid()
        #assert self.reg.is_valid()


@sdataclass(frozen=True)
class Cluster:
    """Container for relevant fields in a SVD cluster node"""

    name: str = elem("name")
    dim: DimElementGroup = elem(".")
    reg: RegisterPropertiesGroup = elem(".")
    derived_from: Optional[Cluster] = attr("derivedFrom", default=None)
    description: Optional[str] = elem("description", default=None)
    alternate_cluster: Optional[Cluster] = elem(
        "alternateCluster", default=None)
    header_struct_name: Optional[str] = elem("headerStructName", default=None)
    address_offset: int = elem("addressOffset", default=0)

    register: List[Register] = elem("register", default_factory=list)
    cluster: List[Cluster] = elem("cluster", default_factory=list)

    @property
    def registers(self):
        return

    def __post_init__(self):
        assert self.dim.is_valid()


@sdataclass(frozen=True)
class Peripheral:
    """"""

    name: str = elem("name")
    base_address: int = elem("baseAddress")
    dim: DimElementGroup = elem(".")
    reg: RegisterPropertiesGroup = elem(".")
    registers: List[Union[Cluster, Register]] = elem(
        "registers//", default_factory=list)
    derived_from: Optional[Peripheral] = attr("derivedFrom", default=None)
    version: Optional[str] = elem("version", default=None)
    description: Optional[str] = elem("description", default=None)
    alternate_peripheral: Optional[Peripheral] = elem(
        "alternatePeripheral", default=None)
    group_name: Optional[str] = elem("groupName", default=None)
    prepend_to_name: Optional[str] = elem("prependToName", default=None)
    append_to_name: Optional[str] = elem("appendToName", default=None)
    header_struct_name: Optional[str] = elem("headerStructName", default=None)
    disable_condition: Optional[str] = elem("disableCondition", default=None)
    address_block: List[AddressBlock] = elem(
        "addressBlock", default_factory=list)
    interrupt: List[Interrupt] = elem("interrupt", default_factory=list)


@enum.unique
class EndianType(CaseInsensitiveStrEnum):
    LITTLE = "little"
    BIG = "big"
    SELECTABLE = "selectable"
    OTHER = "other"


@enum.unique
class SauAccess(CaseInsensitiveStrEnum):
    NON_SECURE = "n"
    SECURE_CALLABLE = "c"


@sdataclass(frozen=True)
class SauRegion:
    base: int = elem("base")
    limit: int = elem("limit")
    access: SauAccess = elem("access")
    enabled: bool = attr("enabled", default=True)
    name: Optional[str] = attr("name", default=None)


@sdataclass(frozen=True)
class SauRegionsConfig:
    enabled: bool = attr("enabled", default=True)
    protection_when_disabled: Optional[Protection] = attr(
        "protectionWhenDisabled", default=None)
    regions: List[SauRegion] = elem("region", default_factory=list)


@sdataclass(frozen=True)
class Cpu:
    name: str = elem("name")
    revision: str = elem("revision")
    endian: EndianType = elem("endian")
    mpu_present: bool = elem("mpuPresent")
    fpu_present: bool = elem("fpuPresent")
    nvic_prio_bits: int = elem("nvicPrioBits")
    vendor_systick_config: bool = elem("vendorSystickConfig")
    fpu_dp: bool = elem("fpuDP", default=False)
    icache_present: bool = elem("icachePresent", default=False)
    dcache_present: bool = elem("dcachePresent", default=False)
    itcm_present: bool = elem("ictmPresent", default=False)
    dtcm_present: bool = elem("dctmPresent", default=False)
    vtor_present: bool = elem("vtorPresent", default=True)
    device_num_interrupts: Optional[int] = elem(
        "deviceNumInterrupts", default=None)
    sau_num_regions: Optional[int] = elem("sauNumRegions", default=None)
    sau_regions_config: Optional[SauRegionsConfig] = elem(
        "sauRegionsConfig", default=None)


@sdataclass(frozen=True)
class Device:
    name: str = elem("name")
    version: str = elem("version")
    reg: RegisterPropertiesGroup = elem(".")
    peripherals: List[Peripheral] = elem("peripherals//", default_factory=list)
    vendor: Optional[str] = elem("vendor", default=None)
    vendor_id: Optional[str] = elem("vendorId", default=None)
    series: Optional[str] = elem("series", default=None)
    description: Optional[str] = elem("description", default=None)
    cpu: Optional[Cpu] = elem("cpu", default=None)
    header_system_filename: Optional[str] = elem(
        "headerSystemName", default=None)
    header_definitions_prefix: Optional[str] = elem(
        "headerDefinitionsPrefix", default=None)
    address_unit_bits: int = elem("addressUnitBits", default=8)
    width: int = elem("width", default=32)
    # TODO
    # vendor_extensions: List[ET.Element] = xml_field(
    #     path="./vendorExtensions//*", default_factory=list)


@enum.unique
class Source(enum.Enum):
    ATTR = enum.auto()
    ELEM = enum.auto()


class Spec(NamedTuple):
    source: Source
    path: str


def sdataclass(*args, **kwargs):
    return dc.dataclass(*args, **kwargs)


def sfield(_source: Source, _path: str, /, *args, **kwargs):
    spec = Spec(source=_source, path=_path)
    return dc.field(*args, **kwargs, metadata={"spec": spec})


def elem(_path: str, /, *args, **kwargs):
    return sfield(Source.ELEM, _path, *args, **kwargs)


def attr(_path: str, /, *args, **kwargs):
    return sfield(Source.ATTR, _path, *args, **kwargs)


def extract_children(elem: ET.Element, name: str) -> Optional[List[ET.Element]]:
    matches = elem.findall(name)
    return matches if matches else None


def extract_child(elem: ET.Element, name: str) -> Optional[ET.Element]:
    return elem.find(name)


def extract_element_text(elem: ET.Element) -> Optional[str]:
    return elem.text


def extract_attribute(elem: ET.Element, name: str) -> Optional[str]:
    return AttrNode(text=elem.attrib.get(name))

class AttrNode(NamedTuple):
    text: Optional[str]

def to_int(value: str) -> int:
    """Convert an SVD integer string to an int"""
    if value.startswith("0x"):
        return int(value, base=16)
    if value.startswith("#"):
        return int(value[1:], base=2)
    return int(value)


def to_bool(value: str) -> bool:
    """Convert an SVD boolean string to a bool"""
    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False
    raise ValueError(f"Invalid boolean value: {value}")


R = TypeVar("R")

def make_element_converter(result_type: type, simple_only: bool = False):
    if result_type is int:
        return lambda e: to_int(e.text)
    if result_type is bool:
        return lambda e: to_bool(e.text)
    if result_type is str:
        return lambda e: e.text
    if hasattr(result_type, "from_str"):
        return lambda e: result_type.from_str(e.text)
    if not simple_only and hasattr(result_type, "_svd_extractors"):
        return result_type.from_xml
    raise NotImplementedError(
        f"Conversion not implemented for type {result_type}")


def from_xml(cls, elem: ET.Element):
    svd_fields = {}
    for name, extract in cls._svd_extractors.items():
        if (value := extract(elem)) is not None:
            svd_fields[name] = value
    return cls(**svd_fields)


class TypeNode(NamedTuple):
    value: type
    children: List[TypeNode]


def normalize_type_tree(base):
    origin = typing.get_origin(base)
    if origin is None:
        return TypeNode(base, [])
    children = [normalize_type_tree(t) for t in typing.get_args(base)]
    return TypeNode(origin, children)


def extract_list(elem, field_name: str, converter):
    children = extract_children(elem, field_name)
    if not children:
        return None
    return [converter(c) for c in children]


def extract_base(elem, field_name: str, extractor, converter):
    child = extractor(elem, field_name)
    if child is None:
        return None
    return converter(child)


def make_extractor_elem(type_tree: TypeNode, field_name: str) -> ExtractFunction:
    if type_tree.value is list:
        type_tree = type_tree.children[0]
        extractor = ft.partial(extract_list, field_name=field_name)
    else:
        if type_tree.value is Union and type_tree.children[1].value is type(None):
            type_tree = type_tree.children[0]
        extractor = ft.partial(
            extract_base, field_name=field_name, extractor=extract_child)

    converter = make_element_converter(type_tree.value)
    extractor.keywords["converter"] = converter
    return extractor


def make_extractor_attr(type_tree: TypeNode, field_name: str) -> ExtractFunction:
    if type_tree.value is Union and type_tree.children[1].value is type(None):
        type_tree = type_tree.children[0]
    extractor = ft.partial(
        extract_base, field_name=field_name, extractor=extract_attribute)

    converter = make_element_converter(type_tree.value, simple_only=True)
    extractor.keywords["converter"] = converter
    return extractor


def _parse_object(cls, elem: ET.Element):
    type_hints = typing.get_type_hints(cls)

    for field in dc.fields(cls):
        field_spec: Optional[Spec] = field.metadata.get("spec")
        if field_spec is None:
            continue

        type_tree = normalize_type_tree(type_hints[field.name])

        if field_spec.source == Source.ELEM:
            pass

        elif field.spec.source == Source.ATTR:
            pass

        else:
            raise ValueError("Invalid source")


def parse_device(root: ET.Element):
    inherit: Dict[Tuple[type, str], Set[str]] = {}
    device = _parse_object(Device, root)
    # TODO: inheritance
    return device

    
def main():
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("svd_file", type=Path)
    args = p.parse_args()

    with open(args.svd_file, "r") as f:
        root = ET.parse(f).getroot()

    device = Device.from_xml(root)
    print(device)


if __name__ == "__main__":
    main()
