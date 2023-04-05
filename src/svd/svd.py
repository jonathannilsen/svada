#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import dataclasses as dc
import enum
import re
import xml.etree.ElementTree as ET
from typing import Dict, Generic, List, Optional, NewType, TypeVar

from deserialize import Attr, Elem, svd_dataclass


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
class Source(enum.Enum):
    ATTR = enum.auto()
    ELEM = enum.auto()


def sfield(_source: Source, _path: str, /, *args, **kwargs):
    pass


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


@svd_dataclass(frozen=True)
class RegisterPropertiesGroup:
    """Container for SVD registerPropertiesGroup properties"""

    size: Optional[int] = sfield(Source.ELEM, "size", default=None)
    access: Optional[Access] = sfield(Source.ELEM, "access", default=None)
    reset_value: Optional[int] = sfield(
        Source.ELEM, "resetValue", default=None)
    reset_mask: Optional[int] = sfield(Source.ELEM, "resetMask", default=None)

    def is_valid(self):
        return all(
            v is not None
            for v in (self.size, self.access, self.reset_value, self.reset_mask)
        )


@svd_dataclass(frozen=True)
class RegisterPropertiesGroup:
    """Container for SVD registerPropertiesGroup properties"""

    size: Elem[Optional[int]] = None
    access: Elem[Optional[Access]] = None
    reset_value: Elem[Optional[int]] = None
    reset_mask: Elem[Optional[int]] = None

    def is_valid(self):
        return all(
            v is not None
            for v in (self.size, self.access, self.reset_value, self.reset_mask)
        )


@svd_dataclass(frozen=True)
class BitRangeLsbMsbStyle:
    lsb: Elem[int]  # = xml_field(tag="lsb", factory=util.to_int)
    msb: Elem[int]  # = xml_field(tag="msb", factory=util.to_int)


@svd_dataclass(frozen=True)
class BitRangeOffsetWidthStyle:
    bit_offset: Elem[int]  # = xml_field(tag="bitOffset", factory=util.to_int)
    bit_width: Elem[int]  # = xml_field(tag="bitWidth", factory=util.to_int)

    def to_lsb_msb(self) -> BitRangeLsbMsbStyle:
        return BitRangeLsbMsbStyle(lsb=self.bit_offset, msb=self.bit_offset + self.bit_width - 1)


@svd_dataclass(frozen=True)
class BitRangePattern:
    bit_range: Elem[str]  # = xml_field(tag="bitRange")

    def to_lsb_msb(self) -> BitRangeLsbMsbStyle:
        match = re.fullmatch(r"\[(?P<lsb>[0-9]+):(?P<msb>[0-9]+)[\]")
        assert match is not None
        return BitRangeLsbMsbStyle(lsb=int(match["lsb"]), msb=int(match["msb"]))


@svd_dataclass(frozen=True)
class BitRange:
    """Container for SVD bitRange properties"""

    offset_width: Elem[Optional[BitRangeOffsetWidthStyle], "."] = None  # = xml_field(
    #    factory=BitRangeOffsetWidthStyle.from_xml, default=None
    # )
    lsb_msb: Elem[Optional[BitRangeLsbMsbStyle], "."] = None  # = xml_field(
    #    factory=BitRangeLsbMsbStyle.from_xml, default=None
    # )
    pattern: Elem[Optional[BitRangePattern], "."] = None  # xml_field(
    #    factory=BitRangePattern.from_xml, default=None
    # )

    # TODO: how to have a uniform interface

    def __post_init__(self):
        assert sum(1 for v in dc.astuple(self) if v is not None) == 1


"""
@svd_dataclass(frozen=True)
class BitRange:

    # = xml_field(tag="bitOffset", factory=util.to_int)
    bit_offset: Elem[Optional[int]]
    # int = xml_field(tag="bitWidth", factory=util.to_int)
    bit_width: Elem[Optional[int]]
    lsb: Elem[Optional[int]]  # = xml_field(tag="lsb", factory=util.to_int)
    msb: Elem[Optional[int]]  # = xml_field(tag="msb", factory=util.to_int)
    bit_range: Elem[Optional[str]]  # = xml_field(tag="bitRange")

    def __post_init__(self):
        if self.bit_range is not None:
            match = re.fullmatch(
                r"\[(?P<lsb>[0-9]+):(?P<msb>[0-9]+)[\]", self.bit_range)
            assert match is not None
            self.lsb = int(match["lsb"])
            self.msb = int(match["msb"])
            self.bit_offset = self.lsb
            self.bit_width = self.msb - self.lsb + 1
        elif self.lsb is not None and self.msb is not None:
            self.bit_offset = self.lsb
            self.bit_width = self.msb - self.lsb + 1
            self.bit_range = f"{self.lsb}:{self.msb}"
        elif self.bit_offset is not None and self.bit_width is not None:
            self.lsb = self.bit_offset
            self.msb = self.lsb + self.bit_width - 1
            self.bit_range = f"{self.lsb}:{self.msb}"
        else:
            raise ValueError(f"Invalid {self.__class__.__name__}: {self}")
"""


@svd_dataclass(frozen=True)
class DimElementGroup:
    """Container for SVD dimElementGroup properties"""

    # = xml_field(tag="dim", default=None, factory=util.to_int)
    dim: Elem[Optional[int]] = None
    # = xml_field(    tag="dimIncrement", default=None, factory=util.to_int)
    dim_increment: Elem[Optional[int]] = None

    def is_valid(self):
        return self.dim_increment is None or self.dim is not None

    def is_specified(self):
        return self.dim_increment is not None


@svd_dataclass(frozen=True)
class RangeWriteConstraint:
    minimum: Elem[int]  # = xml_field(tag="minimum", factory=util.to_int)
    maximum: Elem[int]  # = xml_field(tag="maximum", factory=util.to_int)


@svd_dataclass(frozen=True)
class WriteConstraint:
    """Write constraint for a register"""

    # = xml_field(tag="writeAsRead", default=None)
    write_as_read: Elem[Optional[bool]] = None
    # = xml_field(tag="useEnumeratedValues", default=None)
    use_enumerated_values: Elem[Optional[bool]] = None
    range_write_constraint: Elem[Optional[RangeWriteConstraint], "range"] = None  # = xml_field(

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


@svd_dataclass(frozen=True)
class Field:
    """Container for relevant fields in a SVD field node"""

    name: Elem[str]  # = xml_field(tag="name")
    bit_range: Elem[BitRange, "."]  # = xml_field(factory=BitRange.from_xml)

    # = xml_field(attrib="derivedFrom", default=None)
    derived_from: Attr[Optional[str]] = None

    # = xml_field(tag="description", default=None)
    description: Elem[Optional[str]] = None
    # = xml_field(tag="access", default=None)
    access: Elem[Optional[Access]] = None
    modified_write_values: Elem[Optional[str]] = None  # = xml_field(
    #    tag="modifiedWriteValues", default=None
    # )
    write_constraint: Elem[Optional[WriteConstraint]] = None  # = xml_field(
    #     factory=WriteConstraint.from_xml, default=None
    # )
    # = xml_field(tag="readAction", default=None)
    read_action: Elem[Optional[ReadAction]] = None


@enum.unique
class EnumUsage(CaseInsensitiveStrEnum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read-write"


@svd_dataclass(frozen=True)
class EnumeratedValue:
    """Container for relevant fields in a SVD enumeratedValue node"""

    name: Elem[Optional[str]] = None  # = xml_field(tag="name", default=None)
    # = xml_field(tag="description", default=None)
    description: Elem[Optional[str]] = None
    value: Elem[Optional[int]] = None  # = xml_field(
    #    tag="value", default=None, factory=util.to_int)
    is_default: Elem[bool] = False  # = xml_field(
    #    tag="isDefault", default=False, factory=util.to_bool)

    def __post_init__(self):
        assert self.value is not None or self.is_default


@svd_dataclass(frozen=True)
class EnumeratedValues:
    """Container for relevant fields in a SVD enumeratedValues node"""

    name: Elem[str]  # = xml_field(tag="name")

    enumerated_value: Elem[List[EnumeratedValue]]  # = xml_field(
    #    tag="enumeratedValue",
    #    multiple=True,
    #    factory=EnumeratedValue.from_xml,
    #    default_factory=list,
    # )
    derived_from: Attr[Optional[str]] = None

    header_enum_name: Elem[Optional[str]] = None  # = xml_field(
    #    tag="headerEnumName", default=None)
    usage: Elem[EnumUsage] = EnumUsage.READ_WRITE  # = xml_field(
    #    tag="usage", default=EnumUsage.READ_WRITE, factory=EnumUsage.from_str
    # )


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


@svd_dataclass(frozen=True)
class AddressBlock:
    """Container for relevant fields in a SVD addressBlock node"""

    # = xml_field(tag="offset", default=0, factory=util.to_int)
    size: Elem[int]  # = xml_field(tag="size", factory=util.to_int)
    offset: Elem[int] = 0
    # = xml_field(tag="usage", default=None)
    usage: Elem[Optional[AddressBlockUsage]] = None
    protection: Elem[Optional[Protection]] = None  # = xml_field(
    #    tag="protection", default=None)


@svd_dataclass(frozen=True)
class Interrupt:
    """Container for relevant fields in a SVD interrupt node"""

    name: Elem[str]  # = xml_field(tag="name")
    value: Elem[int]  # = xml_field(tag="value", factory=util.to_int)
    # = xml_field(tag="description", default=None)
    description: Elem[Optional[str]] = None


@svd_dataclass(frozen=True)
class Fields:
    fields: Elem[List[Field], "field"] = dc.field(default_factory=list)


@svd_dataclass(frozen=True)
class Register:
    """Container for relevant fields in a SVD register node"""

    name: Elem[str]
    dim: Elem[DimElementGroup, "."]
    reg: Elem[RegisterPropertiesGroup, "."]
    derived_from: Attr[Optional[str]] = None
    display_name: Elem[Optional[str]] = None
    description: Elem[Optional[str]] = None
    alternate_group: Elem[Optional[str]] = None
    alternate_register: Elem[Optional[str]] = None
    header_struct_name: Elem[Optional[str]] = None
    address_offset: Elem[int] = 0
    data_type: Elem[Optional[str]] = None
    modified_write_values: Elem[Optional[str]] = None
    write_constraint: Elem[Optional[WriteConstraint]] = None
    fields: Elem[Fields] = dc.field(default_factory=Fields)

    def __post_init__(self):
        pass
        #assert self.dim.is_valid()
        #assert self.reg.is_valid()


@svd_dataclass(frozen=True)
class Cluster:
    """Container for relevant fields in a SVD cluster node"""

    name: Elem[str]  # = xml_field(tag="name")
    # = xml_field(factory=DimElementGroup.from_xml)
    dim: Elem[DimElementGroup, "."]
    reg: Elem[RegisterPropertiesGroup, "."]  # = xml_field(
    # = xml_field(tag="description", default=None)
    derived_from: Attr[Optional[str]] = None
    description: Elem[Optional[str]] = None
    # = xml_field(tag="alternateCluster", default=None)
    alternate_cluster: Elem[Optional[str]] = None
    # = xml_field(tag="headerStructName", default=None)
    header_struct_name: Elem[Optional[str]] = None
    # = xml_field(tag="addressOffset", default=0, factory=util.to_int)
    address_offset: Elem[int] = 0
    #    factory=RegisterPropertiesGroup.from_xml
    # )
    # = xml_field(tag="register", multiple=True, factory=Register.from_xml, default_factory=list)
    registers: Elem[List[Register], "register"] = dc.field(
        default_factory=list)
    # = xml_field(tag="cluster", multiple=True, factory=Cluster.from_xml, default_factory=list)
    # FIXME: how to do this recursive reference??
    #clusters: Elem[List[Cluster], "cluster"] = dc.field(default_factory=list)

    def __post_init__(self):
        assert self.dim.is_valid()


# FIXME: can this class be made obsolete?
@svd_dataclass(frozen=True)
class Registers:
    clusters: Elem[List[Cluster], "cluster"] = dc.field(default_factory=list)
    registers: Elem[List[Register], "register"] = dc.field(
        default_factory=list)


@svd_dataclass(frozen=True)
class Peripheral:
    """"""

    name: Elem[str]
    base_address: Elem[int]
    dim: Elem[DimElementGroup, "."]
    reg: Elem[RegisterPropertiesGroup, "."]
    registers: Elem[Registers] = dc.field(default_factory=Registers)

    derived_from: Attr[Optional[str]] = None
    version: Elem[Optional[str]] = None
    description: Elem[Optional[str]] = None
    alternate_peripheral: Elem[Optional[str]] = None
    group_name: Elem[Optional[str]] = None
    prepend_to_name: Elem[Optional[str]] = None
    append_to_name: Elem[Optional[str]] = None
    header_struct_name: Elem[Optional[str]] = None
    disable_condition: Elem[Optional[str]] = None
    address_block: Elem[List[AddressBlock]] = dc.field(default_factory=list)
    interrupt: Elem[List[Interrupt]] = dc.field(default_factory=list)


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


@svd_dataclass(frozen=True)
class SauRegion:
    base: Elem[int]
    limit: Elem[int]
    access: Elem[SauAccess]
    enabled: Attr[bool] = True
    name: Attr[Optional[str]] = None


@svd_dataclass(frozen=True)
class SauRegionsConfig:
    enabled: Attr[bool] = True
    protection_when_disabled: Attr[Optional[Protection]] = None
    regions: Elem[List[SauRegion], "region"] = dc.field(default_factory=list)


@svd_dataclass(frozen=True)
class Cpu:
    name: Elem[str]
    revision: Elem[str]
    endian: Elem[EndianType]
    mpu_present: Elem[bool]
    fpu_present: Elem[bool]
    nvic_prio_bits: Elem[int]
    vendor_systick_config: Elem[bool]
    # FIXME
    fpu_dp: Elem[bool, "fpuDP"] = False
    icache_present: Elem[bool] = False
    dcache_present: Elem[bool] = False
    itcm_present: Elem[bool] = False
    dtcm_present: Elem[bool] = False
    vtor_present: Elem[bool] = True
    device_num_interrupts: Elem[Optional[int]] = None
    sau_num_regions: Elem[Optional[int]] = None
    sau_regions_config: Elem[Optional[SauRegionsConfig]] = None


# FIXME: can this class be made obsolete?
@svd_dataclass(frozen=True)
class Peripherals:
    peripherals: Elem[List[Peripheral],
                      "peripheral"] = dc.field(default_factory=list)


@svd_dataclass(frozen=True)
class Device:

    name: Elem[str]
    version: Elem[str]
    reg: Elem[RegisterPropertiesGroup, "."]
    peripherals: Elem[Peripherals]
    vendor: Elem[Optional[str]] = None
    vendor_id: Elem[Optional[str]] = None
    series: Elem[Optional[str]] = None
    description: Elem[Optional[str]] = None
    cpu: Elem[Optional[Cpu]] = None
    header_system_filename: Elem[Optional[str]] = None
    header_definitions_prefix: Elem[Optional[str]] = None
    address_unit_bits: Elem[int] = 8
    width: Elem[int] = 32
    # TODO
    # vendor_extensions: List[ET.Element] = xml_field(
    #     path="./vendorExtensions//*", default_factory=list)


def parse_device(root: ET.Element):
    inherit_graphs: Dict[type, Dict[str, str]] = {}


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
