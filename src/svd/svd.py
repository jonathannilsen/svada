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
from itertools import chain
from typing import Dict, Generic, List, Optional, NewType, TypeVar, Union

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
