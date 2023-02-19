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
from typing import Generic, List, Optional, NewType, TypeVar

from deserialize import Attr, Elem, svd_object


class Access(enum.StrEnum):
    READ_ONLY = "read-only"
    READ_WRITE = "read-write"
    WRITE_ONCE = "writeOnce"
    READ_WRITE_ONCE = "read-writeOnce"

    def from_str(cls, value: str):
        return cls(value.lower())


class ReadAction(enum.StrEnum):
    CLEAR = "clear"
    SET = "set"
    MODIFY = "modify"
    MODIFY_EXTERNAL = "modifyExternal"

    def from_str(cls, value: str):
        return cls(value.lower())


@svd_object
@dc.dataclass(frozen=True)
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


@dc.dataclass(frozen=True)
class BitRangeOffsetWidthStyle(XmlDeserialize):
    bit_offset: int = xml_field(tag="bitOffset", factory=util.to_int)
    bit_width: int = xml_field(tag="bitWidth", factory=util.to_int)


@dc.dataclass(frozen=True)
class BitRangeLsbMsbStyle(XmlDeserialize):
    lsb: int = xml_field(tag="lsb", factory=util.to_int)
    msb: int = xml_field(tag="msb", factory=util.to_int)


@dc.dataclass(frozen=True)
class BitRangePattern(XmlDeserialize):
    bit_range: str = xml_field(tag="bitRange")

    lsb: int = dc.field(init=False)
    msb: int = dc.field(init=False)

    def __post_init__(self):
        match = re.fullmatch(r"\[(?P<lsb>[0-9]+):(?P<msb>[0-9]+)[\]")
        assert match is not None
        self.lsb = int(match["lsb"])
        self.msb = int(match["msb"])


@dc.dataclass(frozen=True)
class BitRange(XmlDeserialize):
    """Container for SVD bitRange properties"""

    offset_width: Optional[BitRangeOffsetWidthStyle] = xml_field(
        factory=BitRangeOffsetWidthStyle.from_xml, default=None
    )
    lsb_msb: Optional[BitRangeLsbMsbStyle] = xml_field(
        factory=BitRangeLsbMsbStyle.from_xml, default=None
    )
    pattern: Optional[BitRangePattern] = xml_field(
        factory=BitRangePattern.from_xml, default=None
    )

    def __post_init__(self):
        assert sum(1 for v in dc.astuple(self) if v is not None) == 1


@dc.dataclass
class DimElementGroup(XmlDeserialize):
    """Container for SVD dimElementGroup properties"""

    dim: int = xml_field(tag="dim", default=None, factory=util.to_int)
    dim_increment: int = xml_field(
        tag="dimIncrement", default=None, factory=util.to_int
    )

    def is_valid(self):
        return self.dim_increment is None or self.dim is not None

    def is_specified(self):
        return self.dim_increment is not None


@dc.dataclass(frozen=True)
class RangeWriteConstraint:
    minimum: int = xml_field(tag="minimum", factory=util.to_int)
    maximum: int = xml_field(tag="maximum", factory=util.to_int)


@dc.dataclass(frozen=True)
class WriteConstraint(XmlDeserialize):
    """Write constraint for a register"""

    write_as_read: Optional[bool] = xml_field(tag="writeAsRead", default=None)
    use_enumerated_values: Optional[bool] = xml_field(
        tag="useEnumeratedValues", default=None
    )
    range_constraint: Optional[RangeWriteConstraint] = xml_field(
        factory=RangeWriteConstraint.from_xml, default=None
    )

    def __post_init__(self):
        assert sum(1 for v in dc.astuple(self) if v is not None) == 1


@dc.dataclass(frozen=True)
class Cluster(XmlDeserialize):
    """Container for relevant fields in a SVD cluster node"""

    name: str = xml_field(tag="name")
    description: Optional[str] = xml_field(tag="description", default=None)
    alternate_cluster: Optional[str] = xml_field(
        tag="alternateCluster", default=None)
    header_struct_name: Optional[str] = xml_field(
        tag="headerStructName", default=None)
    address_offset: int = xml_field(
        tag="addressOffset", default=0, factory=util.to_int)
    dim: DimElementGroup = xml_field(factory=DimElementGroup.from_xml)
    reg_prop: RegisterPropertiesGroup = xml_field(
        factory=RegisterPropertiesGroup.from_xml
    )
    register: List[Register] = xml_field(
        tag="register", multiple=True, factory=Register.from_xml, default_factory=list
    )
    cluster: List[Cluster] = xml_field(
        tag="cluster", multiple=True, factory=Cluster.from_xml, default_factory=list
    )

    def __post_init__(self):
        assert self.dim.is_valid()


class ModifiedWriteValues(enum.StrEnum):
    ONE_TO_CLEAR = "oneToClear"
    ONE_TO_SET = "oneToSet"
    ONE_TO_TOGGLE = "oneToToggle"
    ZERO_TO_CLEAR = "zeroToClear"
    ZERO_TO_SET = "zeroToSet"
    ZERO_TO_TOGGLE = "zeroToToggle"
    CLEAR = "clear"
    SET = "set"
    MODIFY = "modify"


@dc.dataclass(frozen=True)
class Register(XmlDeserialize):
    """Container for relevant fields in a SVD register node"""

    name: str = xml_field(tag="name")
    display_name: Optional[str] = xml_field(tag="displayName", default=None)
    description: Optional[str] = xml_field(tag="description", default=None)
    alternate_group: Optional[str] = xml_field(
        tag="alternateGroup", default=None)
    alternate_register: Optional[str] = xml_field(
        tag="alternateRegister", default=None)
    header_struct_name: Optional[str] = xml_field(
        tag="headerStructName", default=None)
    address_offset: int = xml_field(
        tag="addressOffset", default=0, factory=util.to_int)
    dim: DimElementGroup = xml_field(factory=DimElementGroup.from_xml)
    reg_prop: RegisterPropertiesGroup = xml_field(
        factory=RegisterPropertiesGroup.from_xml
    )
    data_type: Optional[str] = xml_field(tag="dataType", default=None)
    modified_write_values: Optional[str] = xml_field(
        tag="modifiedWriteValues", default=None
    )
    write_constraint: Optional[WriteConstraint] = xml_field(
        factory=WriteConstraint.from_xml, default=None
    )
    fields: List[Field] = xml_field(
        tag="field", multiple=True, factory=Field.from_xml, default_factory=list
    )

    def __post_init__(self):
        assert self.dim.is_valid()
        assert self.reg_prop.is_valid()


@dc.dataclass(frozen=True)
class Field(XmlDeserialize):
    """Container for relevant fields in a SVD field node"""

    derived_from: Optional[str] = xml_field(attrib="derivedFrom", default=None)

    name: str = xml_field(tag="name")
    description: Optional[str] = xml_field(tag="description", default=None)
    bit_range: BitRange = xml_field(factory=BitRange.from_xml)
    access: Optional[Access] = xml_field(tag="access", default=None)
    modified_write_values: Optional[str] = xml_field(
        tag="modifiedWriteValues", default=None
    )
    write_constraint: Optional[WriteConstraint] = xml_field(
        factory=WriteConstraint.from_xml, default=None
    )
    read_action: Optional[ReadAction] = xml_field(
        tag="readAction", default=None)


class EnumUsage(enum.Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read-write"

    @classmethod
    def from_str(cls, value: str) -> EnumUsage:
        return cls(value.lower())


@dc.dataclass(frozen=True)
class EnumeratedValue(XmlDeserialize):
    """Container for relevant fields in a SVD enumeratedValue node"""

    name: Optional[str] = xml_field(tag="name", default=None)
    description: Optional[str] = xml_field(tag="description", default=None)
    value: Optional[int] = xml_field(
        tag="value", default=None, factory=util.to_int)
    is_default: bool = xml_field(
        tag="isDefault", default=False, factory=util.to_bool)

    def __post_init__(self):
        assert self.value is not None or self.is_default


@dc.dataclass(frozen=True)
class EnumeratedValues(XmlDeserialize):
    """Container for relevant fields in a SVD enumeratedValues node"""

    name: str = xml_field(tag="name")
    header_enum_name: Optional[str] = xml_field(
        tag="headerEnumName", default=None)
    usage: EnumUsage = xml_field(
        tag="usage", default=EnumUsage.READ_WRITE, factory=EnumUsage.from_str
    )
    enumerated_value: List[EnumeratedValue] = xml_field(
        tag="enumeratedValue",
        multiple=True,
        factory=EnumeratedValue.from_xml,
        default_factory=list,
    )


class AddressBlockUsage(enum.Enum):
    REGISTER = "register"
    BUFFERS = "buffers"
    RESERVED = "reserved"

    @classmethod
    def from_str(cls, value: str) -> AddressBlockUsage:
        return cls(value.lower())


class Protection(enum.Enum):
    SECURE = "s"
    NON_SECURE = "n"
    PRIVILEGED = "p"

    @classmethod
    def from_str(cls, value: str) -> Protection:
        return cls(value.lower())


@dc.dataclass(frozen=True)
class AddressBlock(XmlDeserialize):
    """Container for relevant fields in a SVD addressBlock node"""

    offset: int = xml_field(tag="offset", default=0, factory=util.to_int)
    size: int = xml_field(tag="size", factory=util.to_int)
    usage: Optional[AddressBlockUsage] = xml_field(tag="usage", default=None)
    protection: Optional[Protection] = xml_field(
        tag="protection", default=None)


@dc.dataclass(frozen=True)
class Interrupt(XmlDeserialize):
    """Container for relevant fields in a SVD interrupt node"""

    name: str = xml_field(tag="name")
    description: Optional[str] = xml_field(tag="description", default=None)
    value: int = xml_field(tag="value", factory=util.to_int)


@dc.dataclass(frozen=True)
class Peripheral(XmlDeserialize):
    """"""

    derived_from: Optional[str] = xml_field(attrib="derivedFrom", default=None)

    name: str = xml_field(tag="name")
    version: Optional[str] = xml_field(tag="version", default=None)
    description: Optional[str] = xml_field(tag="description", default=None)
    alternate_peripheral: Optional[str] = xml_field(
        tag="alternatePeripheral", default=None
    )
    group_name: Optional[str] = xml_field(tag="groupName", default=None)
    prepend_to_name: Optional[str] = xml_field(
        tag="prependToName", default=None)
    append_to_name: Optional[str] = xml_field(tag="appendToName", default=None)
    header_struct_name: Optional[str] = xml_field(
        tag="headerStructName", default=None)
    disable_condition: Optional[str] = xml_field(
        tag="disableCondition", default=None)
    base_address: int = xml_field(tag="baseAddress", factory=util.to_int)
    dim: DimElementGroup = xml_field(factory=DimElementGroup.from_xml)
    reg_prop: RegisterPropertiesGroup = xml_field(
        factory=RegisterPropertiesGroup.from_xml
    )
    address_block: List[AddressBlock] = xml_field(
        factory=AddressBlock.from_xml, multiple=True, default_factory=list
    )
    interrupt: List[Interrupt] = xml_field(
        factory=Interrupt.from_xml, multiple=True, default_factory=list
    )
    registers: List[Register] = xml_field(
        factory=Register.from_xml, multiple=True, default_factory=list
    )


class EndianType(enum.Enum):
    LITTLE = "little"
    BIG = "big"
    SELECTABLE = "selectable"
    OTHER = "other"

    @classmethod
    def from_str(cls, value: str) -> EndianType:
        return cls(value.lower())


class SauAccess(enum.StrEnum):
    NON_SECURE = "n"
    SECURE_CALLABLE = "c"


@dc.dataclass(frozen=True)
class SauRegion(XmlDeserialize):
    enabled: bool = xml_field(attr="enabled", default=True, factory=util.to_bool)
    name: Optional[str] = xml_field(attr="name", default=None)

    base: int = xml_field(tag="base", factory=util.to_int)
    limit: int = xml_field(tag="limit", factory=util.to_int)
    access: SauAccess = xml_field(tag="access", factory=SauAccess)


@dc.dataclass(frozen=True)
class SauRegionsConfig(XmlDeserialize):
    enabled: bool = xml_field(attr="enabled", factory=util.to_bool, default=True)
    protection_when_disabled: Optional[str] = xml_field(attr="protectionWhenDisabled", default=None)
    region: List[SauRegion] = xml_field(tag="region", factory=SauRegion.from_xml, default_factory=list)


@dc.dataclass(frozen=True)
class Cpu(XmlDeserialize):
    name: str = xml_field(tag="name")
    revision: str = xml_field(tag="revision")
    endian: EndianType = xml_field(tag="endian", factory=EndianType.from_str)
    mpu_present: bool = xml_field(tag="mpuPresent", factory=util.to_bool)
    fpu_present: bool = xml_field(tag="fpuPresent", factory=util.to_bool)
    fpu_dp: bool = xml_field(tag="fpuDP", factory=util.to_bool, default=False)
    icache_present: bool = xml_field(
        tag="icachePresent", factory=util.to_bool, default=False)
    dcache_present: bool = xml_field(
        tag="dcachePresent", factory=util.to_bool, default=False)
    itcm_present: bool = xml_field(
        tag="itcmPresent", factory=util.to_bool, default=False)
    dtcm_present: bool = xml_field(
        tag="dtcmPresent", factory=util.to_bool, default=False)
    vtor_present: bool = xml_field(
        tag="vtorPresent", factory=util.to_bool, default=True)
    nvic_prio_bits: int = xml_field(tag="nvicPrioBits", factory=util.to_int)
    vendor_systick_config: bool = xml_field(
        tag="vendorSystickConfig", factory=util.to_bool)
    device_num_interrupts: Optional[int] = xml_field(
        tag="deviceNumInterrupts", factory=util.to_int, default=None)
    sau_num_regions: Optional[int] = xml_field(
        tag="sauNumRegions", factory=util.to_int, default=None)
    sau_regions_config: Optional[SauRegionsConfig] = xml_field(
        "sauRegionsConfig", factory=SauRegionsConfig.from_xml, default=None)


@dc.dataclass(frozen=True)
class Device(XmlDeserialize):

    vendor: Optional[str] = xml_field(tag="vendor", default=None)
    vendor_id: Optional[str] = xml_field(tag="vendorID", default=None)
    name: str = xml_field(tag="name")
    series: Optional[str] = xml_field(tag="series", default=None)
    version: str = xml_field(tag="version")
    description: Optional[str] = xml_field(tag="description", default=None)
    cpu: Optional[Cpu] = xml_field(factory=Cpu.from_xml, default=None)
    header_system_filename: Optional[str] = xml_field(
        tag="headerSystemFilename", default=None)
    header_definitions_prefix: Optional[str] = xml_field(
        tag="headerDefinitionsPrefix", default=None)
    address_unit_bits: int = xml_field(
        tag="addressUnitBits", default=8, factory=util.to_int)
    width: int = xml_field(tag="width", default=32, factory=util.to_int)
    reg_props: RegisterPropertiesGroup = xml_field(
        factory=RegisterPropertiesGroup.from_xml)
    peripherals: List[Peripheral] = xml_field(
        factory=Peripheral.from_xml, multiple=True, default_factory=list)
    # TODO
    # vendor_extensions: List[ET.Element] = xml_field(
    #     path="./vendorExtensions//*", default_factory=list)


@dc.dataclass
class SvdPeripheralInfo(XmlDeserialize):
    """Container for relevant fields in a SVD peripheral node"""

    name: str = xml_field(tag="name")
    base_address: int = xml_field(tag="baseAddress", factory=parse_svd_int)

    _header_struct_name: str = xml_field(default=None, tag="headerStructName")
    derived_from: SvdPeripheralInfo = dc.field(default=None)

    def __hash__(self):
        return hash((self.name, self.base_address))

    @property
    def header_struct_name(self) -> str:
        if self._header_struct_name is None:
            return self.derived_from.header_struct_name
        return self._header_struct_name

    @property
    def is_derived(self) -> bool:
        return self.derived_from is not None


def parse_register(
    register: ET.Element,
    base_props: SvdRegisterPropertiesGroup,
    base_offset: int = 0,
    parent_name="",
):
    common = SvdRegisterElementCommon.from_xml(register)
    dim_props = SvdDimElementGroup.from_xml(register)
    reg_props = SvdRegisterPropertiesGroup.from_xml(register, base=base_props)

    abs_name = make_absolute_name(common.name, parent_name=parent_name)

    if (derived_from := register.findtext("derivedFrom")) is not None:
        print(
            f"WARNING: register element {abs_name} derivedFrom={derived_from} is ignored"
        )

    if register.tag == "cluster":
        child_base_offset = base_offset + common.offset
        children = [
            parse_register(
                r,
                base_props=reg_props,
                base_offset=child_base_offset,
                parent_name=abs_name,
            )
            for r in register.findall("*/[name]")
        ]
        cluster_base_offset = base_offset
        return SvdCluster(
            common,
            dim_props,
            reg_props,
            base_offset=cluster_base_offset,
            parent_name=parent_name,
            children=children,
        )

    elif register.tag == "register":
        return SvdRegister(
            common,
            dim_props,
            reg_props,
            base_offset=base_offset,
            parent_name=parent_name,
            rewritable_field_mask=get_rewritable_field_mask(
                register, reg_props.access),
        )

    else:
        raise UnsupportedElementError(
            f"Unsupported register level element: {register.tag}"
        )


def parse_peripheral(
    peripheral: ET.Element,
    base_props: SvdRegisterPropertiesGroup,
) -> List[SvdRegisterElement]:
    reg_props = SvdRegisterPropertiesGroup.from_xml(
        peripheral, base=base_props)
    return [
        parse_register(r, base_props=reg_props)
        for r in peripheral.findall("./registers/")
    ]


def parse_device_peripherals(
    device: ET.Element, peripheral_info: Dict[str, SvdPeripheralInfo]
) -> Dict[SvdPeripheralInfo, List[SvdRegisterElement]]:
    result = {}
    reg_props = SvdRegisterPropertiesGroup.from_xml(device)

    for peripheral in device.findall("peripherals/peripheral"):
        info = peripheral_info[peripheral.findtext("name")]
        result[info] = parse_peripheral(peripheral, base_props=reg_props)

    for info in result:
        if info.is_derived:
            if result[info]:
                raise NotImplementedError(
                    "Merging of inherited registers is not implemented"
                )
            result[info] = deepcopy(result[info.derived_from])

    return result


def parse_peripheral_info(device: ET.Element) -> Dict[str, SvdPeripheralInfo]:
    """Parse the peripherals in device to a list of SvdPeripheralInfo"""
    peripherals = {}
    peripheral_derived_from = {}
    for peripheral in device.findall("peripherals/peripheral"):
        info = SvdPeripheralInfo.from_xml(peripheral)
        if "derivedFrom" in peripheral.attrib:
            peripheral_derived_from[info.name] = peripheral.attrib["derivedFrom"]
        peripherals[info.name] = info
    for name, peripheral in peripherals.items():
        if name in peripheral_derived_from:
            peripheral.derived_from = peripherals[
                peripheral_derived_from[peripheral.name]
            ]
    return peripherals


@dc.dataclass
class SvdDevice:
    name: str
    peripherals: Dict[str, SvdPeripheralInfo]
    registers: Dict[SvdPeripheralInfo, List[SvdRegisterElement]]


def parse_svd(device: ET.Element):
    name = device.findtext("name")
    peripherals = parse_peripheral_info(device)
    registers = parse_device_peripherals(device, peripherals)
    return SvdDevice(name, peripherals, registers)
