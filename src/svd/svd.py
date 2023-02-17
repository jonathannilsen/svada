#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#


import dataclasses as dc
import xml.etree.ElementTree as ET
from typing import Optional

from . import util


class XmlDeserialize:
    """Helper class for deserializing a dataclass from an XML element"""

    @classmethod
    def from_xml(cls, elem: ET.Element, base=None):
        """Construct class from an XML element"""

        init_values = {} if base is None else dc.asdict(base)
        for field in dc.fields(cls):
            if not field.metadata.get("xml", False):
                continue
            if "tag" in field.metadata:
                value = elem.findtext(field.metadata["tag"])
            elif "attrib" in field.metadata:
                value = elem.findtext(field.metadata["attrib"])
            else:
                raise NotImplementedError(
                    f'{cls.__qualname__}.{field.name} does not have a "tag" or "attrib" metadata'
                )
            if value is not None:
                if field.metadata.get("factory", None) is not None:
                    value = field.metadata["factory"](value)
                init_values[field.name] = value
        return cls(**init_values)


# xml_field with just factory to deserialize to object?

def xml_field(*, tag=None, attrib=None, factory=None, **kwargs):
    metadata = kwargs.get("metadata", {})
    metadata["xml"] = True
    if tag is not None:
        metadata["tag"] = tag
    elif attrib is not None:
        metadata["attrib"] = attrib
    else:
        raise ValueError('Either "tag" or "attrib" must be specified')
    metadata["factory"] = factory
    return dc.field(metadata=metadata, **kwargs)


@dc.dataclass
class RegisterPropertiesGroup(XmlDeserialize):
    """Container for SVD registerPropertiesGroup properties"""

    size: int = xml_field(tag="size", default=None, factory=util.to_int)
    access: str = xml_field(tag="access", default=None)
    reset_value: int = xml_field(tag="resetValue", default=None, factory=util.to_int)
    reset_mask: int = xml_field(tag="resetMask", default=None, factory=util.to_int)

    def is_valid(self):
        return all(
            v is not None
            for v in (self.size, self.access, self.reset_value, self.reset_mask)
        )


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


@dc.dataclass
class RegisterElementCommon(XmlDeserialize):
    """Container for relevant SVD register level attributes that are common to register and cluster"""

    name: str = xml_field(tag="name")
    address_offset: int = xml_field(tag="addressOffset", default=0, factory=util.to_int)


@dc.dataclass(frozen=True)
class RegisterElement:
    """ """
    common: RegisterElementCommon
    dim: DimElementGroup
    reg: RegisterPropertiesGroup


@dc.dataclass(frozen=True)
class Register:
    """Container for relevant fields in a SVD register node"""

    name: str = xml_field(tag="name")
    description: Optional[str] = xml_field(tag="description", default=None)
    address_offset: int = xml_field(tag="addressOffset", default=0, factory=util.to_int)
    alternate_cluster: str = xml_field(tag="alternateCluster", default=None)



    def __post_init__(self):
        assert self.dim.is_valid()
        assert self.reg.is_valid()


@dc.dataclass(frozen=True)
class SvdCluster(SvdRegisterElement):
    children: List[Union[SvdCluster, SvdRegister]]

    def __post_init__(self):
        assert self.dim.is_valid()


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

    @property
    def space_address(self):
        """Mask the region/secure part of address, returning only the address space part"""
        return self.base_address & ADDRESS_SPACE_MASK

    @property
    def is_global(self) -> bool:
        """Check if peripheral is in the global peripheral address space"""
        return any(self.base_address in space for space in GLOBAL_ADDRESS_RANGES)


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
                r, base_props=reg_props, base_offset=child_base_offset, parent_name=abs_name
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
            rewritable_field_mask=get_rewritable_field_mask(register, reg_props.access),
        )

    else:
        raise UnsupportedElementError(
            f"Unsupported register level element: {register.tag}"
        )


def parse_peripheral(
    peripheral: ET.Element,
    base_props: SvdRegisterPropertiesGroup,
) -> List[SvdRegisterElement]:
    reg_props = SvdRegisterPropertiesGroup.from_xml(peripheral, base=base_props)
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
                raise NotImplementedError("Merging of inherited registers is not implemented")
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


