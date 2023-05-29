#
# Copyright (c) 2023 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Optional, Tuple, get_type_hints

from lxml import objectify
from lxml.objectify import BoolElement, StringElement

from . import svd_enums
from . import util


# TODO: attributes
# TODO: validation
# TODO: assert that element class is found

class _XmlDataElementBinding(objectify.ObjectifiedDataElement):
    def get_pyval(self, default: Optional[Any] = None) -> Any:
        ...


class AccessElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> svd_enums.Access:
        return svd_enums.Access.from_str(self.text)


class ReadActionElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> svd_enums.ReadAction:
        return svd_enums.ReadAction.from_str(self.text)


class EndianTypeElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> svd_enums.Endian:
        return svd_enums.Endian.from_str(self.text)


class SauAccessElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> svd_enums.SauAccess:
        return svd_enums.SauAccess.from_str(self.text)


class AddressBlockUsageElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> svd_enums.AddressBlockUsage:
        return svd_enums.AddressBlockUsage.from_str(self.text)


class ProtectionElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> svd_enums.Protection:
        return svd_enums.Protection.from_str(self.text)


class EnumUsageElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> svd_enums.EnumUsage:
        return svd_enums.EnumUsage.from_str(self.text)


class ModifiedWriteValuesElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> svd_enums.ModifiedWriteValues:
        return svd_enums.ModifiedWriteValues.from_str(self.text)


class DataTypeElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> svd_enums.DataType:
        return svd_enums.DataType.from_str(self.text)


class CpuNameElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> svd_enums.CpuName:
        return svd_enums.CpuName.from_str(self.text)


class SvdIntElement(objectify.IntElement):
    def _init(self):
        self._setValueParser(util.to_int)


class _Missing:
    ...


MISSING = _Missing


@dataclass
class _SvdPropertyInfo:
    name: str
    klass: type
    default: Any


class _Elem(_SvdPropertyInfo):
    ...


class _Attr(_SvdPropertyInfo):
    ...


def elem(name: str, klass: type = MISSING, /, *, default: Optional[Any] = MISSING, ) -> _Elem:
    return _Elem(name, klass, default)


def attr(name: str, klass: type = MISSING, /, *, default: Optional[Any] = MISSING,) -> _Attr:
    return _Attr(name, klass, default)


def binding(klass: type) -> type:
    dict_arg = {}
    for field_name, _field_type in get_type_hints(klass).items():
        try:
            prop_info = getattr(klass, field_name)
        except AttributeError:
            continue
        if isinstance(prop_info, _Elem):
            def getter(self):
                try:
                    svd_obj = getattr(super(klass, self), prop_info.name)
                except AttributeError:
                    if prop_info.default == MISSING:
                        raise
                    return prop_info.default

                if issubclass(prop_info.klass, objectify.ObjectifiedDataElement):
                    return svd_obj.pyval
                else:
                    return svd_obj

        elif isinstance(prop_info, _Attr):
            raise NotImplementedError("Attributes are not implemented")
        else:
            continue

        dict_arg[field_name] = property(fget=getter)

    new_klass = type(
        klass.__name__, (objectify.ObjectifiedElement, klass), dict_arg)
    functools.update_wrapper(new_klass, klass)

    return new_klass


@binding
class RangeWriteConstraintElement:
    TAG = "range"

    minimum: int = elem("minimum", SvdIntElement)
    maximum: int = elem("maximum", SvdIntElement)


@binding
class WriteConstraintElement:
    TAG = "writeConstraint"

    writeAsRead: bool = elem("writeAsRead", BoolElement)
    useEnumeratedValues: bool = elem("useEnumeratedValues", BoolElement)
    range: RangeWriteConstraintElement = elem(
        "range", RangeWriteConstraintElement)


@binding
class SauRegionElement:
    TAG = "region"

    base: int = elem("base", SvdIntElement)
    limit: int = elem("limit", SvdIntElement)
    access: SauAccessElement = elem("access", SauAccessElement)
    enabled: bool = elem("enabled", BoolElement, default=True)


@binding
class SauRegionsConfigElement:
    TAG = "sauRegions"

    region: SauRegionElement = elem("region", SauRegionElement)


@binding
class CpuElement:
    TAG = "cpu"

    name: CpuNameElement = elem("name", CpuNameElement)
    revision: str = elem("revision", StringElement)
    endian: EndianTypeElement = elem("endian", EndianTypeElement)
    mpuPresent: Optional[bool] = elem("mpuPresent", BoolElement, default=None)
    fpuPresent: Optional[bool] = elem("fpuPresent", BoolElement, default=None)
    fpuDP: Optional[bool] = elem("fpuDP", BoolElement, default=None)
    dspPresent: Optional[bool] = elem("dspPresent", BoolElement, default=None)
    icachePresent: Optional[bool] = elem(
        "icachePresent", BoolElement, default=None)
    dcachePresent: Optional[bool] = elem(
        "dcachePresent", BoolElement, default=None)
    itcmPresent: Optional[bool] = elem(
        "itcmPresent", BoolElement, default=None)
    dtcmPresent: Optional[bool] = elem(
        "dtcmPresent", BoolElement, default=None)
    vtorPresent: Optional[bool] = elem(
        "vtorPresent", BoolElement, default=None)
    nvicPrioBits: int = elem("nvicPrioBits", SvdIntElement)
    vendorSystickConfig: bool = elem("vendorSystickConfig", BoolElement)
    deviceNumInterrupts: Optional[int] = elem(
        "deviceNumInterrupts", SvdIntElement, default=None)
    sauNumRegions: Optional[int] = elem(
        "sauNumRegions", SvdIntElement, default=None)
    sauRegionsConfig: Optional[SauRegionsConfigElement] = elem(
        "sauRegionsConfig", SauRegionsConfigElement, default=None)


@binding
class AddressBlockElement:
    TAG = "addressBlock"

    offset: int = elem("offset", SvdIntElement)
    size: int = elem("size", SvdIntElement)
    usage: AddressBlockUsageElement = elem("usage", AddressBlockUsageElement)
    protection: Optional[ProtectionElement] = elem(
        "protection", ProtectionElement)


@binding
class EnumeratedValueElement:
    TAG = "enumeratedValue"

    name: str = elem("name", StringElement)
    description: Optional[str] = elem(
        "description", StringElement, default=None)
    value: int = elem("value", SvdIntElement)
    isDefault: bool = elem("isDefault", BoolElement, default=False)


@binding
class EnumerationElement:
    TAG = "enumeratedValues"

    name: Optional[str] = elem("name", StringElement, default=None)
    headerEnumName: Optional[str] = elem(
        "headerEnumName", StringElement, default=None)
    usage: Optional[EnumUsageElement] = elem(
        "usage", EnumUsageElement, default=None)
    enumeratedValue: EnumeratedValueElement = elem(
        "enumeratedValue", EnumeratedValueElement)


@binding
class DimArrayIndexElement:
    TAG = "dimArrayIndex"

    headerEnumName: Optional[str] = elem(
        "headerEnumName", StringElement, default=None)
    enumeratedValue: EnumeratedValueElement = elem(
        "enumeratedValue", EnumeratedValueElement)


class InterruptElement(_XmlElementBinding):
    TAG = "interrupt"

    name: StringElement
    description: StringElement
    value: SvdIntElement


class BitRangeElement(StringElement):
    TAG = "bitRange"

    ...


@binding
class PeripheralElement:
    TAG = "peripheral"

    name: str = elem("name", StringElement)
    version: Optional[str] = elem("version", StringElement, default=None)
    description: Optional[str] = elem(
        "description", StringElement, default=None)
    alternatePeripheral: Optional[str] = elem(
        "alternatePeripheral", StringElement, default=None)
    groupName: Optional[str] = elem("groupName", StringElement, default=None)
    prependToName: Optional[str] = elem(
        "prependToName", StringElement, default=None)
    appendToName: Optional[str] = elem(
        "appendToName", StringElement, default=None)
    headerStructName: Optional[str] = elem(
        "headerStructName", StringElement, default=None)
    disableCondition: Optional[str] = elem(
        "disableCondition", StringElement, default=None)
    baseAddress: int = elem("baseAddress", SvdIntElement)
    addressBlock: AddressBlockElement = elem(
        "addressBlock", AddressBlockElement)
    interrupt: Optional[InterruptElement] = elem(
        "interrupt", InterruptElement, default=None)

    size: SvdIntElement
    access: AccessElement
    protection: ProtectionElement
    resetValue: SvdIntElement
    resetMask: SvdIntElement

    dim: SvdIntElement
    dimIncrement: SvdIntElement
    dimIndex: StringElement
    dimName: StringElement
    dimArrayIndex: DimArrayIndexElement

    registers: RegistersElement


class PeripheralsElement(_XmlElementBinding):
    TAG = "peripherals"

    peripheral: PeripheralElement


class DeviceElement(_XmlElementBinding):
    TAG = "device"

    vendor: StringElement
    vendorID: StringElement
    name: StringElement
    series: StringElement
    version: StringElement
    description: StringElement
    licenseText: StringElement
    cpu: CpuElement
    headerSystemFilename: StringElement
    headerDefinitionsPrefix: StringElement
    addressUnitBits: SvdIntElement
    width: SvdIntElement

    size: SvdIntElement
    access: AccessElement
    protection: ProtectionElement
    resetValue: SvdIntElement
    resetMask: SvdIntElement

    peripherals: PeripheralsElement


class FieldElement(_XmlElementBinding):
    TAG = "field"

    name: StringElement
    description: StringElement

    lsb: SvdIntElement
    msb: SvdIntElement

    bitOffset: SvdIntElement
    bitWidth: SvdIntElement

    bitRange: BitRangeElement

    access: AccessElement
    modifiedWriteValues: ModifiedWriteValuesElement
    writeConstraint: WriteConstraintElement
    readAction: ReadActionElement

    enumeratedValues: EnumerationElement

    def get_bit_range(self) -> Tuple[int, int]:
        """
        Get the bit range of a field.

        :param field: ElementTree representation of an SVD Field.

        :return: Tuple of the field's bit offset, and its bit width.
        """

        if hasattr(self, "lsb"):
            return self.lsb.pyval, self.msb.pyval - self.lsb.pyval + 1

        if hasattr(self, "bitOffset"):
            width = self.width.pyval if hasattr(self, "bitWidth") else 32
            return self.bitOffset.pyval, width

        if hasattr(self, "bitRange"):
            msb_string, lsb_string = self.bitRange.pyval[1:-1].split(":")
            msb, lsb = util.to_int(msb_string), util.to_int(lsb_string)
            return (lsb, msb - lsb + 1)

        return 0, 32


class FieldsElement(_XmlElementBinding):
    TAG = "fields"

    field: FieldElement


class RegisterElement(_XmlElementBinding):
    TAG = "register"

    name: StringElement
    displayName: StringElement
    description: StringElement
    alternateGroup: StringElement
    alternateRegister: StringElement
    addressOffset: SvdIntElement

    dim: SvdIntElement
    dimIncrement: SvdIntElement
    dimIndex: SvdIntElement
    dimName: StringElement
    dimArrayIndex: DimArrayIndexElement

    size: SvdIntElement
    access: AccessElement
    protection: ProtectionElement
    resetValue: SvdIntElement
    resetMask: SvdIntElement

    dataType: DataTypeElement
    modifiedWriteValues: ModifiedWriteValuesElement
    writeConstraint: WriteConstraintElement
    readAction: ReadActionElement

    fields: FieldsElement


class ClusterElement(_XmlElementBinding):
    TAG = "cluster"

    name: StringElement
    description: StringElement
    alternateCluster: StringElement
    headerStructName: StringElement
    addressOffset: SvdIntElement

    dim: SvdIntElement
    dimIncrement: SvdIntElement
    dimIndex: SvdIntElement
    dimName: StringElement
    dimArrayIndex: DimArrayIndexElement

    size: SvdIntElement
    access: AccessElement
    protection: ProtectionElement
    resetValue: SvdIntElement
    resetMask: SvdIntElement

    register: RegisterElement
    cluster: ClusterElement


class RegistersElement(_XmlElementBinding):
    TAG = "registers"

    cluster: ClusterElement
    register: RegisterElement


ELEMENT_CLASSES = [
    AddressBlockElement,
    BitRangeElement,
    ClusterElement,
    CpuElement,
    DeviceElement,
    DimArrayIndexElement,
    EnumerationElement,
    EnumeratedValueElement,
    FieldElement,
    FieldsElement,
    InterruptElement,
    PeripheralElement,
    PeripheralsElement,
    RangeWriteConstraintElement,
    RegisterElement,
    RegistersElement,
    SauRegionElement,
    SauRegionsConfigElement,
    WriteConstraintElement,
]
