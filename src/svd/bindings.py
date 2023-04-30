#
# Copyright (c) 2023 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from typing import Any, Optional, Tuple

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



class _XmlElementBinding(objectify.ObjectifiedElement):
    def __getattr__(self, name: str) -> Any:
        """
        Get the attribute with the given name.
        Overridden to make it so properties that have annotations in the class
        default to returning None rather than raising an exception.
        """
        try:
            # TODO: consider whether this should just return the pyval
            return super().__getattr__(name)
        except AttributeError:
            if name not in self.__annotations__:
                raise
            return None


class RangeWriteConstraintElement(_XmlElementBinding):
    TAG = "range"

    minimum: SvdIntElement
    maximum: SvdIntElement


class WriteConstraintElement(_XmlElementBinding):
    TAG = "writeConstraint"

    writeAsRead: BoolElement
    useEnumeratedValues: BoolElement
    range: RangeWriteConstraintElement


class SauRegionElement(_XmlElementBinding):
    TAG = "region"

    base: SvdIntElement
    limit: SvdIntElement
    access: SauAccessElement
    enabled: BoolElement


class SauRegionsConfigElement(_XmlElementBinding):
    TAG = "sauRegions"

    region: SauRegionElement


class CpuElement(_XmlElementBinding):
    TAG = "cpu"

    name: CpuNameElement
    revision: StringElement
    endian: EndianTypeElement
    mpuPresent: BoolElement
    fpuPresent: BoolElement
    fpuDP: BoolElement
    dspPresent: BoolElement
    icachePresent: BoolElement
    dcachePresent: BoolElement
    itcmPresent: BoolElement
    dtcmPresent: BoolElement
    vtorPresent: BoolElement
    nvicPrioBits: SvdIntElement
    vendorSystickConfig: BoolElement
    deviceNumInterrupts: SvdIntElement
    sauNumRegions: SvdIntElement
    sauRegionsConfig: SauRegionsConfigElement


class PeripheralElement(_XmlElementBinding):
    TAG = "peripheral"

    name: StringElement
    version: StringElement
    description: StringElement
    alternatePeripheral: StringElement
    groupName: StringElement
    prependToName: StringElement
    appendToName: StringElement
    headerStructName: StringElement
    disableCondition: StringElement
    baseAddress: SvdIntElement
    addressBlock: AddressBlockElement
    interrupt: InterruptElement

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

    def _init(self):
        self.reverse_lookup = None


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

    def _init(self):
        self.reverse_lookup = None


class DimArrayIndexElement(_XmlElementBinding):
    TAG = "dimArrayIndex"

    headerEnumName: StringElement
    enumeratedValue: EnumeratedValueElement


class EnumerationElement(_XmlElementBinding):
    TAG = "enumeratedValues"

    name: StringElement
    headerEnumName: StringElement
    usage: EnumUsageElement
    enumeratedValue: EnumeratedValueElement


class EnumeratedValueElement(_XmlElementBinding):
    TAG = "enumeratedValue"

    name: StringElement
    description: StringElement
    value: SvdIntElement
    isDefault: BoolElement


class AddressBlockElement(_XmlElementBinding):
    TAG = "addressBlock"

    offset: SvdIntElement
    size: SvdIntElement
    usage: AddressBlockUsageElement
    protection: ProtectionElement


class InterruptElement(_XmlElementBinding):
    TAG = "interrupt"

    name: StringElement
    description: StringElement
    value: SvdIntElement


class BitRangeElement(StringElement):
    TAG = "bitRange"

    ...


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
