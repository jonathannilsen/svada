#
# Copyright (c) 2023 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from typing import Optional, Tuple

from lxml import objectify
from lxml.objectify import BoolElement, StringElement

from . import svd_enums
from . import util


# TODO: attributes
# TODO: validation
# TODO: assert that element class is found

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
    def pyval(self) -> svd_enums.EndianType:
        return svd_enums.EndianType.from_str(self.text)


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


class RangeWriteConstraintElement(objectify.ObjectifiedElement):
    TAG = "range"

    minimum: SvdIntElement
    maximum: SvdIntElement


class WriteConstraintElement(objectify.ObjectifiedElement):
    TAG = "writeConstraint"

    writeAsRead: BoolElement
    useEnumeratedValues: BoolElement
    range: RangeWriteConstraintElement


class SauRegionElement(objectify.ObjectifiedElement):
    TAG = "region"

    base: SvdIntElement
    limit: SvdIntElement
    access: SauAccessElement
    enabled: BoolElement


class SauRegionsConfigElement(objectify.ObjectifiedElement):
    TAG = "sauRegions"

    region: SauRegionElement


class CpuElement(objectify.ObjectifiedElement):
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


class PeripheralElement(objectify.ObjectifiedElement):
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
    dimIndex: SvdIntElement
    dimName: StringElement
    dimArrayIndex: DimArrayIndexElement

    registers: RegistersElement

    def find_derived_from(self) -> Optional[PeripheralElement]:
        derived_from = self.get("derivedFrom")
        if derived_from is None:
            return None
        peripherals = self.getparent()
        base_peripheral = peripherals.xpath(f"//peripheral[name='{derived_from}']")
        if not base_peripheral:
            raise LookupError(f"Did not find a peripheral with name'{derived_from}'")
        return base_peripheral[0]

    def _init(self):
        self.reverse_lookup = None


class PeripheralsElement(objectify.ObjectifiedElement):
    TAG = "peripherals"

    peripheral: PeripheralElement


class DeviceElement(objectify.ObjectifiedElement):
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


class DimArrayIndexElement(objectify.ObjectifiedElement):
    TAG = "dimArrayIndex"

    headerEnumName: StringElement
    enumeratedValue: EnumeratedValueElement


class EnumerationElement(objectify.ObjectifiedElement):
    TAG = "enumeratedValues"

    name: StringElement
    headerEnumName: StringElement
    usage: EnumUsageElement
    enumeratedValue: EnumeratedValueElement


class EnumeratedValueElement(objectify.ObjectifiedElement):
    TAG = "enumeratedValue"

    name: StringElement
    description: StringElement
    value: SvdIntElement
    isDefault: BoolElement


class AddressBlockElement(objectify.ObjectifiedElement):
    TAG = "addressBlock"

    offset: SvdIntElement
    size: SvdIntElement
    usage: AddressBlockUsageElement
    protection: ProtectionElement


class InterruptElement(objectify.ObjectifiedElement):
    TAG = "interrupt"

    name: StringElement
    description: StringElement
    value: SvdIntElement


class BitRangeElement(StringElement):
    TAG = "bitRange"

    ...


class FieldElement(objectify.ObjectifiedElement):
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


class FieldsElement(objectify.ObjectifiedElement):
    TAG = "fields"

    field: FieldElement


class RegisterElement(objectify.ObjectifiedElement):
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


class ClusterElement(objectify.ObjectifiedElement):
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


class RegistersElement(objectify.ObjectifiedElement):
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
