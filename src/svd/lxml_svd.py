from __future__ import annotations

import typing

import lxml.etree as ET
from lxml import objectify
from lxml.objectify import BoolElement, StringElement

from typing import Optional

from svd_enums import Access, CpuName, DataType, ReadAction, EndianType, SauAccess, AddressBlockUsage, Protection, EnumUsage, ModifiedWriteValues


class AccessElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> Access:
        return Access.from_str(self.text)


class ReadActionElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> ReadAction:
        return ReadAction.from_str(self.text)


class EndianTypeElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> EndianType:
        return EndianType.from_str(self.text)


class SauAccessElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> SauAccess:
        return SauAccess.from_str(self.text)


class AddressBlockUsageElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> AddressBlockUsage:
        return AddressBlockUsage.from_str(self.text)


class ProtectionElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> Protection:
        return Protection.from_str(self.text)


class EnumUsageElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> EnumUsage:
        return EnumUsage.from_str(self.text)


class ModifiedWriteValuesElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> ModifiedWriteValues:
        return ModifiedWriteValues.from_str(self.text)


class DataTypeElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> DataType:
        return DataType.from_str(self.text)


class CpuNameElement(objectify.ObjectifiedDataElement):
    @property
    def pyval(self) -> CpuName:
        return CpuName.from_str(self.text)


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


# TODO: implement derivedFrom as a property that looks up the relevant element using xpath


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

    @property
    def derivedFrom(self) -> Optional[PeripheralElement]:
        ...


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

    @property
    def derivedFrom(self) -> Optional[FieldElement]:
        ...


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

    @property
    def derivedFrom(self) -> Optional[RegisterElement]:
        ...


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

    @property
    def derivedFrom(self) -> Optional[ClusterElement]:
        ...


class RegistersElement(objectify.ObjectifiedElement):
    TAG = "registers"

    cluster: ClusterElement
    register: RegisterElement


class ParentChildTagLookup(ET.PythonElementClassLookup):
    def lookup(self, _document, element: ET.Element):
        if (parent := element.getparent()) is not None:
            parent_tag = parent.tag
        else:
            parent_tag = None
        result = _ELEMENT_CLASSES.get((parent_tag, element.tag))
        # print("lookup", (parent_tag, element.tag), "->", result)
        return result


def _add_element_class(element_class: type):
    element_class_tag = element_class.TAG
    for field_name, field_type in typing.get_type_hints(element_class).items():
        key = (element_class_tag, field_name)
        if key in _ELEMENT_CLASSES:
            print(f"Duplicate {key=}")
        _ELEMENT_CLASSES[key] = field_type


_ELEMENT_CLASSES = {
    (None, "device"): DeviceElement
}

for element_class in [RangeWriteConstraintElement, WriteConstraintElement, SauRegionElement, SauRegionsConfigElement, CpuElement, PeripheralElement, PeripheralsElement, DeviceElement, DimArrayIndexElement, EnumerationElement, EnumeratedValueElement, AddressBlockElement, InterruptElement, FieldElement, FieldsElement, RegisterElement, RegistersElement, ClusterElement]:
    _add_element_class(element_class)


def main():
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("svd_file", type=Path)
    args = p.parse_args()

    parser = objectify.makeparser(remove_comments=True)
    parser.set_element_class_lookup(ParentChildTagLookup())

    with open(args.svd_file, "r") as f:
        obj = objectify.parse(f, parser=parser)
        # obj = ET.parse(args.svd_file)

    #    result = schema.validate(obj)

    nodes = list(obj.getroot().iter())
    pass
    print(objectify.dump(obj.getroot()))

    """
    with open(args.svd_file, "r") as f:
        root = ET.parse(f).getroot()

    device = parse_device(root)
    print(device)
    """


if __name__ == "__main__":
    main()
