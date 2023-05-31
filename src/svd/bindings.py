#
# Copyright (c) 2023 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

"""
"Low-level" read-only Python representation of the SVD format.
Each type of XML element the SVD XML tree is represented by a class in this module.
The class properties correspond more or less directly to the XML elements/attributes,
with some abstractions and simplifications added for convenience.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import (
    Iterator,
    NamedTuple,
    Optional,
    Sequence,
    Union,
)

from lxml import objectify
from lxml.objectify import BoolElement, StringElement

from .util import (
    CaseInsensitiveStrEnum,
    attr,
    binding,
    elem,
    enum_wrapper,
    iter_children,
    to_bool,
    to_int,
    SELF_CLASS,
)


# Container for classes that represent non-leaf elements in the SVD XML tree.
ELEMENT_CLASSES = []


class SvdIntElement(objectify.IntElement):
    """
    Element containing an SVD integer value.
    This class uses a custom parser to convert the value to an integer.
    """

    def _init(self):
        self._setValueParser(to_int)


@enum.unique
class Access(CaseInsensitiveStrEnum):
    READ_ONLY = "read-only"
    WRITE_ONLY = "write-only"
    READ_WRITE = "read-write"
    WRITE_ONCE = "writeOnce"
    READ_WRITE_ONCE = "read-writeOnce"


AccessElement = enum_wrapper(Access)


@enum.unique
class ReadAction(CaseInsensitiveStrEnum):
    CLEAR = "clear"
    SET = "set"
    MODIFY = "modify"
    MODIFY_EXTERNAL = "modifyExternal"


ReadActionElement = enum_wrapper(ReadAction)


@enum.unique
class Endian(CaseInsensitiveStrEnum):
    LITTLE = "little"
    BIG = "big"
    SELECTABLE = "selectable"
    OTHER = "other"


EndianElement = enum_wrapper(Endian)


@enum.unique
class SauAccess(CaseInsensitiveStrEnum):
    NON_SECURE = "n"
    SECURE_CALLABLE = "c"


SauAccessElement = enum_wrapper(SauAccess)


@enum.unique
class AddressBlockUsage(CaseInsensitiveStrEnum):
    REGISTER = "registers"
    BUFFERS = "buffers"
    RESERVED = "reserved"


AddressBlockUsageElement = enum_wrapper(AddressBlockUsage)


@enum.unique
class Protection(CaseInsensitiveStrEnum):
    SECURE = "s"
    NON_SECURE = "n"
    PRIVILEGED = "p"


ProtectionElement = enum_wrapper(Protection)


@enum.unique
class EnumUsage(CaseInsensitiveStrEnum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read-write"


EnumUsageElement = enum_wrapper(EnumUsage)


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


ModifiedWriteValuesElement = enum_wrapper(ModifiedWriteValues)


@enum.unique
class DataType(CaseInsensitiveStrEnum):
    UINT8_T = "uint8_t"
    UINT16_T = "uint16_t"
    UINT32_T = "uint32_t"
    UINT64_T = "uint64_t"
    INT8_T = "int8_t"
    INT16_T = "int16_t"
    INT32_T = "int32_t"
    INT64_T = "int64_t"
    UINT8_PTR_T = "uint8_t *"
    UINT16_PTR_T = "uint16_t *"
    UINT32_PTR_T = "uint32_t *"
    UINT64_PTR_T = "uint64_t *"
    INT8_PTR_T = "int8_t *"
    INT16_PTR_T = "int16_t *"
    INT32_PTR_T = "int32_t *"
    INT64_PTR_T = "int64_t *"


DataTypeElement = enum_wrapper(DataType)


@enum.unique
class CpuName(CaseInsensitiveStrEnum):
    CM0 = "CM0"
    CM0_PLUS_ = "CM0PLUS"
    CM0_PLUS = "CM0+"
    CM1 = "CM1"
    CM3 = "CM3"
    CM4 = "CM4"
    CM7 = "CM7"
    CM23 = "CM23"
    CM33 = "CM33"
    CM35P = "CM35P"
    CM55 = "CM55"
    CM85 = "CM85"
    SC000 = "SC000"
    SC300 = "SC300"
    ARMV8MML = "ARMV8MML"
    ARMV8MBL = "ARMV8MBL"
    ARMV81MML = "ARMV81MML"
    CA5 = "CA5"
    CA7 = "CA7"
    CA8 = "CA8"
    CA9 = "CA9"
    CA15 = "CA15"
    CA17 = "CA17"
    CA53 = "CA53"
    CA57 = "CA57"
    CA72 = "CA72"
    SMC1 = "SMC1"
    OTHER = "other"


CpuNameElement = enum_wrapper(CpuName)


@binding(ELEMENT_CLASSES)
class RangeWriteConstraint(objectify.ObjectifiedElement):
    TAG = "range"

    minimum: int = elem("minimum", SvdIntElement)
    maximum: int = elem("maximum", SvdIntElement)


@binding(ELEMENT_CLASSES)
class WriteConstraint(objectify.ObjectifiedElement):
    TAG = "writeConstraint"

    write_as_read: bool = elem("writeAsRead", BoolElement)
    use_enumerated_values: bool = elem("useEnumeratedValues", BoolElement)
    ranges: RangeWriteConstraint = elem("range", RangeWriteConstraint)


@binding(ELEMENT_CLASSES)
class SauRegion(objectify.ObjectifiedElement):
    TAG = "region"

    enabled: bool = attr("enabled", converter=to_bool, default=True)
    name: Optional[str] = attr("name", default=None)

    base: int = elem("base", SvdIntElement)
    limit: int = elem("limit", SvdIntElement)
    access: SauAccess = elem("access", SauAccessElement)


@binding(ELEMENT_CLASSES)
class SauRegionsConfig(objectify.ObjectifiedElement):
    TAG = "sauRegions"

    enabled: bool = attr("enabled", converter=to_bool, default=True)
    protection_when_disabled: Optional[Protection] = attr(
        "name", converter=Protection.from_str, default=None
    )

    def __iter__(self) -> Iterator[SauRegion]:
        """Iterate over all preset SAU regions."""
        return iter_children(self, SauRegion.TAG)

    _region: SauRegion = elem("region", SauRegion)


@binding(ELEMENT_CLASSES)
class Cpu(objectify.ObjectifiedElement):
    TAG = "cpu"

    name: CpuName = elem("name", CpuNameElement)
    revision: str = elem("revision", StringElement)
    endian: Endian = elem("endian", EndianElement)
    has_mpu: Optional[bool] = elem("mpuPresent", BoolElement, default=None)
    has_fpu: Optional[bool] = elem("fpuPresent", BoolElement, default=None)
    fpu_is_double_precision: Optional[bool] = elem("fpuDP", BoolElement, default=None)
    has_dsp: Optional[bool] = elem("dspPresent", BoolElement, default=None)
    has_icache: Optional[bool] = elem("icachePresent", BoolElement, default=None)
    has_dcache: Optional[bool] = elem("dcachePresent", BoolElement, default=None)
    has_ictm: Optional[bool] = elem("itcmPresent", BoolElement, default=None)
    has_dctm: Optional[bool] = elem("dtcmPresent", BoolElement, default=None)
    has_vtor: bool = elem("vtorPresent", BoolElement, default=True)
    num_nvic_priority_bits: int = elem("nvicPrioBits", SvdIntElement)
    has_vendor_systick: bool = elem("vendorSystickConfig", BoolElement)
    num_interrupts: Optional[int] = elem(
        "deviceNumInterrupts", SvdIntElement, default=None
    )
    num_sau_regions: int = elem("sauNumRegions", SvdIntElement, default=0)
    preset_sau_regions: Optional[SauRegionsConfig] = elem(
        "sauRegionsConfig", SauRegionsConfig, default=None
    )


@binding(ELEMENT_CLASSES)
class AddressBlock(objectify.ObjectifiedElement):
    TAG = "addressBlock"

    offset: int = elem("offset", SvdIntElement)
    size: int = elem("size", SvdIntElement)
    usage: AddressBlockUsage = elem("usage", AddressBlockUsageElement)
    protection: Optional[Protection] = elem("protection", ProtectionElement)


class DerivedMixin:
    """Common functionality for elements that contain a SVD 'derivedFrom' attribute."""

    derived_from: Optional[str] = attr("derivedFrom", default=None)

    @property
    def is_derived(self) -> bool:
        """Return True if the element is derived from another element."""
        return self.derived_from is not None


@binding(ELEMENT_CLASSES)
class EnumeratedValue(objectify.ObjectifiedElement):
    TAG = "enumeratedValue"

    name: str = elem("name", StringElement)
    description: Optional[str] = elem("description", StringElement, default=None)
    value: int = elem("value", SvdIntElement)
    is_default: bool = elem("isDefault", BoolElement, default=False)


@binding(ELEMENT_CLASSES)
class Enumeration(objectify.ObjectifiedElement, DerivedMixin):
    TAG = "enumeratedValues"

    name: Optional[str] = elem("name", StringElement, default=None)
    header_enum_name: Optional[str] = elem(
        "headerEnumName", StringElement, default=None
    )
    usage: Optional[EnumUsage] = elem("usage", EnumUsageElement, default=None)

    def __iter__(self) -> Iterator[EnumeratedValue]:
        """Iterate over all enumerated values."""
        return iter_children(self, EnumeratedValue.TAG)

    _enumerated_values: EnumeratedValue = elem("enumeratedValue", EnumeratedValue)


@binding(ELEMENT_CLASSES)
class DimArrayIndex(objectify.ObjectifiedElement):
    TAG = "dimArrayIndex"

    header_enum_name: Optional[str] = elem(
        "headerEnumName", StringElement, default=None
    )
    enumerated_values: Iterator[EnumeratedValue] = elem(
        "enumeratedValue", EnumeratedValue
    )


@binding(ELEMENT_CLASSES)
class InterruptElement(objectify.ObjectifiedElement):
    TAG = "interrupt"

    name: str = elem("name", StringElement)
    description: Optional[str] = elem("description", StringElement, default=None)
    value: int = elem("value", SvdIntElement)


class BitRangeElement(StringElement):
    TAG = "bitRange"


class BitRange(NamedTuple):
    offset: int
    width: int


@binding(ELEMENT_CLASSES)
class FieldElement(objectify.ObjectifiedElement, DerivedMixin):
    TAG = "field"

    name: str = elem("name", StringElement)
    description: Optional[str] = elem("description", StringElement, default=None)

    access: Optional[Access] = elem("access", AccessElement, default=None)
    modified_write_values: Optional[ModifiedWriteValues] = elem(
        "modifiedWriteValue", ModifiedWriteValuesElement, default=None
    )
    write_constraint: Optional[WriteConstraint] = elem(
        "writeConstraint", WriteConstraint, default=None
    )
    read_action: Optional[ReadAction] = elem(
        "readAction", ReadActionElement, default=None
    )

    enumerated_values: Optional[Enumeration] = elem(
        "enumeratedValues", Enumeration, default=None
    )

    @property
    def bit_range(self) -> BitRange:
        """
        Get the bit range of the field.
        :return: Tuple of the field's bit offset and bit width.
        """

        if self._lsb is not None and self._msb is not None:
            return BitRange(offset=self._lsb, width=self._msb - self._lsb + 1)

        if self._bit_offset is not None:
            width = self._bit_width if self._bit_width is not None else 32
            return BitRange(offset=self._bit_offset, width=width)

        if self._bit_range is not None:
            msb_string, lsb_string = self._bit_range[1:-1].split(":")
            msb, lsb = to_int(msb_string), to_int(lsb_string)
            return BitRange(offset=lsb, width=msb - lsb + 1)

        return BitRange(offset=0, width=32)

    _lsb: Optional[int] = elem("lsb", SvdIntElement, default=None)
    _msb: Optional[int] = elem("msb", SvdIntElement, default=None)

    _bit_offset: Optional[int] = elem("bitOffset", SvdIntElement, default=None)
    _bit_width: Optional[int] = elem("bitWidth", SvdIntElement, default=None)

    _bit_range: Optional[BitRangeElement] = elem(
        "bitRange", BitRangeElement, default=None
    )


@binding(ELEMENT_CLASSES)
class FieldsElement(objectify.ObjectifiedElement):
    TAG = "fields"

    field: FieldElement = elem("field", FieldElement)


@dataclass
class RegisterProperties:
    """Common SVD device/peripheral/register level properties."""

    size: Optional[int]
    access: Optional[Access]
    protection: Optional[Protection]
    reset_value: Optional[int]
    reset_mask: Optional[int]


class RegisterPropertiesMixin:
    """Common functionality for elements that contain a SVD 'registerPropertiesGroup'."""

    def get_register_properties(
        self, base_props: Optional[RegisterProperties] = None
    ) -> RegisterProperties:
        """
        Get the register properties of the element, optionally inheriting from a
        base set of properties.
        """
        if base_props is None:
            return RegisterProperties(
                size=self._size,
                access=self._access,
                protection=self._protection,
                reset_value=self._reset_value,
                reset_mask=self._reset_mask,
            )

        return RegisterProperties(
            size=self._size if self._size is not None else base_props.size,
            access=self._access if self._access is not None else base_props.access,
            protection=(
                self._protection
                if self._protection is not None
                else base_props.protection
            ),
            reset_value=(
                self._reset_value
                if self._reset_value is not None
                else base_props.reset_value
            ),
            reset_mask=(
                self._reset_mask
                if self._reset_mask is not None
                else base_props.reset_mask
            ),
        )

    _size: Optional[int] = elem("size", SvdIntElement, default=None)
    _access: Optional[Access] = elem("access", AccessElement, default=None)
    _protection: Optional[Protection] = elem(
        "protection", ProtectionElement, default=None
    )
    _reset_value: Optional[int] = elem("resetValue", SvdIntElement, default=None)
    _reset_mask: Optional[int] = elem("resetMask", SvdIntElement, default=None)


@dataclass
class Dimensions:
    """Dimensions of a repeated SVD element"""

    length: int
    step: int

    def to_range(self) -> Sequence[int]:
        """Convert to a range of offsets"""
        return range(0, (self.length - 1) * self.step + 1, self.step)


class DimensionMixin:
    """Common functionality for elements that contain a SVD 'dimElementGroup'."""

    dim_index: Optional[int] = elem("dimIndex", SvdIntElement, default=None)
    dim_name: Optional[str] = elem("dimName", StringElement, default=None)
    dim_array_index: Optional[DimArrayIndex] = elem(
        "dimArrayIndex", DimArrayIndex, default=None
    )

    @property
    def dimensions(self) -> Optional[Dimensions]:
        """Get the dimensions of the element, if it is repeated."""
        if self._dim is None or self._dim_increment is None:
            return None

        return Dimensions(
            length=self._dim,
            step=self._dim_increment,
        )

    _dim: Optional[int] = elem("dim", SvdIntElement, default=None)
    _dim_increment: Optional[int] = elem("dimIncrement", SvdIntElement, default=None)


@binding(ELEMENT_CLASSES)
class RegisterElement(
    objectify.ObjectifiedElement, DimensionMixin, RegisterPropertiesMixin, DerivedMixin
):
    TAG: str = "register"

    name: str = elem("name", StringElement)
    display_name: Optional[str] = elem("displayName", StringElement, default=None)
    description: Optional[str] = elem("description", StringElement, default=None)
    alternate_group: Optional[str] = elem("alternateGroup", StringElement, default=None)
    alternate_register: Optional[str] = elem(
        "alternateRegister", StringElement, default=None
    )
    offset: Optional[int] = elem("addressOffset", SvdIntElement, default=None)

    data_type: Optional[DataType] = elem("dataType", DataTypeElement, default=None)
    modified_write_values: Optional[ModifiedWriteValues] = elem(
        "modifiedWriteValues", ModifiedWriteValuesElement, default=None
    )
    writeConstraint: Optional[WriteConstraint] = elem(
        "writeConstraint", WriteConstraint, default=None
    )
    readAction: Optional[ReadAction] = elem(
        "readAction", ReadActionElement, default=None
    )

    @property
    def fields(self) -> Iterator[FieldElement]:
        return iter_children(self._fields, FieldElement.TAG)

    _fields: Optional[FieldElement] = elem("fields", FieldsElement, default=None)


@binding(ELEMENT_CLASSES)
class ClusterElement(
    objectify.ObjectifiedElement, DimensionMixin, RegisterPropertiesMixin, DerivedMixin
):
    TAG: str = "cluster"

    name: str = elem("name", StringElement)
    description: Optional[str] = elem("description", StringElement, default=None)
    alternate_cluster: Optional[str] = elem(
        "alternateCluster", StringElement, default=None
    )
    header_struct_name: Optional[str] = elem(
        "headerStructName", StringElement, default=None
    )
    offset: Optional[int] = elem("addressOffset", SvdIntElement, default=None)

    @property
    def registers(self) -> Iterator[Union[RegisterElement, ClusterElement]]:
        return iter_children(self, RegisterElement.TAG, ClusterElement.TAG)

    _register: Optional[RegisterElement] = elem(
        "register", RegisterElement, default=None
    )
    _cluster: Optional[ClusterElement] = elem("cluster", SELF_CLASS, default=None)


@binding(ELEMENT_CLASSES)
class RegistersElement(objectify.ObjectifiedElement):
    TAG = "registers"

    cluster: Optional[ClusterElement] = elem("cluster", ClusterElement, default=None)
    register: Optional[RegisterElement] = elem(
        "register", RegisterElement, default=None
    )


@binding(ELEMENT_CLASSES)
class PeripheralElement(
    objectify.ObjectifiedElement, DimensionMixin, RegisterPropertiesMixin, DerivedMixin
):
    TAG = "peripheral"

    name: str = elem("name", StringElement)
    version: Optional[str] = elem("version", StringElement, default=None)
    description: Optional[str] = elem("description", StringElement, default=None)
    alternate_peripheral: Optional[str] = elem(
        "alternatePeripheral", StringElement, default=None
    )
    group_name: Optional[str] = elem("groupName", StringElement, default=None)
    prepend_to_name: Optional[str] = elem("prependToName", StringElement, default=None)
    append_to_name: Optional[str] = elem("appendToName", StringElement, default=None)
    header_struct_name: Optional[str] = elem(
        "headerStructName", StringElement, default=None
    )
    disable_condition: Optional[str] = elem(
        "disableCondition", StringElement, default=None
    )
    base_address: int = elem("baseAddress", SvdIntElement)
    address_block: AddressBlock = elem("addressBlock", AddressBlock)
    interrupt: Optional[InterruptElement] = elem(
        "interrupt", InterruptElement, default=None
    )

    @property
    def registers(self) -> Iterator[Union[RegisterElement, ClusterElement]]:
        return iter_children(self._registers, RegisterElement.TAG, ClusterElement.TAG)

    _registers: Optional[RegistersElement] = elem(
        "registers", RegistersElement, default=None
    )


@binding(ELEMENT_CLASSES)
class PeripheralsElement(objectify.ObjectifiedElement):
    TAG = "peripherals"

    peripheral: Optional[PeripheralElement] = elem(
        "peripheral", PeripheralElement, default=None
    )


@binding(ELEMENT_CLASSES)
class DeviceElement(objectify.ObjectifiedElement, RegisterPropertiesMixin):
    TAG = "device"

    schema_version: float = attr("schemaVersion", converter=float)

    vendor: Optional[str] = elem("vendor", StringElement, default=None)
    vendor_id: Optional[str] = elem("vendorID", StringElement, default=None)
    name: str = elem("name", StringElement)
    series: Optional[str] = elem("series", StringElement, default=None)
    version: str = elem("version", StringElement)
    description: str = elem("description", StringElement)
    license_text: Optional[str] = elem("licenseText", StringElement)
    cpu: Optional[Cpu] = elem("cpu", Cpu, default=None)
    header_system_filename: Optional[str] = elem(
        "headerSystemFilename", StringElement, default=None
    )
    header_definitions_prefix: Optional[str] = elem(
        "headerDefinitionsPrefix", StringElement, default=None
    )
    address_unit_bits: int = elem("addressUnitBits", SvdIntElement)
    width: int = elem("width", SvdIntElement)

    @property
    def peripherals(self) -> Iterator[PeripheralElement]:
        """Iterate over all peripherals in the device"""
        return iter_children(self._peripherals, PeripheralElement.TAG)

    _peripherals: PeripheralElement = elem(
        "peripherals",
        PeripheralsElement,
    )
