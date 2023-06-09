#
# Copyright (c) 2023 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

"""
"Low-level" read-only Python representation of the SVD format.
Each type of element in the SVD XML tree is represented by a class in this module.
The class properties correspond more or less directly to the XML elements/attributes,
with some abstractions and simplifications added for convenience.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import (
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    Union,
)

from lxml import objectify
from lxml.objectify import BoolElement, StringElement

from .util import (
    CaseInsensitiveStrEnum,
    attr,
    binding,
    elem,
    make_enum_wrapper,
    iter_element_children,
    to_bool,
    to_int,
    SELF_CLASS,
)


# Container for classes that represent non-leaf elements in the SVD XML tree.
ELEMENT_CLASSES: List[Type[objectify.ObjectifiedElement]] = []


class SvdIntElement(objectify.IntElement):
    """
    Element containing an SVD integer value.
    This class uses a custom parser to convert the value to an integer.
    """

    def _init(self):
        self._setValueParser(to_int)


@enum.unique
class Access(CaseInsensitiveStrEnum):
    """Access rights for a given register or field."""

    # Read access is permitted. Write operations have an undefined result.
    READ_ONLY = "read-only"
    # Write access is permitted. Read operations have an undefined result.
    WRITE_ONLY = "write-only"
    # Read and write accesses are permitted.
    READ_WRITE = "read-write"
    # Only the first write after reset has an effect. Read operations have an undefined results.
    WRITE_ONCE = "writeOnce"
    # Only the first write after reset has an effect. Read access is permitted.
    READ_WRITE_ONCE = "read-writeOnce"


AccessElement = make_enum_wrapper(Access)


@enum.unique
class ReadAction(CaseInsensitiveStrEnum):
    """Side effect following a read operation of a given register or field."""

    # The register/field is set to zero following a read operation.
    CLEAR = "clear"
    # The register/field is set to ones following a read operation.
    SET = "set"
    # The register/field is modified by a read operation.
    MODIFY = "modify"
    # A dependent resource is modified by a read operation.
    MODIFY_EXTERNAL = "modifyExternal"


ReadActionElement = make_enum_wrapper(ReadAction)


@enum.unique
class Endian(CaseInsensitiveStrEnum):
    """Processor endianness."""

    # Little endian
    LITTLE = "little"
    # Big endian
    BIG = "big"
    # Endianness is configurable for the device, taking effect on the next reset.
    SELECTABLE = "selectable"
    # Neither big nor little endian
    OTHER = "other"


EndianElement = make_enum_wrapper(Endian)


@enum.unique
class SauAccess(CaseInsensitiveStrEnum):
    """SAU region access type"""

    # Non-secure accessible
    NON_SECURE = "n"
    # Secure callable
    SECURE_CALLABLE = "c"


SauAccessElement = make_enum_wrapper(SauAccess)


@enum.unique
class AddressBlockUsage(CaseInsensitiveStrEnum):
    """Defined usage type of a peripheral address block."""

    REGISTER = "registers"
    BUFFERS = "buffers"
    RESERVED = "reserved"


AddressBlockUsageElement = make_enum_wrapper(AddressBlockUsage)


@enum.unique
class Protection(CaseInsensitiveStrEnum):
    """Security privilege required to access an address region"""

    # Secure permission required for access
    SECURE = "s"
    # Non-secure or secure permission required for access
    NON_SECURE = "n"
    # Privileged permission required for access
    PRIVILEGED = "p"


ProtectionElement = make_enum_wrapper(Protection)


@enum.unique
class EnumUsage(CaseInsensitiveStrEnum):
    """Usage of an enumerated value."""

    # The value is relevant for read operations.
    READ = "read"
    # The value is relevant for write operations.
    WRITE = "write"
    # The value is relevant for read and write operations.
    READ_WRITE = "read-write"


EnumUsageElement = make_enum_wrapper(EnumUsage)


@enum.unique
class WriteAction(CaseInsensitiveStrEnum):
    """Side effect following a write operation of a given register or field."""

    # Bits written to one are set to zero in the register/field.
    ONE_TO_CLEAR = "oneToClear"
    # Bits written to one are set to one in the register/field.
    ONE_TO_SET = "oneToSet"
    # Bits written to one are inverted in the register/field.
    ONE_TO_TOGGLE = "oneToToggle"
    # Bits written to zero are set to zero in the register/field.
    ZERO_TO_CLEAR = "zeroToClear"
    # Bits written to zero are set to one in the register/field.
    ZERO_TO_SET = "zeroToSet"
    # Bits written to zero are inverted in the register/field.
    ZERO_TO_TOGGLE = "zeroToToggle"
    # All bits are set to zero on writing to the register/field.
    CLEAR = "clear"
    # All bits are set to one on writing to the register/field.
    SET = "set"
    # All bits are modified on writing to the register/field.
    MODIFY = "modify"


ModifiedWriteValuesElement = make_enum_wrapper(WriteAction)


@enum.unique
class DataType(CaseInsensitiveStrEnum):
    """Data types defined in the SVD specification."""

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


DataTypeElement = make_enum_wrapper(DataType)


@enum.unique
class CpuName(CaseInsensitiveStrEnum):
    """CPU names defined in the SVD specification."""

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


CpuNameElement = make_enum_wrapper(CpuName)


@binding(ELEMENT_CLASSES)
class RangeWriteConstraint(objectify.ObjectifiedElement):
    """Value range constraint for a register or field."""

    TAG = "range"

    # Minimum permitted value
    minimum: int = elem("minimum", SvdIntElement)

    # Maximum permitted value
    maximum: int = elem("maximum", SvdIntElement)


@enum.unique
class WriteConstraint(enum.Enum):
    """Type of write constraint for a register or field."""

    # Only the last read value can be written.
    WRITE_AS_READ = enum.auto()
    # Only enumerated values can be written.
    USE_ENUMERATED_VALUES = enum.auto()
    # Only values within a given range can be written.
    RANGE = enum.auto()


@binding(ELEMENT_CLASSES)
class WriteConstraintElement(objectify.ObjectifiedElement):
    TAG = "writeConstraint"

    # Value range constraint
    value_range: Optional[RangeWriteConstraint] = elem("range", RangeWriteConstraint, default=None)

    def as_enum(self) -> Optional[WriteConstraint]:
        """Return the write constraint as an enum value."""
        if self._write_as_read:
            return WriteConstraint.WRITE_AS_READ
        if self._use_enumerated_values:
            return WriteConstraint.USE_ENUMERATED_VALUES
        if self.value_range is not None:
            return WriteConstraint.RANGE

    # (internal) If true, only the last read value can be written.
    _write_as_read: bool = elem("writeAsRead", BoolElement, default=False)

    # (internal) If true, only enumerated values can be written.
    _use_enumerated_values: bool = elem("useEnumeratedValues", BoolElement, default=False)


@binding(ELEMENT_CLASSES)
class SauRegion(objectify.ObjectifiedElement):
    """Predefined Secure Attribution Unit (SAU) region."""

    TAG = "region"

    # If true, the SAU region is enabled.
    enabled: bool = attr("enabled", converter=to_bool, default=True)

    # Name of the SAU region.
    name: Optional[str] = attr("name", default=None)

    # Base address of the SAU region.
    base: int = elem("base", SvdIntElement)

    # Limit address of the SAU region.
    limit: int = elem("limit", SvdIntElement)

    # Access permissions of the SAU region.
    access: SauAccess = elem("access", SauAccessElement)


@binding(ELEMENT_CLASSES)
class SauRegionsConfig(objectify.ObjectifiedElement):
    """Container for predefined Secure Attribution Unit (SAU) regions."""

    TAG = "sauRegions"

    # If true, the SAU is enabled.
    enabled: bool = attr("enabled", converter=to_bool, default=True)

    # Default protection for disabled SAU regions.
    protection_when_disabled: Optional[Protection] = attr(
        "name", converter=Protection.from_str, default=None
    )

    def __iter__(self) -> Iterator[SauRegion]:
        """Iterate over all predefined SAU regions."""
        return iter_element_children(self, SauRegion.TAG)

    # (internal) SAU regions.
    _region: SauRegion = elem("region", SauRegion)


@binding(ELEMENT_CLASSES)
class Cpu(objectify.ObjectifiedElement):
    """Description of the device processor"""

    TAG = "cpu"

    # CPU name. See CpuName for possible values.
    name: CpuName = elem("name", CpuNameElement)

    # CPU hardware revision with the format "rNpM".
    revision: str = elem("revision", StringElement)

    # Default endianness of the CPU.
    endian: Endian = elem("endian", EndianElement)

    # True if the CPU has a memory protection unit (MPU).
    has_mpu: Optional[bool] = elem("mpuPresent", BoolElement, default=None)

    # True if the CPU has a floating point unit (FPU).
    has_fpu: Optional[bool] = elem("fpuPresent", BoolElement, default=None)

    # True if the CPU has a double precision floating point unit (FPU).
    fpu_is_double_precision: Optional[bool] = elem("fpuDP", BoolElement, default=None)

    # True if the CPU implements the SIMD DSP extensions.
    has_dsp: Optional[bool] = elem("dspPresent", BoolElement, default=None)

    # True if the CPU has an instruction cache.
    has_icache: Optional[bool] = elem("icachePresent", BoolElement, default=None)

    # True if the CPU has a data cache.
    has_dcache: Optional[bool] = elem("dcachePresent", BoolElement, default=None)

    # True if the CPU has an instruction tightly coupled memory (ITCM).
    has_ictm: Optional[bool] = elem("itcmPresent", BoolElement, default=None)

    # True if the CPU has a data tightly coupled memory (DTCM).
    has_dctm: Optional[bool] = elem("dtcmPresent", BoolElement, default=None)

    # True if the CPU has a Vector Table Offset Register (VTOR).
    has_vtor: bool = elem("vtorPresent", BoolElement, default=True)

    # Bit width of interrupt priority levels in the Nested Vectored Interrupt Controller (NVIC).
    num_nvic_priority_bits: int = elem("nvicPrioBits", SvdIntElement)

    # True if the CPU has a vendor-specific SysTick Timer.
    # If False, the Arm-defined System Tick Timer is used.
    has_vendor_systick: bool = elem("vendorSystickConfig", BoolElement)

    # Maximum interrupt number in the CPU plus one.
    num_interrupts: Optional[int] = elem(
        "deviceNumInterrupts", SvdIntElement, default=None
    )

    # Number of supported Secure Attribution Unit (SAU) regions.
    num_sau_regions: int = elem("sauNumRegions", SvdIntElement, default=0)

    # Predefined Secure Attribution Unit (SAU) regions, if any.
    preset_sau_regions: Optional[SauRegionsConfig] = elem(
        "sauRegionsConfig", SauRegionsConfig, default=None
    )


@binding(ELEMENT_CLASSES)
class AddressBlock(objectify.ObjectifiedElement):
    """Address range mapped to a peripheral."""

    TAG = "addressBlock"

    # Start address of the address block, relative to the peripheral base address.
    offset: int = elem("offset", SvdIntElement)

    # Number of address unit bits covered by the address block.
    size: int = elem("size", SvdIntElement)

    # Address block usage. See AddressBlockUsage for possible values.
    usage: AddressBlockUsage = elem("usage", AddressBlockUsageElement)

    # Protection level for the address block.
    protection: Optional[Protection] = elem("protection", ProtectionElement)


class DerivedMixin:
    """Common functionality for elements that contain a SVD 'derivedFrom' attribute."""

    # Name of the element that this element is derived from.
    derived_from: Optional[str] = attr("derivedFrom", default=None)

    @property
    def is_derived(self) -> bool:
        """Return True if the element is derived from another element."""
        return self.derived_from is not None


@binding(ELEMENT_CLASSES)
class EnumeratedValue(objectify.ObjectifiedElement):
    """Value definition for a field."""

    TAG = "enumeratedValue"

    # Name of the enumerated value.
    name: str = elem("name", StringElement)

    # Description of the enumerated value.
    description: Optional[str] = elem("description", StringElement, default=None)

    # Value of the enumerated value.
    value: int = elem("value", SvdIntElement)

    # True if the enumerated value is the default value of the field.
    is_default: bool = elem("isDefault", BoolElement, default=False)


@binding(ELEMENT_CLASSES)
class Enumeration(objectify.ObjectifiedElement, DerivedMixin):
    """Container for enumerated values."""

    TAG = "enumeratedValues"

    # Name of the enumeration.
    name: Optional[str] = elem("name", StringElement, default=None)

    # Identifier of the enumeration in the device header file.
    header_enum_name: Optional[str] = elem(
        "headerEnumName", StringElement, default=None
    )

    # Description of which types of operations the enumeration is used for.
    usage: EnumUsage = elem("usage", EnumUsageElement, default=EnumUsage.READ_WRITE)

    def __iter__(self) -> Iterator[EnumeratedValue]:
        """Iterate over all enumerated values."""
        return iter_element_children(self, EnumeratedValue.TAG)

    # (internal) Enumerated values
    _enumerated_values: EnumeratedValue = elem("enumeratedValue", EnumeratedValue)


@binding(ELEMENT_CLASSES)
class DimArrayIndex(objectify.ObjectifiedElement):
    """Description of the index used for an array of registers."""

    TAG = "dimArrayIndex"

    # The base name of enumerations.
    header_enum_name: Optional[str] = elem(
        "headerEnumName", StringElement, default=None
    )

    # Values contained in the enumeration.
    enumerated_values: Iterator[EnumeratedValue] = elem(
        "enumeratedValue", EnumeratedValue
    )


@binding(ELEMENT_CLASSES)
class Interrupt(objectify.ObjectifiedElement):
    """Peripheral interrupt description."""

    TAG = "interrupt"

    # Name of the interrupt.
    name: str = elem("name", StringElement)

    # Description of the interrupt.
    description: Optional[str] = elem("description", StringElement, default=None)

    # Interrupt number.
    value: int = elem("value", SvdIntElement)


class BitRange(NamedTuple):
    """Bit range of a field."""

    # Bit offset of the field.
    offset: int

    # Bit width of the field.
    width: int


@binding(ELEMENT_CLASSES)
class FieldElement(objectify.ObjectifiedElement, DerivedMixin):
    """SVD field element."""

    TAG = "field"

    # Name of the field.
    name: str = elem("name", StringElement)

    # Description of the field.
    description: Optional[str] = elem("description", StringElement, default=None)

    # Access rights of the field.
    access: Optional[Access] = elem("access", AccessElement, default=None)

    # Side effect when writing to the field.
    modified_write_values: Optional[WriteAction] = elem(
        "modifiedWriteValue", ModifiedWriteValuesElement, default=None
    )

    # Constraints on writing to the field.
    write_constraint: Optional[WriteConstraintElement] = elem(
        "writeConstraint", WriteConstraintElement, default=None
    )

    # Side effect when reading from the field.
    read_action: Optional[ReadAction] = elem(
        "readAction", ReadActionElement, default=None
    )

    # Permitted values of the field.
    enumerated_values: Optional[Enumeration] = elem(
        "enumeratedValues", Enumeration, default=None
    )

    @property
    def bit_range(self) -> BitRange:
        """
        Bit range of the field.
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

    # (internal) Least significant bit of the field, if specified in the bitRangeLsbMsbStyle style.
    _lsb: Optional[int] = elem("lsb", SvdIntElement, default=None)

    # (internal) Most significant bit of the field, if specified in the bitRangeLsbMsbStyle style.
    _msb: Optional[int] = elem("msb", SvdIntElement, default=None)

    # (internal) Bit offset of the field, if specified in the bitRangeOffsetWidthStyle style.
    _bit_offset: Optional[int] = elem("bitOffset", SvdIntElement, default=None)

    # (internal) Bit width of the field, if specified in the bitRangeOffsetWidthStyle style.
    _bit_width: Optional[int] = elem("bitWidth", SvdIntElement, default=None)

    # (internal) Bit range of the field, given in the form "[msb:lsb]", if specified in the
    # bitRangePattern style.
    _bit_range: Optional[str] = elem(
        "bitRange", StringElement, default=None
    )


@binding(ELEMENT_CLASSES)
class FieldsElement(objectify.ObjectifiedElement):
    """Container for SVD field elements."""

    TAG = "fields"

    # Field elements.
    field: FieldElement = elem("field", FieldElement)


@dataclass
class RegisterProperties:
    """Common SVD device/peripheral/register level properties."""

    # Size of the register in bits.
    size: Optional[int]

    # Access rights of the register.
    access: Optional[Access]

    # Protection level of the register.
    protection: Optional[Protection]

    # Reset value of the register.
    reset_value: Optional[int]

    # Reset mask of the register.
    reset_mask: Optional[int]


class RegisterPropertiesGroup:
    """Common functionality for elements that contain a SVD 'registerPropertiesGroup'."""

    @property
    def register_properties(self) -> RegisterProperties:
        """Register properties specified in the element itself."""
        return self.get_register_properties()

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

    # Number of times the element is repeated.
    length: int

    # Increment between each element.
    step: int

    def to_range(self) -> Sequence[int]:
        """Convert to a range of offsets"""
        return range(0, (self.length - 1) * self.step + 1, self.step)


@typing.no_
class DimElementGroup:
    """Common functionality for elements that contain a SVD 'dimElementGroup'."""

    # Index of the element, if it is repeated.
    dim_index: Optional[int] = elem("dimIndex", SvdIntElement, default=None)

    # Name of the dimension, if it is repeated.
    dim_name: Optional[str] = elem("dimName", StringElement, default=None)

    # Array index of the element, if it is repeated.
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
    objectify.ObjectifiedElement, DimElementGroup, RegisterPropertiesGroup, DerivedMixin
):
    """SVD register element."""

    TAG: str = "register"

    # Name of the register.
    name: str = elem("name", StringElement)

    # Display name of the register.
    display_name: Optional[str] = elem("displayName", StringElement, default=None)

    # Description of the register.
    description: Optional[str] = elem("description", StringElement, default=None)

    # Alternate group of the register.
    alternate_group: Optional[str] = elem("alternateGroup", StringElement, default=None)

    # Name of a different register that corresponds to this register.
    alternate_register: Optional[str] = elem(
        "alternateRegister", StringElement, default=None
    )

    # Address offset of the register, relative to the parent element.
    offset: Optional[int] = elem("addressOffset", SvdIntElement, default=None)

    # C data type to use when accessing the register.
    data_type: Optional[DataType] = elem("dataType", DataTypeElement, default=None)

    # Side effect of writing the register.
    modified_write_values: Optional[WriteAction] = elem(
        "modifiedWriteValues", ModifiedWriteValuesElement, default=WriteAction.MODIFY
    )

    # Write constraint of the register.
    write_constraint: Optional[WriteConstraintElement] = elem(
        "writeConstraint", WriteConstraintElement, default=None
    )

    # Side effect of reading the register.
    read_action: Optional[ReadAction] = elem(
        "readAction", ReadActionElement, default=None
    )

    @property
    def fields(self) -> Iterator[FieldElement]:
        """Iterator over the fields of the register."""
        return iter_element_children(self._fields, FieldElement.TAG)

    # (internal) Fields of the register.
    _fields: Optional[FieldElement] = elem("fields", FieldsElement, default=None)


@binding(ELEMENT_CLASSES)
class ClusterElement(
    objectify.ObjectifiedElement, DimElementGroup, RegisterPropertiesGroup, DerivedMixin
):
    """SVD cluster element."""

    TAG: str = "cluster"

    # Name of the cluster.
    name: str = elem("name", StringElement)

    # Description of the cluster.
    description: Optional[str] = elem("description", StringElement, default=None)

    # Name of a different cluster that corresponds to this cluster.
    alternate_cluster: Optional[str] = elem(
        "alternateCluster", StringElement, default=None
    )

    # Name of the C struct used to represent the cluster.
    header_struct_name: Optional[str] = elem(
        "headerStructName", StringElement, default=None
    )

    # Address offset of the cluster, relative to the parent peripheral element.
    offset: Optional[int] = elem("addressOffset", SvdIntElement, default=None)

    @property
    def registers(self) -> Iterator[Union[RegisterElement, ClusterElement]]:
        """Iterator over the registers and clusters that are direct children of this cluster."""
        return iter_element_children(self, RegisterElement.TAG, ClusterElement.TAG)

    # (internal) Register elements in the cluster.
    _register: Optional[RegisterElement] = elem(
        "register", RegisterElement, default=None
    )

    # (internal) Cluster elements in the cluster.
    _cluster: Optional[ClusterElement] = elem("cluster", SELF_CLASS, default=None)


@binding(ELEMENT_CLASSES)
class RegistersElement(objectify.ObjectifiedElement):
    """Container for SVD register/cluster elements."""

    TAG = "registers"

    # Cluster elements in the container.
    cluster: Optional[ClusterElement] = elem("cluster", ClusterElement, default=None)

    # Register elements in the container.
    register: Optional[RegisterElement] = elem(
        "register", RegisterElement, default=None
    )


@binding(ELEMENT_CLASSES)
class PeripheralElement(
    objectify.ObjectifiedElement, DimElementGroup, RegisterPropertiesGroup, DerivedMixin
):
    """SVD peripheral element."""

    TAG = "peripheral"

    # Name of the peripheral.
    name: str = elem("name", StringElement)

    # Version of the peripheral.
    version: Optional[str] = elem("version", StringElement, default=None)

    # Description of the peripheral.
    description: Optional[str] = elem("description", StringElement, default=None)

    # Base address of the peripheral.
    base_address: int = elem("baseAddress", SvdIntElement)

    @property
    def interrupts(self) -> Iterator[Interrupt]:
        """Iterator over the interrupts of the peripheral."""
        return iter_element_children(self, Interrupt.TAG)

    @property
    def address_blocks(self) -> Iterator[AddressBlock]:
        """Iterator over the address blocks of the peripheral."""
        return iter_element_children(self, AddressBlock.TAG)

    @property
    def registers(self) -> Iterator[Union[RegisterElement, ClusterElement]]:
        """Iterator over the registers and clusters that are direct children of this peripheral."""
        return iter_element_children(self._registers, RegisterElement.TAG, ClusterElement.TAG)

    # Name of a different peripheral that corresponds to this peripheral.
    alternate_peripheral: Optional[str] = elem(
        "alternatePeripheral", StringElement, default=None
    )

    # Name of the group that the peripheral belongs to.
    group_name: Optional[str] = elem("groupName", StringElement, default=None)

    # String to prepend to the names of registers contained in the peripheral.
    prepend_to_name: Optional[str] = elem("prependToName", StringElement, default=None)

    # String to append to the names of registers contained in the peripheral.
    append_to_name: Optional[str] = elem("appendToName", StringElement, default=None)

    # Name of the C struct that represents the peripheral.
    header_struct_name: Optional[str] = elem(
        "headerStructName", StringElement, default=None
    )

    disable_condition: Optional[str] = elem(
        "disableCondition", StringElement, default=None
    )

    # (internal) Interrupt elements in the peripheral.
    _interrupts: Optional[Interrupt] = elem(
        "interrupt", Interrupt, default=None
    )

    # (internal) Address block elements in the peripheral.
    _address_blocks: Optional[AddressBlock] = elem("addressBlock", AddressBlock, default=None)

    # (internal) Register/cluster container.
    _registers: Optional[RegistersElement] = elem(
        "registers", RegistersElement, default=None
    )


@binding(ELEMENT_CLASSES)
class PeripheralsElement(objectify.ObjectifiedElement):
    """Container for SVD peripheral elements."""

    TAG = "peripherals"

    # Peripheral elements in the container.
    peripheral: Optional[PeripheralElement] = elem(
        "peripheral", PeripheralElement, default=None
    )


@binding(ELEMENT_CLASSES)
class DeviceElement(objectify.ObjectifiedElement, RegisterPropertiesGroup):
    """SVD device element."""

    TAG = "device"

    # Version of the CMSIS schema that the SVD file conforms to.
    schema_version: float = attr("schemaVersion", converter=float)

    # Name of the device.
    name: str = elem("name", StringElement)

    # Device series name.
    series: Optional[str] = elem("series", StringElement, default=None)

    # Version of the device.
    version: str = elem("version", StringElement)

    # Full device vendor name.
    vendor: Optional[str] = elem("vendor", StringElement, default=None)

    # Abbreviated device vendor name.
    vendor_id: Optional[str] = elem("vendorID", StringElement, default=None)

    # Description of the device.
    description: str = elem("description", StringElement)

    # The license to use for the device header file.
    license_text: Optional[str] = elem("licenseText", StringElement)

    # Description of the device processor.
    cpu: Optional[Cpu] = elem("cpu", Cpu, default=None)

    # Device header filename without extension.
    header_system_filename: Optional[str] = elem(
        "headerSystemFilename", StringElement, default=None
    )

    # String to prepend to all type definitions in the device header file.
    header_definitions_prefix: Optional[str] = elem(
        "headerDefinitionsPrefix", StringElement, default=None
    )

    # Number of data bits selected by each address.
    address_unit_bits: int = elem("addressUnitBits", SvdIntElement)

    # Width of the maximum data transfer supported by the device.
    width: int = elem("width", SvdIntElement)

    @property
    def peripherals(self) -> Iterator[PeripheralElement]:
        """Iterate over all peripherals in the device"""
        return iter_element_children(self._peripherals, PeripheralElement.TAG)

    # (internal) Peripheral elements in the device.
    _peripherals: PeripheralElement = elem(
        "peripherals",
        PeripheralsElement,
    )
