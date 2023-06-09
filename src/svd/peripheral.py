#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Python representation of a SVD device.
"""

from __future__ import annotations

import operator as op
import textwrap
from collections import defaultdict
from collections.abc import Mapping
from functools import cached_property
from math import log2
from typing import (
    Collection,
    List,
    Tuple,
    Union,
    Dict,
    Iterable,
    Iterator,
    NamedTuple,
    Optional,
    Sequence,
)
from pprint import pformat

from . import bindings
from .bindings import (
    Access,
    AddressBlock,
    Cpu,
    RegisterProperties,
    Dimensions,
    WriteAction,
    ReadAction,
)
from . import util
from .util import LazyStaticList, LazyStaticMapping


class Device(Mapping):
    """
    Representation of a SVD device.
    """

    def __init__(self, device: bindings.DeviceElement):
        self._device: bindings.DeviceElement = device
        self._reg_props: RegisterProperties = self._device.register_properties

        peripherals = {}

        # Process peripherals in topological order to ensure that base peripherals are processed
        # before derived peripherals.
        for peripheral_element in _topo_sort_derived_peripherals(device.peripherals):
            if peripheral_element.is_derived:
                base_peripheral = peripherals[peripheral_element.derived_from]
            else:
                base_peripheral = None

            peripheral = Peripheral(
                peripheral_element,
                base_reg_props=self._reg_props,
                base_peripheral=base_peripheral,
            )

            peripherals[peripheral.name] = peripheral

        self._peripherals: Dict[str, Peripheral] = dict(
            sorted(peripherals.items(), key=lambda kv: kv[1].base_address)
        )

    @property
    def name(self) -> str:
        """Name of the device."""
        return self._device.name

    @property
    def series(self) -> Optional[str]:
        """Device series name."""
        return self._device.series

    @property
    def vendor_id(self) -> Optional[str]:
        """Device vendor ID."""
        return self._device.vendor_id

    @property
    def cpu(self) -> Cpu:
        """Device CPU information"""
        return self._device.cpu

    @property
    def address_unit_bits(self) -> int:
        """Number of data bits corresponding to an address."""
        return self._device.address_unit_bits

    @property
    def bus_bit_width(self) -> int:
        """Maximum data bits supported by the data bus in a single transfer."""
        return self._device.width

    @property
    def peripherals(self) -> Mapping[str, Peripheral]:
        """Map of peripherals in the device, indexed by name."""
        return self._peripherals

    def __getitem__(self, name: str) -> Peripheral:
        """Get a peripheral by name."""
        try:
            return self._peripherals[name]
        except LookupError as e:
            raise KeyError(
                f"{self.__class__} {self.name} does not contain a peripheral named '{name}'"
            ) from e

    def __iter__(self) -> Iterator[str]:
        """Iterate over the names of peripherals in the device."""
        return iter(self._peripherals)

    def __len__(self) -> int:
        """Get the number of peripherals in the device."""
        return len(self._peripherals)


class Peripheral(Mapping):
    """
    Representation of a specific device peripheral.

    Internally, this class maintains a representation of a peripheral that is always guaranteed to
    be correct when compared to the allowable values prescribed by the SVD file the class was
    instantiated from. This representation starts off by having the default values defined within
    the SVD.
    """

    def __init__(
        self,
        peripheral: bindings.PeripheralElement,
        base_reg_props: bindings.RegisterProperties,
        base_peripheral: Optional[Peripheral] = None,
    ):
        """
        Initialize the class attribute(s).

        :param peripheral: Element for the peripheral node in the device SVD file.
        :param base_reg_props: Register properties inherited from the parent device.
        :param base_peripheral: Base peripheral to derive this peripheral from, if any.
        """
        self._peripheral: bindings.PeripheralElement = peripheral
        self._base_peripheral: Optional[Peripheral] = base_peripheral
        self._base_address: int = peripheral.base_address
        self._reg_props: bindings.RegisterProperties = (
            self._peripheral.get_register_properties(base_props=base_reg_props)
        )
        self._values: Dict[int, int] = {}

    @property
    def name(self) -> str:
        """Name of the peripheral."""
        return self._peripheral.name

    @property
    def version(self) -> Optional[str]:
        """Optional version of the peripheral."""
        return self._peripheral.version

    @property
    def description(self) -> Optional[str]:
        """Optional description of the peripheral."""
        return self._peripheral.description

    @property
    def base_address(self) -> int:
        """Base address of the peripheral in memory."""
        return self._base_address

    # FIXME: reconsider. Maybe make into a constructor or give as an offset to memory iter
    @base_address.setter
    def base_address(self, address: int):
        """Set the base address of the peripheral."""
        self._base_address = address

    @property
    def interrupts(self) -> Mapping[str, int]:
        """Interrupts associated with the peripheral, a mapping from interrupt name to value."""
        return {interrupt.name: interrupt.value for interrupt in self._peripheral.interrupts}

    @property
    def address_blocks(self) -> List[AddressBlock]:
        """List of address blocks associated with the peripheral."""
        return list(self._peripheral.address_blocks)

    @cached_property
    def registers(self) -> Mapping[str, RegisterType]:
        """Mapping of top-level registers in the peripheral, indexed by name."""
        return LazyStaticMapping(
            keys=self._register_tree.keys(), factory=self._register_factory
        )

    def recursive_iter(self, leaf_only: bool = False) -> Iterator[RegisterType]:
        """
        Recursive iterator over the registers in the peripheral in pre-order.
        Registers are ordered by increasing offset/address.

        :param absolute_addresses: If true,
        :param leaf_only: Only
        """
        for register_name in self._register_tree:
            yield from self[register_name].recursive_iter(leaf_only)

    def memory_iter(self, absolute_addresses: bool = True):
        """TODO"""
        for register in self.recursive_iter(leaf_only=True):
            address = register.address if absolute_addresses else register.offset
            yield address, register

    @cached_property
    def memory_map(self) -> Dict[int, RegisterType]:
        """Map of the peripheral register contents in memory."""
        return dict(self.memory_iter())

    @cached_property
    def _register_tree(self) -> Mapping[str, _RegisterDescription]:
        """TODO"""
        # Add the registers defined in this peripheral
        registers = _extract_register_descriptions(
            self._peripheral.registers, self._reg_props
        )

        # Add the registers defined in the base peripheral, if any
        if (
            self._base_peripheral is not None
            and self._base_peripheral._peripheral.registers
        ):
            # If the register properties are equal, then it is possible to reuse all the immutable
            # properties from the base peripheral.
            if self._base_peripheral._reg_props == self._reg_props:
                base_registers = self._base_peripheral._register_tree
            # Otherwise, traverse the base registers again, because the difference in
            # register properties propagates down to the register elements.
            else:
                base_registers = _extract_register_descriptions(
                    self._base_peripheral._peripheral.registers, self._reg_props
                )

            # The register maps are each sorted internally, but need to be merged by address
            # to ensure sorted order in the combined map
            registers = dict(
                util.iter_merged(
                    registers.items(),
                    base_registers.items(),
                    key=lambda kv: kv[1].start_offset,
                )
            )

        return registers

    def _register_factory(self, name: str) -> RegisterType:
        """Instantiate the register with the given name."""
        return _create_register_instance(self._register_tree[name], peripheral=self)

    def __getitem__(self, name: str) -> Register:
        """
        :param name: Name of the register to get.

        :return: The instance of the specified register.
        """
        try:
            return self.registers[name]
        except LookupError as e:
            raise KeyError(
                f"Peripheral {self} does not contain a register named '{name}'"
            ) from e

    def __setitem__(self, name: str, value: int):
        """
        :param name: Name of the register to update.
        :param value: The raw register value to write to the specified register.
        """
        self[name].value = value

    def __iter__(self):
        """Iterate over the names of registers in the peripheral."""
        return iter(self.registers)

    def __len__(self):
        """Get the number of registers in the peripheral."""
        return len(self.registers)

    def __repr__(self) -> str:
        return f"Peripheral {self.name.upper()}@{hex(self.base_address)}"

    def __str__(self) -> str:
        periph = {hex(k): v for k, v in self.memory_map.items() if v.modified}
        return f"Peripheral {self.name.upper()}@{hex(self.base_address)}:\n{pformat(periph)}"


class _RegisterDescription(NamedTuple):
    """
    Class containing immutable data describing a SVD register/cluster element.
    This is separated from the register classes to optimize construction of derived peripherals.
    Since the description is not tied to a specific Peripheral instance, it can be
    instantiated once per register/cluster in the SVD document and reused for derived peripherals.
    """

    name: str
    start_offset: int
    reg_props: bindings.RegisterProperties
    dim_props: Optional[bindings.Dimensions]
    registers: Optional[Dict[str, _RegisterDescription]]
    fields: Optional[Dict[str, _FieldDescription]]
    element: Union[bindings.RegisterElement, bindings.ClusterElement]


class _RegisterBase:
    """Base class for all register types"""

    __slots__ = [
        "_description",
        "_peripheral",
        "_instance_offset",
        "_index",
        "_full_name_prefix",
    ]

    def __init__(
        self,
        description: _RegisterDescription,
        peripheral: Peripheral,
        instance_offset: int = 0,
        index: Optional[int] = None,
        full_name_prefix: str = "",
    ):
        """
        Initialize the class attribute(s).

        :param description: Register description
        :param peripheral: Register name
        :param instance_offset: Address offset inherited from the parent register
        :param index: Index of this register in the parent register, if applicable
        :param full_name_prefix: String prefixed to the register name to get the full name
        """
        self._description: _RegisterDescription = description
        self._peripheral: Peripheral = peripheral
        self._instance_offset: int = instance_offset
        self._index: Optional[int] = index
        self._full_name_prefix: str = full_name_prefix

    @property
    def name(self) -> str:
        """Name of the register."""
        if self._index is not None:
            return f"{self._description.name}[{self._index}]"
        return self._description.name

    @property
    def full_name(self) -> str:
        """Full qualified name of the register."""
        return f"{self._full_name_prefix}{self.name}"

    @property
    def address(self) -> int:
        """Absolute address of the peripheral in memory"""
        return self._peripheral.base_address + self.offset

    @property
    def offset(self) -> int:
        """Address offset of the register, relative to the peripheral it is contained in"""
        return self._description.start_offset + self._instance_offset

    @property
    def bit_width(self) -> int:
        """Bit width of the register."""
        return self._description.reg_props.size

    @property
    def access(self) -> Access:
        """Register access."""
        return self._description.reg_props.access

    @property
    def reset_value(self) -> int:
        """Register reset value."""
        return self._description.reg_props.reset_value

    @property
    def reset_mask(self) -> int:
        """"""
        return self._description.reg_props.reset_mask

    @property
    def write_action(self) -> ModifiedWriteValues:


    @property
    def dimensions(self) -> Optional[Dimensions]:
        """Dimensions of the register, if specified."""
        return self._description.dim_props

    @property
    def has_value(self) -> bool:
        return False

    def __str__(self) -> str:
        return f"{self.__class__.__name__} {self.full_name} @ {hex(self.offset)}"

    @property
    def _peripheral_qualified_name(self) -> str:
        """Full name of the register including the parent peripheral"""
        return f"{self._peripheral.name}.{self.full_name}"


class RegisterStruct(_RegisterBase, Mapping):
    """
    Register structure representing a group of registers.
    Represents either a SVD cluster element without dimensions,
    or a specific index of a cluster array.
    """

    __slots__ = ["_registers"]

    def __init__(self, **kwargs):
        """
        Initialize the class attribute(s).
        See parent class for a description of parameters.
        """
        super().__init__(**kwargs)
        self._registers = LazyStaticMapping(
            keys=self._description.registers.keys(),
            factory=self._register_factory,
        )

    def recursive_iter(self, leaf_only: bool = False):
        """TODO"""
        if not leaf_only:
            yield self

        for register in self.values():
            yield from register.recursive_iter(leaf_only)

    def __getitem__(self, name: str) -> RegisterType:
        """Get a register by name"""
        try:
            return self._registers[name]
        except LookupError as e:
            raise KeyError(
                f"{self.__class__} {self._peripheral_qualified_name} "
                f"does not contain a register named '{name}'"
            ) from e

    def __setitem__(self, name: str, value: int) -> None:
        """Set the value of a register by name"""
        self[name].value = value

    def __iter__(self) -> Iterator[str]:
        """Iterate over the names of registers in the register structure"""
        return iter(self._registers)

    def __len__(self) -> int:
        """Get the number of registers in the register structure"""
        return len(self._registers)

    def _register_factory(self, name: str) -> RegisterType:
        """Instantiate the child register with the given name"""
        return _create_register_instance(
            description=self._description.registers[name],
            peripheral=self._peripheral,
            instance_offset=self._instance_offset,
            full_name_prefix=f"{self.full_name}.",
        )


class Register(_RegisterBase, Mapping):
    """
    Physical register instance containing a value.
    Represents either a SVD register element without dimensions,
    or a specific index of a register array.
    """

    __slots__ = ["_fields"]

    def __init__(self, **kwargs):
        """
        Initialize the class attribute(s).
        See parent class for a description of parameters.
        """
        super().__init__(**kwargs)
        self._fields = LazyStaticMapping(
            keys=self._description.fields.keys(), factory=self._field_factory
        )

    @property
    def modified(self) -> bool:
        """True if the register contains a different value now than at reset."""
        return self.value != self.reset_value

    @property
    def value(self) -> int:
        """Current value of the register."""
        return self._peripheral._values.setdefault(self.offset, self.reset_value)

    @value.setter
    def value(self, new_value: int) -> None:
        """
        Set the value of the register.

        :param new_value: New value for the register.
        """
        self.set_value(new_value)

    def set_value(self, new_value: int, mask: Optional[int] = None):
        """
        Set the value of the register.

        :param value: New value for the register.
        :param mask: Mask of the bits to copy from the given value. If None, all bits are copied.
        """
        if new_value > 0 and log2(new_value) > self._description.reg_props.size:
            raise ValueError(
                f"Value {hex(new_value)} is too large for {self._description.reg_props.size}-bit "
                f"register {self.full_name}."
            )

        for field in self.values():
            # Only check fields that are affected by the mask
            if mask is None or mask & field.mask:
                field_value = field._extract_value_from_register(new_value)
                if field_value not in field.allowed_values:
                    raise ValueError(
                        f"Value {hex(new_value)} is invalid for register {self.full_name}, as field "
                        f"{field.full_name} does not accept the value {hex(field_value)}."
                    )

        if mask is not None:
            # Update only the bits indicated by the mask
            new_value = (self.value & ~mask) | (new_value & mask)
        else:
            new_value = new_value

        self._peripheral._values[self.offset] = new_value

    @property
    def fields(self) -> Mapping[str, Field]:
        """Map of fields in the register, indexed by name"""
        return self._fields

    def unconstrain(self) -> None:
        """
        Remove all value constraints imposed on the register.
        """
        for field in self.values():
            field.unconstrain()

    def recursive_iter(self, _leaf_only: bool = False):
        """TODO"""
        yield self

    def __getitem__(self, name: str) -> Field:
        """
        :param name: Field name.

        :return: The instance of the specified field.
        """
        try:
            return self._fields[name]
        except LookupError as e:
            raise KeyError(
                f"{self.__class__} {self._peripheral_qualified_name} "
                f"does not define a field with name '{name}'"
            ) from e

    def __setitem__(self, key: str, value: Union[str, int]) -> None:
        """
        :param key: Either the bit offset of a field, or the field's name.
        :param value: A raw numeric value, or a field enumeration, to write
            to the selected register field.
        """
        self[key].value = value

    def __iter__(self) -> Iterator[str]:
        """Iterate over the field names in the register."""
        return iter(self._fields)

    def __len__(self) -> int:
        """Number of fields in the register."""
        return len(self._fields)

    def _field_factory(self, name: str) -> Field:
        """Instantiate the child field with the given name."""
        return Field(description=self._description.fields[name], register=self)

    def __repr__(self) -> str:
        """Basic representation of the class object."""
        return f"{super().__repr__()} {'(modified) ' if self.modified else ''}= 0x{self.value:08x}"

    def __str__(self) -> str:
        """String representation of the class."""

        attr_str = pformat(
            {
                "Modified": self.modified,
                "Value": f"{self.value:08x}",
                "Fields": {k: str(v) for k, v in self.items()},
            }
        )

        return f"{super().__str__()}:\n{textwrap.indent(attr_str, '  ')}"


class _DimensionedRegister(_RegisterBase, Sequence):
    """
    Base class for register arrays.
    """

    __slots__ = ["_array_offsets", "_array"]

    # Register type contained in the register array, to be set by child classes
    member_type: type

    def __init__(self, description: _RegisterDescription, **kwargs):
        self._array_offsets: Sequence[int] = description.dim_props.to_range()
        self._array = LazyStaticList(
            length=len(self._array_offsets), factory=self._register_factory
        )
        super().__init__(description=description, **kwargs)

    def recursive_iter(self, leaf_only: bool = False) -> Iterator[RegisterType]:
        """"""
        if not leaf_only:
            yield self

        for child in self:
            yield from child.recursive_iter(leaf_only)

    def _register_factory(self, index: int) -> RegisterType:
        """Initialize the register at the given index"""
        return self.member_type(
            self._description,
            self._peripheral,
            instance_offset=self._instance_offset + self._array_offsets[index],
            index=index,
            full_name_prefix=self._full_name_prefix,
        )

    def __getitem__(self, index: int) -> RegisterType:
        try:
            return self._array[index]
        except IndexError as e:
            raise IndexError(
                f"{self.__class__} {self._peripheral_qualified_name}<{len(self)}>: "
                f"array index {index} is out of range"
            ) from e

    def __iter__(self) -> Iterator[RegisterType]:
        """Iterate over the registers in the register array"""
        return iter(self._array)

    def __len__(self) -> int:
        """Number of registers in the register array"""
        return len(self._array)


class RegisterStructArray(_DimensionedRegister):
    """
    Array of RegisterStruct objects.
    SVD cluster elements with dimensions are represented using this class.
    """

    @property
    def member_type(self) -> type:
        return RegisterStruct


class RegisterArray(_DimensionedRegister):
    """
    Array of Register objects.
    SVD register elements with dimensions are represented using this class.
    """

    @property
    def member_type(self) -> type:
        return Register


# Union of all register types
RegisterType = Union[Register, RegisterArray, RegisterStruct, RegisterStructArray]


def _create_register_instance(
    description: _RegisterDescription, index: Optional[int] = None, **kwargs
) -> RegisterType:
    """
    Create a mutable register instance from a register description.

    :param description: Register description
    :param index: Index of the register in the parent register array, if applicable
    :return: Register instance
    """
    if description.registers is not None:
        if description.dim_props is not None and index is None:
            return RegisterStructArray(description=description, **kwargs)
        return RegisterStruct(description=description, index=index, **kwargs)
    else:
        if description.dim_props is not None and index is None:
            return RegisterArray(description=description, **kwargs)
        return Register(description=description, index=index, **kwargs)


class _FieldDescription(NamedTuple):
    """
    Class containing immutable data describing a SVD field element.
    This is separated from the Field class to optimize construction of derived peripherals.
    Since the description is not tied to a specific Peripheral instance, it can be
    instantiated once per field in the SVD document and reused for derived peripherals.
    """

    name: str
    bit_range: bindings.BitRange
    enums: Dict[str, int]
    allowed_values: Collection[int]
    element: bindings.FieldElement

    @classmethod
    def from_element(cls, element: bindings.FieldElement) -> _FieldDescription:
        """
        Construct a Field class from a SVD element.

        :param element: ElementTree representation of an SVD Field element.
        """

        name = element.name
        bit_range = element.bit_range

        # We do not support "do not care" bits, as by marking bits "x", see
        # SVD docs "/device/peripherals/peripheral/registers/.../enumeratedValue"
        if element.enumerated_values is not None:
            enums = {e.name: e.value for e in element.enumerated_values}
        else:
            enums = {}

        allowed_values = set(enums.values()) if enums else range(2**bit_range.width)

        return cls(
            name=name,
            bit_range=bit_range,
            enums=enums,
            allowed_values=allowed_values,
            element=element,
        )


class Field:
    """
    Register field instance.
    Represents a SVD field element.
    """

    __slots__ = ["_description", "_register", "_allowed_values"]

    def __init__(
        self,
        description: _FieldDescription,
        register: Register,
    ):
        """
        Initialize the class attribute(s).

        :param description: Field description.
        :param register: Register to which the field belongs.
        """
        self._description: _FieldDescription = description
        self._register: Register = register
        self._allowed_values = description.allowed_values

    @property
    def name(self) -> str:
        """Name of the field."""
        return self._description.name

    @property
    def full_name(self) -> str:
        """The full name of the field, including the register name."""
        return f"{self._register.full_name}.{self.name}"

    @property
    def value(self) -> int:
        """The value of the field."""
        return self._extract_value_from_register(self._register.value)

    @value.setter
    def value(self, new_value: Union[int, str]):
        """
        Set the value of the field.

        :param value: A numeric value, or the name of a field enumeration (if applicable), to
            write to the field.
        """

        if not isinstance(new_value, (int, str)):
            raise TypeError(
                f"Field does not accept write of '{new_value}' of type '{type(new_value)}'"
                " Permitted values types are 'str' (field enum) and 'int' (bit value)."
            )

        if isinstance(new_value, int):
            val = self._trailing_zero_adjusted(new_value)

            if val not in self.allowed_values:
                raise ValueError(
                    f"Field '{self.full_name}' does not accept"
                    f" the bit value '{val}' ({hex(val)})."
                    " Are you sure you have an up to date .svd file?"
                )
            resolved_value = val
        else:
            if new_value not in self.enums:
                raise ValueError(
                    f"Field '{self.full_name}' does not accept"
                    f" the enum '{new_value}'."
                    " Are you sure you have an up to date .svd file?"
                )
            resolved_value = self.enums[new_value]

        self._register.set_value(resolved_value << self.bit_offset, self.mask)

    @property
    def reset_value(self) -> int:
        """Default field value."""
        return self._extract_value_from_register(self._register.reset_value)

    @property
    def bit_offset(self) -> int:
        """Bit offset of the field."""
        return self._description.bit_range.offset

    @property
    def bit_width(self) -> int:
        """Bit width of the field."""
        return self._description.bit_range.width

    @property
    def mask(self) -> int:
        """Bitmask of the field."""
        return ((1 << self.bit_width) - 1) << self.bit_offset

    @property
    def access(self) -> Access:
        ...

    @property
    def allowed_values(self) -> Collection[int]:
        """
        Possible valid values for the bitfield.
        By default, the values allowed for the field are defined by the field enumeration
        values. If the field does not have enumerations, all values that fit within the
        field bit width are allowed.
        """
        return self._allowed_values

    @property
    def enums(self) -> Mapping[str, int]:
        """
        A mapping between field enumerations and their corresponding values.
        Field enumerations are values such as "Allowed" = 1, "NotAllowed" = 0
        and are defined by the device's SVD file. This may be an empty map,
        if enumerations are not applicable to the field.
        """
        return self._description.enums

    @property
    def parent_register(self) -> Register:
        """Register to which the field belongs."""
        return self._register

    @property
    def modified(self) -> bool:
        """True if the field contains a different value now than at reset."""
        return self.value != self.reset_value

    def unconstrain(self) -> None:
        """
        Remove restrictions on values that may be entered into this field. After this,
        the field will accept any value that can fit inside its bit width.
        """
        self._allowed_values = range(2**self.bit_width)

    def _extract_value_from_register(self, register_value: int) -> int:
        """
        Internal method for extracting the field value from the parent register value.

        :param register_value: Value of the parent register
        :return: Field value extracted based on the field bit range
        """
        return (register_value & self.mask) >> self.bit_offset

    def _trailing_zero_adjusted(self, value):
        """
        Internal method that checks and adjusts a given value for trailing zeroes if it exceeds
        the bit width of its field. Some values are simplest to encode as a full 32-bit value even
        though their field is comprised of less than 32 bits, such as an address.

        :param value: A numeric value to check against the field bits

        :return: Field value without any trailing zeroes
        """

        width_max = 2**self.bit_width

        if value > width_max:
            max_val = width_max - 1
            max_val_hex_len = len(f"{max_val:x}")
            hex_val = f"{value:0{8}x}"  # leading zeros, 8-byte max, in hex
            trailing = hex_val[max_val_hex_len:]  # Trailing zeros

            if int(trailing, 16) != 0:
                raise ValueError(f"Unexpected trailing value: {trailing}")

            cropped = hex_val[:max_val_hex_len]  # value w/o trailing
            adjusted = int(cropped, 16)

            if adjusted <= max_val:
                return adjusted

        return value

    def __repr__(self):
        """Basic representation of the class."""
        return f"Field {self.name} {'(modified) ' if self.modified else ''}= {hex(self.value)}"

    def __str__(self):
        attrs = {
            "Allowed": self.allowed_values,
            "Modified": self.modified,
            "Bit offset": self.bit_offset,
            "Bit width": self.bit_width,
            "Enums": self.enums,
        }
        return f"Field {self.name}: {pformat(attrs)}"


def _topo_sort_derived_peripherals(
    peripherals: Iterable[bindings.PeripheralElement],
) -> List[bindings.PeripheralElement]:
    """
    Topologically sort the peripherals based on 'derivedFrom' attributes using Kahn's algorithm.
    The returned list has the property that the peripheral element at index i does not derive from
    any of the peripherals at indices 0..(i - 1).

    :param peripherals: List of peripheral elements to sort
    :return: List of peripheral elements topologically sorted based on the 'derivedFrom' attribute.
    """

    sorted_peripherals: List[bindings.PeripheralElement] = []
    no_dep_peripherals: List[bindings.PeripheralElement] = []
    dep_graph: Dict[str, List[bindings.PeripheralElement]] = defaultdict(list)

    for peripheral in peripherals:
        if peripheral.is_derived:
            dep_graph[peripheral.derived_from].append(peripheral)
        else:
            no_dep_peripherals.append(peripheral)

    while no_dep_peripherals:
        peripheral = no_dep_peripherals.pop()
        sorted_peripherals.append(peripheral)
        # Each peripheral has a maximum of one in-edge since they can only derive from one
        # peripheral. Therefore, once they are encountered here they have no remaining dependencies.
        no_dep_peripherals.extend(dep_graph[peripheral.name])
        dep_graph.pop(peripheral.name, None)

    if dep_graph:
        raise ValueError(
            "Unable to determine order in which peripherals are derived. "
            "This is likely caused either by a cycle in the "
            "'derivedFrom' attributes, or a 'derivedFrom' attribute pointing to a "
            "nonexistent peripheral."
        )

    return sorted_peripherals


def _extract_register_descriptions(
    elements: Iterable[Union[bindings.RegisterElement, bindings.ClusterElement]],
    base_reg_props: bindings.RegisterProperties,
) -> Mapping[str, _RegisterDescription]:
    """
    Extract register descriptions for the given SVD register level elements.
    The returned structure mirrors the structure of the SVD elements.
    Each level of the structure is internally sorted by ascending address.

    :param elements: Register level elements to process.
    :param base_reg_props: Register properties inherited from the parent peripheral.
    :return: Map of register descriptions, indexed by name.
    """
    result, _ = _extract_register_descriptions_helper(elements, base_reg_props)

    return result


def _extract_register_descriptions_helper(
    elements: Iterable[Union[bindings.RegisterElement, bindings.ClusterElement]],
    base_reg_props: bindings.RegisterProperties,
    base_address: int = 0,
) -> Tuple[Mapping[str, _RegisterDescription], Optional[int]]:
    """
    Helper that recursively extracts the names, addresses, register properties, dimensions,
    fields etc. of a collection of SVD register level elements.

    :param elements: SVD register level elements.
    :param base_reg_props: Base address of the parent SVD element.
    :param base_address: Base address of the parent SVD element.

    :return: Tuple of two elements. The first element is a mapping of register descriptions indexed
             by name. The second element is the minimum address offset of any of the returned
             registers, used inside this function to sort registers while traversing.
    """
    address_descriptions: List[Tuple[int, _RegisterDescription]] = []
    min_address_total = float("inf")

    for element in elements:
        # Remove suffixes used for elements with dimensions
        name = util.strip_prefixes_suffixes(element.name, [], ["[%s]"])

        reg_props = element.get_register_properties(base_props=base_reg_props)
        dim_props = element.dimensions
        address_offset = element.offset

        if isinstance(element, bindings.RegisterElement):
            # Register addresses are defined relative to the enclosing element
            if address_offset is not None:
                address = base_address + address_offset
            else:
                address = base_address

            min_address = address
            registers = None
            fields = _extract_field_descriptions(element)

        else:  # ClusterElement
            # By the SVD specification, cluster addresses are defined relative to the peripheral
            # base address, but some SVDs don't follow this rule.
            if address_offset is not None:
                address = base_address + address_offset
                # address = address_offset
            else:
                address = base_address

            registers, min_child_address = _extract_register_descriptions_helper(
                elements=element.registers,
                base_reg_props=reg_props,
                base_address=address,
            )

            if registers:
                min_address = min(address, min_child_address)

            fields = None

        description = _RegisterDescription(
            element=element,
            name=name,
            start_offset=address,
            reg_props=reg_props,
            dim_props=dim_props,
            registers=registers,
            fields=fields,
        )

        address_descriptions.append((min_address, description))
        min_address_total = min(min_address_total, min_address)

    # No elements; return a dummy value
    if not address_descriptions:
        return {}, None

    sorted_address_descriptions = sorted(address_descriptions, key=op.itemgetter(0))

    descriptions = {
        register.name: register for _, register in sorted_address_descriptions
    }

    return descriptions, int(min_address_total)


def _extract_field_descriptions(
    elements: Iterable[bindings.FieldElement],
) -> Optional[Mapping[str, _FieldDescription]]:
    """
    Extract field descriptions for the given SVD field elements.
    The resulting mapping is internally sorted by ascending field bit offset.

    :param elements: Field elements to process.
    :return: Mapping of field descriptions, indexed by name.
    """
    field_descriptions = sorted(
        [_FieldDescription.from_element(field) for field in elements],
        key=lambda fd: fd.bit_range.offset,
    )

    fields = {description.name: description for description in field_descriptions}

    if not fields:
        return None

    return fields
