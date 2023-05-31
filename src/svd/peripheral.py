#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Python representation of an SVD Peripheral unit.
"""

from __future__ import annotations

import operator as op
from collections import ChainMap, defaultdict
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
    NamedTuple,
    Optional,
    Sequence,
)
from pprint import pformat

from . import bindings
from .bindings import (
    RegisterProperties, Dimensions
)
from . import util

# TODO: add option for not expanding arrays during recursive iter


class Device(Mapping):
    """Representation of a SVD device."""

    def __init__(self, device: bindings.DeviceElement):
        self._device: bindings.DeviceElement = device
        self._reg_props: RegisterProperties = (
            self._device.get_register_properties()
        )

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

    def __getitem__(self, name: str) -> Peripheral:
        """Get a peripheral by name."""
        try:
            return self._peripherals[name]
        except KeyError as e:
            raise KeyError(f"Device does not contain a peripheral named {name}") from e

    def __iter__(self) -> Iterable[str]:
        """Iterate over the names of peripherals in the device."""
        return iter(self._peripherals)

    def __len__(self) -> int:
        """Get the number of peripherals in the device."""
        return len(self._peripherals)

    @property
    def peripherals(self) -> Dict[str, Peripheral]:
        """Map of peripherals in the device, indexed by name."""
        return self._peripherals


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

        self._registers: Dict[str, RegisterType] = {}

    @property
    def name(self) -> str:
        """Name of the peripheral."""
        return self._peripheral.name

    @property
    def base_address(self) -> int:
        """Base address of the peripheral in memory."""
        return self._base_address

    # FIXME: reconsider. Maybe make into a constructor or give as an offset to memory iter
    @base_address.setter
    def base_address(self, address: int):
        """Set the base address of the peripheral."""
        self._base_address = address

    def recursive_iter(self, leaf_only: bool = False):
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
    def _instance_map(self) -> Dict[str, int]:
        """Map of the peripheral register instances in memory."""
        return {reg.name: address for address, reg in self.memory_map.items()}

    @cached_property
    def _register_tree(self) -> Mapping[str, _RegisterDescription]:
        # Add the registers defined in this peripheral
        registers = _expand_registers(self._peripheral.registers, self._reg_props)

        # Add the registers defined in the base peripheral, if any
        # FIXME: need to merge registers to ensure sorted
        if self._base_peripheral is not None:
            # If the register properties are equal, then it is possible to reuse all the immutable
            # properties from the base peripheral.
            if self._base_peripheral._reg_props == self._reg_props:
                registers = ChainMap(registers, self._base_peripheral._register_tree)
            # Otherwise, traverse the base registers again, because the difference in
            # register properties propagates down to the register elements.
            elif base_registers := self._base_peripheral._peripheral.registers:
                base_registers = _expand_registers(base_registers, self._reg_props)
                registers = ChainMap(registers, base_registers)

        return registers

    def _register_factory(self, name: str) -> RegisterType:
        try:
            return _create_register(self._register_tree[name], peripheral=self)
        except LookupError as e:
            raise KeyError(
                f"Peripheral does not contain a register named '{name}'"
            ) from e

    def __getitem__(self, name: str) -> Register:
        """
        :param name: Name of the register to get.

        :return: The instance of the specified register.
        """
        if (existing_register := self._registers.get(name)) is not None:
            return existing_register


        self._registers[name] = register

        return register

    def __setitem__(self, name: str, value: int):
        """
        :param name: Name of the register to update.
        :param value: The raw register value to write to the specified register.
        """

        self[name].value = value

    def __iter__(self):
        """Iterate over the names of registers in the peripheral."""
        return iter(self._registers)

    def __len__(self):
        """Get the number of registers in the peripheral."""
        return len(self._registers)

    def __repr__(self) -> str:
        return f"Peripheral {self.name.upper()}@{hex(self.base_address)}"

    def __str__(self) -> str:
        periph = {hex(k): v for k, v in self.memory_map.items() if v.modified}
        return f"Peripheral {self.name.upper()}@{hex(self.base_address)}:\n{pformat(periph)}"


class _RegisterDescription(NamedTuple):
    """
    Description of a register, extracted from the SVD nodes.
    This description is static
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
        "_qualified_prefix",
    ]

    def __init__(
        self,
        description: _RegisterDescription,
        peripheral: Peripheral,
        instance_offset: int = 0,
        index: Optional[int] = None,
        qualified_prefix: str = "",
    ):
        """
        Initialize the class attribute(s).

        :param name: Register name
        :param fields: Dictionary of bitfields present in the register
        :param reset_value: Register reset value
        """
        self._description: _RegisterDescription = description
        self._peripheral: Peripheral = peripheral
        self._instance_offset: int = instance_offset
        self._index: Optional[int] = index
        self._qualified_prefix: str = qualified_prefix

    @property
    def name(self) -> str:
        """Name of the register."""
        if self._index is not None:
            return f"{self._description.name}[{self._index}]"
        return self._description.name

    @property
    def full_name(self) -> str:
        """Full qualified name of the register."""
        return f"{self._qualified_prefix}{self.name}"

    @property
    def address(self) -> int:
        """Absolute address of the peripheral in memory"""
        return self._peripheral.base_address + self.offset

    @property
    def offset(self) -> int:
        """Address offset of the register, relative to the peripheral it is contained in"""
        return self._description.start_offset + self._instance_offset

    # FIXME: should this be different in any way?
    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.full_name} @ {hex(self.offset)}"

    def __str__(self) -> str:
        return f"{self.__class__.__name__} {self.full_name} @ {hex(self.offset)}"


class _DimensionedRegister(_RegisterBase, Sequence):
    """
    Base class for register arrays.
    """

    __slots__ = ["_array_offsets", "_array"]

    member_type: type

    def __init__(self, description: _RegisterDescription, *args, **kwargs):
        self._array_offsets: Sequence[int] = description.dim_props.to_range()
        self._array = [None for _ in range(len(self._array_offsets))]
        super().__init__(description, *args, **kwargs)

    def recursive_iter(self, leaf_only: bool = False):
        """TODO"""
        if not leaf_only:
            yield self

        for child in self:
            yield from child.recursive_iter(leaf_only)

    def __getitem__(self, index: int):
        if (existing_child := self._array[index]) is not None:
            return existing_child

        new_register = self.member_type(
            self._description,
            self._peripheral,
            instance_offset=self._instance_offset + self._array_offsets[index],
            index=index,
            qualified_prefix=self._qualified_prefix,
        )
        self._array[index] = new_register

        return new_register

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return len(self._array)


class RegisterStruct(_RegisterBase, Mapping):
    """TODO"""

    __slots__ = ["_registers"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize with None values - this ensures that iteration happens in address sorted order
        self._registers = {name: None for name in self._description.registers}

    def recursive_iter(self, leaf_only: bool = False):
        """TODO"""
        if not leaf_only:
            yield self

        for register in self.values():
            yield from register.recursive_iter(leaf_only)

    def __getitem__(self, name: str):
        """Get a register by name"""
        if (existing_register := self._registers.get(name)) is not None:
            return existing_register

        try:
            child_description = self._description.registers[name]
        except LookupError as e:
            raise KeyError(
                f"Register structure {self.full_name} does not contain a register named {name}"
            ) from e

        new_register = _create_register(
            child_description,
            peripheral=self._peripheral,
            instance_offset=self._instance_offset,
            qualified_prefix=f"{self.full_name}.",
        )
        self._registers[name] = new_register

        return new_register

    def __setitem__(self, name: str, value: int):
        """Set a register by name"""
        register = self[name]

        try:
            self[name].value = value
        except AttributeError as e:
            raise AttributeError(
                f"Register {register.full_name} does not have a value"
            ) from e

    def __len__(self) -> int:
        return len(self._registers)

    def __iter__(self):
        return iter(self._registers)


class Register(_RegisterBase, Mapping):
    """
    Internal representation of a peripheral register.
    Not intended for direct user interaction.
    """

    __slots__ = ["_fields"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fields = {}

    @property
    def reset_value(self) -> int:
        """Register reset value."""
        return self._description.reg_props.reset_value

    @property
    def modified(self) -> bool:
        """True if the register contains a different value now than at reset."""
        return self.value != self.reset_value

    @property
    def value(self) -> int:
        """Current value of the register."""
        return self._peripheral._values.setdefault(self.offset, self.reset_value)

    @value.setter
    def value(self, value: int):
        """
        Set the value of the register.

        :param value: New value for the register.
        """
        self.set_value(value)

    def set_value(self, value: int, mask: Optional[int] = None):
        """
        Set the value of the register.

        :param value: New value for the register.
        :param mask: Mask of the bits to copy from the given value. If None, all bits are copied.
        """
        if value > 0 and log2(value) > self._description.reg_props.size:
            raise ValueError(
                f"Value {hex(value)} is too large for {self._description.reg_props.size}-bit "
                f"register {self.full_name}."
            )

        for field in self.values():
            # Only check fields that are affected by the mask
            if mask is None or mask & field.mask:
                field_value = field._extract_value_from_register(value)
                if field_value not in field.allowed_values:
                    raise ValueError(
                        f"Value {hex(value)} is invalid for register {self.full_name}, as field "
                        f"{field.full_name} does not accept the value {hex(field_value)}."
                    )

        if mask is not None:
            # Update only the bits indicated by the mask
            new_value = (self.value & ~mask) | (value & mask)
        else:
            new_value = value

        self._peripheral._values[self.offset] = new_value

    def unconstrain(self):
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
        if (existing_field := self._fields.get(name)) is not None:
            return existing_field

        try:
            field_description = self._description.fields[name]
        except LookupError as e:
            raise KeyError(
                f"Register '{self.full_name}' does not define a field with name '{name}'"
            ) from e

        field = Field(field_description, self)
        self._fields[name] = field

        return field

    def __setitem__(self, key: str, value: Union[str, int]):
        """
        :param key: Either the bit offset of a field, or the field's name.
        :param value: A raw numeric value, or a field enumeration, to write
            to the selected register field.
        """
        self[key].value = value

    def __iter__(self):
        """Iterate over the fields of the register."""
        return iter(self._fields)

    def __len__(self):
        """Number of fields in the register."""
        return len(self._fields)

    def __repr__(self):
        """Basic representation of the class object."""
        return f"{super().__repr__()} {'(modified) ' if self.modified else ''}= 0x{self.value:08x}"

    def __str__(self):
        """String representation of the class."""

        attrs = {
            "Modified": self.modified,
            "Value": f"{self.value:08x}",
            "Fields": {k: str(v) for k, v in self.items()},
        }

        return f"{super().__str__()}: {pformat(attrs)}"


class RegisterStructArray(_DimensionedRegister):
    """Array of RegisterStruct objects."""

    member_type = RegisterStruct


class RegisterArray(_DimensionedRegister):
    """Array of Register objects."""

    member_type = Register


# Union of all register types
RegisterType = Union[Register, RegisterArray, RegisterStruct, RegisterStructArray]


def _create_register(
    description: _RegisterDescription, index: Optional[int] = None, **kwargs
) -> RegisterType:
    """
    Create a register object from a register description.
    The type of the register object returned depends on the description.
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
    name: str
    bit_range: bindings.BitRange
    enums: Dict[str, int]
    allowed_values: Collection[int]
    element: bindings.FieldElement

    @classmethod
    def from_element(cls, element: bindings.FieldElement) -> _FieldDescription:
        """
        Construct a Field class from an SVD element.

        :param element: ElementTree representation of an SVD Field element.
        :param name: Name of field.
        :param reset_value: Reset value of field.
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
    """Register field instance"""

    __slots__ = ["_description", "_register", "_allowed_values"]

    def __init__(
        self,
        description: _FieldDescription,
        register: Register,
    ):
        """
        Initialize the class attribute(s).

        :param name: Name of field.
        :param bit_offset: The field's bit offset from position 0. Same as bit position.
        :param bit_width: The bit width of the field.
        :param default_value: The value the field has at reset.
        :param enums: A mapping between field enumerations and their corresponding raw
            values. Field enumerations are values such as "Allowed" = 1, "NotAllowed" = 0
            and are defined by the device's SVD file. This is allowed to be an empty map,
            if enumerations are not applicable to the field.
        :param allowed_values: The values this field accepts. May be either a list of
            allowed values, such as [0, 1], or a range - in case the field consists of
            several bits that may all either be set or not.
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

        :param value: A numeric value, or a field enumeration (if applicable), to
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
        """Default bitfield value."""
        return self._extract_value_from_register(self._register.reset_value)

    @property
    def bit_offset(self) -> int:
        """Bit offset of the field. Same as the field's bit position."""
        return self._description.bit_range.offset

    @property
    def bit_width(self) -> int:
        """Width of bits in the field."""
        return self._description.bit_range.width

    @property
    def mask(self) -> int:
        """Bitmask of the field."""
        return ((1 << self.bit_width) - 1) << self.bit_offset

    @property
    def allowed_values(self) -> Collection[int]:
        """Possible valid values for the bitfield."""
        return self._allowed_values

    @property
    def enums(self) -> Dict[str, int]:
        """Dictionary of the bitfield enumerations in the field."""
        return self._description.enums

    @property
    def parent_register(self) -> Register:
        """Register to which the field belongs."""
        return self._register

    @property
    def modified(self) -> bool:
        """True if the field contains a different value now than at reset."""
        return self.value != self.reset_value

    def unconstrain(self):
        """
        Remove restrictions on values that may be entered into this field. After this,
        the field will accept any value that can fit inside its bit width.
        """
        self._allowed_values = range(2**self.bit_width)

    def _extract_value_from_register(self, register_value: int) -> int:
        """Extract the field value from a register value."""
        return (register_value & self.mask) >> self.bit_offset

    def _trailing_zero_adjusted(self, value):
        """
        Checks and adjusts a given value for trailing zeroes if it exceeds the bit width of its
        field. Some values are simplest to encode as a full 32-bit value even though their field
        is comprised of less than 32 bits, such as an address.

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
        """String representation of the class."""

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


def _expand_registers(
    elements: Iterable[Union[bindings.RegisterElement, bindings.ClusterElement]],
    base_reg_props: bindings.RegisterProperties,
) -> Dict[int, Register]:
    """
    Get the memory map of a peripheral unit given by an SVD element and its
    base address.

    :param element: ElementTree representation of the SVD peripheral.
    :param base_address: Base address of peripheral unit.

    :return: Mapping from addresses to Registers.
    """
    result, _ = _get_register_elements(elements, base_reg_props)

    return result


# TODO: maybe cache the leaf nodes here
def _get_register_elements(
    elements: Iterable[Union[bindings.RegisterElement, bindings.ClusterElement]],
    base_reg_props: bindings.RegisterProperties,
    base_address: int = 0,
) -> Tuple[Optional[Dict[str, RegisterType]], Optional[int]]:
    """
    Helper that recursively extracts the addresses, names, and ElementTree representations of all
    SVD Registers that are children of a given SVD element.

    :param element: ElementTree representation of an enclosing SVD element.
    :param base_address: Base address of the parent SVD element.
    :param prefix: Name prefix of the current Register. This is primarily meaningful for nested SVD
        Clusters and Registers.
    :param reset_value: Last observed reset value thus far. May be overridden if a more specific
        reset value is observed.

    :return: Mapping between Register addresses and their ElementTree representing element, names,
        and reset values.
    """
    result_list = []
    min_address_total = float("inf")

    for element in elements:
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
            fields = _expand_element_fields(element)

        else:  # ClusterElement
            if address_offset is not None:
                # Nordic SVDs are broken wrt. addressOffset in cluster elements
                address = base_address + address_offset
                # address = address_offset
            else:
                address = base_address

            registers, min_child_address = _get_register_elements(
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

        result_list.append((min_address, description))
        min_address_total = min(min_address_total, min_address)

    if not result_list:
        return None, None

    result = {
        register.name: register
        for _, register in sorted(result_list, key=op.itemgetter(0))
    }

    return result, min_address_total


def _expand_element_fields(element):
    # FIXME: sorted?
    fields = {
        field.name: _FieldDescription.from_element(field) for field in element.fields
    }

    if not fields:
        return None

    return fields

