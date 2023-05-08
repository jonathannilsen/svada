#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Python representation of an SVD Peripheral unit.
"""

from __future__ import annotations

import copy
import operator as op
from collections import ChainMap, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain, product
from typing import List, Tuple, Union, Dict, Iterable, NamedTuple, Optional, Set
from pprint import pformat

import lxml.etree as ET

from . import bindings
from . import util
from .svd_enums import *


class Device:
    """
    Representation of a device.
    TODO
    """

    def __init__(self, device: bindings.DeviceElement):
        self._device = device
        self._reg_props = RegisterProperties(self._device)

        peripherals = {}
        for peripheral_element in topo_sort_derived_peripherals(
            device.peripherals.iterchildren()
        ):
            if (derived_from := peripheral_element.get("derivedFrom")) is not None:
                base_peripheral = peripherals[derived_from]
            else:
                base_peripheral = None

            peripheral = Peripheral(
                peripheral_element,
                base_reg_props=self._reg_props,
                base_peripheral=base_peripheral,
            )
            peripherals[peripheral.name] = peripheral

        self._peripherals = peripherals

    @property
    def peripherals(self) -> Dict[str, Peripheral]:
        """Map of Peripherals in the device, indexed by name"""
        return self._peripherals


# Support __delitem__ to remove registers
# (internally this could just hide the registers if necessary)


class Peripheral:
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
        base_reg_props: RegisterProperties,
        base_peripheral: Optional[Peripheral] = None,
    ):
        """
        Initialize the class attribute(s).

        :param peripheral_node: ElementTree element for the peripheral node in the device SVD file.
        :param base_address: Specific base address to use for the peripheral.
        """
        self._peripheral: bindings.PeripheralElement = peripheral
        self._base_peripheral: Optional[Peripheral] = base_peripheral
        self._name: str = peripheral.name.pyval
        self._base_address: int = peripheral.baseAddress.pyval
        self._reg_props: RegisterProperties = RegisterProperties(
            self._peripheral, base=base_reg_props
        )
        self._registers = {}

    @property
    def name(self) -> str:
        """Simplified name of the peripheral."""
        return self._name

    # FIXME: iterator?
    def memory_iter(self):
        """Iterator over registers in the peripheral
        TODO: option for leaf-only? or have a different function for recursive
        """
        for description in self._register_tree.values():
            register = _create_register(description, self)
            yield from register.memory_iter()

    # TODO: register_walk() ?

    @cached_property
    def memory_map(self) -> Dict[int, Register]:
        """Map of the peripheral register contents in memory."""
        return dict(self.memory_iter())

    @cached_property
    def _instance_map(self) -> Dict[str, int]:
        """Map of the peripheral register instances in memory."""
        return {reg.name: address for address, reg in self.memory_map.items()}

    @cached_property
    def _register_tree(self) -> Mapping[str, _RegisterDescription]:
        # Add the registers defined in this peripheral
        if (registers_element := self._peripheral.registers) is not None:
            own_registers = registers_element.iterchildren()
            registers = expand_registers(own_registers, self._reg_props)
        else:
            registers = {}

        # Add the registers defined in the base peripheral, if any
        # FIXME: need to merge registers to ensure sorted
        if self._base_peripheral is not None:
            # If the register properties are equal, then it is possible to reuse all the immutable
            # properties from the base peripheral.
            if self._base_peripheral._reg_props == self._reg_props:
                registers = ChainMap(registers, self._base_peripheral._registers)
            # Otherwise, traverse the base registers again, because the difference in
            # register properties propagates down to the register elements.
            elif (
                base_registers_element := self._base_peripheral._peripheral.registers
            ) is not None:
                base_registers = base_registers_element.iterchildren()
                base_registers = expand_registers(base_registers, self._reg_props)
                registers = ChainMap(registers, base_registers)

        return registers

    @property
    def base_address(self) -> int:
        """Base address of the peripheral space in memory."""
        return self._base_address

    @base_address.setter
    def base_address(self, value: int):
        self._base_address = value

    def __getitem__(self, key: Union[int, str]) -> Register:
        """
        :param key: Either the address of a register, or the register's name and instance; for
            example "mem.config.0".

        :return: The instance of the specified register.
        """

        if not isinstance(key, (int, str)):
            raise TypeError(
                f"Peripheral does not allow key of type '{type(key)}' for register"
                " lookup. Permitted key types are 'str' (register name)"
                " and 'int' (address)."
            )

        if isinstance(key, int):
            if key not in self.memory_map:
                raise LookupError(
                    f"Peripheral does not contain a memory map for address '{hex(key)}'"
                )
            return self.memory_map[key]

        if (existing_register := self._registers.get(key)) is not None:
            return existing_register

        try:
            register = _create_register(self._register_tree[key], self)
            self._registers[key] = register
            return register
        except LookupError as e:
            raise LookupError(f"Peripheral does not contain a register named '{key}'") from e

    def __setitem__(self, key: Union[int, str], value: int):
        """
        :param key: Either the address of a register, or the register's name and instance; for
            example "mem.config.0".
        :param value: The raw register value to write to the specified register.
        """

        self[key].set(value)

    def __repr__(self) -> str:
        """Base representation of the class."""
        return f"Peripheral {self.name.upper()}@{hex(self.base_address)}"

    def __str__(self) -> str:
        """String representation of the class."""
        periph = {hex(k): v for k, v in self.memory_map.items() if v.modified}
        return f"Peripheral {self.name.upper()}@{hex(self.base_address)}:\n{pformat(periph)}"

    def unconstrain_path(self, path: str):
        """
        Remove all value constraints imposed on a given register. Optionally, only
        selected fields within a register may be unconstrained by adding a dot (".")
        to the register name, followed by the name of the field to unconstrain.
        Register names must include the instance qualifier.

        :param path: Fully qualified register path, with optional field information.
        """

        register_field = path.strip(".").split(".")

        if len(register_field) not in [1, 2]:
            raise ValueError(
                f"Cannot unconstrain the register path '{path}'."
                " The accepted format is the name of a register along with its"
                " instance number, optionally followed by a dot '.' and a field name"
            )

        affected_register = next(
            (r for r in self.memory_map.values() if r.name == register_field[0]), None
        )

        if affected_register is None:
            raise ValueError(
                f"Register '{register_field[0]}' not found in peripheral."
                " Maybe you misspelled the name or forgot the instance number?"
            )

        field_match = (
            lambda f: f.name == register_field[1]
            if len(register_field) == 2
            else lambda _: True
        )

        affected_fields = list(filter(field_match, affected_register.field_iter()))

        if not any(affected_fields):
            raise ValueError(
                f"No field named '{register_field[1]}' found in"
                f" register '{register_field[0]}'"
            )

        for field in filter(field_match, affected_register.field_iter()):
            field.unconstrain()


class _RegisterDescription(NamedTuple):
    name: str
    offset: int
    reg_props: RegisterProperties
    dim_props: Optional[DimensionProperties]
    registers: Dict[str, _RegisterDescription]
    fields: Dict[str, _FieldDescription]
    element: Union[bindings.RegisterElement, bindings.ClusterElement]


class _RegisterBase:
    __slots__ = ["_description", "_peripheral", "_qualified_prefix", "_offset"]

    # FIXME: need to pass down parent offset
    def __init__(
        self, description, peripheral, qualified_prefix: str = "", offset=None
    ):
        """
        Initialize the class attribute(s).

        :param name: Register name
        :param fields: Dictionary of bitfields present in the register
        :param reset_value: Register reset value
        """
        self._description: _RegisterDescription = description
        self._peripheral: Peripheral = peripheral
        self._qualified_prefix: str = qualified_prefix
        # FIXME: change the name of this - it's really confusing
        self._offset = offset

    @property
    def name(self) -> str:
        """Name of the register."""
        if self._offset is not None:
            return f"{self._description.name}[{self._offset.index}]"
        return self._description.name

    @property
    def full_name(self) -> str:
        return f"{self._qualified_prefix}{self.name}"

    @property
    def address(self) -> int:
        return self._peripheral.base_address + self.offset

    @property
    def offset(self) -> int:
        if self._offset is not None:
            return self._description.offset + self._offset.offset
        return self._description.offset

    def memory_iter(self):
        if isinstance(self, Iterable):
            for child in self:
                yield from child.memory_iter()
        else:
            yield (self.address, self)

    def __repr__(self) -> str:
        return str(self)  # FIXME

    def __str__(self) -> str:
        return f"{self.__class__.__name__} '{self.full_name}' @ 0x{self.address:08x}"


class Offset(NamedTuple):
    index: int
    offset: int


class _DimensionedRegister(_RegisterBase):
    __slots__ = ["_offsets", "_array"]

    member_type: type
    _description: _RegisterDescription
    _peripheral: Peripheral
    _qualified_prefix: str

    def __init__(self, description: _RegisterDescription, *args, **kwargs):
        self._offsets = description.dim_props.to_range()
        self._array = [None for _ in range(len(self._offsets))]
        super().__init__(description, *args, **kwargs)

    def memory_iter(self):
        for child in self:
            yield from child.memory_iter()

    def __getitem__(self, index: int):
        if (existing_child := self._array[index]) is not None:
            return existing_child

        offset = Offset(index, self._offsets[index])
        return self.member_type(
            self._description,
            self._peripheral,
            qualified_prefix=self._qualified_prefix,
            offset=offset,
        )

    # FIXME: iter type breaks with how mapping does it
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return len(self._array)


class RegisterStruct(_RegisterBase, Mapping):
    __slots__ = ["_registers"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._registers = {}

    def memory_iter(self):
        for register in self:
            yield from register.memory_iter()

    def __getitem__(self, name: str):
        # TODO: could implement a lookup in cached regs in the parent peripheral here
        if (existing_register := self._registers.get(name)) is not None:
            return existing_register

        try:
            child_description = self._description.registers[name]
            return _create_register(
                child_description,
                self._peripheral,
                qualified_prefix=f"{self.full_name}.",
                offset=self._offset
            )
        except LookupError as e:
            raise LookupError(
                f"no register named {name} in register structure {self.name}"
            ) from e

    def __setitem__(self, name: str, value: int):
        self[name].value = value

    def __len__(self) -> int:
        return len(self._description.registers)

    def __iter__(self):
        for key in self._description.registers:
            yield self[key]


class RegisterStructArray(_DimensionedRegister):
    member_type = RegisterStruct


class Register(_RegisterBase):
    """
    Internal representation of a peripheral register.
    Not intended for direct user interaction.
    """
    __slots__ = ["_fields"]

    # FIXME
    #def __init__(self, *):
    #    ...
    #    self._fields = {}

    @cached_property
    def fields(self) -> Dict[int, Field]:
        """Register bitfields."""
        return {
            field.bit_offset: Field(field, self)
            for field in self._description.fields.values()
        }

    @property
    def reset_value(self) -> int:
        """Register reset value."""
        return self._description.reg_props.reset_value

    @property
    def modified(self) -> bool:
        """True if the register contains a different value now than at reset."""
        return any([field.modified for field in self.fields.values()])

    def raw(self) -> int:
        """The raw numeric value the register contains."""
        return self._peripheral._values.setdefault(
            self._description.address, self.reset_value
        )
        # value = self.reset_value
        # for offset, field in self.fields.items():
        #    value = (value & ~field.mask) | ((field.raw << offset) & field.mask)
        # return value

    # def __repr__(self):
    #    """Basic representation of the class object."""
    #    return f"Register {self.name} {'(modified) ' if self.modified else ''}= {hex(self.raw)}"
    #
    # def __str__(self):
    #    """String representation of the class."""
    #
    #    attrs = {
    #        "modified": self.modified,
    #        "value": self.raw,
    #        "Fields": {k: str(v) for k, v in self.fields.items()},
    #    }
    #    return f"Register {self.name}: {pformat(attrs)}"

    def __getitem__(self, key: str) -> Field:
        """
        :param key: Either the bit offset of a field, or the field's name.

        :return: The instance of the specified field.
        """

        if not isinstance(key, str):
            raise TypeError(
                f"Register does not allow key of type '{type(key)}' for field lookup."
                " Permitted key types are 'str' (field name) and 'int' (bit offset)."
            )

        try:
            field_description = self._description.fields[key]
            return Field(field_description, self)
        except LookupError as e:
            raise LookupError(
                f"Register '{self.name}' does not define a field with name '{key}'"
            ) from e

    def __setitem__(self, key: Union[int, str], value: Union[str, int]):
        """
        :param key: Either the bit offset of a field, or the field's name.
        :param value: A raw numeric value, or a field enumeration, to write
            to the selected register field.
        """

        self[key].set(value)

    def set(self, value: int, mask: Optional[int] = None):
        """
        Set all the fields in the register, given an overall raw register value.

        :param value: Raw register value.
        """
        # TODO: validate against fields
        # for field in self.fields.values():
        #     field.set_from_register(value)

        if mask is not None:
            prev_value = self._peripheral._values.get(
                self._description.address, self.reset_value
            )
            new_value = (prev_value & ~mask) | (value & mask)
        else:
            new_value = value

        self._peripheral.values[self._description.address] = new_value

    def field_iter(self) -> Iterable[Field]:
        """
        :return: Iterator over the register's fields.
        """
        return iter(self.fields.values())


class RegisterArray(_DimensionedRegister):
    member_type = Register



RegisterType = Union[Register, RegisterStruct]


def _create_register(description: _RegisterDescription, peripheral: Peripheral, qualified_prefix: str = "", offset=None):
    if description.registers is not None:
        if description.dim_props is not None and offset is None:
            return RegisterStructArray(description=description, peripheral=peripheral, qualified_prefix=qualified_prefix)
        return RegisterStruct(description=description, peripheral=peripheral, qualified_prefix=qualified_prefix, offset=offset)
    else:
        if description.dim_props is not None and offset is None:
            return RegisterArray(description=description, peripheral=peripheral, qualified_prefix=qualified_prefix)
        return Register(description=description, peripheral=peripheral, qualified_prefix=qualified_prefix, offset=offset)


class _FieldDescription(NamedTuple):
    name: str
    bit_offset: int
    bit_width: int
    enums: Dict[str, int]
    allowed_values: Union[List[int], range]
    element: bindings.FieldElement

    @classmethod
    def from_element(cls, element: bindings.FieldElement) -> Field:
        """
        Construct a Field class from an SVD element.

        :param element: ElementTree representation of an SVD Field element.
        :param name: Name of field.
        :param reset_value: Reset value of field.
        """

        name = element.name.pyval
        # FIXME move get_bit_range() back to util function
        bit_offset, bit_width = element.get_bit_range()

        # We do not support "do not care" bits, as by marking bits "x", see
        # SVD docs "/device/peripherals/peripheral/registers/.../enumeratedValue"
        if (enum_values := element.enumeratedValues) is not None:
            enums = {
                enum.name.pyval: enum.value.pyval
                for enum in enum_values.enumeratedValue
            }
        else:
            enums = {}

        allowed_values = set(enums.values()) if enums else range(2**bit_width)

        return cls(
            name=name,
            bit_offset=bit_offset,
            bit_width=bit_width,
            enums=enums,
            allowed_values=allowed_values,
            element=element,
        )


class Field:
    """
    Internal representation of a register field.
    Not intended for direct user interaction.
    """

    __slots__ = ["_description", "_register"]

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

    @property
    def name(self) -> str:
        """Name of the field."""
        return self._description.name

    @property
    def default_value(self) -> int:
        """Default bitfield value."""
        default_register_value = self._register.reset_value
        field_default_value = (
            default_register_value >> self.bit_offset
        ) & self.bit_mask
        return field_default_value

    @property
    def bit_offset(self) -> int:
        """Bit offset of the field. Same as the field's bit position."""
        return self._description.bit_offset

    @property
    def bit_width(self) -> int:
        """Width of bits in the field."""
        return self._description.bit_width

    @property
    def mask(self) -> int:
        """Bitmask of the field."""
        return ((1 << self.bit_width) - 1) << self.bit_offset

    @property
    def raw(self) -> int:
        """The raw numeric value the field contains."""
        # TODO: take from parent
        return self._value

    @property
    def allowed_values(self) -> Union[List[int], range]:
        """Possible valid values for the bitfield."""
        return self._description.allowed_values

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
        return self.raw != self.default_value

    def __repr__(self):
        """Basic representation of the class."""
        return f"Field {self.name} {'(modified) ' if self.modified else ''}= {hex(self.raw)}"

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

    def set(self, value: Union[int, str]):
        """
        Set the value of the field.

        :param value: A numeric value, or a field enumeration (if applicable), to
            write to the field.
        """

        if not isinstance(value, (int, str)):
            raise TypeError(
                f"Field does not accept write of '{value}' of type '{type(value)}'"
                " Permitted values types are 'str' (field enum) and 'int' (bit value)."
            )

        if isinstance(value, int):
            val = self._trailing_zero_adjusted(value)

            if val not in self.allowed_values:
                raise ValueError(
                    f"Field '{self._register.name}.{self.name}' does not accept"
                    f" the bit value '{val}' ({hex(val)})."
                    " Are you sure you have an up to date .svd file?"
                )
            new_value = val
        else:
            if value not in self.enums:
                raise ValueError(
                    f"Field '{self._register.name}.{self.name}' does not accept"
                    f" the enum '{value}'."
                    " Are you sure you have an up to date .svd file?"
                )
            new_value = self.enums[value]

        self._register.set(new_value, self.bit_mask)

    def unconstrain(self):
        """
        Remove restrictions on values that may be entered into this field. After this,
        the field will accept any value that can fit inside its bit width.
        """
        self._allowed_values = range(2**self._bit_width)


def topo_sort_derived_peripherals(
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
        if (derived_from := peripheral.get("derivedFrom")) is not None:
            dep_graph[derived_from].append(peripheral)
        else:
            no_dep_peripherals.append(peripheral)

    while no_dep_peripherals:
        peripheral = no_dep_peripherals.pop()
        sorted_peripherals.append(peripheral)
        # Each peripheral has a maximum of one in-edge since they can only derive from one
        # peripheral. Therefore, once they are encountered here they have no remaining dependencies.
        no_dep_peripherals.extend(dep_graph[peripheral.name.pyval])
        dep_graph.pop(peripheral.name.pyval, None)

    if dep_graph:
        raise ValueError(
            "Unable to determine order in which peripherals are derived. "
            "This is likely caused either by a cycle in the "
            "'derivedFrom' attributes, or a 'derivedFrom' attribute pointing to a "
            "nonexistent peripheral."
        )

    return sorted_peripherals


def iter_dim(dimensions: List[DimensionProperties]) -> Iterable[Tuple[int, int]]:
    """
    TODO better doc
    Dimensions are ordered from outer to inner dimensions.
    Yields tuples (index, offset)
    """
    if dimensions:
        index_factors = [1 for _ in dimensions]

        for i in range(len(dimensions) - 1, 0, -1):
            index_factors[i - 1] = index_factors[i] * dimensions[i].length

        for indices in product(*(range(dim.length) for dim in dimensions)):
            index = 0
            offset = 0
            for i, factor, dim in zip(indices, index_factors, dimensions):
                index += i * factor
                offset += i * dim.step
            yield index, offset
    else:
        yield 0, 0


def _simplify_register_name(name: str) -> str:
    return util.strip_prefixes_suffixes(
        util.strip_prefixes_suffixes(name, [], ["[%s]"]),
        ["_"],
        ["_"],
    )


def _expand_element_fields(element: Iterable[bindings.RegisterElement]):
    # FIXME: sorted?
    if (fields_element := element.fields) is not None:
        field_desc = (
            _FieldDescription.from_element(e) for e in fields_element.iterchildren()
        )
        return {desc.name: desc for desc in field_desc}
    return None


# TODO: maybe cache the leaf nodes here
def get_register_elements(
    elements: Iterable[Union[bindings.RegisterElement, bindings.ClusterElement]],
    base_reg_props: RegisterProperties,
    base_address: int = 0,
) -> Tuple[Dict[str, RegisterType], int]:
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
        name = _simplify_register_name(element.name.pyval)

        reg_props = RegisterProperties(element, base=base_reg_props)
        dim_props = DimensionProperties.from_binding(element)

        if (address_offset_data := element.addressOffset) is not None:
            address_offset = address_offset_data.pyval
        else:
            address_offset = None

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
                address = address_offset
            else:
                address = base_address

            registers, min_child_address = get_register_elements(
                elements=element.iterchildren(
                    bindings.RegisterElement.TAG, bindings.ClusterElement.TAG
                ),
                base_reg_props=reg_props,
                base_address=address,
            )

            if registers:
                min_address = min(address, min_child_address)

            fields = None

        description = _RegisterDescription(
            element=element,
            name=name,
            offset=address,
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


def expand_registers(
    elements: Iterable[Union[bindings.RegisterElement, bindings.ClusterElement]],
    base_reg_props: RegisterProperties,
) -> Dict[int, Register]:
    """
    Get the memory map of a peripheral unit given by an SVD element and its
    base address.

    :param element: ElementTree representation of the SVD peripheral.
    :param base_address: Base address of peripheral unit.

    :return: Mapping from addresses to Registers.
    """
    result, _ = get_register_elements(elements, base_reg_props)

    return result
