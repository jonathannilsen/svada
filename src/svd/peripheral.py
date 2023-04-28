#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Python representation of an SVD Peripheral unit.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import chain
from typing import List, Tuple, Union, Dict, Iterable, NamedTuple, Optional, Set
from pprint import pformat
import lxml.etree as ET

from . import bindings
from . import util
from .svd_enums import *


_COMMON_PREFIXES = ["global"]
_COMMON_SUFFIXES = ["ns", "s"]


def _simplify_name(name: str) -> str:
    return util.strip_prefixes_suffixes(
        name.lower(), _COMMON_PREFIXES, _COMMON_SUFFIXES
    )


def find_peripheral_node(device: ET.Element, peripheral_name: str) -> ET.Element:
    """
    Find the first peripheral node with a name matching peripheral_name.
    Note that peripheral names are simplified before comparison, by removing common prefixes and
    suffixes.

    :param device: ElementTree element for the device definition in the SVD file.
    :param peripheral_name: Name of the peripheral to instantiate. Example "uicr".
    """
    peripheral_name = _simplify_name(peripheral_name)

    for peripheral in device.findall("peripherals/peripheral"):
        name = _simplify_name(peripheral.findtext("name").lower())
        if name == peripheral_name:
            return peripheral

    raise LookupError(f"Peripheral '{peripheral_name}' was not found in the device SVD")


class Device:
    def __init__(self, device: bindings.DeviceElement):
        self._device = device
        peripherals = {}
        for peripheral_element in topo_sort_derived_peripherals(
            device.peripherals.iterchildren()
        ):
            if (derived_from := peripheral_element.get("derivedFrom")) is not None:
                base_peripheral = peripherals[_simplify_name(derived_from)]
            else:
                base_peripheral = None

            peripheral = Peripheral(peripheral_element, base_peripheral=base_peripheral)
            peripherals[peripheral.name] = peripheral

        self._peripherals = peripherals

    @property
    def peripherals(self) -> Dict[str, Peripheral]:
        """Map of Peripherals in the device, indexed by name"""
        return self._peripherals


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
        base_peripheral: Optional[bindings.PeripheralElement],
        base_address: int = None,
    ):
        """
        Initialize the class attribute(s).

        :param peripheral_node: ElementTree element for the peripheral node in the device SVD file.
        :param base_address: Specific base address to use for the peripheral.
        """
        self._peripheral = peripheral
        self._base_peripheral = base_peripheral

        self._name: str = _simplify_name(peripheral.name.pyval)
        self._base_address: int = (
            peripheral.baseAddress.pyval if base_address is None else base_address
        )

        # Maybe reuse the computed registers from _base_peripheral?
        # This is possible as long as the peripheral element has the same register properties
        # as the base element (which is probably most cases)
        # Need to make it memory map etc. is relative to the base address to reuse it

        # Find a way to avoid hasattr
        # registers = (
        #     self._peripheral.registers.iterchildren()
        #     if hasattr(self, "registers")
        #     else []
        # )
        # if self._base_peripheral is not None and hasattr(
        #     self._base_peripheral._peripheral, "registers"
        # ):
        #     registers = chain(
        #         self._base_peripheral._peripheral.registers.iterchildren(), registers
        #     )
        # just for benchmarking
        registers = (
            self._peripheral.registers.iterdescendants()
            if hasattr(self, "registers")
            else []
        )
        if self._base_peripheral is not None and hasattr(
            self._base_peripheral._peripheral, "registers"
        ):
            registers = chain(
                self._base_peripheral._peripheral.registers.iterdescendants(), registers
            )

        reg_list = list(registers)
        #self._memory_map: Dict[int, Register] = get_memory_map(
        #   registers, self._base_address
        #)
        #self._instance_map: Dict[str, int] = {
        #   register.name: address for address, register in self._memory_map.items()
        #}

    @property
    def name(self) -> str:
        """Simplified name of the peripheral."""
        return self._name

    @property
    def memory_map(self) -> Dict[int, Register]:
        """Map of the peripheral register contents in memory."""
        return self._memory_map

    @property
    def base_address(self) -> int:
        """Base address of the peripheral space in memory."""
        return self._base_address

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
            if key not in self._memory_map:
                raise LookupError(
                    f"Peripheral does not contain a memory map for address '{hex(key)}'"
                )
            return self._memory_map[key]

        if key not in self._instance_map:
            raise LookupError(f"Peripheral does not contain a register named '{key}'")

        return self._memory_map[self._instance_map[key]]

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
            (r for r in self._memory_map.values() if r.name == register_field[0]), None
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


class RegisterElement(NamedTuple):
    """
    Basic tuple representation of a register used for element mapping.
    """

    element: ET.Element
    name: str
    reset_value: int


class Register:
    """
    Internal representation of a peripheral register.
    Not intended for direct user interaction.
    """

    def __init__(self, name: str, fields: Dict[int, Field], reset_value: int = 0):
        """
        Initialize the class attribute(s).

        :param name: Register name
        :param fields: Dictionary of bitfields present in the register
        :param reset_value: Register reset value
        """

        self._name: str = name
        self._fields: Dict[int, Field] = fields
        self._reset_value: int = reset_value

        for _bit_offset, field in self._fields.items():
            field.set_parent(self)

    @classmethod
    def from_element(
        cls, element: bindings.RegisterElement, name: str, reset_value: int
    ) -> Register:
        """
        Construct a Register class from an SVD element.

        :param element: ElementTree representation of an SVD Register element.
        :param name: Name of register.
        :param reset_value: Reset value of register.
        """

        element_fields = (
            element.fields.iterchildren() if hasattr(element, "fields") else []
        )

        fields = {
            field.bit_offset: field
            for field_element in element_fields
            for field in [Field.from_element(field_element, reset_value)]
        }

        if len(fields) == 0:
            fields = {0: Field.from_default(reset_value)}

        return cls(name, fields, reset_value)

    @property
    def name(self) -> str:
        """Name of the register."""
        return self._name

    @property
    def fields(self) -> Dict[int, Field]:
        """Register bitfields."""
        return self._fields

    @property
    def reset_value(self) -> int:
        """Register reset value."""
        return self._reset_value

    @property
    def modified(self) -> bool:
        """True if the register contains a different value now than at reset."""
        return any([field.modified for field in self.fields.values()])

    @property
    def raw(self) -> int:
        """The raw numeric value the register contains."""
        value = self.reset_value
        for offset, field in self.fields.items():
            value = (value & ~field.mask) | ((field.raw << offset) & field.mask)
        return value

    def __repr__(self):
        """Basic representation of the class object."""
        return f"Register {self.name} {'(modified) ' if self.modified else ''}= {hex(self.raw)}"

    def __str__(self):
        """String representation of the class."""

        attrs = {
            "modified": self.modified,
            "value": self.raw,
            "Fields": {k: str(v) for k, v in self.fields.items()},
        }
        return f"Register {self.name}: {pformat(attrs)}"

    def __getitem__(self, key: Union[int, str]) -> Field:
        """
        :param key: Either the bit offset of a field, or the field's name.

        :return: The instance of the specified field.
        """

        if not isinstance(key, (int, str)):
            raise TypeError(
                f"Register does not allow key of type '{type(key)}' for field lookup."
                " Permitted key types are 'str' (field name) and 'int' (bit offset)."
            )

        if isinstance(key, int):
            if key not in self.fields:
                raise LookupError(
                    f"Register '{self._name}' does not define a field at bit"
                    " offset '{key}'"
                )
            return self.fields[key]

        names_to_offsets = {
            field.name: field.bit_offset for field in self.fields.values()
        }

        if key not in names_to_offsets:
            raise LookupError(
                f"Register '{self._name}' does not define a field with name '{key}'"
            )

        return self.fields[names_to_offsets[key]]

    def __setitem__(self, key: Union[int, str], value: Union[str, int]):
        """
        :param key: Either the bit offset of a field, or the field's name.
        :param value: A raw numeric value, or a field enumeration, to write
            to the selected register field.
        """

        self[key].set(value)

    def set(self, value: int):
        """
        Set all the fields in the register, given an overall raw register value.

        :param value: Raw register value.
        """

        for field in self.fields.values():
            field.set_from_register(value)

    def field_iter(self) -> Iterable[Field]:
        """
        :return: Iterator over the register's fields.
        """
        return iter(self.fields.values())


class Field:
    """
    Internal representation of a register field.
    Not intended for direct user interaction.
    """

    def __init__(
        self,
        name: str,
        bit_offset: int,
        bit_width: int,
        default_value: int,
        enums: Dict[str, int],
        allowed_values: Union[List[int], range],
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
        self._name: str = name
        self._bit_offset: int = bit_offset
        self._bit_width: int = bit_width
        self._default_value: int = default_value
        self._value: int = default_value
        self._enums: Dict[str, int] = enums
        self._allowed_values: Union[List[int], range] = allowed_values
        self._parent_register: Register = None

    @classmethod
    def from_default(cls, default_register_value: int) -> Field:
        """
        Construct a single 32 bit wide Field with a given reset value.

        :param default_register_value: Reset value of register.
        """

        return cls("", 0, 32, default_register_value, {}, range(2**32))

    @classmethod
    def from_element(
        cls, element: bindings.FieldElement, default_register_value: int
    ) -> Field:
        """
        Construct a Field class from an SVD element.

        :param element: ElementTree representation of an SVD Field element.
        :param name: Name of field.
        :param reset_value: Reset value of field.
        """

        name = element.name.pyval
        bit_offset, bit_width = element.get_bit_range()

        bit_mask = 2**bit_width - 1
        default_value = (default_register_value >> bit_offset) & bit_mask

        # We do not support "do not care" bits, as by marking bits "x", see
        # SVD docs "/device/peripherals/peripheral/registers/.../enumeratedValue"
        if hasattr(element, "enumeratedValues"):
            enums = {
                enum.name.pyval: enum.value.pyval
                for enum in element.enumeratedValues.enumeratedValue
            }
        else:
            enums = {}

        allowed_values = enums.values() if len(enums) != 0 else range(bit_mask + 1)

        return cls(name, bit_offset, bit_width, default_value, enums, allowed_values)

    @property
    def name(self) -> str:
        """Name of the field."""
        return self._name

    @property
    def default_value(self) -> int:
        """Default bitfield value."""
        return self._default_value

    @property
    def bit_offset(self) -> int:
        """Bit offset of the field. Same as the field's bit position."""
        return self._bit_offset

    @property
    def bit_width(self) -> int:
        """Width of bits in the field."""
        return self._bit_width

    @property
    def mask(self) -> int:
        """Bitmask of the field."""
        return ((1 << self.bit_width) - 1) << self.bit_offset

    @property
    def raw(self) -> int:
        """The raw numeric value the field contains."""
        return self._value

    @property
    def allowed_values(self) -> Union[List[int], range]:
        """Possible valid values for the bitfield."""
        return self._allowed_values

    @property
    def enums(self) -> Dict[str, int]:
        """Dictionary of the bitfield enumerations in the field."""
        return self._enums

    @property
    def parent_register(self) -> Register:
        """Register to which the field belongs."""
        return self._parent_register

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

    def set_parent(self, register: Register):
        """
        Link the Field to its parent Register.

        :param register: Parent Register.
        """

        if not isinstance(register, Register):
            raise TypeError("Parent of a field must be a register")
        self._parent_register = register

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
                    f"Field '{self._parent_register.name}.{self.name}' does not accept"
                    f" the bit value '{val}' ({hex(val)})."
                    " Are you sure you have an up to date .svd file?"
                )
            self._value = val
        else:
            if value not in self.enums:
                raise ValueError(
                    f"Field '{self._parent_register.name}.{self.name}' does not accept"
                    f" the enum '{value}'."
                    " Are you sure you have an up to date .svd file?"
                )
            self._value = self.enums[value]

    def set_from_register(self, register_value: int):
        """
        Set the field value based on a value applicable to its containing register.

        :param register_value: Value applicable to its parent register.
        """

        bit_mask = 2**self._bit_width - 1
        value = (register_value >> self._bit_offset) & bit_mask
        self.set(value)

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


def get_register_elements(
    element: Iterable[Union[bindings.RegisterElement, bindings.ClusterElement]],
    base_address: int,
    prefix: str = "",
    reset_value: int = 0,
) -> Dict[int, RegisterElement]:
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
    name = element.name.pyval
    full_name = util.strip_prefixes_suffixes(
        util.strip_prefixes_suffixes(prefix + "_" + name.lower(), [], ["[%s]"]),
        ["_"],
        ["_"],
    )

    reset_value = (
        util.to_int(element.find("resetValue").text)
        if element.find("resetValue") is not None
        else reset_value
    )

    address_offset = (
        util.to_int(element.find("addressOffset").text)
        if element.find("addressOffset") is not None
        else 0
    )

    if name.endswith("[%s]"):
        array_length = util.to_int(element.find("dim").text)
        step = util.to_int(element.find("dimIncrement").text)
    else:
        array_length = 1
        step = 0

    occurences = [address_offset + step * s for s in range(array_length)]

    registers: Dict[int, RegisterElement] = {}
    children = element.findall("cluster") + element.findall("register")

    for child in children:
        registers = {
            **registers,
            **get_register_elements(
                child,
                base_address,
                prefix=full_name,
                reset_value=reset_value,
            ),
        }

    if len(children) == 0:
        registers = {base_address: RegisterElement(element, full_name, reset_value)}

    expanded_registers: Dict[int, RegisterElement] = {}
    for address, register_bundle in registers.items():
        for offset in occurences:
            expanded_registers[address + offset] = register_bundle

    return expanded_registers


def get_memory_map(
    element: Iterable[bindings.RegisterElement], base_address: int
) -> Dict[int, Register]:
    """
    Get the memory map of a peripheral unit given by an SVD element and its
    base address.

    :param element: ElementTree representation of the SVD peripheral.
    :param base_address: Base address of peripheral unit.

    :return: Mapping from addresses to Registers.
    """

    register_bundles = {}
    for register in element:
        register_bundles.update(get_register_elements(register, base_address))

    instance_counter = {name: 0 for _, name, _ in register_bundles.values()}

    memory_map: Dict[int, Register] = {}

    for address, reg in register_bundles.items():
        instance = instance_counter[reg.name]
        instance_counter[reg.name] += 1

        memory_map[address] = Register.from_element(
            reg.element, f"{reg.name}_{instance}", reg.reset_value
        )

    return memory_map
