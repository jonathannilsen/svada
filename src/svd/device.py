#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

"""
High level representation of a SVD device.
"""

from __future__ import annotations

import enum
import math
from collections import ChainMap, defaultdict, deque
from functools import cached_property
from itertools import chain
from types import MappingProxyType
from typing import (
    Any,
    Collection,
    List,
    MutableMapping,
    Tuple,
    Union,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

import numpy as np
import numpy.ma as ma

from . import bindings
from .bindings import (
    Access,
    AddressBlock,
    Cpu,
    RegisterProperties,
    Dimensions,
    WriteAction,
    ReadAction,
    WriteConstraint,
)
from .path import SPath
from . import util
from .util import LazyStaticList, LazyStaticMapping

from time import perf_counter_ns

LOG_TIME = False


class Device(Mapping[str, "Peripheral"]):
    """
    Representation of a SVD device.
    """

    def __init__(self, device: bindings.DeviceElement):
        self._device: bindings.DeviceElement = device
        self._reg_props: RegisterProperties = self._device.register_properties

        peripherals_unsorted: Dict[str, Peripheral] = {}

        # Process peripherals in topological order to ensure that base peripherals are processed
        # before derived peripherals.
        for peripheral_element in _topo_sort_derived_peripherals(device.peripherals):
            if peripheral_element.is_derived:
                base_peripheral = peripherals_unsorted[peripheral_element.derived_from]
            else:
                base_peripheral = None

            peripheral = Peripheral(
                peripheral_element,
                device=self,
                base_reg_props=self._reg_props,
                base_peripheral=base_peripheral,
            )

            peripherals_unsorted[peripheral.name] = peripheral

        self._peripherals: Dict[str, Peripheral] = dict(
            sorted(peripherals_unsorted.items(), key=lambda kv: kv[1].base_address)
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
    def _qualified_name(self) -> str:
        """Name of the device, including vendor and series."""
        return f"{self.vendor_id or ''}::{self.series or ''}::{self.name}"

    @property
    def cpu(self) -> Optional[Cpu]:
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
        """
        :param name: Peripheral name.

        :return: Peripheral with the given name.
        """
        try:
            return self._peripherals[name]
        except LookupError as e:
            raise KeyError(
                f"{self.__class__} {self.name} does not contain a peripheral named '{name}'"
            ) from e

    def __iter__(self) -> Iterator[str]:
        """
        :return: Iterator over the names of peripherals in the device.
        """
        return iter(self._peripherals)

    def __len__(self) -> int:
        """
        :return: Number of peripherals in the device.
        """
        return len(self._peripherals)

    def __repr__(self) -> str:
        return _svd_element_repr(self.__class__, self._qualified_name, length=len(self))


class Peripheral(Mapping[str, "RegisterType"]):
    """
    Representation of a specific device peripheral.

    Internally, this class maintains a representation of a peripheral that is always guaranteed to
    be correct when compared to the allowable values prescribed by the SVD file the class was
    instantiated from. This representation starts off by having the default values defined within
    the SVD.
    """

    def __init__(
        self,
        element: bindings.PeripheralElement,
        device: Device,
        base_reg_props: bindings.RegisterProperties,
        base_peripheral: Optional[Peripheral] = None,
    ):
        self._peripheral: bindings.PeripheralElement = element
        self._device: Device = device
        self._base_peripheral: Optional[Peripheral] = base_peripheral
        self._base_address: int = element.base_address
        self._reg_props: bindings.RegisterProperties = (
            self._peripheral.get_register_properties(base_props=base_reg_props)
        )

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

    @property
    def interrupts(self) -> Mapping[str, int]:
        """Interrupts associated with the peripheral, a mapping from interrupt name to value."""
        return {
            interrupt.name: interrupt.value for interrupt in self._peripheral.interrupts
        }

    @property
    def address_blocks(self) -> List[AddressBlock]:
        """List of address blocks associated with the peripheral."""
        return list(self._peripheral.address_blocks)

    @cached_property
    def registers(self) -> Mapping[str, Register]:
        """Mapping of top-level registers in the peripheral, indexed by name."""
        return (
            LazyStaticMapping(
                keys=self._register_descriptions.keys(),
                factory=lambda name: _create_register_instance(
                    description=self._register_descriptions[name],
                    peripheral=self,
                    path=SPath(name),
                ),
            ),
        )

    def register_iter(
        self, flat: bool = False, leaf_only: bool = False
    ) -> Iterator[RegisterType]:
        """
        Recursive iterator over the registers in the peripheral in pre-order.
        Registers are ordered by increasing offset/address.

        :param flat: Do not yield individual registers in register arrays.
        :param leaf_only: Only yield physical registers, i.e. those that have a value.

        :return: Iterator over the registers in the peripheral.
        """

        return _register_iter_helper(
            peripheral=self,
            registers=self._register_descriptions,
            flat=flat,
            leaf_only=leaf_only,
        )

    @property
    def contents(self) -> Mapping[int, int]:
        """Read-only view of the peripheral register contents in memory."""
        return MappingProxyType(self._mutable_contents)

    @cached_property
    def _mutable_contents(self) -> MutableMapping[int, int]:
        """
        Mutable mapping of the peripheral register contents in memory.
        Intended for internal use only.
        """
        return ChainMap({}, dict(self._immutable_register_info.reset_contents))

    @property
    def _register_descriptions(self) -> Mapping[str, _RegisterDescription]:
        """Mapping of register descriptions in the peripheral, indexed by name."""
        return self._immutable_register_info.descriptions

    @cached_property
    def _immutable_register_info(self) -> _ExtractedRegisterInfo:
        """
        Compute the immutable descriptions of the registers contained in the peripheral, taking into
        account registers derived from the base peripheral, if any.
        """

        base_info: Optional[_ExtractedRegisterInfo] = None

        if self._base_peripheral is not None:
            # If the register properties are equal, then it is possible to reuse all the immutable
            # properties from the base peripheral.
            if self._base_peripheral._reg_props == self._reg_props:
                base_info = self._base_peripheral._immutable_register_info
            # Otherwise, traverse the base registers again, because the difference in
            # register properties propagates down to the register elements.
            else:
                base_info = _extract_register_info(
                    self._base_peripheral.address_blocks,
                    self._base_peripheral._peripheral.registers,
                    self._reg_props,
                )

            address_blocks = self.address_blocks + self._base_peripheral.address_blocks

        else:
            address_blocks = self.address_blocks

        info = _extract_register_info(
            address_blocks,
            self._peripheral.registers,
            self._reg_props,
            base_info=base_info,
        )

        return info

    def __getitem__(self, path: Union[str, Sequence[Union[str, int]]]) -> Register:
        """
        :param path: Name of the register to get, or a path to a register. # FIXME

        :return: The instance of the specified register.
        """
        try:
            return self.registers[path]
        except LookupError as e:
            raise KeyError(
                f"Peripheral {self} does not contain a register named '{path}'"
            ) from e

    def __setitem__(self, name: str, value: int) -> None:
        """
        :param name: Name of the register to update.
        :param value: The raw register value to write to the specified register.
        """
        self[name].content = value

    def __iter__(self) -> Iterator[str]:
        """
        :return: Iterator over the names of registers in the peripheral.
        """
        return iter(self.registers)

    def __len__(self) -> int:
        """
        :return: Number of registers in the peripheral.
        """
        return len(self.registers)

    def __repr__(self) -> str:
        return _svd_element_repr(self.__class__, self.name, address=self.base_address)


def _register_iter_helper(
    peripheral: Peripheral,
    registers: Mapping[str, _RegisterDescription],
    flat: bool = False,
    leaf_only: bool = False,
) -> Iterator[Register]:
    """ """

    queue = deque((SPath(name), 0, desc) for name, desc in registers.items())

    while queue:
        path, offset, description = queue.pop()

        register = Register(
            description=description,
            peripheral=peripheral,
            path=path,
            instance_offset=offset,
        )

        is_leaf = description.registers is None and (
            flat or (description.dim_props is None or path.is_array_element())
        )

        if not leaf_only or is_leaf:
            yield register

        if (
            not flat
            and description.dim_props is not None
            and not path.is_array_element()
        ):
            # Expand dimensioned register
            for i, additional_offset in enumerate(description.dim_props.to_range()):
                queue.append((path.join(i), offset + additional_offset, description))
        elif description.registers is not None:
            # Expand register structure
            for name, child_description in description.registers.items():
                queue.append((path.join(name), offset, child_description))


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
    registers: Optional[Mapping[str, _RegisterDescription]]
    fields: Optional[Mapping[str, _FieldDescription]]
    element: Union[bindings.RegisterElement, bindings.ClusterElement]


class _RegisterBase:
    """Base class for all register types"""

    __slots__ = [
        "_description",
        "_peripheral",
        "_path",
        "_instance_offset",
    ]

    def __init__(
        self,
        description: _RegisterDescription,
        peripheral: Peripheral,
        path: SPath,
        instance_offset: int = 0,
    ):
        """
        :param description: Register description.
        :param peripheral: Parent peripheral.
        :param instance_offset: Address offset inherited from the parent register.
        :param index: Index of this register in the parent register, if applicable.
        :param path_prefix: String prefixed to the register name to get the register path.
        """
        self._description: _RegisterDescription = description
        self._peripheral: Peripheral = peripheral
        self._path: SPath = path
        self._instance_offset: int = instance_offset

    @property
    def name(self) -> str:
        """Name of the register."""
        return self.path.name

    @property
    def path(self) -> SPath:
        """Full path to the register."""
        return self._path

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
    def reset_content(self) -> int:
        """Register reset value."""
        return self._description.reg_props.reset_value

    @property
    def reset_mask(self) -> int:
        """Mask of bits in the register that are affected by a reset."""
        return self._description.reg_props.reset_mask

    @property
    def write_action(self) -> WriteAction:
        """Side effect of writing the register"""
        return self._description.element.modified_write_values

    @property
    def modified(self) -> bool:
        """True if the register contains a different value now than at reset."""
        return self.content != self.reset_content

    @property
    def dimensions(self) -> Optional[Dimensions]:
        """Dimensions of the register, if any."""
        return self._description.dim_props

    # TODO: Rewrite to use np api
    @property
    def content(self) -> int:
        """Current value of the register."""
        return self._peripheral._mutable_contents[self.offset]

    @property
    def registers(self) -> Optional[Mapping[str, RegisterType]]:
        return None

    @property
    def fields(self) -> Optional[Mapping[str, Field]]:
        """Map of fields in the register, indexed by name"""
        return None

    def unconstrain(self) -> None:
        """
        Remove all value constraints imposed on the register.
        """
        for field in self.values():
            field.unconstrain()

    def register_iter(
        self, flat: bool = False, leaf_only: bool = False
    ) -> Iterator[Register]:
        """
        Recursive iterator over the registers in the peripheral in pre-order.
        See Peripheral.register_iter().
        """
        # FIXME: can this be here?
        return _register_iter_helper(
            peripheral=self._peripheral,
            registers=self._description.registers,
            flat=flat,
            leaf_only=leaf_only,
        )

    def __getitem__(self, path: Union[int, str, Sequence[Union[str, int]]]) -> Register:
        """
        :param path: Register name.

        :return: Register with the given name.
        """
        try:
            return self._registers[path]
        except LookupError as e:
            raise KeyError(
                f"{self.__class__} {self._path_with_peripheral} "
                f"does not contain a register named '{path}'"
            ) from e

#    def __iter__(self) -> Iterator[MT]:
#        """
#        :return: Iterator over the registers in the register array.
#        """
#        return iter(self._array)
#
#    def __len__(self) -> int:
#        """
#        :return: Number of registers in the register array.
#        """
#        return len(self._array)

    def __repr__(self) -> str:
        return _svd_element_repr(self.__class__, self.path, address=self.offset)

    @property
    def _path_with_peripheral(self) -> str:
        """Full path of the register including the parent peripheral"""
        return f"{self._peripheral.name}.{self.path}"

    # FIXME: move this to SPath
    @property
    def _array_index(self) -> Optional[int]:
        """Index of the register in the parent array, if applicable."""
        if not isinstance(self._path.parts[-1], int):
            return None
        return self._path.parts[-1]


class RegisterStruct(_RegisterBase, Mapping[str, Union["RegisterArray", "Register"]]):
    """
    Register structure representing a group of registers.
    Represents either a SVD cluster element without dimensions,
    or a specific index of a cluster array.
    """

    __slots__ = ["_registers"]

    def __init__(self, **kwargs):
        """
        See parent class for a description of parameters.
        """
        super().__init__(**kwargs)

        self._registers = LazyStaticMapping(
            keys=self._description.registers.keys(),
            factory=lambda name: _create_register_instance(
                description=self._description.registers[name],
                peripheral=self._peripheral,
                instance_offset=self._instance_offset,
                path=self.path.join(name),
            ),
        )

    def register_iter(
        self, flat: bool = False, leaf_only: bool = False
    ) -> Iterator[RegisterType]:
        """
        Recursive iterator over the registers in the peripheral in pre-order.
        Registers are ordered by increasing offset/address.

        :param flat: Do not yield individual registers in register arrays.
        :param leaf_only: Only yield physical registers, i.e. those that have a value.

        :return: Iterator over the registers in the peripheral.
        """
        if not leaf_only and not (flat and self._array_index is not None):
            yield self

        for register in self.values():
            yield from register.register_iter(flat=flat, leaf_only=leaf_only)

    def __getitem__(self, path: Union[str, Sequence[Union[str, int]]]) -> RegisterType:
        """
        :param name: Register name.

        :return: Register with the given name.
        """
        try:
            return self._registers[path]
        except LookupError as e:
            raise KeyError(
                f"{self.__class__} {self._path_with_peripheral} "
                f"does not contain a register named '{path}'"
            ) from e

    def __setitem__(self, name: str, content: int) -> None:
        """
        :param name: Register name.
        :param content: Register value.
        """
        try:
            self[name].content = content
        except AttributeError as e:
            raise TypeError(f"{self[name]} does not have content") from e

    def __iter__(self) -> Iterator[str]:
        """
        :return: Iterator over the names of registers in the register structure.
        """
        return iter(self._registers)

    def __len__(self) -> int:
        """
        :return: Number of registers in the register structure
        """
        return len(self._registers)


class Register(_RegisterBase, Mapping[str, "Field"]):
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

        if self._description.fields is not None:
            self._fields = LazyStaticMapping(
                keys=self._description.fields.keys(),
                factory=lambda name: Field(
                    description=self._description.fields[name], register=self
                ),
            )
        else:
            self._fields = {}

    @property
    def modified(self) -> bool:
        """True if the register contains a different value now than at reset."""
        return self.content != self.reset_content

    @property
    def content(self) -> int:
        """Current value of the register."""
        return self._peripheral._mutable_contents[self.offset]

    @content.setter
    def content(self, new_content: int) -> None:
        """
        Set the value of the register.

        :param new_content: New value for the register.
        """
        self.set_content(new_content)

    def set_content(self, new_content: int, mask: Optional[int] = None):
        """
        Set the value of the register.

        :param new_content: New value for the register.
        :param mask: Mask of the bits to copy from the given value. If None, all bits are copied.
        """
        if (
            new_content > 0
            and math.ceil(math.log2(new_content)) > self._description.reg_props.size
        ):
            raise ValueError(
                f"Value {hex(new_content)} is too large for {self._description.reg_props.size}-bit "
                f"register {self.path}."
            )

        for field in self.values():
            # Only check fields that are affected by the mask
            if mask is None or mask & field.mask:
                field_content = field._extract_content_from_register(new_content)
                if field_content not in field.allowed_values:
                    raise ValueError(
                        f"Value {hex(new_content)} is invalid for register {self.path}, as field "
                        f"{field.full_name} does not accept the value {hex(field_content)}."
                    )

        if mask is not None:
            # Update only the bits indicated by the mask
            new_content = (self.content & ~mask) | (new_content & mask)
        else:
            new_content = new_content

        self._peripheral._mutable_contents[self.offset] = new_content

    @property
    def fields(self) -> Mapping[str, Field]:
        """Map of fields in the register, indexed by name"""
        return MappingProxyType(self._fields)

    def unconstrain(self) -> None:
        """
        Remove all value constraints imposed on the register.
        """
        for field in self.values():
            field.unconstrain()

    def register_iter(
        self, flat: bool = False, leaf_only: bool = False
    ) -> Iterator[RegisterType]:
        """
        Recursive iterator over the registers in the peripheral in pre-order.
        See Peripheral.register_iter().
        """
        if not flat or self._array_index is None:
            yield self

    def __getitem__(self, path: Union[str, Sequence[str]]) -> Field:
        """
        :param name: Field name.

        :return: The instance of the specified field.
        """
        try:
            return self._fields[path]
        except LookupError as e:
            raise KeyError(
                f"{self.__class__} {self._path_with_peripheral} "
                f"does not define a field with name '{path}'"
            ) from e

    def __setitem__(self, key: str, value: Union[str, int]) -> None:
        """
        :param key: Either the bit offset of a field, or the field's name.
        :param value: A raw numeric value, or a field enumeration, to write
            to the selected register field.
        """
        self[key].content = value

    def __iter__(self) -> Iterator[str]:
        """
        :return: Iterator over the field names in the register.
        """
        return iter(self._fields)

    def __len__(self) -> int:
        """
        :return: Number of fields in the register.
        """
        return len(self._fields)

    def __repr__(self) -> str:
        bool_props = ("modified",) if self.modified else ()

        return _svd_element_repr(
            self.__class__,
            self.path,
            address=self.offset,
            content=self.content,
            bool_props=bool_props,
        )


# Member type in a dimensioned register
MT = TypeVar("MT")


class _DimensionedRegister(_RegisterBase, Sequence[MT]):
    """
    Base class for register arrays.
    """

    __slots__ = ["_array_offsets", "_array"]

    # Register type contained in the register array, to be set by child classes
    # This annotation is just here to satisfy the type checker
    member_type: Type[MT]

    def __init__(self, description: _RegisterDescription, **kwargs):
        self._array_offsets: Sequence[int] = description.dim_props.to_range()

        self._array: Sequence[MT] = LazyStaticList(
            length=len(self._array_offsets),
            factory=lambda i: self.member_type(
                description=self._description,
                peripheral=self._peripheral,
                instance_offset=self._instance_offset + self._array_offsets[i],
                path=self.path.join(i),
            ),
        )

        super().__init__(description=description, **kwargs)

    @property
    def dimensions(self) -> Dimensions:
        """Dimensions of the register array."""
        return self._description.dim_props

    def __getitem__(self, path: Union[int, Sequence[str, int]]) -> MT:
        """
        :param index: Index of the register in the register array.

        :return: The instance of the specified register.
        """
        try:
            return self._array[path]
        except IndexError as e:
            raise IndexError(f"{self!s}: array index {path} is out of range") from e

    def __iter__(self) -> Iterator[MT]:
        """
        :return: Iterator over the registers in the register array.
        """
        return iter(self._array)

    def __len__(self) -> int:
        """
        :return: Number of registers in the register array.
        """
        return len(self._array)

    def __repr__(self) -> str:
        """Short description of the register."""
        return _svd_element_repr(
            self.__class__,
            self.path,
            address=self.offset,
            length=len(self),
        )


class RegisterStructArray(_DimensionedRegister[RegisterStruct]):
    """
    Array of RegisterStruct objects.
    SVD cluster elements with dimensions are represented using this class.
    """

    @property
    def member_type(self) -> Type:
        return RegisterStruct


class RegisterArray(_DimensionedRegister[Register]):
    """
    Array of Register objects.
    SVD register elements with dimensions are represented using this class.
    """

    @property
    def member_type(self) -> Type:
        return Register


# Union of all register types
RegisterType = Union[Register, RegisterArray, RegisterStruct, RegisterStructArray]


def _create_register_instance(
    description: _RegisterDescription, is_element: bool = False, **kwargs
) -> RegisterType:
    """
    Create a mutable register instance from a register description.

    :param description: Register description
    :param index: Index of the register in the parent register array, if applicable
    :return: Register instance
    """
    if description.registers is not None:
        if description.dim_props is not None and not is_element:
            return RegisterStructArray(description=description, **kwargs)
        return RegisterStruct(description=description, **kwargs)
    else:
        if description.dim_props is not None and not is_element:
            return RegisterArray(description=description, **kwargs)
        return Register(description=description, **kwargs)




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
        :param element: SVD field element binding object.
        :return: Description of the field.
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
    def path(self) -> SPath:
        """The full name of the field, including the register name."""
        return self._register.path.join(self.name)

    @property
    def content(self) -> int:
        """The value of the field."""
        return self._extract_content_from_register(self._register.content)

    @content.setter
    def content(self, new_value: Union[int, str]):
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
                    f"Field '{self.path}' does not accept"
                    f" the bit value '{val}' ({hex(val)})."
                    " Are you sure you have an up to date .svd file?"
                )
            resolved_value = val
        else:
            if new_value not in self.enums:
                raise ValueError(
                    f"Field '{self.path}' does not accept"
                    f" the enum '{new_value}'."
                    " Are you sure you have an up to date .svd file?"
                )
            resolved_value = self.enums[new_value]

        self._register.set_content(resolved_value << self.bit_offset, self.mask)

    @property
    def reset_content(self) -> int:
        """Default field value."""
        return self._extract_content_from_register(self._register.reset_content)

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
        """Access property of the field."""
        if (field_access := self._description.element.access) is not None:
            return field_access
        return self._register.access

    @property
    def write_action(self) -> WriteAction:
        """Side effect of writing to the field."""
        if (
            field_write_action := self._description.element.modified_write_values
        ) is not None:
            return field_write_action
        return self._register.write_action

    @property
    def read_action(self) -> ReadAction:
        """Side effect of reading from the field."""
        if (field_read_action := self._description.element.read_action) is not None:
            return field_read_action
        return self._register.read_action

    @property
    def write_constraint(self) -> Optional[WriteConstraint]:
        """Constraints on writing to the field."""
        if (
            field_write_constraint := self._description.element.write_constraint
        ) is not None:
            return field_write_constraint
        return self._register.write_constraint

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
        return self.content != self.reset_content

    def unconstrain(self) -> None:
        """
        Remove restrictions on values that may be entered into this field. After this,
        the field will accept any value that can fit inside its bit width.
        """
        self._allowed_values = range(2**self.bit_width)

    def _extract_content_from_register(self, register_content: int) -> int:
        """
        Internal method for extracting the field value from the parent register value.

        :param register_value: Value of the parent register
        :return: Field value extracted based on the field bit range
        """
        return (register_content & self.mask) >> self.bit_offset

    def _trailing_zero_adjusted(self, content: int) -> int:
        """
        Internal method that checks and adjusts a given value for trailing zeroes if it exceeds
        the bit width of its field. Some values are simplest to encode as a full 32-bit value even
        though their field is comprised of less than 32 bits, such as an address.

        :param value: A numeric value to check against the field bits

        :return: Field value without any trailing zeroes
        """

        width_max = 2**self.bit_width

        if content > width_max:
            max_val = width_max - 1
            max_val_hex_len = len(f"{max_val:x}")
            hex_val = f"{content:0{8}x}"  # leading zeros, 8-byte max, in hex
            trailing = hex_val[max_val_hex_len:]  # Trailing zeros

            if int(trailing, 16) != 0:
                raise ValueError(f"Unexpected trailing value: {trailing}")

            cropped = hex_val[:max_val_hex_len]  # value w/o trailing
            adjusted = int(cropped, 16)

            if adjusted <= max_val:
                return adjusted

        return content

    def __repr__(self):
        bool_props = ("modified",) if self.modified else ()

        return _svd_element_repr(
            self.__class__,
            self.path,
            content=self.content,
            content_max_width=self.bit_width,
            bool_props=bool_props,
        )


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


class _ExtractedRegisterInfo(NamedTuple):
    """Container for register descriptions and reset values."""

    descriptions: Mapping[str, _RegisterDescription]
    reset_contents: Mapping[int, int]


def numpy_full(length: int, value: int, dtype: np.dtype):
    # Apparently this special case is not handled by numpy.full.
    # Explicitly handling it here saves a lot of time.
    if value == 0:
        return np.zeros(length, dtype=dtype)
    else:
        return np.full(
            length,
            value,
            dtype=dtype,
        )


def _extract_register_info(
    address_blocks: List[AddressBlock],
    elements: Iterable[Union[bindings.RegisterElement, bindings.ClusterElement]],
    base_reg_props: bindings.RegisterProperties,
    base_info: Optional[_ExtractedRegisterInfo] = None,
) -> _ExtractedRegisterInfo:
    """
    Extract register descriptions for the given SVD register level elements.
    The returned structure mirrors the structure of the SVD elements.
    Each level of the structure is internally sorted by ascending address.

    :param elements: Register level elements to process.
    :param base_reg_props: Register properties inherited from the parent peripheral.

    :return: Map of register descriptions, indexed by name.
    """
    if len(address_blocks) != 1:
        raise NotImplementedError(
            "The implementation assumes exactly one peripheral address block"
        )

    t = perf_counter_ns()

    if base_info is not None:
        reset_content = np.copy(base_info.reset_contents)
    else:
        # TODO: take dtype from SVD?
        reset_content = numpy_full(
            address_blocks[0].size, base_reg_props.reset_value, dtype=np.uint8
        )

    if LOG_TIME:
        print(f"1: {(perf_counter_ns() - t) / 1_000_000:.2f}")
    t = perf_counter_ns()

    helper_result = _extract_register_descriptions_helper(
        reset_content, elements, base_reg_props
    )

    if LOG_TIME:
        print(f"2: {(perf_counter_ns() - t) / 1_000_000:.2f}")
    t = perf_counter_ns()

    descriptions = {}

    for res in helper_result:
        descriptions[res.description.name] = res.description

    if base_info is not None:
        # The register maps are each sorted internally, but need to be merged by address
        # to ensure sorted order in the combined map
        descriptions = dict(
            util.iter_merged(
                descriptions.items(),
                base_info.descriptions.items(),
                key=lambda kv: kv[1].start_offset,
            )
        )

    if LOG_TIME:
        print(f"3: {(perf_counter_ns() - t) / 1_000_000:.2f}")
    t = perf_counter_ns()

    if LOG_TIME:
        for k, v in HELPER_TIMES.items():
            print(f"{k}: {v / 1_000_000:.2f} ms")
        print()

    HELPER_TIMES.clear()

    return _ExtractedRegisterInfo(descriptions, reset_content)


class _ExtractHelperResult(NamedTuple):
    """Container for intermediary results of the register extraction helper."""

    description: _RegisterDescription
    # TODO: consider creating an abstraction over this once things are done
    address_mask: np.ndarray
    address_start: int
    address_end: int


def _expand_dimensioned_content(
    content: List[Tuple[int, int]], dimensions: Dimensions
) -> List[Tuple[int, int]]:
    """Expand the given content along the given dimension."""

    if dimensions.length <= 1:
        return content

    if len(content) == 1:
        # Handling this case separately provides a substantial speed increase for large repetitive
        # register arrays.
        addr, value = content[0]
        expanded_content = [(addr + offset, value) for offset in dimensions.to_range()]
    else:
        expanded_content: List[Tuple[int, int]] = list(content)
        for offset in dimensions.to_range()[1:]:
            expanded_content.extend(((addr + offset, value) for addr, value in content))

    return expanded_content


# TODO: use SVD endianness
SIZE_TO_DTYPE = {
    1: np.uint8,
    2: np.dtype((np.dtype("<u2"), (np.uint8, 2))),
    4: np.dtype((np.dtype("<u4"), (np.uint8, 4))),
}

HELPER_TIMES = defaultdict(int)


def _extract_register_descriptions_helper(
    content: np.ndarray,
    elements: Iterable[Union[bindings.RegisterElement, bindings.ClusterElement]],
    base_reg_props: bindings.RegisterProperties,
    base_address: int = 0,
) -> List[_ExtractHelperResult]:
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
    total_result: List[_ExtractHelperResult] = []

    for element in elements:
        # Remove suffixes used for elements with dimensions
        name = util.strip_suffix(element.name, "[%s]")
        reg_props = element.get_register_properties(base_props=base_reg_props)
        dim_props = element.dimensions
        address_offset = element.offset

        if isinstance(element, bindings.RegisterElement):
            t = perf_counter_ns()

            # Register addresses are defined relative to the enclosing element
            if address_offset is not None:
                address_start = base_address + address_offset
            else:
                address_start = base_address

            size_bytes = reg_props.size // 8

            k = perf_counter_ns()

            if dim_props is not None and dim_props.length > 1:
                if dim_props.step == size_bytes:
                    address_mask = np.ones(size_bytes * dim_props.length, bool)
                    address_end = address_start + size_bytes * dim_props.length
                # else ?
            else:
                address_end = address_start + size_bytes
                address_mask = np.ones(size_bytes, bool)

            HELPER_TIMES["reg1"] += perf_counter_ns() - k
            k = perf_counter_ns()

            if reg_props.reset_value != 0:  # FIXME: compare to peripheral reset_value
                reset_value_dtype = SIZE_TO_DTYPE[size_bytes]
                content[address_start:address_end].view(reset_value_dtype)[
                    :
                ] = reg_props.reset_value

            HELPER_TIMES["reg2"] += perf_counter_ns() - k
            k = perf_counter_ns()

            registers = None
            fields = _extract_field_descriptions(element.fields)

            HELPER_TIMES["reg3"] += perf_counter_ns() - k

            HELPER_TIMES["reg"] += perf_counter_ns() - t

        else:  # ClusterElement
            t = perf_counter_ns()
            # By the SVD specification, cluster addresses are defined relative to the peripheral
            # base address, but some SVDs don't follow this rule.
            if address_offset is not None:
                address_start = base_address + address_offset
                # address = address_offset
            else:
                address_start = base_address

            HELPER_TIMES["clu"] += perf_counter_ns() - t

            child_results = _extract_register_descriptions_helper(
                content=content,
                elements=element.registers,
                base_reg_props=reg_props,
                base_address=address_start,
            )

            t = perf_counter_ns()

            registers = {}
            last_address_end = None
            address_mask = np.asarray([], dtype=bool)

            for result in child_results:
                registers[result.description.name] = result.description

                if last_address_end is not None:
                    address_gap = result.address_start - last_address_end
                    assert address_gap >= 0
                else:
                    address_gap = 0

                last_address_end = result.address_end

                address_mask = np.pad(address_mask, address_gap)
                address_mask = np.concatenate([address_mask, result.address_mask])

            address_end = last_address_end
            fields = None

            # TODO: optimize for the contiguous fill case
            if dim_props is not None and dim_props.length > 1:
                # Duplicate content across the dimension
                for element_offset in dim_props.to_range():
                    np.copyto(
                        dst=content[
                            address_start
                            + element_offset : address_end
                            + element_offset
                        ],
                        src=content[address_start:address_end],
                        where=address_mask,
                    )

                # Duplicate the address mask across the dimension
                pad_bytes = address_end - address_start - len(address_mask)
                if pad_bytes > 0:
                    address_mask = np.pad(address_mask, pad_bytes)
                address_mask = np.tile(address_mask, dim_props.length)
                address_end = (
                    address_end + dim_props.to_range()[-1]
                )  # TODO: is this correct?

                HELPER_TIMES["clu"] += perf_counter_ns() - t

        description = _RegisterDescription(
            element=element,
            name=name,
            start_offset=address_start,
            reg_props=reg_props,
            dim_props=dim_props,
            registers=registers,
            fields=fields,
        )

        total_result.append(
            _ExtractHelperResult(
                description=description,
                address_mask=address_mask,
                address_start=address_start,
                address_end=address_end,
            )
        )

    t = perf_counter_ns()

    sorted_result = sorted(total_result, key=lambda r: r.address_start)

    HELPER_TIMES["sort"] += perf_counter_ns() - t

    return sorted_result


def _extract_field_descriptions(
    elements: Iterable[bindings.FieldElement],
) -> Optional[Mapping[str, _FieldDescription]]:
    """
    Extract field descriptions for the given SVD field elements.
    The resulting mapping is internally sorted by ascending field bit offset.

    :param elements: Field elements to process.
    :return: Mapping of field descriptions, indexed by name.
    """

    t = perf_counter_ns()
    field_descriptions_unsorted = [
        _FieldDescription.from_element(field) for field in elements
    ]
    HELPER_TIMES["fields_con"] += perf_counter_ns() - t

    t = perf_counter_ns()
    field_descriptions = sorted(
        field_descriptions_unsorted,
        key=lambda field: field.bit_range.offset,
    )

    fields = {description.name: description for description in field_descriptions}

    HELPER_TIMES["fields_rest"] = perf_counter_ns() - t

    if not fields:
        return None

    return fields


def _svd_element_repr(
    klass: type,
    name: str,
    /,
    *,
    address: Optional[int] = None,
    length: Optional[int] = None,
    content: Optional[int] = None,
    content_max_width: int = 32,
    bool_props: Iterable[Any] = (),
    kv_props: Mapping[Any, Any] = MappingProxyType({}),
) -> str:
    """
    Common pretty print function for SVD elements.

    :param klass: Class of the element.
    :param name: Name of the element.
    :param address: Address of the element.
    :param content: Length of the element.
    :param width: Available width of the element, used to zero-pad the value.
    :param value: Value of the element.
    :param kwargs: Additional keyword arguments to include in the pretty print.

    :return: Pretty printed string.
    """

    address_str: str = f" @ 0x{address:08x}" if address is not None else ""
    length_str: str = f"<{length}>" if length is not None else ""

    if content is not None:
        leading_zeros: str = "0" * ((content_max_width - content.bit_length()) // 4)
        value_str: str = f" = 0x{leading_zeros}{content:x}"
    else:
        value_str: str = ""

    if bool_props or kv_props:
        bool_props_str: str = (
            f"{', '.join(f'{v!s}' for v in bool_props)}" if bool_props else ""
        )
        kv_props_str: str = (
            f"{', '.join(f'{k}: {v!s}' for k, v in kv_props.items())})"
            if kv_props
            else ""
        )
        props_str = f" ({bool_props_str}{', ' if kv_props else ''}{kv_props_str})"
    else:
        props_str = ""

    return (
        f"[{name}{length_str}{address_str}{value_str}{props_str} {{{klass.__name__}}}]"
    )
