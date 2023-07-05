#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

"""
High level representation of a SVD device.
This representation does not aim to represent all the information contained in the SVD file,
but instead focuses on certain key features of the device description.
"""

from __future__ import annotations

import math
from functools import cached_property
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Collection,
    List,
    Tuple,
    Union,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Reversible,
    Sequence,
    Type,
    TypeVar,
)

from . import bindings
from .bindings import (
    Access,
    Cpu,
    RegisterProperties,
    Dimensions,
    WriteAction,
    ReadAction,
    WriteConstraint,
)
from ._device import topo_sort_derived_peripherals, svd_element_repr
from .memory_block import MemoryBlock
from .path import SPath
from . import util
from .util import LazyStaticList, LazyStaticMapping


class SvdPathError(IndexError, KeyError):
    """Error raised when trying to access a nonexistent/invalid SVD path."""

    def __init__(self, path: SPath, source: Any, explanation: str = "") -> None:
        formatted_explanation = "" if not explanation else f" ({explanation})"
        message = (
            f"{source!s} does not contain an element '{path}'{formatted_explanation}"
        )

        super().__init__(message)


class SvdTypeError(TypeError):
    def __init__(self, source: RegisterType, explanation: str = "") -> None:
        formatted_explanation = "" if not explanation else f" ({explanation})"
        message = f"{source!s} is what????{formatted_explanation}"

        super().__init__(message)


class SvdFlatArrayError:
    """Error raised when trying to access an element in a flat array"""

    def __init__(self, path: SPath, source: RegisterType) -> None:
        extra_info = "flat arrays do not have elements"

        super.__init__(path, source, extra_info=extra_info)


class SvdMemoryError(BufferError):
    ...


class Device(Mapping[str, "Peripheral"]):
    """
    Representation of a SVD device.
    """

    def __init__(self, device: bindings.DeviceElement):
        self._device: bindings.DeviceElement = device
        self._reg_props: RegisterProperties = self._device.register_properties

        if self._device.address_unit_bits != 8:
            raise NotImplementedError(
                f"the implementation assumes a byte-addressable device"
            )

        peripherals_unsorted: Dict[str, Peripheral] = {}

        # Initialize peripherals in topological order to ensure that base peripherals are
        # initialized before derived peripherals.
        for peripheral_element in topo_sort_derived_peripherals(device.peripherals):
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
    def qualified_name(self) -> str:
        """Name of the device, including vendor and series."""
        return f"{self.vendor_id or ''}::{self.series or ''}::{self.name}"

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
        return MappingProxyType(self._peripherals)

    def __getitem__(self, name: str) -> Peripheral:
        """
        :param name: Peripheral name.

        :return: Peripheral with the given name.
        """
        try:
            return self.peripherals[name]
        except LookupError as e:
            raise SvdPathError(SPath(name), self, "peripheral not found") from e

    def __iter__(self) -> Iterator[str]:
        """
        :return: Iterator over the names of peripherals in the device.
        """
        return iter(self.peripherals)

    def __len__(self) -> int:
        """
        :return: Number of peripherals in the device.
        """
        return len(self.peripherals)

    def __repr__(self) -> str:
        """Short description of the device."""
        return svd_element_repr(self.__class__, self.qualified_name, length=len(self))


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
        self._flat_registers: Dict[SPath, RegisterType] = {}
        self._dim_registers: Dict[SPath, RegisterType] = {}

    @property
    def name(self) -> str:
        """Name of the peripheral."""
        return self._peripheral.name

    @property
    def base_address(self) -> int:
        """Base address of the peripheral."""
        return self._base_address

    @property
    def interrupts(self) -> Mapping[str, int]:
        """Interrupts associated with the peripheral, a mapping from interrupt name to value."""
        return {
            interrupt.name: interrupt.value for interrupt in self._peripheral.interrupts
        }

    @cached_property
    def registers(self) -> Mapping[str, RegisterType]:
        """
        Mapping of top-level registers in the peripheral, indexed by name.
        """
        return LazyStaticMapping(
            keys=self._register_descriptions.keys(), factory=self.__getitem__
        )

    def register_iter(
        self, flat: bool = False, leaf_only: bool = False
    ) -> Iterator[RegisterType]:
        """
        Iterator over the registers in the peripheral in pre-order.
        Registers are ordered by increasing offset/address.

        :param flat: Do not yield individual registers in register arrays.
        :param leaf_only: Only yield registers at the bottom of the register tree.

        :return: Iterator over the registers in the peripheral.
        """
        stack: List[RegisterType] = [
            self._get_or_create_register(SPath(name), flat=flat)
            for name in reversed(self._register_descriptions.keys())
        ]

        while stack:
            register = stack.pop()
            is_leaf = (flat or not isinstance(register, Array)) and not isinstance(
                register, _Struct
            )

            if not leaf_only or is_leaf:
                yield register

            if not is_leaf:
                stack.extend(register.get_children(reverse=True))

    def get_content(
        self,
        absolute_addresses: bool = False,
        item_size: int = 1,
        byte_order: str = "little",
    ) -> Dict[int, int]:
        """Memory map of the peripheral register contents."""
        return self._memory_block.as_dict()

    def __getitem__(self, path: Union[str, Sequence[Union[str, int]]]) -> Register:
        """
        :param path: Name of the register to get, or a path to a register. # FIXME

        :return: The instance of the specified register.
        """
        return self._get_or_create_register(SPath(path))

    def __setitem__(self, name: str, value: int) -> None:
        """
        :param name: Name of the register to update.
        :param value: The raw register value to write to the specified register.
        """
        try:
            self[name].content = value
        except AttributeError:
            raise  # TODO

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

    @cached_property
    def _memory_block(self) -> MemoryBlock:
        """
        TODO
        """
        return self._register_info.memory_builder.build()

    def _get_or_create_register(
        self,
        path: Union[int, str, SPath],
        flat: bool = False,
        relative_to: Optional[RegisterType] = None,
    ) -> RegisterType:
        if relative_to is not None:
            full_path = relative_to.path.join(path)
        else:
            full_path = path

        storage = self._flat_registers if flat else self._dim_registers

        try:
            register = storage[full_path]
        except KeyError:
            if len(full_path) > 1:
                if relative_to is None:
                    # Build all the registers from the first available ancestor to the requested
                    # register.
                    search_path = path.parent
                    while (
                        relative_to := storage.get(search_path, None)
                    ) is None and len(search_path) > 1:
                        search_path = search_path.parent

                    if relative_to is None:
                        # We reached the top level without finding an initialized register
                        relative_to = self._create_top_level_register(path, flat)

                    # Create any registers on the path from the ancestor
                    for i in range(len(search_path) + 1, len(full_path)):
                        relative_to = self._create_register(
                            full_path[:i], flat, relative_to
                        )

                # Create the register itself
                register = self._create_register(full_path, flat, relative_to)

            else:
                # Top-level register
                register = self._create_top_level_register(full_path, flat)

            storage[full_path] = register

        return register

    @staticmethod
    def _reg_class_resolve(
        description: _RegisterDescription, path: SPath, flat: bool
    ) -> Type:
        return (
            Array
            if not flat and description.dimensions is not None and path.index is None
            else (Struct if description.registers is not None else Register)
        )

    def _create_top_level_register(self, path: SPath, flat: bool) -> RegisterType:
        """ """
        try:
            description = self._register_descriptions[path[-1]]
        except KeyError as e:
            raise SvdPathError(
                path,
                self,
                explanation="register was not found",
            ) from e

        reg_class = self._reg_class_resolve(description, path, flat)

        # Note: no need to handle array element creation here as it is impossible for a top-level
        # register to be an array element.
        return reg_class(description=description, peripheral=self, path=path)

    def _create_register(
        self, path: SPath, flat: bool, parent: RegisterType
    ) -> RegisterType:
        """ """
        """
        if not self.is_array():
            raise SvdPathError(path, self)

        if self.flat:
            raise SvdFlatArrayError(path, self)
        """

        # FIXME: error checking in general
        # FIXME: move some logic to child classes again!
        if isinstance(parent, Array):
            description = parent._description
            reg_class = self._reg_class_resolve(description, path, False)
            # TODO: check index/handle oob error
            instance_offset = (
                parent._instance_offset + description.dimensions.to_range()[path.index]
            )
        else:
            description = parent._description.registers[path.name]
            reg_class = self._reg_class_resolve(description, path, flat)
            instance_offset = parent._instance_offset

        return reg_class(
            description=description,
            peripheral=self,
            path=path,
            instance_offset=instance_offset,
        )

    @property
    def _register_descriptions(self) -> Mapping[str, _RegisterDescription]:
        """Mapping of register descriptions in the peripheral, indexed by name."""
        return self._register_info.descriptions

    @cached_property
    def _register_info(self) -> _ExtractedRegisterInfo:
        """Compute the immutable descriptions of the registers contained in the peripheral, taking into
        account registers derived from the base peripheral, if any.
        """

        base_descriptions: Optional[Mapping[str, _RegisterDescription]] = None
        base_memory: Optional[Callable[[], MemoryBlock]] = None

        if self._base_peripheral is not None:
            # If the register properties are equal, then it is possible to reuse all the immutable
            # properties from the base peripheral.
            if self._base_peripheral._reg_props == self._reg_props:
                base_descriptions = self._base_peripheral._register_descriptions
                base_memory = lambda: self._base_peripheral._memory_block
            # Otherwise, traverse the base registers again, because the difference in
            # register properties propagates down to the register elements.
            else:
                base_info = _extract_register_info(
                    self._base_peripheral._peripheral.registers,
                    self._reg_props,
                )
                base_descriptions = base_info.descriptions
                base_memory = lambda: base_info.memory_builder.build()

        info = _extract_register_info(
            self._peripheral.registers,
            self._reg_props,
            base_descriptions=base_descriptions,
            base_memory=base_memory,
        )

        return info

    def __repr__(self) -> str:
        """Short peripheral description."""
        return svd_element_repr(self.__class__, self.name, address=self.base_address)


class _RegisterDescription(NamedTuple):
    """
    Class containing immutable data describing a SVD register/cluster element.
    This is separated from the register classes to optimize construction of derived peripherals.
    Since the description is not tied to a specific Peripheral instance, it can be
    instantiated once per register/cluster in the SVD document and reused for derived peripherals.
    """

    name: str
    offset_start: int
    offset_end: int
    reg_props: bindings.RegisterProperties
    dimensions: Optional[bindings.Dimensions]
    registers: Optional[Mapping[str, _RegisterDescription]]
    fields: Optional[Mapping[str, _FieldDescription]]
    element: Union[bindings.RegisterElement, bindings.ClusterElement]

    def is_array(self) -> bool:
        return self.dimensions is not None

    def is_struct(self) -> bool:
        return self.registers is not None

    def has_fields(self) -> bool:
        return self.fields is not None


class _RegisterNode:
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
        # FIXME: should only use the instance offset in non-flat classes
        return self._description.offset_start + self._instance_offset

    # FIXME: remove these?
    @property
    def content(self) -> Optional[int]:
        """Current value of the register."""
        return None

    @property
    def registers(self) -> Optional[Mapping[str, RegisterType]]:
        return None

    @property
    def fields(self) -> Optional[Mapping[str, Field]]:
        """Map of fields in the register, indexed by name"""
        return None


# FIXME: generic typing
class _RegisterContainerMixin:
    def __getitem__(self, path: Union[int, Sequence[str, int]]) -> RegisterType:
        """
        :param index: Index of the register in the register array.

        :return: The instance of the specified register.
        """
        return self._peripheral._get_or_create_register(
            path=SPath(path), flat=False, relative_to=self
        )

    def get_children(self, reverse: bool = False) -> Iterator[RegisterType]:
        if reverse:
            for name in reversed(self):
                yield self[name]
        else:
            yield from self.values()


class _MutableRegisterContainerMixin(_RegisterContainerMixin):
    def __setitem__(self, path: Union[int, Sequence[str, int]], content: int) -> None:
        """
        :param name: Register name.
        :param content: Register value.
        """
        try:
            self[path].content = content
        # FIXME
        except AttributeError as e:
            raise TypeError(f"{self[path]!s} does not have content") from e


class Array(_RegisterNode, _MutableRegisterContainerMixin, Sequence["RegisterType"]):
    """Container of Structs and Registers"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __iter__(self) -> Iterator[RegisterType]:
        for i in range(len(self)):
            yield self[i]

    def __reversed__(self) -> Iterator[RegisterType]:
        for i in reversed(range(len(self))):
            yield self[i]

    def __len__(self) -> int:
        """
        :return: Number of registers in the register array.
        """
        return self._description.dimensions.length

    def get_children(self, reverse: bool = False) -> Iterator[RegisterType]:
        return reversed(self) if reverse else iter(self)

    def __repr__(self) -> str:
        """Short register description."""
        return svd_element_repr(
            self.__class__, self.path, address=self.offset, length=len(self)
        )


class _Struct(_RegisterNode, Reversible):
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

        self._registers: Optional[Mapping[str, RegisterType]] = None

    @property
    def registers(self) -> Mapping[str, RegisterType]:
        if self._registers is None:
            self._registers = LazyStaticMapping(
                keys=iter(self), factory=self.__getitem__
            )

        return self._registers

    def __iter__(self) -> Iterator[str]:
        """
        :return: Iterator over the names of registers in the register structure.
        """
        return iter(self._description.registers.keys())

    def __reversed__(self) -> Iterator[str]:
        return reversed(self._description.registers.keys())

    def __len__(self) -> int:
        """
        :return: Number of registers in the register structure
        """
        return len(self._description.registers)

    def __repr__(self) -> str:
        """Short register description."""
        return svd_element_repr(
            self.__class__,
            self.path,
            address=self.offset,
        )


class FlatStruct(_Struct, _RegisterContainerMixin):
    @property
    def dimensions(self) -> Optional[Dimensions]:
        """Dimensions of the register, if any."""
        return self._description.dimensions


class Struct(_Struct, _MutableRegisterContainerMixin, Mapping[str, "RegisterType"]):
    ...


class _Register(_RegisterNode, Mapping[str, "Field"]):
    """
    Physical register instance containing a value.
    Represents a SVD register element.
    """

    __slots__ = ["_fields"]

    def __init__(self, **kwargs):
        """
        See parent class for a description of parameters.
        """
        super().__init__(**kwargs)

        self._fields: Optional[Mapping[str, Field]] = None

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

    def __getitem__(self, name: str) -> Field:
        """
        :param name: Field name.

        :return: The instance of the specified field.
        """
        try:
            return self.fields[name]
        except LookupError as e:
            raise SvdPathError(
                name, self, explanation="no field matching the given path was found"
            ) from e

    def __iter__(self) -> Union[Iterator[RegisterType], Iterator[str]]:
        """
        :return: Iterator over the field names in the register.
        """
        return iter(self.fields)

    def __len__(self) -> int:
        """
        :return: Number of fields in the register.
        """
        return len(self.fields)

    def __repr__(self) -> str:
        bool_props = ("modified",) if self.modified else ()

        return svd_element_repr(
            self.__class__,
            self.path,
            address=self.offset,
            content=self.content,
            bool_props=bool_props,
        )


class FlatRegister(_Register):
    @property
    def dimensions(self) -> Optional[Dimensions]:
        """Dimensions of the register, if any."""
        return self._description.dimensions

    @property
    def fields(self) -> Mapping[str, FlatField]:
        """Map of fields in the register, indexed by name"""
        if self._fields is None:
            self._fields = LazyStaticMapping(
                keys=self._description.fields.keys(),
                factory=lambda name: FlatField(
                    description=self._description.fields[name], register=self
                ),
            )

        return MappingProxyType(self._fields)

    def __repr__(self) -> str:
        return svd_element_repr(
            self.__class__,
            self.path,
            address=self.offset,
            length=self.dimensions.length if self.dimensions is not None else None
        )



class Register(_Register):
    @property
    def fields(self) -> Mapping[str, Field]:
        """Map of fields in the register, indexed by name"""
        if self._fields is None:
            self._fields = LazyStaticMapping(
                keys=self._description.fields.keys(),
                factory=lambda name: Field(
                    description=self._description.fields[name], register=self
                ),
            )

        return MappingProxyType(self._fields)

    @property
    def modified(self) -> bool:
        """True if the register contains a different value now than at reset."""
        return self.content != self.reset_content

    @property
    def content(self) -> int:
        """Current value of the register."""
        content = self._peripheral._memory_block.at(self.offset, self.bit_width // 8)
        return int(content)

    @content.setter
    def content(self, new_content: int) -> None:
        """
        Set the value of the register.

        :param new_content: New value for the register.
        """
        self.set_content(new_content)

    def set_content(self, new_content: int, mask: Optional[int] = None) -> None:
        """
        Set the value of the register.

        :param new_content: New value for the register.
        :param mask: Mask of the bits to copy from the given value. If None, all bits are copied.
        """
        reg_width = self.bit_width

        if new_content > 0 and math.ceil(math.log2(new_content)) > reg_width:
            raise SvdMemoryError(
                f"Value {hex(new_content)} is too large for {reg_width}-bit register {self.path}."
            )

        for field in self.values():
            # Only check fields that are affected by the mask
            if mask is None or mask & field.mask:
                field_content = field._extract_content_from_register(new_content)
                if field_content not in field.allowed_values:
                    raise SvdMemoryError(
                        f"Value {hex(new_content)} is invalid for register {self.path}, as field "
                        f"{field.full_name} does not accept the value {hex(field_content)}."
                    )

        if mask is not None:
            # Update only the bits indicated by the mask
            new_content = (self.content & ~mask) | (new_content & mask)
        else:
            new_content = new_content

        self._peripheral._memory_block.set_at(
            self.offset, new_content, elem_size=reg_width // 8
        )

    def unconstrain(self) -> None:
        """
        Remove all value constraints imposed on the register.
        """
        for field in self.values():
            field.unconstrain()

    def __repr__(self) -> str:
        bool_props = ("modified",) if self.modified else ()

        return svd_element_repr(
            self.__class__,
            self.path,
            address=self.offset,
            content=self.content,
            bool_props=bool_props,
        )


# Regular register types
RegisterType = Union[Register, Struct]

# Flat register types
FlatRegisterType = Union[FlatRegister, FlatStruct]


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


class _Field:
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


class FlatField(_Field):
    ...


class Field(_Field):
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

        return svd_element_repr(
            self.__class__,
            self.path,
            content=self.content,
            content_max_width=self.bit_width,
            bool_props=bool_props,
        )


class _ExtractedRegisterInfo(NamedTuple):
    """Container for register descriptions and reset values."""

    descriptions: Mapping[str, _RegisterDescription]
    memory_builder: MemoryBlock.Builder


def _extract_register_info(
    elements: Iterable[Union[bindings.RegisterElement, bindings.ClusterElement]],
    base_reg_props: bindings.RegisterProperties,
    base_descriptions: Optional[Mapping[str, _RegisterDescription]] = None,
    base_memory: Optional[Callable[[], MemoryBlock]] = None,
) -> _ExtractedRegisterInfo:
    """
    Extract register descriptions for the given SVD register level elements.
    The returned structure mirrors the structure of the SVD elements.
    Each level of the structure is internally sorted by ascending address.

    :param elements: Register level elements to process.
    :param base_reg_props: Register properties inherited from the parent peripheral.

    :return: Map of register descriptions, indexed by name.
    """
    memory_builder = MemoryBlock.Builder()

    if base_memory is not None:
        memory_builder.lazy_copy_from(base_memory)

    description_list, min_addresss, max_address = _extract_register_descriptions_helper(
        memory_builder, elements, base_reg_props
    )

    descriptions = {d.name: d for d in description_list}

    if base_descriptions is not None:
        # The register maps are each sorted internally, but need to be merged by address
        # to ensure sorted order in the combined map
        descriptions = dict(
            util.iter_merged(
                descriptions.items(),
                base_descriptions.items(),
                key=lambda kv: kv[1].offset_start,
            )
        )

    if description_list:
        # Use the child address range if there is at least one child
        memory_builder.set_extent(
            offset=min_addresss, length=max_address - min_addresss
        )

    memory_builder.set_default_value(base_reg_props.reset_value)

    return _ExtractedRegisterInfo(descriptions, memory_builder)


def _extract_register_descriptions_helper(
    memory: MemoryBlock.Builder,
    elements: Iterable[Union[bindings.RegisterElement, bindings.ClusterElement]],
    base_reg_props: bindings.RegisterProperties,
    base_address: int = 0,
) -> Tuple[List[_RegisterDescription], int, int]:
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

    descriptions: List[_RegisterDescription] = []
    min_address: int = 2**32
    max_address: int = 0

    for element in elements:
        # Remove suffixes used for elements with dimensions
        name = util.strip_suffix(element.name, "[%s]")
        reg_props = element.get_register_properties(base_props=base_reg_props)
        dim_props = element.dimensions
        address_offset = element.offset

        registers: Optional[Mapping[str, _RegisterDescription]] = None
        fields: Optional[Mapping[str, _FieldDescription]] = None

        if isinstance(element, bindings.RegisterElement):
            # Register addresses are defined relative to the enclosing element
            if address_offset is not None:
                address_start = base_address + address_offset
            else:
                address_start = base_address

            size_bytes = reg_props.size // 8

            # Contiguous fill
            if dim_props is None or dim_props.step == size_bytes:
                length = dim_props.length if dim_props is not None else 1
                address_end = address_start + size_bytes * length
                memory.fill(
                    start=address_start,
                    end=address_end,
                    value=reg_props.reset_value,
                    elem_size=size_bytes,
                )

            # Fill with gaps
            elif dim_props is not None and dim_props.step > size_bytes:
                memory.fill(
                    start=address_start,
                    end=address_start + size_bytes,
                    value=reg_props.reset_value,
                    elem_size=size_bytes,
                )
                memory.tile(
                    start=address_start,
                    end=address_start + dim_props.step,
                    times=dim_props.length,
                )

            else:
                raise ValueError("step less than size")

            fields = _extract_field_descriptions(element.fields)

        else:  # ClusterElement
            # By the SVD specification, cluster addresses are defined relative to the peripheral
            # base address, but some SVDs don't follow this rule.
            if address_offset is not None:
                address_start = base_address + address_offset
                # address = address_offset
            else:
                address_start = base_address

            (
                sub_descriptions,
                sub_min_address,
                sub_max_address,
            ) = _extract_register_descriptions_helper(
                memory=memory,
                elements=element.registers,
                base_reg_props=reg_props,
                base_address=address_start,
            )

            if sub_descriptions:
                registers = {d.name: d for d in sub_descriptions}

                if address_offset is None:
                    address_start = sub_min_address

                if dim_props is not None and dim_props.length > 1:
                    if dim_props.step < sub_max_address - address_start:
                        raise ValueError("step less than size")

                    address_end = address_start + dim_props.step * dim_props.length

                    # Copy memory from sub elements along the struct array dimension
                    memory.tile(
                        start=address_start,
                        end=address_start + dim_props.step,
                        times=dim_props.length,
                    )

                else:  # Not an array
                    address_end = sub_max_address

            else:  # Empty struct
                registers = {}
                address_end = address_start

        description = _RegisterDescription(
            element=element,
            name=name,
            offset_start=address_start,
            offset_end=address_end,
            reg_props=reg_props,
            dimensions=dim_props,
            registers=registers,
            fields=fields,
        )

        descriptions.append(description)
        min_address = min(min_address, address_start)
        max_address = max(max_address, address_end)

    sorted_result = sorted(descriptions, key=lambda r: r.offset_start)

    # Check that our structural assumptions hold.
    # (they don't lmao)
    # if len(sorted_result) > 1:
    #    for i in range(1, len(sorted_result)):
    #        r1 = sorted_result[i - 1]
    #        r2 = sorted_result[i]
    #        if r1.offset_end > r2.offset_start and not r1.compatible_with(r2):
    #            raise ValueError(
    #                "overlapping structures"
    #            )  # FIXME: better error message

    return sorted_result, min_address, max_address


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
        key=lambda field: field.bit_range.offset,
    )

    fields = {description.name: description for description in field_descriptions}

    return fields
