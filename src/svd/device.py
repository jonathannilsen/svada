#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

"""
High level representation of a SVD device.
This representation does not aim to expose all the information contained in the SVD file,
but instead focuses on certain key features of the device description.

Since processing all the information contained in the SVD file can be computationally expensive,
many of the operations in this module lazily compute the data needed on first access.
"""

from __future__ import annotations

import dataclasses as dc
import math
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property, singledispatchmethod
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Collection,
    List,
    Tuple,
    Union,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    NoReturn,
    Optional,
    Protocol,
    Reversible,
    Sequence,
    Type,
    TypeVar,
    overload,
)
from typing_extensions import TypeGuard

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
from ._device import (
    ChildIter,
    LazyFixedMapping,
    remove_registers,
    topo_sort_derived_peripherals,
    svd_element_repr,
    iter_merged,
    strip_suffix,
)
from .errors import (
    SvdIndexError,
    SvdKeyError,
    SvdMemoryError,
    SvdPathError,
    SvdDefinitionError,
)
from .memory_block import MemoryBlock
from .path import SPath, FSPath, SPathType


@dataclass(frozen=True)
class Options:
    ignore_structural_errors: bool = False
    parent_relative_cluster_address: bool = False
    skip_registers: Mapping[str, Sequence[str]] = dc.field(
        default_factory=lambda: defaultdict(list)
    )


# Regular register types
RegisterUnion = Union["Array", "Register", "Struct"]

# Flat register types
FlatRegisterUnion = Union["FlatRegister", "FlatStruct"]

# Type variable constrained to either a regular or a flat register (but not a mix of regular/flat)
RegisterClass = TypeVar("RegisterClass", "RegisterUnion", "FlatRegisterUnion")

# Type variable constrained to either a regular or flat field
FieldClass = TypeVar("FieldClass", "Field", "FlatField")


class Device(Mapping[str, "Peripheral"]):
    """Representation of a SVD device."""

    def __init__(
        self, device: bindings.DeviceElement, options: Optional[Options] = None
    ):
        self._device: bindings.DeviceElement = device
        self._reg_props: RegisterProperties = self._device.register_properties

        if self._device.address_unit_bits != 8:
            raise NotImplementedError(
                "the implementation assumes a byte-addressable device"
            )

        peripherals_unsorted: Dict[str, Peripheral] = {}

        # Initialize peripherals in topological order to ensure that base peripherals are
        # initialized before derived peripherals.
        for peripheral_element in topo_sort_derived_peripherals(device.peripherals):
            if peripheral_element.derived_from is not None:
                base_peripheral = peripherals_unsorted[peripheral_element.derived_from]
            else:
                base_peripheral = None

            if options is not None and options.skip_registers:
                remove_registers(peripheral_element, options.skip_registers)

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
    def qualified_name(self) -> str:
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
        """
        Map of peripherals in the device, indexed by name.
        The peripherals are sorted by ascending base address.
        """
        return MappingProxyType(self._peripherals)

    def __getitem__(self, name: str) -> Peripheral:
        """
        :param name: Peripheral name.
        :raises SvdPathError: if the peripheral was not found.
        :return: Peripheral with the given name.
        """
        try:
            return self.peripherals[name]
        except LookupError as e:
            raise SvdKeyError(name, self, "peripheral not found") from e

    def __iter__(self) -> Iterator[str]:
        """:return: Iterator over the names of peripherals in the device."""
        return iter(self.peripherals)

    def __len__(self) -> int:
        """:return: Number of peripherals in the device."""
        return len(self.peripherals)

    def __repr__(self) -> str:
        """Short description of the device."""
        return svd_element_repr(self.__class__, self.qualified_name, length=len(self))


# FIXME: make sure that copied memory is clean, otherwise fall back to retraversing
class Peripheral(Mapping[str, RegisterUnion]):
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
        device: Optional[Device],
        base_reg_props: bindings.RegisterProperties,
        base_peripheral: Optional[Peripheral] = None,
        new_base_address: Optional[int] = None,
    ):
        """
        :param element: SVD peripheral element.
        :param device: Parent Device object (may be None if this is a copy).
        :param base_reg_props: Register properties of the parent device.
        :param base_peripheral: Base peripheral that this peripheral is derived from, if any.
        :param new_base_address: Overridden base address of this peripheral, if any.
        """

        self._peripheral: bindings.PeripheralElement = element
        self._device: Optional[Device] = device
        self._base_peripheral: Optional[Peripheral] = base_peripheral
        self._base_address: int = (
            element.base_address if new_base_address is None else new_base_address
        )
        self._reg_props: bindings.RegisterProperties = (
            self._peripheral.get_register_properties(base_props=base_reg_props)
        )

        # These dicts store every register associated with the peripheral.
        self._flat_registers: Dict[FSPath, FlatRegisterUnion] = {}
        self._dim_registers: Dict[SPath, RegisterUnion] = {}

    def copy_to(self, new_base_address: int) -> Peripheral:
        """
        Copy the peripheral to a new base address.

        :param new_base_address: Base address of the new peripheral.
        :returns: A copy of this peripheral at the new base address.
        """
        return Peripheral(
            element=self._peripheral,
            device=None,
            base_reg_props=self._reg_props,
            base_peripheral=self,
            new_base_address=new_base_address,
        )

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
        """Interrupts associated with the peripheral. Mapping from interrupt name to value."""
        return {
            interrupt.name: interrupt.value for interrupt in self._peripheral.interrupts
        }

    @cached_property
    def registers(self) -> Mapping[str, RegisterUnion]:
        """Mapping of top-level registers in the peripheral, indexed by name."""
        return LazyFixedMapping(
            keys=self._register_descriptions.keys(), factory=self.__getitem__
        )

    def register_iter(self, leaf_only: bool = False) -> Iterator[RegisterUnion]:
        """
        Iterator over the registers in the peripheral in pre-order.
        Registers are ordered by increasing offset/address.

        :param flat: Do not yield individual registers in register arrays.
        :param leaf_only: Only yield registers at the bottom of the register tree.
        :return: Iterator over the registers in the peripheral.
        """
        return self._register_iter(SPath, leaf_only)

    @cached_property
    def flat_registers(self) -> Mapping[str, FlatRegisterUnion]:
        """
        Mapping of top-level flat registers in the peripheral, indexed by name.
        The
        """
        return LazyFixedMapping(
            keys=self._register_descriptions.keys(),
            factory=lambda n: self._get_or_create_register(FSPath(n)),
        )

    def flat_register_iter(
        self, leaf_only: bool = False
    ) -> Iterator[FlatRegisterUnion]:
        """
        Iterator over the registers in the peripheral in pre-order.
        Registers are ordered by increasing offset/address.

        :param flat: Do not yield individual registers in register arrays.
        :param leaf_only: Only yield registers at the bottom of the register tree.
        :return: Iterator over the registers in the peripheral.
        """
        return self._register_iter(FSPath, leaf_only)

    def _register_iter(self, path_cls: Type, leaf_only: bool = False) -> Iterator:
        """Commmon register iteration implementation."""
        stack = [
            self._get_or_create_register(path_cls(name))
            for name in reversed(self._register_descriptions.keys())
        ]

        while stack:
            register = stack.pop()

            if register.leaf or not leaf_only:
                yield register

            if not register.leaf:
                stack.extend(reversed(register.child_iter()))

    def memory_iter(
        self,
        item_size: int = 1,
        absolute_addresses: bool = False,
        native_byteorder: bool = False,
    ) -> Iterator[Tuple[int, int]]:
        """
        Get an iterator over the peripheral register contents.

        :param item_size: Byte granularity of the iterator.
        :param absolute_addresses: If True, use absolute instead of peripheral relative addresses.
        :param native_byteorder: If true, use native byte order instead of device byte order.
        :return: Iterator over the peripheral register contents.
        """
        # TODO: actually use byteorder
        address_offset = self.base_address if absolute_addresses else 0
        yield from self._memory_block.memory_iter(item_size, with_offset=address_offset)

    def __getitem__(self, path: Union[str, Sequence[Union[str, int]]]) -> RegisterUnion:
        """
        :param path: Name or path of the register.
        :return: Register instance.
        """
        return self._get_or_create_register(SPath(path))

    def __setitem__(self, name: str, content: int) -> None:
        """
        :param name: Name of the register to update.
        :param content: The raw register value to write to the specified register.
        """
        _register_set_content(self, name, content)

    def __iter__(self) -> Iterator[str]:
        """:return: Iterator over the names of top-level registers in the peripheral."""
        return iter(self._register_descriptions)

    def __len__(self) -> int:
        """:return: Number of top-level registers in the peripheral."""
        return len(self._register_descriptions)

    @singledispatchmethod
    def _get_or_create_register(self, path: Any) -> Any:
        """
        Common method for accessing a register contained in the peripheral.
        If the register is accessed for the first time, it is first initialized.
        Otherwise, a cached register is returned.
        Note that if the requested register is below the top level, all the registers that are
        ancestors of the register are also initialized if needed.

        :param path: Path to the register.
        :return: The register instance at the given path.
        """
        raise ValueError(f"Invalid path {path}")

    @_get_or_create_register.register
    def _(self, path: SPath) -> RegisterUnion:
        return self._do_get_or_create_register(self._dim_registers, path)

    @_get_or_create_register.register
    def _(self, path: FSPath) -> FlatRegisterUnion:
        return self._do_get_or_create_register(self._flat_registers, path)

    def _do_get_or_create_register(
        self, storage: Dict[SPathType, RegisterClass], path: SPathType
    ) -> RegisterClass:
        try:
            register = storage[path]
        except KeyError:
            ancestor_path = path.parent
            if ancestor_path is not None:
                # Build all the registers from the first available ancestor to the requested
                # register.
                while (register := storage.get(ancestor_path, None)) is None and (
                    parent := ancestor_path.parent
                ) is not None:
                    ancestor_path = parent

                if register is None:
                    register = self._create_register(ancestor_path)

                for i in range(len(ancestor_path), len(path)):
                    register = self._create_register(path[:i + 1], register)
            else:
                register = self._create_register(path)

            storage[path] = register

        return register

    @singledispatchmethod
    def _create_register(self, path: Any, parent: Any = None) -> Any:
        raise ValueError(f"Invalid path: {path}")

    @_create_register.register
    def _(self, path: SPath, parent: Optional[RegisterUnion] = None) -> RegisterUnion:
        if parent is None:
            try:
                description = self._register_descriptions[path.stem]
            except KeyError:
                raise SvdPathError(path, self)

            instance_offset = 0

        elif isinstance(parent, Array):
            description = parent._description

            index = path.element_index
            if index is None:
                # TypeError?
                raise SvdIndexError(path, parent, f"expected an array index")

            try:
                array_offset = description.dimensions.to_range()[index]
            except IndexError:
                raise SvdIndexError(
                    path,
                    parent,
                    f"index {index} is out of range for array with length "
                    f"{description.dimensions.length}",
                )

            instance_offset = parent._instance_offset + array_offset

        elif isinstance(parent, Struct):
            try:
                description = parent._description.registers[path.stem]
            except KeyError:
                raise SvdKeyError(path, parent)

            instance_offset = parent._instance_offset

        else:
            raise ValueError(f"Invalid parent register: {parent}")

        reg_class = (
            Array
            if description.is_array() and path.element_index is None
            else (Struct if description.is_struct() else Register)
        )

        return reg_class(
            description=description,
            peripheral=self,
            path=path,
            instance_offset=instance_offset,
        )

    @_create_register.register
    def _(
        self, path: FSPath, parent: Optional[FlatRegisterUnion] = None
    ) -> FlatRegisterUnion:
        """Create a flat register instance."""
        if parent is None:
            try:
                description = self._register_descriptions[path.stem]
            except KeyError as e:
                raise SvdKeyError(path, self) from e

            instance_offset = 0

        elif isinstance(parent, FlatStruct):
            try:
                description = parent._description.registers[path.stem]
            except KeyError as e:
                raise SvdKeyError(path, parent) from e

            instance_offset = parent._instance_offset
        else:
            raise ValueError(f"Invalid parent register: {parent}")

        reg_class = FlatStruct if description.is_struct() else FlatRegister

        return reg_class(
            description=description,
            peripheral=self,
            path=path,
            instance_offset=instance_offset,
        )

    @cached_property
    def _memory_block(self) -> MemoryBlock:
        """
        The memory block describing the values in the peripheral.
        This is computed the first time the property is accessed and cached for later.
        Initially this contains the reset values described in the SVD file.

        Note that accessing this property in a derived peripheral may also cause the memory blocks
        of base peripherals to be computed.
        """
        return self._register_info.memory_builder.build()

    @property
    def _register_descriptions(self) -> Mapping[str, _RegisterSpec]:
        """Mapping of register descriptions in the peripheral, indexed by name."""
        return self._register_info.descriptions

    @cached_property
    def _register_info(self) -> _ExtractedRegisterInfo:
        """
        Compute the descriptions of the registers contained in the peripheral, taking into
        account registers derived from the base peripheral, if any.
        """
        base_descriptions: Optional[Mapping[str, _RegisterSpec]] = None
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


def _register_set_content(container: Any, path: Union[int, str], content: int) -> None:
    """Common function for setting the content of a register."""
    try:
        container[path].content = content
    except AttributeError as e:
        raise SvdMemoryError(f"{container[path]!s} does not have content") from e


class _RegisterSpec(NamedTuple):
    """
    Immutable description of a SVD register/cluster element.
    This is separated from the register classes to optimize construction of derived peripherals.
    Since the description is not tied to a specific Peripheral instance, it can be
    instantiated once per register/cluster in the SVD document and reused for derived peripherals,
    as long as inherited properties are the same for the base and derived peripheral.
    """

    # Register name
    name: str
    # Lowest address offset contained within this element and any descendants
    offset_start: int
    # Highest address offset contained within this element and any descendants
    offset_end: int
    # Effective register properties, either inherited or specified on the element itself
    reg_props: bindings.RegisterProperties
    # Register dimensions
    dimensions: Optional[bindings.Dimensions]
    # Direct child registers of this element
    registers: Optional[Mapping[str, _RegisterSpec]]
    # Child fields of this element
    fields: Optional[Mapping[str, _FieldSpec]]
    # The SVD element itself
    element: Union[bindings.RegisterElement, bindings.ClusterElement]

    def is_array(self) -> bool:
        """True if the SVD element describes an array, i.e. has dimensions"""
        return self.dimensions is not None

    def is_struct(self) -> bool:
        """True if the SVD element describes a structure, i.e. contains other registers"""
        return self.registers is not None

    def has_fields(self) -> bool:
        """True if the SVD element contains fields."""
        return self.fields is not None


class _RegisterNode(ABC, Generic[SPathType]):
    """Base class for all register types"""

    __slots__ = [
        "_description",
        "_peripheral",
        "_path",
        "_instance_offset",
    ]

    def __init__(
        self,
        description: _RegisterSpec,
        peripheral: Peripheral,
        path: SPathType,
        instance_offset: int = 0,
    ):
        """
        :param description: Register description.
        :param peripheral: Parent peripheral.
        :param path: Path of the register.
        :param instance_offset: Address offset inherited from the parent register.
        """
        self._description: _RegisterSpec = description
        self._peripheral: Peripheral = peripheral
        self._path: SPathType = path
        self._instance_offset: int = instance_offset

    @property
    def name(self) -> str:
        """Name of the register."""
        return self.path.name

    @property
    def path(self) -> SPathType:
        """Full path to the register."""
        return self._path

    @property
    def address(self) -> int:
        """Absolute address of the peripheral in memory"""
        return self._peripheral.base_address + self.offset

    @property
    def offset(self) -> int:
        """Address offset of the register, relative to the peripheral it is contained in"""
        return self._description.offset_start + self._instance_offset

    @property
    @abstractmethod
    def leaf(self) -> bool:
        ...


class Array(_RegisterNode[SPath], Sequence[RegisterUnion]):
    """Container of Structs and Registers"""

    @overload
    def __getitem__(self, path: int, /) -> RegisterUnion:
        ...

    @overload
    def __getitem__(self, path: slice, /) -> Sequence[RegisterUnion]:
        ...

    @overload
    def __getitem__(self, path: Sequence[Union[int, str]], /) -> RegisterUnion:
        ...

    def __getitem__(self, path: Any, /) -> Any:
        """ """
        if isinstance(path, slice):
            return [self[i] for i in range(*path.indices(len(self)))]

        return self._peripheral._get_or_create_register(self.path.join(path))

    def __setitem__(self, path: int, content: int, /) -> None:
        _register_set_content(self, path, content)

    def __iter__(self) -> Iterator[RegisterUnion]:
        for i in range(len(self)):
            yield self[i]

    def __reversed__(self) -> Iterator[RegisterUnion]:
        for i in reversed(range(len(self))):
            yield self[i]

    def __len__(self) -> int:
        """:return: Number of registers in the register array."""
        return self._description.dimensions.length

    @property
    def leaf(self) -> bool:
        return False

    def child_iter(self) -> Reversible[RegisterUnion]:
        return ChildIter(range(len(self)), self.__getitem__)

    def __repr__(self) -> str:
        """Short register description."""
        return svd_element_repr(
            self.__class__, str(self.path), address=self.offset, length=len(self)
        )


class _Struct(_RegisterNode, Mapping[str, RegisterClass]):
    """Class implementing common struct functionality."""

    __slots__ = ["_registers"]

    def __init__(self, **kwargs: Any) -> None:
        """See parent class for a description of parameters."""
        super().__init__(**kwargs)

        self._registers: Optional[Mapping[str, RegisterClass]] = None

    @property
    def leaf(self) -> bool:
        """"""
        return False

    @property
    def registers(self) -> Mapping[str, RegisterClass]:
        """:return A mapping of registers in the structure, ordered by ascending address."""
        if self._registers is None:
            self._registers = LazyFixedMapping(
                keys=iter(self), factory=self.__getitem__
            )

        return self._registers

    def __getitem__(self, path: Union[str, Sequence[Union[str, int]]]) -> RegisterClass:
        """
        :param index: Index of the register in the register array.
        :return: The instance of the specified register.
        """
        return self._peripheral._get_or_create_register(self.path.join(path))

    def __iter__(self) -> Iterator[str]:
        """:return: Iterator over the names of registers in the register structure."""
        return iter(self._description.registers)

    def __len__(self) -> int:
        """:return: Number of registers in the register structure"""
        return len(self._description.registers)

    def child_iter(self) -> Reversible[RegisterClass]:
        return ChildIter(self._description.registers.keys(), self.__getitem__)

    def __repr__(self) -> str:
        """Short register description."""
        return svd_element_repr(self.__class__, self.path, address=self.offset)


class FlatStruct(_Struct[FlatRegisterUnion]):
    """
    Register structure representing a group of registers.
    Represents a SVD cluster element.
    """

    @property
    def dimensions(self) -> Optional[Dimensions]:
        """Dimensions of the register, if any."""
        return self._description.dimensions

    def __repr__(self) -> str:
        """Short struct description."""
        return svd_element_repr(
            self.__class__,
            self.path,
            address=self.offset,
            length=self.dimensions.length if self.dimensions is not None else None,
        )


class Struct(_Struct[RegisterUnion]):
    """
    Register structure representing a group of registers.
    Represents either a SVD cluster element without dimensions,
    or a specific index of a cluster array.
    """

    def __setitem__(self, path: str, content: int) -> None:
        _register_set_content(self, path, content)


class _Register(_RegisterNode, Mapping[str, FieldClass]):
    __slots__ = ["_fields"]

    def __init__(self, **kwargs: Any) -> None:
        """See parent class for a description of parameters."""
        super().__init__(**kwargs)

        self._fields: Optional[Mapping[str, FieldClass]] = None

    @property
    def leaf(self) -> bool:
        return True

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
    def read_action(self) -> Optional[ReadAction]:
        """Side effect of reading from the register"""
        return self._description.element.read_action

    @property
    def fields(self) -> Mapping[str, FieldClass]:
        """Map of fields in the register, indexed by name"""
        if self._fields is None:
            self._fields = LazyFixedMapping(
                keys=self._description.fields.keys(),
                factory=self._create_field,
            )

        return MappingProxyType(self._fields)

    def __getitem__(self, name: str) -> FieldClass:
        """
        :param name: Field name.
        :return: The instance of the specified field.
        """
        try:
            return self.fields[name]
        except LookupError as e:
            raise SvdKeyError(
                name, self, explanation="no field matching the given path was found"
            ) from e

    def __iter__(self) -> Iterator[str]:
        """:return: Iterator over the field names in the register."""
        return iter(self._description.fields)

    def __len__(self) -> int:
        """:return: Number of fields in the register."""
        return len(self._description.fields)

    @abstractmethod
    def _create_field(self, name: str) -> FieldClass:
        ...


class FlatRegister(_Register["FlatField"]):
    """
    Represents a SVD register element.
    """

    @property
    def dimensions(self) -> Optional[Dimensions]:
        """Dimensions of the register, if any."""
        return self._description.dimensions

    def _create_field(self, name: str) -> FlatField:
        return FlatField(description=self._description.fields[name], register=self)

    def __repr__(self) -> str:
        return svd_element_repr(
            self.__class__,
            self.path,
            address=self.offset,
            length=self.dimensions.length if self.dimensions is not None else None,
        )


class Register(_Register["Field"]):
    """
    Represents a SVD register element.
    """

    @property
    def leaf(self) -> bool:
        return True

    @property
    def address_range(self) -> range:
        return range(self.address, self.address + self.bit_width // 8)

    @property
    def offset_range(self) -> range:
        """Range of addresses covered by the register."""
        return range(self.offset, self.offset + self.bit_width // 8)

    def __setitem__(self, key: str, content: Union[str, int]) -> None:
        """
        :param key: Either the bit offset of a field, or the field's name.
        :param content: A raw numeric value, or a field enumeration, to write
            to the selected register field.
        """
        self[key].content = content

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
                        f"{field.name} does not accept the value {hex(field_content)}."
                    )

        if mask is not None:
            # Update only the bits indicated by the mask
            new_content = (self.content & ~mask) | (new_content & mask)
        else:
            new_content = new_content

        self._peripheral._memory_block.set_at(
            self.offset, new_content, item_size=reg_width // 8
        )

    def unconstrain(self) -> None:
        """Remove all value constraints imposed on the register."""
        for field in self.values():
            field.unconstrain()

    def _create_field(self, name: str) -> Field:
        return Field(description=self._description.fields[name], register=self)

    def __repr__(self) -> str:
        """Short register description."""
        bool_props = ("modified",) if self.modified else ()

        return svd_element_repr(
            self.__class__,
            self.path,
            address=self.offset,
            content=self.content,
            bool_props=bool_props,
        )


class _FieldSpec(NamedTuple):
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
    def from_element(cls, element: bindings.FieldElement) -> _FieldSpec:
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


FieldParent = TypeVar("FieldParent", Register, FlatRegister)


class _Field(Generic[FieldParent]):
    """
    Register field instance.
    Represents a SVD field element.
    """

    __slots__ = ["_description", "_register", "_allowed_values"]

    def __init__(
        self,
        description: _FieldSpec,
        register: FieldParent,
    ):
        """
        Initialize the class attribute(s).

        :param description: Field description.
        :param register: Register to which the field belongs.
        """
        self._description: _FieldSpec = description
        self._register: FieldParent = register
        self._allowed_values = description.allowed_values

    @property
    def name(self) -> str:
        """Name of the field."""
        return self._description.name

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
    def read_action(self) -> Optional[ReadAction]:
        """Side effect of reading from the field."""
        if (field_read_action := self._description.element.read_action) is not None:
            return field_read_action
        return self._register.read_action

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

    def _extract_content_from_register(self, register_content: int) -> int:
        """
        Internal method for extracting the field value from the parent register value.

        :param register_value: Value of the parent register
        :return: Field value extracted based on the field bit range
        """
        return (register_content & self.mask) >> self.bit_offset


class FlatField(_Field):
    """SVD field"""

    def __repr__(self) -> str:
        """Short field description."""
        return svd_element_repr(
            self.__class__,
            self.name,
            content_max_width=self.bit_width,
        )


class Field(_Field):
    """SVD field"""

    @property
    def content(self) -> int:
        """The value of the field."""
        return self._extract_content_from_register(self._register.content)

    @content.setter
    def content(self, new_content: Union[int, str]) -> None:
        """
        Set the value of the field.

        :param value: A numeric value, or the name of a field enumeration (if applicable), to
        write to the field.
        """
        if isinstance(new_content, int):
            val = self._trailing_zero_adjusted(new_content)

            if val not in self.allowed_values:
                raise ValueError(
                    f"{self!r} does not accept"
                    f" the bit value '{val}' ({hex(val)})."
                    " Are you sure you have an up to date .svd file?"
                )
            resolved_value = val
        elif isinstance(new_content, str):
            if new_content not in self.enums:
                raise ValueError(
                    f"{self!r} does not accept"
                    f" the enum '{new_content}'."
                    " Are you sure you have an up to date .svd file?"
                )
            resolved_value = self.enums[new_content]
        else:
            raise TypeError(
                f"Field does not accept write of '{new_content}' of type '{type(new_content)}'"
                " Permitted values types are 'str' (field enum) and 'int' (bit value)."
            )

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
                raise SvdMemoryError(f"Unexpected trailing value: {trailing}")

            cropped = hex_val[:max_val_hex_len]  # value w/o trailing
            adjusted = int(cropped, 16)

            if adjusted <= max_val:
                return adjusted

        return content

    def __repr__(self) -> str:
        """Short field description."""
        bool_props = ("modified",) if self.modified else ()

        return svd_element_repr(
            self.__class__,
            self.name,
            content=self.content,
            content_max_width=self.bit_width,
            bool_props=bool_props,
        )


class _ExtractedRegisterInfo(NamedTuple):
    """Container for register descriptions and reset values."""

    descriptions: Mapping[str, _RegisterSpec]
    memory_builder: MemoryBlock.Builder


def _extract_register_info(
    elements: Iterable[Union[bindings.RegisterElement, bindings.ClusterElement]],
    base_reg_props: bindings.RegisterProperties,
    base_descriptions: Optional[Mapping[str, _RegisterSpec]] = None,
    base_memory: Optional[Callable[[], MemoryBlock]] = None,
) -> _ExtractedRegisterInfo:
    """
    Extract register descriptions for the given SVD register level elements.
    The returned structure mirrors the structure of the SVD elements.
    Each level of the structure is internally sorted by ascending address.

    :param elements: Register level elements to process.
    :param base_reg_props: Register properties of the peripheral.
    :param base_descriptions: Register descriptions inherited from the base peripheral, if any.
    :param base_memory: Memory inherited from the base peripheral, if any.
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
            iter_merged(
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

    memory_builder.set_default_content(base_reg_props.reset_value)

    return _ExtractedRegisterInfo(descriptions, memory_builder)


class _ExtractHelperResult(NamedTuple):
    descriptions: List[_RegisterSpec]
    min_address: int
    max_address: int


def _extract_register_descriptions_helper(
    memory: MemoryBlock.Builder,
    elements: Iterable[Union[bindings.RegisterElement, bindings.ClusterElement]],
    base_reg_props: bindings.RegisterProperties,
    base_address: int = 0,
    validate_overlap: bool = False  # True, # FIXME
) -> _ExtractHelperResult:
    """
    Helper that recursively extracts the names, addresses, register properties, dimensions,
    fields etc. of a collection of SVD register level elements.

    :param memory: Memory builder.
    :param elements: SVD register level elements.
    :param base_reg_props: Base address of the parent SVD element.
    :param base_address: Base address of the parent SVD element.
    :param validate_overlap: If True, raise an exception if overlapping registers are detected.
    :return: Extraction result.
    """

    descriptions: List[_RegisterSpec] = []
    min_address: int = 2**32
    max_address: int = 0

    for element in elements:
        # Remove suffixes used for elements with dimensions
        name = strip_suffix(element.name, "[%s]")
        reg_props = element.get_register_properties(base_props=base_reg_props)
        dim_props = element.dimensions
        address_offset = element.offset

        registers: Optional[Mapping[str, _RegisterSpec]] = None
        fields: Optional[Mapping[str, _FieldSpec]] = None

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
                    content=reg_props.reset_value,
                    item_size=size_bytes,
                )

            # Fill with gaps
            elif dim_props is not None and dim_props.step > size_bytes:
                memory.fill(
                    start=address_start,
                    end=address_start + size_bytes,
                    content=reg_props.reset_value,
                    item_size=size_bytes,
                )
                memory.tile(
                    start=address_start,
                    end=address_start + dim_props.step,
                    times=dim_props.length,
                )

            else:
                raise SvdDefinitionError(
                    element,
                    f"step of 0x{dim_props.step:x} is less than the size of the "
                    f"array (0x{size_bytes})",
                )

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
                        raise SvdDefinitionError(
                            element,
                            f"step of 0x{dim_props.step:x} is less than the size required to "
                            f"cover all the child elements (0x{sub_max_address - address_start})",
                        )

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

        description = _RegisterSpec(
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
    if validate_overlap:
        if len(sorted_result) > 1:
            for i in range(1, len(sorted_result)):
                r1 = sorted_result[i - 1]
                r2 = sorted_result[i]
                if r1.offset_end > r2.offset_start:
                    raise SvdDefinitionError(
                        [r1.element, r2.element], "element addresses overlap"
                    )

    return _ExtractHelperResult(sorted_result, min_address, max_address)


def _extract_field_descriptions(
    elements: Iterable[bindings.FieldElement],
) -> Optional[Mapping[str, _FieldSpec]]:
    """
    Extract field descriptions for the given SVD field elements.
    The resulting mapping is internally sorted by ascending field bit offset.

    :param elements: Field elements to process.
    :return: Mapping of field descriptions, indexed by name.
    """

    field_descriptions = sorted(
        [_FieldSpec.from_element(field) for field in elements],
        key=lambda field: field.bit_range.offset,
    )

    fields = {description.name: description for description in field_descriptions}

    return fields
