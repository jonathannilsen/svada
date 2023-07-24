#
# Copyright (c) 2023 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Various internal functionality used by the bindings module.
"""

from __future__ import annotations

import enum
import inspect
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Type, TypeVar, Union

from lxml import objectify

from .util import CaseInsensitiveStrEnum


class CaseInsensitiveStrEnum(enum.Enum):
    """String enum class that can be constructed from a case-insensitive string."""

    @classmethod
    def from_str(cls, value: str) -> CaseInsensitiveStrEnum:
        """Construct the enum from a case-insensitive string."""
        value_lower = value.lower()
        for member in cls:
            if member.value.lower() == value_lower:
                return member
        raise ValueError(
            f"Class {cls.__qualname__} has no member corresponding to '{value}'"
        )


def to_int(number: str) -> int:
    """
    Convert a string representation of an integer following the SVD format to its corresponding
    integer representation.

    :param number: String representation of the integer.

    :return: Decoded integer.
    """

    if number.startswith("0x"):
        return int(number, base=16)

    if number.startswith("#"):
        return int(number[1:], base=2)

    return int(number)


def to_bool(value: str) -> bool:
    """
    Convert a string representation of a boolean following the SVD format to its corresponding
    boolean representation.

    :param value: String representation of the boolean.

    :return: Decoded boolean.
    """
    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False
    raise ValueError(f"Invalid boolean value: {value}")


class SvdIntElement(objectify.IntElement):
    """
    Element containing an SVD integer value.
    This class uses a custom parser to convert the value to an integer.
    """

    def _init(self):
        self._setValueParser(to_int)


class _Self:
    """
    Sentinel value used for element properties where the result class is equal to the class of the
    parent object. This is required since self-referential class members are not allowed.
    """

    ...


SELF_CLASS = _Self


class _Missing:
    """Sentinel value used to indicate that a default value is missing."""

    ...


MISSING = _Missing


class elem:
    """Data descriptor class used to access a XML element."""

    def __init__(
        self,
        name: str,
        klass: type,
        /,
        *,
        default: Union[Any, MISSING] = MISSING,
        default_factory: Union[Callable[[], Any], MISSING] = MISSING,
    ) -> None:
        """
        Create a data descriptor object that extracts an element from an XML node.
        Only one of default or default_factory can be set.

        :param name: Name of the element.
        :param klass: Class to use for the extracted element.
        :param default: Default value to return if the element is not found.
        :param default_factory: Callable that returns the default value to return if the element is
                                not found.
        """
        if default != MISSING and default_factory != MISSING:
            raise ValueError("Cannot set both default and default_factory")

        self.name: str = name
        self.klass: type = klass
        self.default: Union[Any, MISSING] = default
        self.default_factory: Union[Callable[[], Any], MISSING] = default_factory

    def __get__(self, node: Optional[objectify.ObjectifiedElement], owner: Any = None):
        """Get the element value from the given node."""

        # If the node argument is None, we are being accessed through the class object.
        # In that case, return the descriptor itself.
        if node is None:
            # Return self when accessed through the class, e.g., NodeClass.my_elem
            return self
        # else: the elem is being accessed through an instance, e.g. node_obj.my_elem
        try:
            svd_obj = node.__getattr__(self.name)
        except AttributeError:
            if self.default_factory != MISSING:
                return self.default_factory()
            if self.default != MISSING:
                return self.default
            raise

        if issubclass(self.klass, objectify.ObjectifiedDataElement):
            return svd_obj.pyval
        else:
            return svd_obj


class attr:
    """Data descriptor used to access a XML attribute."""

    def __init__(
        self,
        name: str,
        /,
        *,
        converter: Optional[Callable[[str], Any]] = None,
        default: Union[Any, MISSING] = MISSING,
        default_factory: Union[Callable[[], Any], MISSING] = MISSING,
    ) -> None:
        """
        Create a data descriptor object that extracts an attribute from an XML node.
        Only one of default or default_factory can be set.

        :param name: Name of the attribute.
        :param converter: Optional callable that converts the attribute value from a string to another
                        type.
        :param default: Default value to return if the element is not found.
        :param default_factory: Callable that returns the default value to return if the element is
                                not found.
        """
        if default != MISSING and default_factory != MISSING:
            raise ValueError("Cannot set both default and default_factory")

        self.name: str = name
        self.converter: Optional[Callable[[str], Any]] = converter
        self.default: Union[Any, MISSING] = default
        self.default_factory: Union[Callable[[], Any], MISSING] = default_factory

    def __get__(self, node: Optional[objectify.ObjectifiedElement], _owner: Any = None):
        """Get the attribute value from the given node."""

        # If the node argument is None, we are being accessed through the class object.
        # In that case, return the descriptor itself.
        if node is None:
            return self

        value = node.get(self.name)

        if value is None:
            if self.default_factory != MISSING:
                return self.default_factory()
            if self.default != MISSING:
                return self.default
            raise AttributeError(f"Attribute {self.name} was not found")

        if self.converter is None:
            return value

        try:
            return self.converter(value)
        except Exception as e:
            raise ValueError(f"Error converting attribute {self.name}") from e


C = TypeVar("C")


class BindingRegistry:
    """Simple container for XML binding classes."""

    def __init__(self) -> None:
        self._element_classes: List[Type[objectify.ObjectifiedElement]] = []

    def add(
        self,
        klass: Type[C],
        /,
    ) -> Type[C]:
        """
        Add a class to the binding registry.

        This is intended to be used as a class decorator.
        """

        elem_props: Dict[str, elem] = getattr(klass, "_xml_elem_props", {})
        attr_props: Dict[str, attr] = getattr(klass, "_xml_attr_props", {})

        for name, prop in inspect.getmembers(klass):
            if isinstance(prop, elem):
                if prop.klass == SELF_CLASS:
                    prop.klass = klass
                elem_props[name] = prop

            elif isinstance(prop, attr):
                attr_props[name] = prop

        setattr(klass, "_xml_elem_props", elem_props)
        setattr(klass, "_xml_attr_props", attr_props)

        self._element_classes.append(klass)

        return klass

    @property
    def bindings(self) -> List[Type[objectify.ObjectifiedElement]]:
        """Get the list of registered bindings."""
        return self._element_classes


def get_binding_elem_props(
    klass: type,
) -> Mapping[str, Union[attr, elem]]:
    """Get the XML element properties of a binding class."""
    try:
        return klass._xml_elem_props
    except AttributeError as e:
        raise ValueError(f"Class {klass} is not a binding") from e


def make_enum_wrapper(
    enum_cls: type[CaseInsensitiveStrEnum],
) -> type[objectify.ObjectifiedDataElement]:
    """
    Factory for creating lxml.objectify.ObjectifiedDataElement wrappers around
    CaseInsensitiveStrEnum subclasses.
    """

    class EnumWrapper(objectify.ObjectifiedDataElement):
        @property
        def pyval(self) -> CaseInsensitiveStrEnum:
            return enum_cls.from_str(self.text)

    return EnumWrapper


def iter_element_children(
    element: Optional[objectify.ObjectifiedElement], *tags: str
) -> Iterator[objectify.ObjectifiedElement]:
    """
    Iterate over the children of an lxml element, optionally filtered by tag.
    If the element is None, an empty iterator is returned.
    """
    if element is None:
        return iter(())

    return element.iterchildren(*tags)
