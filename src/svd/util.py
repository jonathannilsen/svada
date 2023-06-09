#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import enum
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

from lxml import objectify


def strip_prefixes_suffixes(word: str, prefixes: List[str], suffixes: List[str]) -> str:
    """
    Emulates the functionality provided by chaining `removeprefix` and `removesuffix`
    to a str object.

    :param word: String to strip prefixes and suffixes from.
    :param prefixes: List of prefixes to strip.
    :param suffixes: List of suffixes to strip.

    :return: String where prefixes and suffixes have been sequentially removed.
    """

    for prefix in prefixes:
        if word.startswith(prefix):
            word = word[len(prefix) :]

    for suffix in suffixes:
        if word.endswith(suffix):
            word = word[: -len(suffix)]

    return word.strip("_")


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


@dataclass
class ElemProperty:
    """Represents an XML element property."""

    name: str
    klass: type
    default: Union[Any, MISSING]
    default_factory: Union[Callable[[], Any], MISSING]

    def __call__(self, node: objectify.ObjectifiedElement):
        """
        Get the element property value from the given node.
        This method is intended to be used as a property getter.
        """
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


def elem(
    name: str,
    klass: type,
    /,
    *,
    default: Union[Any, MISSING] = MISSING,
    default_factory: Union[Callable[[], Any], MISSING] = MISSING,
) -> Any:
    """
    Create a property that extracts an element from an XML node.
    Only one of default or default_factory can be set.

    :param name: Name of the element.
    :param klass: Class to use for the extracted element.
    :param default: Default value to return if the element is not found.
    :param default_factory: Callable that returns the default value to return if the element is
                            not found.
    :return: Property that extracts the element from an XML node.
    """
    if default != MISSING and default_factory != MISSING:
        raise ValueError("Cannot set both default and default_factory")

    return property(
        fget=ElemProperty(
            name=name,
            klass=klass,
            default=default,
            default_factory=default_factory,
        )
    )


@dataclass
class AttrProperty:
    """Represents an XML attribute property."""

    name: str
    converter: Optional[Callable[[str], Any]]
    default: Union[Any, MISSING]
    default_factory: Union[Callable[[], Any], MISSING]

    def __call__(self, node: objectify.ObjectifiedElement):
        """
        Get the attribute property value from the given node.
        This method is intended to be used as a property getter.
        """
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


def attr(
    name: str,
    /,
    *,
    converter: Optional[Callable[[str], Any]] = None,
    default: Union[Any, MISSING] = MISSING,
    default_factory: Union[Callable[[], Any], MISSING] = MISSING,
) -> Any:
    """
    Create a property that extracts an attribute from an XML node.
    Only one of default or default_factory can be set.

    :param name: Name of the attribute.
    :param converter: Optional callable that converts the attribute value from a string to another
                      type.
    :param default: Default value to return if the element is not found.
    :param default_factory: Callable that returns the default value to return if the element is
                            not found.
    :return: Property that extracts the attribute from an XML node.
    """
    if default != MISSING and default_factory != MISSING:
        raise ValueError("Cannot set both default and default_factory")

    return property(
        fget=AttrProperty(
            name=name,
            converter=converter,
            default=default,
            default_factory=default_factory,
        )
    )


def binding(
    element_classes: List[Type[objectify.ObjectifiedElement]],
) -> Callable[[type], type]:
    """ """

    def decorator(klass: type) -> type:
        xml_props: Dict[str, property] = getattr(klass, "_xml_props", {})

        for name, prop in inspect.getmembers(klass):
            if not isinstance(prop, property):
                continue

            prop_info = prop.fget

            if not (
                isinstance(prop_info, ElemProperty)
                or isinstance(prop_info, AttrProperty)
            ):
                continue

            if isinstance(prop_info, ElemProperty) and prop_info.klass == SELF_CLASS:
                prop_info.klass = klass

            xml_props[name] = prop

        setattr(klass, "_xml_props", xml_props)

        element_classes.append(klass)

        return klass

    return decorator


def get_binding_props(klass: type) -> Dict[str, property]:
    """Get the XML properties of a binding class."""
    try:
        return klass._xml_props
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
        def pyval(self):
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


T = TypeVar("T")


class LazyStaticMapping(Mapping[str, T]):
    """
    A mapping that lazily constructs its values.
    The set of keys is fixed at construction time.
    """

    def __init__(self, keys: Iterable[str], factory: Callable[[str], T]):
        self._factory = factory
        self._storage: Dict[str, Optional[T]] = {key: None for key in keys}

    def __getitem__(self, key: str) -> T:
        if (value := self._storage[key]) is not None:
            return value

        new_value = self._factory(key)
        self._storage[key] = new_value

        return new_value

    def __contains__(self, key: str) -> bool:
        return key in self._storage

    def __iter__(self) -> Iterator[str]:
        return iter(self._storage)

    def __len__(self) -> int:
        return len(self._storage)


class LazyStaticList(Sequence[T]):
    """
    A list that lazily constructs its values.
    The length of the list is fixed at construction time.
    """

    def __init__(self, length: int, factory: Callable[[int], T], **kwargs):
        """
        :param length: Length of the list.
        :param factory: Factory function which is called to initialize
                        new elements.
        """
        self._factory: Callable[[int], T] = factory
        self._storage: List[Optional[T]] = [None for _ in range(length)]

    def __getitem__(self, index: int) -> T:
        if (value := self._storage[index]) is not None:
            return value

        new_value = self._factory(index)
        self._storage[index] = new_value

        return new_value

    def __len__(self) -> int:
        return len(self._storage)


def iter_merged(a: Iterable[T], b: Iterable[T], key: Callable[[T], Any]) -> Iterator[T]:
    """Iterator that merges two sorted iterables."""

    class _End:
        """Sentinel value for the end of an iterator."""

        ...

    iter_a = iter(a)
    iter_b = iter(b)

    item_a = next(iter_a, _End)
    item_b = next(iter_b, _End)

    while True:
        if item_a is not _End and item_b is not _End:
            key_a = key(item_a)
            key_b = key(item_b)
            if key_a <= key_b:
                yield item_a
                item_a = next(iter_a, _End)
            else:
                yield item_b
                item_b = next(iter_b, _End)
        elif item_a is not _End:
            yield item_a
            item_a = next(iter_a, _End)
        elif item_b is not _End:
            yield item_b
            item_b = next(iter_b, _End)
        else:
            break
