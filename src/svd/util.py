#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import enum
import inspect
from abc import ABC
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from lxml import objectify


def strip_suffix(word: str, suffix: str) -> str:
    """
    Remove the given suffix from the word, if present.

    :param word: String to strip prefixes and suffixes from.
    :param suffix: Suffix to strip.

    :return: word without the suffix.
    """

    if word.endswith(suffix):
        word = word[: -len(suffix)]

    return word


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


K = TypeVar("K")
T = TypeVar("T")


class LSMCollection(ABC, Generic[K, T]):
    """
    Generic collection data structure used to implement the collection operations
    we suppport in the SVD device structure.
    """

    def __init__(
        self,
        *,
        key_type: Type[K],
        storage: Collection[T],
        factory: Callable[[K], T],
        **kwargs,
    ):
        self._key_type: Type[K] = key_type
        self._storage: Dict[str, Optional[T]] = storage
        self._factory = factory

        super().__init__(**kwargs)

    def __getitem__(self, key: Union[K, Sequence[Any]]) -> Union[T, Any]:
        """
        Get an item from the collection.

        If the item is requested for the first time, it is first constructed using
        the factory function.

        The key parameter may either be a single key or a sequence of keys.
        In the single key case, e.g.
            my_collection["key"],
        the item with key "key" is looked up in this instance.
        A sequence of keys can be used to get elements deeper in the hierarchy.
        If given e.g.
            my_collection["key1", "key2", "key3"],
        the element at
            my_collection["key1"]["key2"]["key3"]
        is returned.

        :param key: Singular key or sequence of keys identifying the item.
        :return:
        """
        this_key, remaining_keys = self.decode_key(key)

        value = self._storage[this_key]
        if value is None:
            value = self._factory(this_key)
            self._storage[this_key] = value

        if remaining_keys:
            return value[remaining_keys]
        else:
            return value

    def __contains__(self, key: K) -> bool:
        return key in self._storage

    def __len__(self) -> int:
        return len(self._storage)

    def recursive_iter():
        ...

    def decode_key(self, key: Union[K, Sequence[Any]]) -> Tuple[K, Sequence[Any]]:
        """Decode a key into a tuple of (initial key, remaining keys)."""
        if isinstance(key, self._key_type):
            return key, ()
        elif isinstance(key, Sequence):
            return key[0], key[1:]
        else:
            raise ValueError(f"Invalid key: {key}")


class LazyStaticMapping(LSMCollection[str, T], Mapping[str, T]):
    """
    A mapping that lazily constructs its values.
    The set of keys is fixed at construction time.
    """

    def __init__(self, keys: Iterable[str], factory: Callable[[str], T], **kwargs):
        """
        :param keys: Keys contained in the mapping.
        :param factory: Factory function which is called to initialize
                        new elements.
        """
        super().__init__(
            key_type=str, storage={key: None for key in keys}, factory=factory, **kwargs
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self._storage)


class LazyStaticList(LSMCollection[int, T], Sequence[T]):
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
        super().__init__(
            key_type=int,
            storage=[None for _ in range(length)],
            factory=factory,
            **kwargs,
        )


def iter_merged(a: Iterable[T], b: Iterable[T], key: Callable[[T], Any]) -> Iterator[T]:
    """
    Iterator that merges two sorted iterables.
    It is implicitly assumed that both iterables are sorted - passing unsorted iterables will not
    cause an error.
    """

    # Sentinel value for the end of an iterator
    __end = object()

    iter_a = iter(a)
    iter_b = iter(b)

    item_a = next(iter_a, __end)
    item_b = next(iter_b, __end)

    while True:
        if item_a is not __end and item_b is not __end:
            key_a = key(item_a)
            key_b = key(item_b)
            if key_a <= key_b:
                yield item_a
                item_a = next(iter_a, __end)
            else:
                yield item_b
                item_b = next(iter_b, __end)
        elif item_a is not __end:
            yield item_a
            item_a = next(iter_a, __end)
        elif item_b is not __end:
            yield item_b
            item_b = next(iter_b, __end)
        else:
            break
