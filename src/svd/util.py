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

# FIXME: rename, consider refactor

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
