#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import dataclasses as dc
import enum
import functools as ft
import re
import typing
import xml.etree.ElementTree as ET
from typing import Callable, Generic, List, NamedTuple, Optional, Tuple, TypeVar, Union

"""
<data_type>     ::= <attr_spec> | <elem_spec>
<attr_spec>     ::= Attr[<optional_type>]
<elem_spec>     ::= Elem[<list_type> | <optional_type>]
<list_type>     ::= List[<scalar_type>]
<optional_type> ::= Optional[<scalar_type>] | <scalar_type>
<scalar_type>   ::= int | bool | str | <enum_type> | <class_type>
<class_type>    ::= class with @svd_object
<enum_type>     ::= class that is a subclass of enum.Enum
"""

# TODO: add supported types
T = TypeVar("T")


class Attr(Generic[T]):
    pass


class Elem(Generic[T]):
    pass


def to_camel_case(value: str) -> str:
    """Convert a string to camel case"""
    parts = value.split("_")
    return parts[0] + "".join(x.title() for x in parts[1:])


def extract_children(elem: ET.Element, name: str) -> Optional[List[ET.Element]]:
    matches = elem.findall(name)
    return matches if matches else None


def extract_child(elem: ET.Element, name: str) -> Optional[ET.Element]:
    return elem.find(name)


def extract_element_text(elem: ET.Element) -> Optional[str]:
    return elem.text


def extract_attribute_text(elem: ET.Element, name: str) -> Optional[str]:
    return elem.attrib.get(name)


def to_int(value: str) -> int:
    """Convert an SVD integer string to an int"""
    if value.startswith("0x"):
        return int(value, base=16)
    if value.startswith("#"):
        return int(value[1:], base=2)
    return int(value)


def to_bool(value: str) -> bool:
    """Convert an SVD boolean string to a bool"""
    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False
    raise ValueError(f"Invalid boolean value: {value}")


R = TypeVar("R")

ExtractFunction = Callable[[ET.Element], R]


def compose_extractor(*functions, exit_on_none: bool = False) -> Callable:
    def f(*args, **kwargs):
        result = functions[0](*args, **kwargs)
        if exit_on_none and result is None:
            return result
        for func in functions[1:]:
            result = func(result)
            if exit_on_none and result is None:
                return result
        return result
    return f


def make_element_converter(result_type: type):
    if result_type is int:
        return to_int
    if result_type is bool:
        return to_bool
    if result_type is str:
        return str
    if result_type is enum.Enum:
        return result_type
    raise NotImplementedError(
        f"Conversion not implemented for type {result_type}")


def make_list_extractor(svd_name: str, converter: Callable):
    def f(elem: ET.Element):
        children = extract_children(elem, svd_name)
        return [
            converter(c) for c in children
        ]
    return f


def unwrap_optional(type_spec: type) -> Tuple[type, bool]:
    if typing.get_origin(type_spec) is Union:
        type_args = typing.get_args(type_spec)
        if len(type_args) != 2 or type_args[1] is not type(None):
            raise TypeError(f"Invalid type hint: {type_spec}")
        return type_args[0], True
    return type_spec, False


def from_xml(cls, elem: ET.Element):
    svd_fields = {
        name: extract(elem) for name, extract in cls._svd_extractors.items()
    }

    return cls(**svd_fields)


@ft.singledispatch
def _make_xml_extractor(field_type: type) -> ExtractFunction:
    raise NotImplementedError(f"Cannot extract type {field_type}")


@_make_xml_extractor.register
def _(field_type: Attr) -> ExtractFunction:
    result_type, is_optional = unwrap_optional(result_type)
    return compose_extractor(
        ft.partial(extract_attribute_text, name=svd_name),
        make_element_converter(result_type),
        exit_on_none=is_optional)


@_make_xml_extractor.register
def _(field_type: Elem) -> ExtractFunction:
    if typing.get_origin(result_type) is list:
        result_type = typing.get_args(result_type)[0]
        if typing.get_origin(result_type) is list:
            raise TypeError("nested not supported")

        if hasattr(result_type, "_svd_extractors"):
            converter = result_type.from_xml
        else:
            converter = compose_extractor(extract_element_text,
                                          make_element_converter(result_type))

        return make_list_extractor(svd_name, converter)

    result_type, is_optional = unwrap_optional(result_type)

    if hasattr(result_type, "_svd_extractors"):
        return compose_extractor(
            ft.partial(extract_child, name=svd_name),
            result_type.from_xml,
            exit_on_none=is_optional
        )

    return compose_extractor(
        ft.partial(extract_child, name=svd_name),
        extract_element_text,
        make_element_converter(result_type), exit_on_none=is_optional)


class TypeInfo(NamedTuple):
    origin: type
    args: Tuple[type, ...]

    @classmethod
    def from_type(cls, field_type: type) -> TypeInfo:
        origin = typing.get_origin(field_type)
        args = typing.get_args(field_type)
        return TypeInfo(origin, args)


def make_extractor_elem(type_info: TypeInfo, field_name: str) -> ExtractFunction:
    pass


def make_extractor_attr(type_info: TypeInfo, field_name: str) -> ExtractFunction:
    pass


def make_extractor_field(type_info: TypeInfo, field_name: str) -> Optional[ExtractFunction]:
    if type_info.origin is not Elem and type_info.origin is not Attr:
        return None

    if len(type_info.args) != 1:
        raise TypeError("Invalid type hint") # FIXME: better error message

    if type_info.origin is Elem:
        pass

    if type_info.origin is Attr:
        pass

    raise TypeError("Invalid type hint") # FIXME: better error message


def svd_object(cls):
    if not dc.is_dataclass(cls):
        raise TypeError("@svd_object can only be used on dataclasses")

    type_hints = typing.get_type_hints(cls)
    extractors = {}

    for field in dc.fields(cls):
        field_type = type_hints[field.name]

        type_info = TypeInfo.from_type(field_type)
        svd_name = to_camel_case(field.name)
        extractor = make_extractor_field(type_info, svd_name)

        if extractor is not None:
            extractors[field.name] = extractor

        elif field.default == dc.MISSING and field.default_factory == dc.MISSING:
                raise ValueError("Non-svd fields must have a default value")

    cls._svd_extractors = extractors
    cls.from_xml = classmethod(from_xml)

    return cls


@svd_object
@dc.dataclass(frozen=True)
class TestSubObj:
    sub_field: Elem[str]


@svd_object
@dc.dataclass(frozen=True)
class TestClass:
    some_attrib: Attr[str]
    some_field: Elem[str]
    some_list: Elem[List[str]]
    some_sub_obj: Elem[List[TestSubObj]]
    some_empty: Elem[Optional[str]]
    # some_sub_obj: Elem[TestSubObj]
    # non_deser_field: int = 0


TEST_STR = \
    """<?xml version="1.0" encoding="utf-8"?>
<testObject someAttrib="theattrib">
    <someField>thevalue</someField>
    <someList>v1</someList>
    <someSubObj>
        <subField>hellosub</subField>
    </someSubObj>
</testObject>
"""


def main():
    test_elem = ET.fromstring(TEST_STR)
    test_obj = TestClass.from_xml(test_elem)
    print(test_obj)


if __name__ == "__main__":
    main()
