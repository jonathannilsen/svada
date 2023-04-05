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
from typing import Callable, Generic, List, Literal, NamedTuple, Optional, Tuple, TypeVar, Union

"""
<data_type>     ::= <attr_spec> | <elem_spec>
<attr_spec>     ::= Attr[<optional_type>]                         # 1/N child elements
<elem_spec>     ::= Elem[<list_type> | <optional_type>]           # 1 tag
<list_type>     ::= List[<scalar_type>]                           # N equal child elements
<optional_type> ::= Optional[<scalar_type>] | <scalar_type>       # 0 or 1 child element
<scalar_type>   ::= int | bool | str | <enum_type> | <class_type> #
<class_type>    ::= class with @svd_object
<enum_type>     ::= class that is a subclass of enum.Enum
"""

"""
(attr, list), (int | bool | str) ->

Maybe just parse until you reach the scalar_type?
Find how to extract the data:
- 1 attribute of current node
- 0 child nodes
- 1 child node
- N child nodes

Then convert the data to the correct type
"""


# TODO: add supported types
T = TypeVar("T")
U = TypeVar("U")


class Attr(Tuple[T, U]):
    pass


class Elem(Tuple[T, U]):
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


def extract_attribute(elem: ET.Element, name: str) -> Optional[str]:
    return AttrNode(text=elem.attrib.get(name))


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

ExtractFunction = Callable[[ET.Element], Optional[R]]


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


def make_element_converter(result_type: type, simple_only: bool = False):
    if result_type is int:
        return lambda e: to_int(e.text)
    if result_type is bool:
        return lambda e: to_bool(e.text)
    if result_type is str:
        return lambda e: e.text
    if hasattr(result_type, "from_str"):
        return lambda e: result_type.from_str(e.text)
    if not simple_only and hasattr(result_type, "_svd_extractors"):
        return result_type.from_xml
    raise NotImplementedError(f"Conversion not implemented for type {result_type}")


def from_xml(cls, elem: ET.Element):
    svd_fields = {}
    for name, extract in cls._svd_extractors.items():
        if (value := extract(elem)) is not None:
            svd_fields[name] = value
    return cls(**svd_fields)


class AttrNode(NamedTuple):
    text: Optional[str]


class TypeNode(NamedTuple):
    value: type
    children: List[TypeNode]


def normalize_type_tree(base):
    origin = typing.get_origin(base)
    if origin is None:
        return TypeNode(base, [])
    children = [normalize_type_tree(t) for t in typing.get_args(base)]
    return TypeNode(origin, children)


def extract_list(elem, field_name: str, converter):
    children = extract_children(elem, field_name)
    if not children:
        return None
    return [converter(c) for c in children]


def extract_base(elem, field_name: str, extractor, converter):
    child = extractor(elem, field_name)
    if child is None:
        return None
    return converter(child)


def make_extractor_elem(type_tree: TypeNode, field_name: str) -> ExtractFunction:
    if type_tree.value is list:
        type_tree = type_tree.children[0]
        extractor = ft.partial(extract_list, field_name=field_name)
    else:
        if type_tree.value is Union and type_tree.children[1].value is type(None):
            type_tree = type_tree.children[0]
        extractor = ft.partial(extract_base, field_name=field_name, extractor=extract_child)

    converter = make_element_converter(type_tree.value)
    extractor.keywords["converter"] = converter
    return extractor


def make_extractor_attr(type_tree: TypeNode, field_name: str) -> ExtractFunction:
    if type_tree.value is Union and type_tree.children[1].value is type(None):
        type_tree = type_tree.children[0]
    extractor = ft.partial(extract_base, field_name=field_name, extractor=extract_attribute)

    converter = make_element_converter(type_tree.value, simple_only=True)
    extractor.keywords["converter"] = converter
    return extractor

# TODO: use field.default/field.default_factory instead of Optional


def make_extractor_field(
    type_tree: TypeNode, field_name: str
) -> Optional[ExtractFunction]:
    if len(type_tree.children) == 2:
        field_name = type_tree.children[1].value

    if type_tree.value is Elem:
        return make_extractor_elem(type_tree.children[0], field_name)

    if type_tree.value is Attr:
        return make_extractor_attr(type_tree.children[0], field_name)

    return None


def svd_dataclass(*args, **kwargs):
    def f(cls):
        cls = dc.dataclass(cls, *args, **kwargs)
        # if not dc.is_dataclass(cls):
        #     raise TypeError("@svd_object can only be used on dataclasses")

        type_hints = typing.get_type_hints(cls)
        extractors = {}

        for field in dc.fields(cls):
            field_type = type_hints[field.name]

            svd_name = to_camel_case(field.name)
            type_tree = normalize_type_tree(field_type)
            extractor = make_extractor_field(type_tree, svd_name)

            if extractor is not None:
                extractors[field.name] = extractor
            elif field.default == dc.MISSING and field.default_factory == dc.MISSING:
                raise ValueError("Non-svd fields must have a default value")

        cls._svd_extractors = extractors
        cls.from_xml = classmethod(from_xml)

        return cls
    return f
"""
Two cases:
- If the field has a default, it should have similar behavior as Optional(?)
  e.g. Attr[bool] = False
  - Or maybe just make all fields not fail if missing in the svd, then let the constructor handle it
"""



@svd_dataclass
@dc.dataclass(frozen=True)
class TestSubObj:
    sub_field: Elem[str]


@svd_dataclass
@dc.dataclass(frozen=True)
class TestClass:
    some_attrib: Attr[str]
    some_field_diff_name: Elem[str, "someField"]
    some_list: Elem[List[str]]
    # some_sub_obj: Elem[List[TestSubObj]]
    some_empty: Elem[Optional[str]]
    some_sub_obj: Elem[TestSubObj, "."]
    non_deser_field: int = 0


TEST_STR = """<?xml version="1.0" encoding="utf-8"?>
<testObject someAttrib="theattrib">
    <someField>thevalue</someField>
    <someList>v1</someList>
    <someSubObj>
        <subField>hellosub</subField>
    </someSubObj>
    <subField>helloSub2</subField>
</testObject>
"""


def main():
    test_elem = ET.fromstring(TEST_STR)
    test_obj = TestClass.from_xml(test_elem)
    print(test_obj)


if __name__ == "__main__":
    main()
