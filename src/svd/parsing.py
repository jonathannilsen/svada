#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import lxml.etree as ET
from lxml import objectify

from . import bindings
from .device import Device, Options


def parse(svd_path: Union[str, Path], options: Optional[Options] = None) -> Device:
    """
    Parse a device described by a SVD file.

    :param svd_path: Path to the SVD file.

    :raises FileNotFoundError: If the SVD file does not exist.
    :raises SvdParseException: If an error occurred while parsing the SVD file.

    :return: Parsed `Device` representation of the SVD file.
    """

    svd_file = Path(svd_path)

    if not svd_file.is_file():
        raise FileNotFoundError(f"No such file: {svd_file.absolute()}")

    try:
        # Note: remove comments as otherwise these are present as nodes in the returned XML tree
        xml_parser = objectify.makeparser(remove_comments=True)
        class_lookup = _TwoLevelTagLookup(bindings.BINDINGS)
        xml_parser.set_element_class_lookup(class_lookup)

        with open(svd_file, "r") as f:
            xml_device = objectify.parse(f, parser=xml_parser)

        device = Device(xml_device.getroot(), options=Options)

    except Exception as e:
        raise SvdParseException(f"Error parsing SVD file {svd_file}") from e

    return device


class SvdParseException(RuntimeError):
    """Exception raised when an error occurs during SVD parsing."""

    ...


class _TwoLevelTagLookup(ET.ElementNamespaceClassLookup):
    """
    XML element class lookup that uses two levels of tag names to map an XML element to a Python
    class. This two-level scheme is used to slightly optimize the time spent by the parser looking
    up looking up the class for an element (which is a sizeable portion of the time spent parsing).

    Element classes that can be uniquely identified by tag only are stored in the first level.
    This level uses the lxml ElementNamespaceClassLookup which is faster than the second level.
    The remaining element classes are assumed to be uniquely identified by a combination of
    the parent tag and the tag itself, and are stored in the second level.
    The second level uses the lxml PythonElementClassLookup which is slower.
    """

    def __init__(self, element_classes: List[Type[objectify.ObjectifiedElement]]):
        """
        :param element_classes: lxml element classes to add to the lookup table.
        """
        super().__init__()

        tag_classes: Dict[str, Set[type]] = defaultdict(set)
        two_tag_classes: Dict[Tuple[Optional[str], str], Set[type]] = defaultdict(set)

        for element_class in element_classes:
            tag = element_class.TAG
            tag_classes[tag].add(element_class)
            for prop in bindings.get_binding_elem_props(element_class).values():
                tag_classes[prop.name].add(prop.klass)
                two_tag_classes[(tag, prop.name)].add(prop.klass)

        one_tag: Set[str] = set()
        namespace = self.get_namespace(None)  # note: None is the empty namespace

        for tag, classes in tag_classes.items():
            if len(classes) == 1:
                # Add the class to the namespace
                element_class = classes.pop()
                # note: namespace is a decorator, so the syntax here is a little odd
                namespace(tag)(element_class)
                one_tag.add(tag)

        two_tag_lookup: Dict[Tuple[Optional[str], str], type] = {}

        for (tag, field_name), classes in two_tag_classes.items():
            if field_name in one_tag:
                continue

            assert (
                len(classes) == 1
            ), f"Multiple classes for ({tag}, {field_name}): {classes}"

            two_tag_lookup[(tag, field_name)] = classes.pop()

        fallback_lookup = _SecondLevelTagLookup(two_tag_lookup)
        self.set_fallback(fallback_lookup)


class _SecondLevelTagLookup(ET.PythonElementClassLookup):
    """XML element class lookup table that uses two levels of tags to look up the class"""

    def __init__(
        self,
        lookup_table: Dict[Tuple[Optional[str], str], objectify.ObjectifiedElement],
    ):
        """
        :param lookup_table: Lookup table mapping a tuple of (parent tag, tag) to an element class.
        """
        self._lookup_table = lookup_table

    def lookup(self, _document, element: ET._Element):
        """Look up the Element class for the given XML element"""
        if (parent := element.getparent()) is not None:
            parent_tag = parent.tag
        else:
            parent_tag = None
        return self._lookup_table.get((parent_tag, element.tag))
