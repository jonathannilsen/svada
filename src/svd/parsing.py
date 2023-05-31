#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, get_type_hints

import lxml.etree as ET
from lxml import objectify

from . import bindings
from . import util
from .peripheral import Device, Peripheral, Register


def parse_peripheral(
    svd_path: Union[str, Path], peripheral_name: str
) -> Dict[int, Register]:
    """
    Parse an SVD for a specific peripheral and return it as a map of a memory offset and the
    register at that offset.

    :param svd_path: SVD file to use
    :param peripheral_name: Peripheral to parse

    :raise FileNotFoundError: If the SVD file does not exist.

    :return: Mapping of offset:registers for the peripheral.
    """

    # TODO: FIXME
    svd_file = Path(svd_path)

    if not svd_file.is_file():
        raise FileNotFoundError(f"No such file: {svd_file.absolute()}")

    device = ET.parse(svd_file).getroot()
    peripheral = Peripheral(device, peripheral_name)

    return peripheral


def parse(svd_path: Union[str, Path]) -> Device:
    """
    Parse an SVD TODO
    """

    svd_file = Path(svd_path)

    if not svd_file.is_file():
        raise FileNotFoundError(f"No such file: {svd_file.absolute()}")

    xml_parser = objectify.makeparser(remove_comments=True)
    class_lookup = _TwoLevelTagLookup(bindings.ELEMENT_CLASSES)
    xml_parser.set_element_class_lookup(class_lookup)

    # TODO: some handling of errors here
    with open(svd_file, "r") as f:
        xml_device = objectify.parse(f, parser=xml_parser)

    device = Device(xml_device.getroot())

    return device


class _TwoLevelTagLookup(ET.ElementNamespaceClassLookup):
    """
    XML element class lookup that supports multiple levels of tag names.
    This two-level scheme is used to slightly optimize the time spent by the parser looking up
    looking up the class for an element (which is a sizeable portion of the time spent parsing).

    Element classes that can be uniquely identified by tag only are stored in the first level.
    This level uses the lxml ElementNamespaceClassLookup which is faster than the second level.
    The remaining element classes are assumed to be uniquely identified by a combination of
    the parent tag and the tag itself, and are stored in the second level.
    The second level uses the lxml PythonElementClassLookup which is slower.
    """

    def __init__(self, element_classes: List[objectify.ObjectifiedElement]):
        super().__init__()

        tag_classes: Dict[str, Set[type]] = defaultdict(set)
        two_tag_classes: Dict[Tuple[Optional[str], str], Set[type]] = defaultdict(set)

        for element_class in element_classes:
            tag = element_class.TAG
            tag_classes[tag].add(element_class)
            for prop in util.get_binding_props(element_class).values():
                prop_info = prop.fget
                try:
                    name = prop_info.name
                    klass = prop_info.klass
                except AttributeError:
                    continue
                tag_classes[name].add(klass)
                two_tag_classes[(tag, name)].add(klass)

        one_tag: Set[str] = set()
        namespace = self.get_namespace(None)  # note: None is the empty namespace

        for tag, classes in tag_classes.items():
            if len(classes) == 1:
                # Add the class to the namespace
                # Namespace is intended to be used as a decorator, so the syntax is a little odd
                element_class = classes.pop()
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
        self._lookup_table = lookup_table

    def lookup(self, _document, element: ET.Element):
        """Look up the Element class for the given XML element"""
        if (parent := element.getparent()) is not None:
            parent_tag = parent.tag
        else:
            parent_tag = None
        return self._lookup_table.get((parent_tag, element.tag))
