#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union, get_type_hints

import lxml.etree as ET
from lxml import objectify

from . import bindings
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
    class_lookup = _MultiLevelTagLookup(bindings.ELEMENT_CLASSES)
    xml_parser.set_element_class_lookup(class_lookup)

    # TODO: some handling of errors here
    with open(svd_file, "r") as f:
        xml_device = objectify.parse(f, parser=xml_parser)

    device = Device(xml_device.getroot())

    return device


class _TwoLevelTagLookup(ET.PythonElementClassLookup):
    def __init__(self, lookup_table: Dict[Tuple[str, str], objectify.ObjectifiedElement]):
        self._lookup_table = lookup_table

    def lookup(self, _document, element: ET.Element):
        """Look up the Element class for the given XML element"""
        if (parent := element.getparent()) is not None:
            parent_tag = parent.tag
        else:
            parent_tag = None
        return self._lookup_table.get((parent_tag, element.tag))


class _MultiLevelTagLookup(ET.ElementNamespaceClassLookup):
    def __init__(self, element_classes: List[objectify.ObjectifiedElement]):
        super().__init__()

        tag_classes: Dict[str, Set[type]] = defaultdict(set)
        two_tag_classes: Dict[Tuple[str, str], Set[type]] = defaultdict(set)

        for element_class in element_classes:
            tag = element_class.TAG
            tag_classes[tag].add(element_class)
            for field_name, field_type in get_type_hints(element_class).items():
                tag_classes[field_name].add(field_type)
                two_tag_classes[(tag, field_name)].add(field_type)

        one_tag: Set[str] = set()
        namespace = self.get_namespace(None) # None is the empty namespace

        for tag, classes in tag_classes.items():
            if len(classes) == 1:
                # Add the class to the namespace
                # Namespace is intended to be used as a decorator, so the syntax is a little odd
                element_class = classes.pop()
                namespace(tag)(element_class)
                one_tag.add(tag)

        two_tag_lookup: Dict[Tuple[str, str], type] = {}

        for (tag, field_name), classes in two_tag_classes.items():
            if field_name in one_tag:
                continue
            assert len(classes) == 1, f"Multiple classes for ({tag}, {field_name}): {classes}"
            two_tag_lookup[(tag, field_name)] = classes.pop()

        fallback_lookup  = _TwoLevelTagLookup(two_tag_lookup)
        self.set_fallback(fallback_lookup)
