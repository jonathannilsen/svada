#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union, get_type_hints

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
    svd_file = Path(svd_path)

    if not svd_file.is_file():
        raise FileNotFoundError(f"No such file: {svd_file.absolute()}")

    xml_parser = objectify.makeparser(remove_comments=True)
    xml_parser.set_element_class_lookup(_ParentChildTagLookup())

    # TODO: some handling of errors here
    with open(svd_file, "r") as f:
        xml_device = objectify.parse(f, parser=xml_parser)

    device = Device(xml_device)

    return device


class _ParentChildTagLookup(ET.PythonElementClassLookup):
    def __init__(self):
        self._element_classes = {
            (None, "device"): bindings.DeviceElement
        }
        
        for bindings_cls in bindings.ELEMENT_CLASSES:
            element_class_tag = bindings_cls.TAG
            for field_name, field_type in get_type_hints(bindings_cls).items():
                key = (element_class_tag, field_name)
                self._element_classes[key] = field_type

    def lookup(self, _document, element: ET.Element):
        """Look up the Element class for the given XML element"""
        if (parent := element.getparent()) is not None:
            parent_tag = parent.tag
        else:
            parent_tag = None
        return self._element_classes.get((parent_tag, element.tag))
