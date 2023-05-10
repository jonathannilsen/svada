#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

PACKAGE_NAME = "svada"

from .parsing import (
    parse_peripheral,
    parse,
)
from .peripheral import (
    Device,
    Peripheral,
    Register,
    Field,
    _expand_registers,
    _get_register_elements,
)
from .util import (
    strip_prefixes_suffixes,
    to_int,
)

import importlib.metadata

try:
    __version__ = importlib.metadata.version(PACKAGE_NAME)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
    #raise RuntimeError(f"{PACKAGE_NAME} is not installed")
