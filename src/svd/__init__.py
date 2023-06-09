#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

PACKAGE_NAME = "svada"

from .bindings import (
    Access,
    ReadAction,
    Endian,
    SauAccess,
    AddressBlockUsage,
    Protection,
    EnumUsage,
    WriteAction,
    DataType,
    CpuName,
    Cpu,
    AddressBlock,
    SauRegion,
)
from .parsing import (
    parse,
    SvdParseException,
)
from .peripheral import (
    Device,
    Peripheral,
    RegisterType,
    Register,
    RegisterArray,
    RegisterStruct,
    RegisterStructArray,
    Field,
)

import importlib.metadata

try:
    __version__ = importlib.metadata.version(PACKAGE_NAME)
except importlib.metadata.PackageNotFoundError:
    # Package is not installed
    # TODO: fix version here
    __version__ = "0.0.0"
