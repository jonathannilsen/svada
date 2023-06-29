#
# Copyright (c) 2022 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

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
from .path import SPath
from .device import (
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
    __version__ = importlib.metadata.version("svada")
except importlib.metadata.PackageNotFoundError:
    # Package is not installed
    import setuptools_scm
    __version__ = setuptools_scm.get_version(root="../..", relative_to=__file__)
