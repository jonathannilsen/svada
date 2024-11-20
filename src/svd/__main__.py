#
# Copyright (c) 2024 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import argparse
import dataclasses
import enum
import importlib.util
import json
import sys
from pathlib import Path
from textwrap import dedent

import svd
from svd.util import BuildSelector, DeviceBuilder

_HAS_INTELHEX = importlib.util.find_spec("intelhex") is not None
_HAS_TOMLKIT = importlib.util.find_spec("tomlkit") is not None


class Format(enum.Enum):
    JSON = enum.auto()
    BIN = enum.auto()
    IHEX = enum.auto()
    TOML = enum.auto()


def cli() -> None:
    top = argparse.ArgumentParser(
        description=dedent(
            """\
            Collection of utility scripts for working with System View Description (SVD) files.

            Use the --help option with each command to see more information about
            them and their individual options/arguments.
            """
        ),
        allow_abbrev=False,
        add_help=False,  # To override -h
    )
    _add_help_option(top)

    sub = top.add_subparsers(title="subcommands")

    gen = sub.add_parser(
        # TODO: what??
        # "content-transcode"?
        "mem",
        help="Encode and decode device content to and from various formats.",
        description=dedent(
            """\
            Encode device content from one of the supported formats and output it to another
            supported format.
            """
        ),
        allow_abbrev=False,
        add_help=False,
    )
    gen.set_defaults(_command="")
    _add_help_option(gen)

    gen_in = gen.add_argument_group("input options")
    gen_in_mutex = gen_in.add_mutually_exclusive_group(required=True)
    gen_in_mutex.add_argument("-j", "--in-json", action="store_true", help="TODO")
    if _HAS_INTELHEX:
        gen_in_mutex.add_argument(
            "-h",
            "--in-hex",
            action="store_true",
            help=(
                "HEX file. If given, the script will use it to initialize the values of "
                "the peripheral. Multiple HEX files can be given, and they will be applied in "
                "the order as they are given. Files passed as an input for this option are applied before"
                "any other inputs."
            ),
        )
    if _HAS_TOMLKIT:
        gen_in_mutex.add_argument(
            "-t",
            "--in-toml",
            action="store_true",
            help="Input TOML configuration.",
        )
    gen_in.add_argument(
        "-i",
        "--input-file",
        type=Path,
        help="File to read the input from. If not given, stdin is used.",
    )

    gen_svd = gen.add_argument_group("SVD options")
    gen_svd.add_argument(
        "-s",
        "--svd-file",
        required=True,
        type=Path,
        help=("Path to the device SVD file."),
    )
    gen_svd.add_argument(
        "-n",
        "--no-strict",
        action="store_true",
        help="Don't enforce constraints on register and field values based on the SVD file.",
    )
    gen_svd.add_argument(
        "--svd-parse-options",
        type=json.loads,
        help=(
            "JSON object used to override fields in the Options object to customize svada parsing "
            "behavior. Mainly intended for advanced use cases such as working around "
            "difficult SVD files. "
        ),
    )

    gen_sel = gen.add_argument_group("selection options")
    gen_sel.add_argument(
        "-p",
        "--peripheral",
        dest="peripherals",
        action="append",
        help="Name of a peripheral to output.",
    )
    gen_sel.add_argument(
        "-a",
        "--address-range",
        nargs=2,
        type=_parse_address_range,
    )
    gen_sel.add_argument(
        "-c",
        "--content-status",
        choices=[c.value for c in BuildSelector.ContentStatus.__members__.values()],
    )

    gen_out = gen.add_argument_group("output options")
    gen_out_mutex = gen_out.add_mutually_exclusive_group(required=True)
    gen_out_mutex.add_argument(
        "-J",
        "--out-json",
        action="store_true",
        help="Output a JSON string",
    )
    gen_out_mutex.add_argument(
        "-B",
        "--out-bin",
        action="store_true",
        help="Output binary data",
    )
    if _HAS_INTELHEX:
        gen_out_mutex.add_argument(
            "-H",
            "--out-hex",
            action="store_true",
            help="Output an intel hex format string",
        )
    if _HAS_TOMLKIT:
        gen_out_mutex.add_argument(
            "-T",
            "--out-toml",
            action="store_true",
            help="Output a TOML string",
        )
    gen_out.add_argument(
        "-o",
        "--output-file",
        type=Path,
        help="File to write the output to. If not given, output is written to stdout.",
    )

    args = top.parse_args()

    input_format = None
    input_mode = None
    if args.in_json:
        input_format = Format.JSON
        input_mode = "rb"
    elif _HAS_INTELHEX and getattr(args, "in_hex", False):
        input_format = Format.IHEX
        input_mode = "r"
    elif _HAS_TOMLKIT and getattr(args, "in_toml", False):
        input_format = Format.TOML
        input_mode = "rb"

    assert input_format is not None
    assert input_mode is not None

    if args.input_file:
        # TODO: encoding
        input_file = open(args.input_file, input_mode)
    else:
        input_file = sys.stdin.buffer if "b" in input_mode else sys.stdin

    options = svd.Options(
        parent_relative_cluster_address=True,
    )
    if args.svd_parse_options:
        options = dataclasses.replace(options, **args.svd_parse_options)

    device = svd.parse(args.svd_file, options=options)
    device_builder = DeviceBuilder(device, enforce_svd_constraints=not args.no_strict)

    if input_format == Format.JSON:
        input_dict = json.load(input_file)
        device_builder.apply_dict(input_dict)
    elif input_format == Format.IHEX:
        from intelhex import IntelHex

        ihex = IntelHex(input_file)
        ihex_memory = {a: ihex[a] for a in ihex.addresses()}
        device_builder.apply_memory(ihex_memory)
    elif input_format == Format.TOML:
        import tomlkit

        input_dict = tomlkit.load(input_file)
        device_builder.apply_dict(input_dict)

    selector = BuildSelector(
        peripherals=args.peripherals if args.peripherals else None,
        address_range=args.address_range if args.address_range is not None else None,
        content_status=(
            BuildSelector.ContentStatus[args.content_status]
            if args.content_status
            else BuildSelector.ContentStatus.ANY
        ),
    )

    output_format = None
    output_mode = None
    if args.out_json:
        output_format = Format.JSON
        output_mode = "w"
    if args.out_bin:
        output_format = Format.BIN
        output_mode = "wb"
    elif _HAS_INTELHEX and getattr(args, "out_hex", False):
        output_format = Format.IHEX
        output_mode = "w"
    elif _HAS_TOMLKIT and getattr(args, "out_toml", False):
        output_format = Format.TOML
        output_mode = "w"

    assert output_format is not None
    assert output_mode is not None

    if args.output_file:
        output_file = open(args.output_file, output_mode, encoding="utf-8")
    else:
        output_file = sys.stdout.buffer if "b" in output_mode else sys.stdout

    if output_format == Format.JSON:
        output_dict = device_builder.build_dict(selector)
        json.dump(output_dict, output_file)
    elif output_format == Format.BIN:
        output_bin = device_builder.build_bytes(selector)
        output_file.write(output_bin)
    elif output_format == Format.IHEX:
        from intelhex import IntelHex

        output_ihex = IntelHex(device_builder.build_memory(selector))
        output_ihex.write_hex_file(output_file)
    elif output_format == Format.TOML:
        import tomlkit

        output_dict = device_builder.build_dict(selector)
        tomlkit.dump(output_dict, output_file)


def _add_help_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--help", action="help", help="Print help message")


def _parse_address_range(addr_range: str) -> tuple[int, int]:
    start, end = [int(a.strip(), 0) for a in addr_range.split()]
    return start, end


# Entry point when running with python -m svd
if __name__ == "__main__":
    cli()
