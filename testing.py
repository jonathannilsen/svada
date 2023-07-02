import argparse
import json
from pathlib import Path
from time import perf_counter_ns

from src.svd import parse

def printer(svd_file, print_fields, flat):
    device = parse(svd_file)
    for peripheral in device.values():
        print(peripheral)
        for register in peripheral.register_iter(flat=flat):
            depth = len(register.path)
            print("  " * depth, register, sep="")
            if print_fields and hasattr(register, "fields"):
                for field in register.fields.values():
                    print("  " * (depth + 1), field, sep="")
        print()

def format_ms(time_ns):
    return f"{time_ns / 1_000_000:.2f} ms"

def timer(svd_file, fields, flat):
    time_table = []

    t_before = perf_counter_ns()
    device = parse(svd_file)
    t_parse = perf_counter_ns()

    for peripheral in device.values():
        t_1 = perf_counter_ns()
        peripheral._immutable_register_info
        t_2 = perf_counter_ns()
        for register in peripheral.register_iter(flat=flat):
            if fields and register.fields is not None:
                for field in register.fields.values():
                    ...
        t_3 = perf_counter_ns()
        for register in peripheral.register_iter(flat=flat):
            if fields and register.fields is not None:
                for field in register.fields.values():
                    ...
        t_4 = perf_counter_ns()
        time_table.append({
            "peripheral": str(peripheral),
            "t_immutable": format_ms(t_2 - t_1),
            "t_iterate_1": format_ms(t_3 - t_2),
            "t_iterate_2": format_ms(t_4 - t_3),
        })

    t_after = perf_counter_ns()

    time_parse = t_parse - t_before
    time_iterate = t_after - t_parse

    total_time_table = {
        "overall": {
            "parse": format_ms(time_parse),
            "iterate": format_ms(time_iterate),
        },
        "peripherals": time_table,
    }

    print(json.dumps(total_time_table))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("svd_file", type=Path)
    p.add_argument("--mode", choices=["time", "print"], default="print")
    p.add_argument("--fields",  action="store_true")
    p.add_argument("--flat", action="store_true")
    args = p.parse_args()

    if args.mode == "time":
        timer(args.svd_file, args.fields, args.flat)
    else:
        printer(args.svd_file, args.fields, args.flat)


if __name__ == "__main__":
    main()
