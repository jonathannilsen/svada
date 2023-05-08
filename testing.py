import argparse
import timeit
from collections.abc import Iterable
from operator import itemgetter
from pathlib import Path
from time import perf_counter

from src.svd.parsing import parse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("svd_file", type=Path)
    args = p.parse_args()

    count = 10

    # time_taken = timeit.timeit(lambda: parse(args.svd_file), number=count)
    #print(f"{time_taken=}, per: {time_taken / count}")
    # print(mem)

    t_parse_start = perf_counter()
    device = parse(args.svd_file)
    t_parse_end = perf_counter()
    t_parse = t_parse_end - t_parse_start
    print(f"{t_parse=:.2f}s")

    t_mem_map_start = perf_counter()
    mem = peripheral.memory_map
    first = next(iter(mem.values()))
    print(first, isinstance(first, Iterable))
    t_mem_map = perf_counter() - t_mem_map_start
    print(f"{t_mem_map=:.2f}s")
    """
    times = {}

    for i, peripheral in enumerate(device.peripherals.values()):
        print(f"{i + 1} / {len(device.peripherals)}: {peripheral.name}...", end=" ")
        t_start = perf_counter()
        memory_map = peripheral.memory_map
        t_end = perf_counter()
        print(f"{len(memory_map)} registers")
        times[peripheral.name] = (t_end - t_start, len(memory_map))

    print()

    for name, (t, count) in sorted(times.items(), key=itemgetter(1), reverse=True):
        print(f"{t:.3f}s - {count} - {name}")

    t_memory_map = sum(map(itemgetter(0), times.values()))

    print(f"{t_parse=}, {t_memory_map=}, total: {t_parse + t_memory_map:.3f}s")
    """


if __name__ == "__main__":
    main()
