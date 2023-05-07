import argparse
import timeit
from pathlib import Path

from src.svd.parsing import parse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("svd_file", type=Path)
    args = p.parse_args()

    count = 10

    # time_taken = timeit.timeit(lambda: parse(args.svd_file), number=count)
    #print(f"{time_taken=}, per: {time_taken / count}")
    device = parse(args.svd_file)
    mem = device.peripherals["FICR_S"].memory_map
    print(mem)


if __name__ == "__main__":
    main()
