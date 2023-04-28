import argparse
from pathlib import Path

from src.svd.parsing import parse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("svd_file", type=Path)
    args = p.parse_args()

    device = parse(args.svd_file)
    print(device)


if __name__ == "__main__":
    main()
