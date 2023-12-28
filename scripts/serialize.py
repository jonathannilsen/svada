import pickle
from pathlib import Path

import svd

from lxml import objectify


def main() -> None:
    device = svd.parse(
        Path(__file__).parents[1] / "nrf52840.svd",
        options=svd.Options(parent_relative_cluster_address=True),
    )

    to_be_serialized = [device, device["FICR"].copy_to(0x1234_5678)]

    to_be_serialized[0]["FICR"]["CODESIZE"] = 0x1234
    to_be_serialized[1]["CODESIZE"] = 0x5678

    print("0)")
    for r in to_be_serialized[0].peripherals["FICR"].register_iter():
        print(r)
    print("\nr)")
    for r in to_be_serialized[1].register_iter():
        print(r)

    print("=== SERIALIZING")

    with Path("test.pickle").open("wb") as f:
        pickle.dump(to_be_serialized, f)

    print("=== DESERIALIZING")

    with svd.UnpickleContext():
        with Path("test.pickle").open("rb") as f:
            deserialized = pickle.load(f)

    print("=== PRINTING")

    print("0)")
    for r in deserialized[0].peripherals["FICR"].register_iter():
        print(r)
    print("\n1)")
    for r in deserialized[1].register_iter():
        print(r)


if __name__ == "__main__":
    main()
