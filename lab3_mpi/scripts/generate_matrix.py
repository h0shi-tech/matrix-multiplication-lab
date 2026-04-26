import argparse
from pathlib import Path

import numpy as np


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    n = matrix.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(" ".join(map(str, row.tolist())) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate two square matrices in text format")
    parser.add_argument("n", type=int, help="Matrix size")
    parser.add_argument("--output-a", default="data/matrix_a.txt")
    parser.add_argument("--output-b", default="data/matrix_b.txt")
    parser.add_argument("--low", type=int, default=0)
    parser.add_argument("--high", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed + args.n)
    a = rng.integers(args.low, args.high + 1, size=(args.n, args.n), dtype=np.int64)
    b = rng.integers(args.low, args.high + 1, size=(args.n, args.n), dtype=np.int64)

    write_matrix(Path(args.output_a), a)
    write_matrix(Path(args.output_b), b)
    print(f"Generated matrices {args.n}x{args.n}")


if __name__ == "__main__":
    main()
