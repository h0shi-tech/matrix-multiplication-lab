from pathlib import Path
import argparse
import numpy as np


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    n = matrix.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(" ".join(map(str, row.tolist())) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low", type=int, default=0)
    parser.add_argument("--high", type=int, default=9)
    parser.add_argument("--out-dir", type=str, default="data")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed + args.size)
    a = rng.integers(args.low, args.high + 1, size=(args.size, args.size), dtype=np.int64)
    b = rng.integers(args.low, args.high + 1, size=(args.size, args.size), dtype=np.int64)

    write_matrix(out_dir / "matrix_a.txt", a)
    write_matrix(out_dir / "matrix_b.txt", b)
    print(f"Generated matrices: {args.size}x{args.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())