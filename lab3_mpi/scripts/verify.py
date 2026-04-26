import sys
import numpy as np


def read_matrix(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        n = int(f.readline().strip())
        rows = []
        for _ in range(n):
            row = list(map(int, f.readline().split()))
            if len(row) != n:
                raise ValueError(f"Invalid row length in {path}")
            rows.append(row)

    matrix = np.array(rows, dtype=np.int64)
    if matrix.shape != (n, n):
        raise ValueError(f"Invalid matrix shape in {path}: {matrix.shape}")
    return matrix


def main() -> int:
    file_a = "data/matrix_a.txt"
    file_b = "data/matrix_b.txt"
    file_c = "data/result_mpi.txt"

    if len(sys.argv) >= 4:
        file_a = sys.argv[1]
        file_b = sys.argv[2]
        file_c = sys.argv[3]

    a = read_matrix(file_a)
    b = read_matrix(file_b)
    c_mpi = read_matrix(file_c)
    c_np = a @ b

    if np.array_equal(c_mpi, c_np):
        print("Verification successful")
        return 0

    diff = c_mpi - c_np
    print("Results do not match")
    print(f"Max absolute difference: {int(np.max(np.abs(diff)))}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
