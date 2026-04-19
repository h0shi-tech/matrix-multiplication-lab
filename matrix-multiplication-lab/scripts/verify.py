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


def main():
    file_a = "data/matrix_a.txt"
    file_b = "data/matrix_b.txt"
    file_c = "data/result_cpp.txt"

    if len(sys.argv) >= 4:
        file_a = sys.argv[1]
        file_b = sys.argv[2]
        file_c = sys.argv[3]

    A = read_matrix(file_a)
    B = read_matrix(file_b)
    C_cpp = read_matrix(file_c)

    C_py = A @ B

    if np.array_equal(C_cpp, C_py):
        print("Verification successful")
        sys.exit(0)
    else:
        diff = C_cpp - C_py
        max_abs_diff = int(np.max(np.abs(diff)))
        print("Results do not match")
        print(f"Max absolute difference: {max_abs_diff}")
        sys.exit(1)


if __name__ == "__main__":
    main()
