import csv
import os
import re
import subprocess
from pathlib import Path

import numpy as np


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    n = matrix.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(" ".join(map(str, row.tolist())) + "\n")


def read_matrix(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        n = int(f.readline().strip())
        rows = [list(map(int, f.readline().split())) for _ in range(n)]
    return np.array(rows, dtype=np.int64)


def generate_matrices(n: int, low: int = 0, high: int = 9, seed: int = 42) -> None:
    rng = np.random.default_rng(seed + n)
    A = rng.integers(low, high + 1, size=(n, n), dtype=np.int64)
    B = rng.integers(low, high + 1, size=(n, n), dtype=np.int64)

    write_matrix(DATA_DIR / "matrix_a.txt", A)
    write_matrix(DATA_DIR / "matrix_b.txt", B)


def verify_result() -> bool:
    A = read_matrix(DATA_DIR / "matrix_a.txt")
    B = read_matrix(DATA_DIR / "matrix_b.txt")
    C_cpp = read_matrix(DATA_DIR / "result_cpp.txt")
    C_py = A @ B
    return np.array_equal(C_cpp, C_py)


def extract_value(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"Cannot parse value from output:\n{text}")
    return match.group(1)


def run_program(exe: str, threads: int):
    result = subprocess.run(
        [exe, "data/matrix_a.txt", "data/matrix_b.txt", "data/result_cpp.txt", str(threads)],
        capture_output=True,
        text=True,
        check=True
    )

    stdout = result.stdout

    matrix_size = extract_value(stdout, r"Matrix size:\s+(\d+x\d+)")
    actual_threads = int(extract_value(stdout, r"Threads used:\s+(\d+)"))
    processors = int(extract_value(stdout, r"Available processors:\s+(\d+)"))
    exec_time = float(extract_value(stdout, r"Execution time \(s\):\s+([0-9.]+)"))
    workload = int(extract_value(stdout, r"Workload \(operations\):\s+(\d+)"))

    return {
        "matrix_size": matrix_size,
        "threads_used": actual_threads,
        "available_processors": processors,
        "execution_time_s": exec_time,
        "workload_ops": workload,
        "stdout": stdout,
    }


def main():
    exe = "./matrix_mul_omp"
    sizes = [200, 400, 800, 1200, 1600, 2000]

    max_threads = os.cpu_count() or 1
    requested_threads = [1, 2, 4, 8]
    requested_threads = [t for t in requested_threads if t <= max_threads]

    if not requested_threads:
        requested_threads = [1]

    rows = []

    for n in sizes:
        print(f"\n=== Matrix size: {n}x{n} ===")
        generate_matrices(n)

        baseline_time = None

        for threads in requested_threads:
            print(f"Running with {threads} thread(s)...")

            run_info = run_program(exe, threads)

            if not verify_result():
                raise RuntimeError(f"Verification failed for n={n}, threads={threads}")

            time_s = run_info["execution_time_s"]

            if baseline_time is None:
                baseline_time = time_s

            speedup = baseline_time / time_s
            efficiency = speedup / run_info["threads_used"]

            row = {
                "n": n,
                "threads": run_info["threads_used"],
                "available_processors": run_info["available_processors"],
                "time_s": f"{time_s:.6f}",
                "workload_ops": run_info["workload_ops"],
                "speedup": f"{speedup:.6f}",
                "efficiency": f"{efficiency:.6f}",
                "verified": "yes",
            }
            rows.append(row)

            print(
                f"OK | time={time_s:.6f}s | "
                f"speedup={speedup:.4f} | efficiency={efficiency:.4f}"
            )

    out_file = RESULTS_DIR / "experiments.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n",
                "threads",
                "available_processors",
                "time_s",
                "workload_ops",
                "speedup",
                "efficiency",
                "verified",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
