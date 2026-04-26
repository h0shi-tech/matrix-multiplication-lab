import csv
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
EXE = ROOT / "matrix_mul_mpi"
CSV_PATH = RESULTS_DIR / "experiments.csv"


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


def generate_matrices(n: int, seed: int = 42) -> None:
    rng = np.random.default_rng(seed + n)
    a = rng.integers(0, 10, size=(n, n), dtype=np.int64)
    b = rng.integers(0, 10, size=(n, n), dtype=np.int64)
    write_matrix(DATA_DIR / "matrix_a.txt", a)
    write_matrix(DATA_DIR / "matrix_b.txt", b)


def verify_result() -> bool:
    a = read_matrix(DATA_DIR / "matrix_a.txt")
    b = read_matrix(DATA_DIR / "matrix_b.txt")
    c_mpi = read_matrix(DATA_DIR / "result_mpi.txt")
    c_np = a @ b
    return np.array_equal(c_mpi, c_np)


def detect_launcher() -> str:
    for name in ("mpirun", "mpiexec"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError("MPI launcher not found. Install OpenMPI or MPICH first.")


def parse_output(text: str) -> dict:
    patterns = {
        "matrix_size": r"Matrix size:\s+(\d+x\d+)",
        "processes": r"MPI processes:\s+(\d+)",
        "time_s": r"Execution time \(s\):\s+([0-9.]+)",
        "workload": r"Workload \(operations\):\s+(\d+)",
    }
    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise RuntimeError(f"Cannot parse {key} from output:\n{text}")
        parsed[key] = match.group(1)
    return parsed


def run_case(launcher: str, processes: int) -> dict:
    cmd = [launcher]
    if os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() == 0:
        cmd.append("--allow-run-as-root")
    cmd.extend([
        "-np",
        str(processes),
        str(EXE),
        "data/matrix_a.txt",
        "data/matrix_b.txt",
        "data/result_mpi.txt",
    ])

    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    parsed = parse_output(completed.stdout)
    return {
        "matrix_size": parsed["matrix_size"],
        "processes": int(parsed["processes"]),
        "time_s": float(parsed["time_s"]),
        "workload": int(parsed["workload"]),
        "stdout": completed.stdout,
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    launcher = detect_launcher()

    if not EXE.exists():
        raise RuntimeError("Executable matrix_mul_mpi not found. Run 'make' first.")

    sizes = [200, 400, 800, 1200, 1600, 2000]
    requested_processes = [1, 2, 4, 8]
    cpu_total = os.cpu_count() or 1
    requested_processes = [p for p in requested_processes if p <= cpu_total]
    if not requested_processes:
        requested_processes = [1]

    rows = []

    for n in sizes:
        print(f"\n=== Matrix size: {n}x{n} ===")
        generate_matrices(n)
        baseline_time = None

        for processes in requested_processes:
            print(f"Running with {processes} MPI process(es)...")
            info = run_case(launcher, processes)

            if not verify_result():
                raise RuntimeError(f"Verification failed for n={n}, processes={processes}")

            if baseline_time is None:
                baseline_time = info["time_s"]

            speedup = baseline_time / info["time_s"]
            efficiency = speedup / info["processes"]

            row = {
                "n": n,
                "processes": info["processes"],
                "time_s": f"{info['time_s']:.6f}",
                "workload_ops": info["workload"],
                "speedup": f"{speedup:.6f}",
                "efficiency": f"{efficiency:.6f}",
                "verified": "yes",
            }
            rows.append(row)
            print(
                f"OK | time={info['time_s']:.6f}s | "
                f"speedup={speedup:.4f} | efficiency={efficiency:.4f}"
            )

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n",
                "processes",
                "time_s",
                "workload_ops",
                "speedup",
                "efficiency",
                "verified",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {CSV_PATH}")
    print("Now run: python3 scripts/plot_results.py")
    print("Then run: python3 scripts/build_report.py")


if __name__ == "__main__":
    main()
