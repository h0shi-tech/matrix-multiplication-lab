from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CSV_PATH = RESULTS_DIR / "experiments.csv"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        raise RuntimeError("experiments.csv is empty")

    plt.figure(figsize=(8, 5))
    for p in sorted(df["processes"].unique()):
        subset = df[df["processes"] == p].sort_values("n")
        plt.plot(subset["n"], subset["time_s"], marker="o", label=f"{p} proc")
    plt.xlabel("Matrix size n")
    plt.ylabel("Execution time, s")
    plt.title("MPI: execution time vs matrix size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "time_vs_size.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for n in sorted(df["n"].unique()):
        subset = df[df["n"] == n].sort_values("processes")
        plt.plot(subset["processes"], subset["speedup"], marker="o", label=f"n={n}")
    plt.xlabel("MPI processes")
    plt.ylabel("Speedup")
    plt.title("MPI: speedup vs number of processes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "speedup_vs_processes.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for n in sorted(df["n"].unique()):
        subset = df[df["n"] == n].sort_values("processes")
        plt.plot(subset["processes"], subset["efficiency"], marker="o", label=f"n={n}")
    plt.xlabel("MPI processes")
    plt.ylabel("Efficiency")
    plt.title("MPI: efficiency vs number of processes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "efficiency_vs_processes.png", dpi=150)
    plt.close()

    print("Plots saved to results/")


if __name__ == "__main__":
    main()
