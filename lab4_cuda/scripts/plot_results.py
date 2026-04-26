from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path("results")
df = pd.read_csv(RESULTS / "experiments.csv")
df["block"] = df["block_x"].astype(str) + "x" + df["block_y"].astype(str)

plt.figure(figsize=(8, 5))
for block, part in df.groupby("block"):
    plt.plot(part["n"], part["time_ms"], marker="o", label=block)
plt.xlabel("Размер матрицы n")
plt.ylabel("Время, мс")
plt.title("Время выполнения CUDA")
plt.grid(True, alpha=0.3)
plt.legend(title="Блок")
plt.tight_layout()
plt.savefig(RESULTS / "time_vs_size.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
for block, part in df.groupby("block"):
    plt.plot(part["n"], part["speedup_vs_8x8"], marker="o", label=block)
plt.xlabel("Размер матрицы n")
plt.ylabel("Ускорение относительно 8x8")
plt.title("Сравнение конфигураций блоков")
plt.grid(True, alpha=0.3)
plt.legend(title="Блок")
plt.tight_layout()
plt.savefig(RESULTS / "speedup_vs_block.png", dpi=150)
plt.close()

print("Plots saved to results/")