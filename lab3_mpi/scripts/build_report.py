from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CSV_PATH = RESULTS_DIR / "experiments.csv"
OUT_PATH = ROOT / "report_lab3_filled.md"


def df_to_md(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(map(str, row.tolist())) + " |")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        raise RuntimeError("experiments.csv is empty")

    df = df.sort_values(["n", "processes"]).reset_index(drop=True)
    table_md = df_to_md(df)

    best = df.loc[df["speedup"].astype(float).idxmax()]
    largest = df[df["n"] == df["n"].max()].sort_values("processes")
    largest_best = largest.loc[largest["time_s"].astype(float).idxmin()]

    report = f"""# Лабораторная работа №3

## Тема
Параллельное умножение квадратных матриц с использованием MPI.

## Цель работы
Модифицировать программу из л/р №1 для параллельной работы по технологии MPI и исследовать влияние размера матрицы и числа MPI-процессов на время выполнения.

## Что сделано
- реализовано распределение строк матрицы `A` между MPI-процессами;
- матрица `B` передаётся всем процессам через `MPI_Bcast`;
- частичные результаты собираются на корневом процессе через `MPI_Gatherv`;
- результат записывается в файл `data/result_mpi.txt`;
- добавлена автоматическая верификация через Python/NumPy;
- подготовлены скрипты для серии экспериментов и построения графиков.

## Структура папки
```text
lab3_mpi/
├── Makefile
├── requirements.txt
├── report_lab3.md
├── report_lab3_filled.md
├── src/
│   └── main.cpp
├── scripts/
│   ├── build_report.py
│   ├── generate_matrix.py
│   ├── plot_results.py
│   ├── run_experiments.py
│   └── verify.py
├── data/
│   ├── matrix_a.txt
│   └── matrix_b.txt
└── results/
    ├── experiments.csv
    ├── efficiency_vs_processes.png
    ├── speedup_vs_processes.png
    └── time_vs_size.png
```

## Краткое описание алгоритма
Каждый MPI-процесс получает часть строк матрицы `A`. Полная матрица `B` рассылается всем процессам. Затем каждый процесс вычисляет свой блок строк результирующей матрицы `C`. После этого все блоки собираются на процессе с рангом 0.

Объём задачи для матриц размера `n × n`:
- умножений: `n^3`;
- сложений: `n^3 - n^2`;
- всего операций: `2n^3 - n^2`.

## Команды запуска
```bash
make
python3 scripts/generate_matrix.py 400
mpirun -np 4 ./matrix_mul_mpi data/matrix_a.txt data/matrix_b.txt data/result_mpi.txt
python3 scripts/verify.py data/matrix_a.txt data/matrix_b.txt data/result_mpi.txt
python3 scripts/run_experiments.py
python3 scripts/plot_results.py
python3 scripts/build_report.py
```

## Результаты экспериментов
{table_md}

## Графики
![Время выполнения](results/time_vs_size.png)

![Ускорение](results/speedup_vs_processes.png)

![Эффективность](results/efficiency_vs_processes.png)

## Анализ результатов
- максимальное зафиксированное ускорение: **{float(best['speedup']):.3f}**;
- этот результат получен для `n = {int(best['n'])}` при `p = {int(best['processes'])}`;
- для наибольшей матрицы `n = {int(largest_best['n'])}` минимальное время получено при `p = {int(largest_best['processes'])}` и составило **{float(largest_best['time_s']):.6f} с**.

По мере увеличения размера матриц MPI-параллелизация становится выгоднее, так как доля накладных расходов на обмен данными уменьшается относительно объёма вычислений. Для небольших матриц выигрыш может быть незначительным.

## Вывод
В ходе работы программа из л/р №1 была модифицирована для параллельного выполнения по технологии MPI. Реализовано распределение данных между процессами, сбор результатов и автоматическая проверка корректности через NumPy. Эксперименты показали, что использование нескольких MPI-процессов позволяет уменьшить время выполнения и получить ускорение, особенно на матрицах большого размера.
"""

    OUT_PATH.write_text(report, encoding="utf-8")
    print(f"Report saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
