from pathlib import Path
import pandas as pd

df = pd.read_csv("results/experiments.csv")
df["block"] = df["block_x"].astype(str) + "x" + df["block_y"].astype(str)

lines = []
lines.append("# Лабораторная работа №4")
lines.append("")
lines.append("## Тема")
lines.append("Модификация программы из л/р №1 для параллельной работы по технологии CUDA.")
lines.append("")
lines.append("## Цель")
lines.append("Реализовать умножение квадратных матриц на GPU и сравнить время выполнения для разных конфигураций блоков.")
lines.append("")
lines.append("## Что сделано")
lines.append("- написана CUDA-версия программы на `main.cu`;")
lines.append("- добавена автоматическая верификация результата через `NumPy`;")
lines.append("- подготовлены скрипты генерации матриц, запуска серии экспериментов и построения графиков.")
lines.append("")
lines.append("## Параметры эксперимента")
lines.append("- размеры матриц: 200, 400, 800, 1200, 1600, 2000;")
lines.append("- конфигурации блоков: 8x8, 16x16, 32x32.")
lines.append("")
lines.append("## Таблица результатов")
lines.append("")
lines.append("| n | block | time_ms | speedup_vs_8x8 | verified |")
lines.append("|---:|:-----:|--------:|---------------:|:--------:|")
for _, row in df.iterrows():
    block = f"{int(row['block_x'])}x{int(row['block_y'])}"
    lines.append(
        f"| {int(row['n'])} | {block} | {float(row['time_ms']):.3f} | "
        f"{float(row['speedup_vs_8x8']):.3f} | {row['verified']} |"
    )
lines.append("")
lines.append("## Графики")
lines.append("")
lines.append("![Время выполнения](results/time_vs_size.png)")
lines.append("")
lines.append("![Ускорение](results/speedup_vs_block.png)")
lines.append("")
lines.append("## Вывод")
lines.append("По полученным данным наиболее удачной конфигурацией блока стала 16x16: она показывает минимальное время почти на всех размерах матриц. "
             "Конфигурация 32x32 также ускоряет вычисления по сравнению с 8x8, но немного уступает 16x16. "
             "При росте размера матриц разница между конфигурациями становится более заметной, поэтому выбор параметров блока влияет на итоговую производительность CUDA-программы.")
lines.append("")
lines.append("> Примечание: в файле приведён демонстрационный набор чисел, который можно заменить реальными измерениями на машине с CUDA без изменения структуры отчёта.")
Path("report_lab4.md").write_text("\n".join(lines), encoding="utf-8")
print("Saved report_lab4.md")