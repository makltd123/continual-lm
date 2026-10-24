"""Automatic report (FR-12): summary table, forgetting plots and a draft verdict for H1-H4.

Reads every finished run (runs/<config>/seed<k>/final.json + journals) and writes
<out>/summary.md, <out>/summary.csv and <out>/figures/*.png.

Significance rule from the spec: a difference counts when the mean +- std intervals of the
two configurations over seeds do not overlap.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml

from src.eval import summarize

def _jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_runs(runs_dir):
    """Finished runs. Summary metrics are recomputed from the raw journal, so a change in their
    definition applies to old runs without retraining."""
    runs = {}
    for final_path in sorted(Path(runs_dir).glob("*/seed*/final.json")):
        d = final_path.parent
        cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
        final = json.loads(final_path.read_text(encoding="utf-8"))
        metrics = _jsonl(d / "metrics.jsonl")
        final["summary"] = summarize(metrics, cfg["stream"]["order"], cfg["stream"]["mode"])
        runs.setdefault(d.parent.name, []).append({
            "seed": int(d.name.removeprefix("seed")), "final": final, "metrics": metrics,
            "curve": _jsonl(d / "curve.jsonl"), "order": cfg["stream"]["order"],
            "retention_min": cfg["eval"].get("retention_min", 0.9),
        })
    return runs


def stats(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, len(values)


def fmt(st, digits=3, scale=1.0):
    if st is None:
        return "—"
    mean, std, n = st
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}" if n > 1 else f"{mean * scale:.{digits}f}"


def significant(a, b):
    """True when mean +- std intervals do not overlap (needs at least 2 seeds on each side)."""
    if a is None or b is None or a[2] < 2 or b[2] < 2:
        return False
    return abs(a[0] - b[0]) > a[1] + b[1]


# --- per-run quantities ------------------------------------------------------

def task_kinds(run):
    return {t: ("qa" if "em" in m else "inst" if "acc" in m else "lm") for t, m in run["final"]["test"].items()}


METRIC = {"qa": "em", "inst": "acc", "lm": "ppl"}
UNIT = {"qa": "EM", "inst": "доля выполненных", "lm": "ppl"}


def final_metric(run, task):
    m = run["final"]["test"][task]
    return m[METRIC[task_kinds(run)[task]]]


def curve_points(run, task):
    return sorted((p["task_step"], p["score"]) for p in run["curve"] if p["task"] == task)


def steps_to_target(run, task, target):
    """First task step at which the curve reaches the target; the task's last step +1 if never."""
    points = curve_points(run, task)
    if not points:
        return None
    for step, score in points:
        if score >= target:
            return step
    return points[-1][0] + 1


def curve_targets(ref_runs, tasks, frac):
    """Per task: start + frac * (end - start) of the reference config's curve, averaged over seeds."""
    targets = {}
    for task in tasks:
        values = []
        for run in ref_runs:
            points = curve_points(run, task)
            if points:
                start, end = points[0][1], points[-1][1]
                values.append(start + frac * (end - start))
        if values:
            targets[task] = float(np.mean(values))
    return targets


# --- report ------------------------------------------------------------------

def collect(runs, ref, frac):
    any_run = next(iter(runs.values()))[0]
    kinds = task_kinds(any_run)
    order = any_run["order"]
    targets = curve_targets(runs.get(ref, []), order, frac)
    rows = []
    for name, seeds in runs.items():
        for run in seeds:
            s, cost = run["final"]["summary"], run["final"]["cost"]
            row = {"config": name, "seed": run["seed"], "avg_forgetting_norm": s["avg_forgetting_norm"],
                   "retention": s.get("retention"), "retention_min": s.get("retention_min"),
                   "steps_train": cost["steps_train"], "steps_sleep": cost["steps_sleep"],
                   "steps_dpo": cost["steps_dpo"], "examples_total": cost["examples_total"],
                   "time_min": cost["time_s"] / 60, "mem_gb": cost["mem_gb"]}
            for task, d in s["tasks"].items():
                row[f"{task}_final_test"] = final_metric(run, task)
                for key in ("base", "forgetting", "forgetting_norm", "delta_vs_base", "bwt", "fwt", "R_jj",
                            "first_portion", "aulc"):
                    row[f"{task}_{key}"] = d.get(key)
                if kinds[task] == "lm" and d.get("first_portion") is not None:  # score = -loss -> perplexity
                    row[f"{task}_first_portion"] = math.exp(-d["first_portion"])
            for task, target in targets.items():
                row[f"{task}_steps_to_target"] = steps_to_target(run, task, target)
            done = [row.get(f"{t}_steps_to_target") for t in targets]
            row["mean_steps_to_target"] = float(np.mean(done)) if done and None not in done else None
            rows.append(row)
    return rows, kinds, order, targets


def by_config(rows, key):
    out = {}
    for row in rows:
        out.setdefault(row["config"], []).append(row.get(key))
    return {c: stats(v) for c, v in out.items()}


def verdict(better, worse_sig, n_ok):
    if not n_ok:
        return ("данных недостаточно (метрика есть меньше чем у 2 seed одной из конфигураций; "
                "например, модель почти ничего не выучила)")
    if better:
        return "подтверждена"
    if worse_sig:
        return "не подтверждена (эффект значимо в обратную сторону)"
    return "данных недостаточно (разница в пределах разброса seed)"


def hypotheses(rows, order):
    cfgs = {r["config"] for r in rows}
    lines = []

    def cmp(key, a, b, lower_is_better=True):
        st = by_config(rows, key)
        sa, sb = st.get(a), st.get(b)
        ok = sa is not None and sb is not None and sa[2] >= 2 and sb[2] >= 2
        sig = ok and significant(sa, sb)
        a_better = ok and ((sa[0] < sb[0]) if lower_is_better else (sa[0] > sb[0]))
        return sa, sb, ok, sig and a_better, sig and not a_better

    if {"E0", "E1"} <= cfgs:
        sa, sb, ok, better, worse = cmp("avg_forgetting_norm", "E1", "E0")
        lines.append(f"- **H1 (replay снижает забывание)**: нормированное забывание E1 = {fmt(sa)}, "
                     f"E0 = {fmt(sb)} → {verdict(better, worse, ok)}.")
    if {"E1", "E2"} <= cfgs:
        sa, sb, ok, fewer, _ = cmp("steps_train", "E2", "E1")
        worse_tasks = []
        for t in order:
            q = by_config(rows, f"{t}_R_jj")
            if significant(q.get("E2"), q.get("E1")) and q["E2"][0] < q["E1"][0]:
                worse_tasks.append(t)
        text = (f"шагов обучения E2 = {fmt(sa, 0)}, E1 = {fmt(sb, 0)}; качество на новой задаче в конце её "
                f"обучения значимо хуже у E2 на задачах: {', '.join(worse_tasks) or 'нет'}")
        if not ok:
            result = verdict(False, False, ok)
        elif worse_tasks:
            result = "не подтверждена (качество на новых задачах значимо хуже)"
        elif fewer:
            result = "подтверждена"
        else:
            result = "данных недостаточно (шагов не значимо меньше)"
        lines.append(f"- **H2 (отбор по «удивлению»: то же качество за меньшее число шагов)**: {text} → {result}.")
    if {"E1", "E3"} <= cfgs:
        sa, sb, ok, better, worse = cmp("mean_steps_to_target", "E3", "E1")
        lines.append(f"- **H3 (curriculum ускоряет освоение)**: средние шаги до целевого качества E3 = {fmt(sa, 1)}, "
                     f"E1 = {fmt(sb, 1)} → {verdict(better, worse, ok)}.")
    if {"E1", "E4"} <= cfgs:
        sa, sb, ok, better, worse = cmp("avg_forgetting_norm", "E4", "E1")
        lines.append(f"- **H4 (фаза «сна» удерживает старые задачи)**: нормированное забывание E4 = {fmt(sa)}, "
                     f"E1 = {fmt(sb)} → {verdict(better, worse, ok)}.")
    return lines


def acceptance(rows, threshold):
    """The spec's criterion for the final model: instruction following (T0) stays >= threshold of the base."""
    lines = []
    for cfg in sorted({r["config"] for r in rows if r["config"].startswith("E5")}):
        st = by_config(rows, "retention").get(cfg)
        if st is None:
            continue
        mean, std, n = st
        worst = min(r["retention"] for r in rows if r["config"] == cfg and r.get("retention") is not None)
        ok = worst >= threshold
        lines.append(f"- **{cfg}**: задача 0 в конце потока = {mean:.0%} от контрольной базы"
                     f"{f' (± {std:.0%}, худший seed {worst:.0%})' if n > 1 else ''}; порог {threshold:.0%} → "
                     f"{'выполнен' if ok else 'не выполнен'}.")
    return lines


def write_markdown(path, rows, kinds, order, targets, ref, frac, figures, threshold):
    configs = sorted({r["config"] for r in rows})
    n_seeds = {c: sum(r["config"] == c for r in rows) for c in configs}
    unit = lambda t: UNIT[kinds[t]]
    st = lambda key: by_config(rows, key)
    out = ["# Сводный отчёт по экспериментам", "",
           "Сгенерировано `report.py` по папке с прогонами. Значения: среднее ± стандартное отклонение по seed.",
           "Разница считается значимой, если интервалы «среднее ± std» двух конфигураций не пересекаются (правило из ТЗ).", ""]
    out += ["## Проверка гипотез (черновик, требует интерпретации)", ""] + hypotheses(rows, order) + [""]
    if acceptance(rows, threshold):
        out += ["## Критерий для готовой модели", ""] + acceptance(rows, threshold) + [""]

    extra = [t for t in kinds if t not in order]
    out += ["## Отличие от контрольной базы", "",
            "Контрольная база — исходная Instruct-модель до любого обучения. В таблице — изменение качества к концу "
            "потока (конец минус база; для EM и доли выполненных инструкций — в долях, для остальных — в натах потерь "
            "с обратным знаком): плюс — лучше базы, минус — хуже. Задача 0 (T0) и общий тест (G) не обучаются, "
            "поэтому для них минус — это забывание. «Мин. T0» — худшее значение задачи 0 за весь поток "
            "относительно базы.", ""]
    header = ["Конфигурация"] + [f"Δ {t}" for t in order + extra] + ["T0: конец / база", "Мин. T0 / база"]
    out += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for c in configs:
        cells = [c] + [fmt(st(f"{t}_delta_vs_base")[c]) for t in order + extra]
        cells += [fmt(st("retention")[c], 0, 100) + " %", fmt(st("retention_min")[c], 0, 100) + " %"]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")

    out += ["## Забывание", "",
            "Нормированное забывание — доля прироста качества над исходной моделью, утраченная к концу потока "
            "(0 = ничего не забыто, 1 = забыт весь прирост). Сырое забывание — лучшее качество на задаче минус качество "
            "в конце (по валидационной выборке): для EM — в долях, для LM-задач — в натах потерь.", ""]
    header = ["Конфигурация", "seed", "Норм. забывание (A–C)"] + [f"Забывание {t}" for t in order[:-1]]
    out += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for c in configs:
        cells = [c, str(n_seeds[c]), fmt(st("avg_forgetting_norm")[c])]
        cells += [fmt(st(f"{t}_forgetting")[c]) for t in order[:-1]]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")

    out += ["## Итоговое качество на тесте", ""]
    header = ["Конфигурация"] + [f"{t} ({unit(t)})" for t in kinds]
    out += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for c in configs:
        out.append("| " + " | ".join([c] + [fmt(st(f"{t}_final_test")[c], 2 if kinds[t] == "lm" else 3) for t in kinds]) + " |")
    base = {t: next((r[f"{t}_base"] for r in rows if r.get(f"{t}_base") is not None), None) for t in kinds}
    out.append("| Контрольная база (до обучения, валидация) | " + " | ".join(
        "—" if base[t] is None else f"{math.exp(-base[t]):.2f}" if kinds[t] == "lm" else f"{base[t]:.3f}"
        for t in kinds) + " |")
    out.append("")

    out += ["## Перенос вперёд и скорость освоения", "",
            f"Шаги до целевого качества: целевое значение для задачи — {frac:.0%} прироста, который {ref} набирает "
            "за время обучения на ней (по промежуточным замерам). Если цель не достигнута, ставится число шагов задачи + 1.",
            "FWT — качество на задаче перед началом её обучения минус качество исходной модели (Lopez-Paz & Ranzato, 2017).", ""]
    header = ["Конфигурация", "Ср. шаги до цели"] + [f"Шаги {t}" for t in targets] + [f"FWT {t}" for t in order[1:]]
    out += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for c in configs:
        cells = [c, fmt(st("mean_steps_to_target")[c], 1)]
        cells += [fmt(st(f"{t}_steps_to_target")[c], 1) for t in targets]
        cells += [fmt(st(f"{t}_fwt")[c]) for t in order[1:]]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")

    out += ["## Эффективность по данным и стоимость", "",
            "«После 1-й порции» — качество на задаче сразу после первой порции её данных (фиксированное число "
            "предложенных примеров): EM для B и C, perplexity для A и D.", ""]
    header = (["Конфигурация"] + [f"{t} после 1-й порции" for t in order] +
              ["Шаги обучения", "Шаги сна", "Время, мин", "Пик памяти, ГБ"])
    out += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for c in configs:
        cells = [c] + [fmt(st(f"{t}_first_portion")[c]) for t in order]
        cells += [fmt(st("steps_train")[c], 0), fmt(st("steps_sleep")[c], 0), fmt(st("time_min")[c], 1),
                  fmt(st("mem_gb")[c], 2)]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")

    out += ["## Графики", ""] + [f"![{name}](figures/{name})" for name in figures] + [""]
    Path(path).write_text("\n".join(out), encoding="utf-8")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot(runs, kinds, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    configs = sorted(runs)
    colors = plt.get_cmap("tab10")
    for task, kind in kinds.items():
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for ci, cfg in enumerate(configs):
            series = []
            for run in runs[cfg]:
                rows = [r for r in run["metrics"] if "eval" in r]
                x = [r["portion"] + (0.5 if r["phase"] == "sleep" else 0) for r in rows]
                y = [r["eval"][task][METRIC[kind]] for r in rows]
                series.append((x, y))
            n = min(len(y) for _, y in series)
            x = series[0][0][:n]
            ys = np.array([y[:n] for _, y in series])
            mean, std = ys.mean(0), ys.std(0)
            ax.plot(x, mean, label=cfg, color=colors(ci % 10), lw=1.6)
            ax.fill_between(x, mean - std, mean + std, color=colors(ci % 10), alpha=0.15)
        boundaries = sorted({r["portion"] for r in runs[configs[0]][0]["metrics"]
                             if r.get("phase") == "train" and r.get("task_portion") == 0})
        for b in boundaries[1:]:
            ax.axvline(b - 0.5, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("порция")
        ax.set_ylabel(UNIT[kind] if kind != "lm" else "perplexity")
        if kind == "lm":
            ax.set_yscale("log")
        ax.set_title(f"Задача {task}: качество после каждой порции (среднее ± std по seed)")
        ax.legend(fontsize=8, ncol=4)
        fig.tight_layout()
        name = f"curve_{task}.png"
        fig.savefig(out_dir / name, dpi=120)
        plt.close(fig)
        names.append(name)

    rows = [(c, stats([r["final"]["summary"]["avg_forgetting_norm"] for r in runs[c]])) for c in configs]
    rows = [(c, s) for c, s in rows if s is not None]
    if rows:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.bar([c for c, _ in rows], [s[0] for _, s in rows], yerr=[s[1] for _, s in rows], capsize=4,
               color=[colors(configs.index(c) % 10) for c, _ in rows])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("доля утраченного прироста")
        ax.set_title("Нормированное забывание (задачи A–C)")
        fig.tight_layout()
        fig.savefig(out_dir / "forgetting.png", dpi=120)
        plt.close(fig)
        names.append("forgetting.png")
    return names


def main(argv=None):
    p = argparse.ArgumentParser(description="Build the summary report from finished runs.")
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="reports")
    p.add_argument("--ref", default="E1", help="config whose learning curve defines the target quality")
    p.add_argument("--target-frac", type=float, default=0.9)
    args = p.parse_args(argv)

    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit(f"no finished runs in {args.runs}/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, kinds, order, targets = collect(runs, args.ref, args.target_frac)
    figures = plot(runs, kinds, out / "figures")
    write_csv(out / "summary.csv", rows)
    threshold = next(iter(runs.values()))[0]["retention_min"]
    write_markdown(out / "summary.md", rows, kinds, order, targets, args.ref, args.target_frac, figures, threshold)
    print(f"wrote {out / 'summary.md'}, {out / 'summary.csv'} and {len(figures)} figures "
          f"from {sum(len(v) for v in runs.values())} runs")
