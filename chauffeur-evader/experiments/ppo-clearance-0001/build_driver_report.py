"""Build and verify the task 09 report from saved evaluation CSV files."""

import argparse
import csv
import json
from collections import Counter
from itertools import combinations, pairwise
from math import hypot, isfinite
from os.path import relpath
from pathlib import Path
from shutil import copyfile
from statistics import mean

import matplotlib
import numpy as np
from chauffeur_evader.chauffeur_benchmarks import BENCHMARKS, _save_figure
from chauffeur_evader.chauffeur_classical import DEFAULT_CONFIG, Rollout, TrajectoryPoint
from chauffeur_evader.chauffeur_evaluation import (
    GRID_MAX,
    GRID_MIN,
    GRID_SCALE,
    EpisodeResult,
    compare,
)
from evaluate_drivers import (
    BUDGETS,
    DRIVERS,
    EPISODE_FIELDS,
    EVALUATION,
    FREEZE,
    GRAY,
    POINT_FIELDS,
    REPORT_SEED,
    ROOT,
    SEEDS,
    cases,
    digest,
    policy,
)

LABELS = {
    "classical": "Классический",
    "heuristic": "Эвристика",
    "archive": "Архивный PPO",
    "new": "Новый PPO",
}
OUTCOMES = {"capture": "захват", "time_limit": "тайм-аут", "unsupported": "unsupported"}
BEGIN = "<!-- clearance-0001:begin -->"
END = "<!-- clearance-0001:end -->"


def csv_rows(path, fields):
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"Unexpected CSV columns: {path}")
        yield from reader


def read_results(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest["status"] != "complete"
        or manifest["mode"] not in ("full", "smoke")
        or manifest["evaluation"] != EVALUATION
        or manifest["report_seed"] != REPORT_SEED
        or manifest["protocol_sha256"] != digest(FREEZE / "protocol.json")
    ):
        raise ValueError("Incomplete run or unexpected evaluation protocol")
    models = manifest["models"]
    chosen = policy("09", REPORT_SEED, BUDGETS[-1])
    if models["new"] != {**models[chosen], "tuning_policy": chosen}:
        raise ValueError("The report model differs from the preselected 09 checkpoint")
    for entry in models.values():
        for path_key, hash_key in (("loaded_path", "sha256"), ("settings", "settings_sha256")):
            if digest(ROOT / entry[path_key]) != entry[hash_key]:
                raise ValueError(f"Input changed since evaluation: {entry[path_key]}")
    for name in ("episodes.csv", "trajectories.csv"):
        if digest(directory / name) != manifest["sha256"][name]:
            raise ValueError(f"Evaluation CSV changed: {name}")

    datasets = cases(manifest["mode"] == "smoke")
    if manifest["starts"] != {ds: len(starts) for ds, (starts, _) in datasets.items()}:
        raise ValueError("Unexpected dataset sizes")
    expected = {
        (ds, label, driver): start
        for ds, (starts, drivers) in datasets.items()
        for label, start in starts.items()
        for driver in drivers
    }
    episodes = {}
    for row in csv_rows(directory / "episodes.csv", EPISODE_FIELDS):
        key = row["dataset"], row["start"], row["policy"]
        start = float(row["x"]), float(row["y"])
        outcome = row["outcome"]
        time = float(row["capture_time"]) if row["capture_time"] else None
        if (
            key in episodes
            or start != expected.get(key)
            or outcome not in OUTCOMES
            or (outcome == "capture") != (time is not None)
            or (time is not None and not 0 <= time <= DEFAULT_CONFIG.horizon)
        ):
            raise ValueError(f"Invalid episode: {key}")
        episodes[key] = EpisodeResult(start, outcome, time)
    if episodes.keys() != expected.keys():
        raise ValueError("Missing episodes")

    points = {
        key: []
        for key, start in expected.items()
        if key[0] == "benchmarks" or (key[0] == "grid" and start == GRAY)
    }
    for row in csv_rows(directory / "trajectories.csv", POINT_FIELDS):
        key = row["dataset"], row["start"], row["policy"]
        trace = points[key]
        point = TrajectoryPoint(
            float(row["time"]),
            (float(row["x"]), float(row["y"])),
            float(row["phi_on_arrival"]) if row["phi_on_arrival"] else None,
            float(row["psi_at_interval_start"]) if row["psi_at_interval_start"] else None,
            event=row["event"] or None,
        )
        if (
            int(row["decision"]) != len(trace)
            or not all(isfinite(v) for v in (point.time, *point.state))
            or not 0 <= point.time <= DEFAULT_CONFIG.horizon + 1e-9
            or (
                trace
                and (
                    not 0 < point.time - trace[-1].time <= DEFAULT_CONFIG.dt + 1e-9
                    or point.phi is None
                    or point.psi is None
                    or not isfinite(point.phi)
                    or not isfinite(point.psi)
                )
            )
            or (not trace and (point.time != 0 or point.state != expected[key]))
        ):
            raise ValueError(f"Invalid trajectory point: {key}, decision {row['decision']}")
        trace.append(point)
    traces = {}
    for key, trace in points.items():
        episode = episodes[key]
        if (
            not trace
            or (
                episode.capture_time is not None
                and (
                    trace[-1].time != episode.capture_time
                    or hypot(*trace[-1].state) > DEFAULT_CONFIG.beta + 1e-6
                )
            )
            or (
                episode.outcome == "time_limit"
                and abs(trace[-1].time - DEFAULT_CONFIG.horizon) > 1e-9
            )
        ):
            raise ValueError(f"Trajectory and outcome disagree: {key}")
        traces[key] = Rollout(episode.outcome, tuple(trace))
    return manifest, datasets, episodes, traces


def time_text(value, *, signed=False):
    return "—" if value is None else format(value, "+.6f" if signed else ".6f").replace(".", ",")


def tuning_report(directory, labels, episodes):
    references = []
    for driver in ("classical", "heuristic", "archive"):
        captures = sum(episodes["tuning", label, driver].outcome == "capture" for label in labels)
        references.append(f"{LABELS[driver]} — {captures}/{len(labels)}")
    lines = [
        f"### Tuning · {len(labels)} стартов",
        "",
        (
            "Обе серии оценены на исправленном пешеходе с зазором 0,001, попарно по "
            "seed, бюджету и старту. «+ / −» считает подтверждённые захваты только одной политики."
        ),
        "",
        "Подтверждённые захваты эталонов. " + ", ".join(references) + ".",
        "",
        (
            "| Решения | Seed | 08 · захваты / тайм-ауты / unsupported | "
            "09 · захваты / тайм-ауты / unsupported | + / − к 08 | Общие захваты | Среднее t₀₉ − t₀₈ |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    totals = []
    with (directory / "tuning-pairs.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "seed",
                "decisions",
                "start",
                "x",
                "y",
                "outcome_08",
                "outcome_09",
                "capture_time_08",
                "capture_time_09",
                "time_delta_09_minus_08",
            )
        )
        for budget in BUDGETS:
            counts = {"08": [], "09": []}
            for seed in SEEDS:
                old, new = (
                    [episodes["tuning", label, policy(group, seed, budget)] for label in labels]
                    for group in ("08", "09")
                )
                pair = compare(new, old)
                gains = losses = 0
                for label, a, b in zip(labels, old, new, strict=True):
                    gains += a.outcome != "capture" and b.outcome == "capture"
                    losses += a.outcome == "capture" and b.outcome != "capture"
                    delta = (
                        b.capture_time - a.capture_time
                        if a.capture_time is not None and b.capture_time is not None
                        else None
                    )
                    writer.writerow(
                        (
                            seed,
                            budget,
                            label,
                            *a.start,
                            a.outcome,
                            b.outcome,
                            a.capture_time,
                            b.capture_time,
                            delta,
                        )
                    )
                triplets = []
                for group, ordered in (("08", old), ("09", new)):
                    outcomes = Counter(e.outcome for e in ordered)
                    counts[group].append(outcomes["capture"])
                    triplets.append(" / ".join(str(outcomes[outcome]) for outcome in OUTCOMES))
                delta = time_text(pair["mean_capture_time_delta"], signed=True)
                lines.append(
                    f"| {budget} | {seed} | {triplets[0]} | {triplets[1]} | {gains} / {losses} "
                    f"| {pair['both_captured']} | {delta} |"
                )
            aggregates = []
            for group, values in counts.items():
                rates = [100 * n / len(labels) for n in values]
                aggregates.append(
                    f"{group} в среднем {mean(values):.2f} из {len(labels)} ({mean(rates):.2f} %), "
                    f"диапазон {min(values)}–{max(values)} ({min(rates):.2f}–{max(rates):.2f} %)".replace(
                        ".", ","
                    )
                )
            totals.append(f"- {budget} решений. " + ". ".join(aggregates) + ".")
    return lines + ["", *totals, ""]


def capture_maps(target, grid, *, zoom):
    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, to_rgba
    from matplotlib.patches import Patch, Rectangle

    colors = {"time_limit": "#b42d68", "unsupported": "#a5abb3"}
    cmap, norm = plt.colormaps["viridis"], Normalize(0, DEFAULT_CONFIG.horizon)
    size = GRID_MAX - GRID_MIN + 1
    low, high = (GRID_MIN - 0.5) / GRID_SCALE, (GRID_MAX + 0.5) / GRID_SCALE
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True, layout="constrained")
    for driver, ax in zip(DRIVERS, axes.flat, strict=True):
        pixels = np.ones((size, size, 4))
        for result in grid[driver]:
            x, y = result.start
            row, col = round(y * GRID_SCALE) - GRID_MIN, round(x * GRID_SCALE) - GRID_MIN
            pixels[row, col] = (
                cmap(norm(result.capture_time))
                if result.capture_time is not None
                else to_rgba(colors[result.outcome])
            )
        ax.imshow(pixels, origin="lower", extent=(low, high, low, high), interpolation="nearest")
        ax.add_patch(
            Rectangle(
                (GRAY[0] - 0.05, GRAY[1] - 0.05),
                0.1,
                0.1,
                fill=False,
                edgecolor="#333333",
                linewidth=1.0,
            )
        )
        ax.set_title(LABELS[driver])
        if zoom:
            ax.set(xlim=(GRAY[0] - 0.26, GRAY[0] + 0.26), ylim=(GRAY[1] - 0.26, GRAY[1] + 0.26))
    fig.supxlabel("x₀")
    fig.supylabel("y₀")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.83, label="Время захвата")
    fig.legend(
        handles=[Patch(facecolor=c, label=OUTCOMES[k]) for k, c in colors.items()],
        loc="outside lower center",
        ncols=2,
        frameon=False,
    )
    fig.savefig(target, dpi=160)
    plt.close(fig)


def outcomes_table(lines, grid):
    lines.extend(
        [
            (
                "| Водитель | Захваты | Тайм-ауты | Unsupported | Доля подтверждённых | "
                "Среднее время захвата (n) |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for driver, results in grid.items():
        times = [r.capture_time for r in results if r.capture_time is not None]
        counts = Counter(r.outcome for r in results)
        share = f"{100 * len(times) / len(results):.2f}".replace(".", ",")
        avg = time_text(mean(times) if times else None)
        lines.append(
            f"| {LABELS[driver]} | {len(times)} | {counts['time_limit']} "
            f"| {counts['unsupported']} | {share} % | {avg} ({len(times)}) |"
        )


def pair_table(lines, results):
    lines.extend(
        [
            "",
            (
                "Средняя разность времени считается только на стартах с захватом у обоих "
                "водителей. Столбец «Общие захваты» показывает знаменатель."
            ),
            "",
            "| Кандидат | Эталон | Общие захваты | Среднее t кандидата − t эталона |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for reference, candidate in combinations(results, 2):
        pair = compare(results[candidate], results[reference])
        lines.append(
            f"| {LABELS[candidate]} | {LABELS[reference]} | {pair['both_captured']} "
            f"| {time_text(pair['mean_capture_time_delta'], signed=True)} |"
        )


def render(directory, manifest, datasets, episodes, traces):
    smoke = manifest["mode"] == "smoke"
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    tables = {
        ds: {
            driver: [episodes[ds, label, driver] for label in datasets[ds][0]] for driver in DRIVERS
        }
        for ds in ("grid", "benchmarks")
    }
    grid_label = next(label for label, start in datasets["grid"][0].items() if start == GRAY)
    lines = [
        "## Новый протокол с зазором 0,001",
        "",
        (
            "Короткий прогон · 25 соседних ячеек вокруг (−2,3; −0,2)."
            if smoke
            else "Финальная оценка · общая сетка 6 532 стартов."
        ),
        "",
        (
            "Классический пешеход, γ игры = 0,5, β = 0,3, шаг до 0,05, "
            "первый контакт, T = 20. Оба PPO действуют детерминированно на CPU. "
            "Для отчёта заранее зафиксирован seed 43 и бюджет 262 144 решений."
        ),
        "",
    ]
    for driver in ("new", "archive"):
        entry = manifest["models"][driver]
        clearance = str(entry["training_clearance"]).replace(".", ",")
        lines.append(
            f"- **{LABELS[driver]}** обучен с зазором {clearance}. "
            f"Веса `{entry['loaded_path']}`, SHA-256 `{entry['sha256']}`. "
            f"Настройки `{entry['settings']}`."
        )
    lines.extend(
        [
            "",
            (
                "[Эпизоды](@DATA@/episodes.csv) · [Траектории](@DATA@/trajectories.csv) · "
                "[Манифест весов и результатов](@DATA@/manifest.json)"
            ),
            "",
            "### Сетка",
            "",
        ]
    )
    outcomes_table(lines, tables["grid"])
    pair_table(lines, tables["grid"])
    lines.extend(
        [
            "",
            "![Карты времени захвата четырёх водителей](@FIGURES@/capture-times.png)",
            "",
            "Малиновый цвет обозначает тайм-аут, серый — unsupported. "
            + (
                "Белые ячейки пропущены в коротком прогоне."
                if smoke
                else "Белые ячейки лежат внутри круга захвата или на его границе."
            ),
            "",
            "### Бывшая серая ячейка (−2,3; −0,2)",
            "",
            (
                "При исторической оценке с зазором 0 архивный PPO завершился на SE "
                "при t = 2,609972 с unsupported. Ниже тот же старт при зазоре 0,001."
            ),
            "",
            (
                "| Водитель | Исход | t конца | min r на сохранённых состояниях | "
                "Последнее записанное событие | Шагов 0 < Δt < 0,001 |"
            ),
            "| --- | --- | ---: | ---: | --- | ---: |",
        ]
    )
    gray = {driver: traces["grid", grid_label, driver] for driver in DRIVERS}
    for driver, run in gray.items():
        path = run.trajectory
        tail = next((p for p in reversed(path) if p.event), None)
        cross = f"`{tail.event}` в {time_text(tail.time)}" if tail else "—"
        short = sum(0 < b.time - a.time < 0.001 for a, b in pairwise(path))
        lines.append(
            f"| {LABELS[driver]} | {OUTCOMES[run.outcome]} | {time_text(path[-1].time)} "
            f"| {time_text(min(hypot(*p.state) for p in path))} | {cross} | {short} |"
        )
    lines.extend(["", "Первый руль φ и события в каждой траектории.", ""])
    for driver, run in gray.items():
        path = run.trajectory
        first = path[1] if len(path) > 1 else None
        events = {}
        for point in path:
            if point.event in {"axis", "barrier", "SE", "safe_contact", "virtual_entry"}:
                events.setdefault(point.event, point.time)
        changes = ", ".join(f"`{name}` при t = {time_text(t)}" for name, t in events.items())
        initial = f"φ = {time_text(first.phi, signed=True)}" if first else "шагов с Δt > 0 нет"
        lines.append(f"- {LABELS[driver]} · {initial}" + (f", {changes}." if changes else "."))
    if not smoke:
        capture_maps(figures / "gray-cell-neighborhood.png", tables["grid"], zoom=True)
        lines.extend(
            ["", "![Окрестность бывшей серой ячейки](@FIGURES@/gray-cell-neighborhood.png)"]
        )
    _save_figure(figures / "gray-cell.png", "Former unsupported cell", GRAY, gray, DEFAULT_CONFIG)
    lines.extend(
        [
            "",
            "![Траектории и управления из бывшей серой ячейки](@FIGURES@/gray-cell.png)",
            "",
            "### Реперы P1–P8",
            "",
            "| Репер | Старт (x₀, y₀) | " + " | ".join(LABELS.values()) + " |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, start in BENCHMARKS.items():
        runs = {driver: traces["benchmarks", label, driver] for driver in DRIVERS}
        _save_figure(figures / f"{label}.png", label, start, runs, DEFAULT_CONFIG)
        results = [episodes["benchmarks", label, driver] for driver in DRIVERS]
        cells = [
            time_text(r.capture_time) if r.capture_time is not None else OUTCOMES[r.outcome]
            for r in results
        ]
        lines.append(f"| {label} | ({start[0]:g}, {start[1]:g}) | " + " | ".join(cells) + " |")
    pair_table(lines, tables["benchmarks"])
    for label in BENCHMARKS:
        lines.extend(
            ["", f"![{label}. Координаты, траектория, руль и курс](@FIGURES@/{label}.png)"]
        )
    lines.extend(
        [
            "",
            *tuning_report(directory, datasets["tuning"][0], episodes),
            "[Парные исходы и времена по стартам](@DATA@/tuning-pairs.csv)",
        ]
    )
    capture_maps(figures / "capture-times.png", tables["grid"], zoom=smoke)
    return "\n".join(lines) + "\n", figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--publish", action="store_true", help="update docs after a full run")
    args = parser.parse_args()
    directory = args.results.resolve()
    manifest, datasets, episodes, traces = read_results(directory)
    if args.publish and manifest["mode"] != "full":
        parser.error("publishing requires the full 6,532-start evaluation")
    matplotlib.use("Agg")
    section, figures = render(directory, manifest, datasets, episodes, traces)
    report = directory / "report.md"
    report.write_text(
        section.replace("@DATA@/", "").replace("@FIGURES@", "figures"), encoding="utf-8"
    )
    if args.publish:
        doc = ROOT / "docs/driver-comparison.md"
        original = doc.read_text(encoding="utf-8")
        before, marker, rest = original.partition(BEGIN)
        _, end, after = rest.partition(END)
        if not marker or not end:
            raise ValueError(f"Report markers missing: {doc}")
        published = doc.parent / "figures/ppo-clearance-0001"
        published.mkdir(parents=True, exist_ok=True)
        for figure in sorted(figures.glob("*.png")):
            copyfile(figure, published / figure.name)
        section = section.replace("@DATA@", relpath(directory, doc.parent))
        section = section.replace("@FIGURES@", relpath(published, doc.parent))
        doc.write_text(f"{before}{BEGIN}\n\n{section}\n{END}{after}", encoding="utf-8")
        print(f"Updated {doc}")
    print(
        f"Verified {manifest['mode']} report: {report} ({len(episodes)} episodes; "
        f"{len(traces)} trajectories)"
    )


if __name__ == "__main__":
    main()
