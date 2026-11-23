"""Repeat task 08 after the pedestrian regression passes and audit the saved run."""

import argparse
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

from chauffeur_evader.chauffeur_task10.clearance import OUTPUT_ROOT, ROOT, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="new run directory under data/ppo-clearance-0001/task09-pdl")
    parser.add_argument("--smoke", action="store_true", help="exercise the pipeline before repair")
    args = parser.parse_args()
    if args.name in {"", ".", ".."} or Path(args.name).name != args.name:
        parser.error("provide a single directory name")
    output = OUTPUT_ROOT / "task09-pdl" / args.name
    if output.exists():
        parser.error(f"choose a new directory name; {output} exists")
    regression = Path(__file__).with_name("task07_pdl_repro.py")
    if not args.smoke:
        subprocess.run([sys.executable, "-m", "pytest", "-q", str(regression)], check=True)

    sources = sorted((ROOT / "chauffeur_evader").rglob("*.py")) + [
        Path(__file__),
        regression,
    ]
    source_hashes = {
        str(p.resolve().relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in sources
    }
    run(output, smoke=args.smoke)

    summary = json.loads((output / "summary.json").read_text())
    mode = "smoke" if args.smoke else "full"
    expected_starts = {"benchmarks": 8, "tuning": 8 if args.smoke else 512}
    if (summary["status"], summary["mode"], summary["starts"]) != (
        "complete",
        mode,
        expected_starts,
    ):
        raise ValueError("Incomplete evaluation")
    manifest = {}
    for line in (output / "models.sha256").read_text().splitlines():
        digest, path = line.split()
        manifest[path] = digest
    models = summary["models"]
    if len(models) != 7 or {m["path"]: m["sha256"] for m in models.values()} != manifest:
        raise ValueError("Checkpoint manifest differs from summary.json")
    for path, digest in manifest.items():
        if sha256((output / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checkpoint digest mismatch: {path}")

    lines = [f"# Зазор 0,001 · {mode}", ""]
    for dataset, size in expected_starts.items():
        lines.extend(
            [
                f"## {dataset} · {size} стартов",
                "",
                "| Политика | Захват | Лимит | Unsupported |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for policy, result in summary["outcomes"][dataset].items():
            counts = result["counts"]
            lines.append(
                f"| {policy} | {counts['capture']} | {counts['time_limit']} "
                f"| {counts['unsupported']} |"
            )
        lines.extend(
            [
                "",
                "| Кандидат | Эталон | Общие захваты | Среднее t кандидата − t эталона |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for pair in summary["comparisons"][dataset]:
            if pair["reference_policy"] not in {
                "heuristic",
                "classical",
                "archive-seed-43",
            }:
                continue
            delta = pair["mean_capture_time_delta"]
            value = "—" if delta is None else f"{delta:+.6f}"
            lines.append(
                f"| {pair['candidate_policy']} | {pair['reference_policy']} "
                f"| {pair['both_captured']} | {value} |"
            )
        lines.append("")
    (output / "source-sha256.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    report = output / "report.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"Verified {mode} run: {report}")


if __name__ == "__main__":
    main()
