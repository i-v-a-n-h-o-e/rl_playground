"""Сравнить wide + potential на 262144 и 524288 решениях на пяти seed."""

import json

from chauffeur_evader.chauffeur_task10.common import ROOT, parser, run
from chauffeur_evader.chauffeur_task10.compare_control import FREEZE, sha256


def main() -> None:
    cli = parser(__doc__)
    args = cli.parse_args()
    protocol = json.loads((FREEZE / "protocol.json").read_text(encoding="utf-8"))
    control = protocol["control"]
    frozen_controls = {}
    for seed, entry in control["checkpoints"].items():
        checkpoint = ROOT / entry["path"]
        if not checkpoint.is_file() or sha256(checkpoint) != entry["sha256"]:
            cli.error(
                f"контрольный чекпойнт отсутствует или не прошёл проверку SHA-256: {checkpoint}"
            )
        frozen_controls[int(seed)] = checkpoint

    seeds = protocol["training_seeds"]
    run(
        args.output / "length-wide-potential-524288",
        architecture=control["architecture"],
        reward=control["reward"],
        budgets=(control["decisions"], 524_288),
        seeds=tuple(seeds["task10"] + seeds["confirmation"]),
        frozen_controls=frozen_controls,
    )


if __name__ == "__main__":
    main()
