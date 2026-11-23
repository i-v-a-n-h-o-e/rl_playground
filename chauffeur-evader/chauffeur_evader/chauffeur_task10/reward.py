"""Сравнение потенциальной добавки с базовой наградой."""

from chauffeur_evader.chauffeur_task10.common import NETWORKS, parser, run

if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--architecture", choices=NETWORKS, required=True)
    args = cli.parse_args()
    control = "baseline" if args.architecture == "baseline" else "width"
    run(
        args.output / f"reward-{args.architecture}",
        architecture=args.architecture,
        reward="potential",
        reference=args.output / control,
    )
