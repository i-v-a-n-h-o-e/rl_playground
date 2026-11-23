"""Сравнение двух чекпойнтов из одного обучения PPO."""

from chauffeur_evader.chauffeur_task10.common import NETWORKS, REWARDS, STEPS, parser, run

if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--architecture", choices=NETWORKS, required=True)
    cli.add_argument("--reward", choices=REWARDS, required=True)
    cli.add_argument("--clearance", type=float, default=0.0, help="запрошенный зазор пешехода")
    args = cli.parse_args()
    run(
        args.output / f"length-{args.architecture}-{args.reward}",
        architecture=args.architecture,
        reward=args.reward,
        clearance=args.clearance,
        budgets=(STEPS, 2 * STEPS),
    )
