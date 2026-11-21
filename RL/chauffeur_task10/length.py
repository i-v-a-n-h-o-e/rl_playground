"""Сравнение двух чекпойнтов из одного обучения PPO."""

from RL.chauffeur_task10.common import NETWORKS, REWARDS, parser, run

if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--architecture", choices=NETWORKS, required=True)
    cli.add_argument("--reward", choices=REWARDS, required=True)
    args = cli.parse_args()
    run(
        args.output / f"length-{args.architecture}-{args.reward}",
        architecture=args.architecture,
        reward=args.reward,
        long=True,
    )
