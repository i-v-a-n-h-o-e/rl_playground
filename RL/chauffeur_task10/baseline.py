"""Три повтора базового PPO с сетями 32×32."""

from RL.chauffeur_task10.common import parser, run

if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    run(args.output / "baseline", architecture="baseline")
