"""Сравнение сети управления 64×64 с базовой 32×32."""

from chauffeur_evader.chauffeur_task10.common import parser, run

if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    run(args.output / "width", architecture="wide", reference=args.output / "baseline")
