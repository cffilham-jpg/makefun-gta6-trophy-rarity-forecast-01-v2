# GTA VI trophy rarity forecast v2

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/cffilham-jpg/makefun-gta6-trophy-rarity-forecast-01-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/cffilham-jpg/makefun-gta6-trophy-rarity-forecast-01-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each PlayStation trophy's Exophase earn rate at the target time. v2 fixes v1's two flaws: the unvalidated −1.25 logit shift and a 90% band 4.6× wider than the house baseline's.

The model decomposes each rate into earners over players tracked and forecasts the two separately, because they move for different reasons: earners grow as players progress, while players tracked jumps at each Exophase refresh (dilution). With enough distinct snapshots it fits log earners and log players against log time since release, shrinking each trophy's slope toward its tier's pooled slope. With a single launch-night snapshot there is no trend to fit, so it applies the documented launch-night refresh as a prior — the tracked sample jumps ~6× (638 → 3,868 players) while new players have barely begun — with a catch-up factor calibrated on the dev board (it reproduces both the house baseline's practice loss and v1's under simulation). Spreads stay near the house scale: a random-walk term plus fit and refresh uncertainty, with a Poisson term for small earner counts.

Bounty: [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## Run locally

Copy the practice cases from the kit into an `input/` directory, then run:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The program uses only Python's standard library and reads every `*.in` case in the input directory.

[Join make.fun](https://make.fun/r/fpna5aje)
