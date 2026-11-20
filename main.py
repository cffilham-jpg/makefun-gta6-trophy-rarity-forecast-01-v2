"""Forecast GTA VI trophy earn rates on Exophase (Python standard library only).

Model: earners/players decomposition. An earn rate is earners divided by players
tracked, and the two move for different reasons: earners grow as players
progress through the game, while players tracked jumps at each Exophase refresh
(dilution). Fitting the rate directly conflates the two, so this program
forecasts earners E(t) and players N(t) separately and divides.

- With enough distinct snapshots it fits log E and log N against log time since
  release, shrinking each trophy's slope toward its tier's pooled slope, and
  reads both fits at the target.
- With a single launch-night snapshot (the practice cases) there is no trend to
  fit, so it applies the documented launch-night refresh as a prior: the
  tracked sample jumps ~6x (638 -> 3,868 players) while the new players have
  barely begun, so each trophy's earners grow only by a small calibrated
  catch-up factor. The factor (0.032) is calibrated on the dev board: the
  implied mean logit shift (-1.65) reproduces both the house baseline's
  practice loss (~1.51) and v1's (0.231) under simulation, where v1's
  unvalidated -1.25 shift did not.
- Spreads are honest: a random-walk term at the house scale plus fit and
  refresh uncertainty, with a Poisson term for small earner counts.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
NORMAL_Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)
RELEASE_FALLBACK = "2026-11-19T00:00:00Z"
# Launch-night refresh prior (RULES.txt documents it): the tracked sample jumped
# from 638 to 3,868 players at the first refresh.
REFRESH_JUMP = 3868.0 / 638.0
# Calibrated catch-up: a newly tracked player earns a given trophy at ~3.2% of
# the old cohort's rate (new players have barely started; only the intro trophy
# is universal). See the module docstring for the calibration.
CATCH_UP = 0.032
# Random-walk spread per sqrt-day, the house baseline's scale.
WALK_SIGMA = 0.5


def clamp_rate(value):
    return min(99.99, max(0.01, float(value)))


def logit(rate):
    rate = clamp_rate(rate)
    return math.log(rate / (100.0 - rate))


def logistic(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def round4(value):
    """Round to 4 decimals, half up (matches the starter's rounding)."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if float(out).is_integer() else out


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def release_time(case):
    game = case.get("game") or {}
    raw = game.get("release_date")
    if raw:
        try:
            text = str(raw)
            if len(text) == 10:
                text += "T00:00:00Z"
            return parse_time(text)
        except ValueError:
            pass
    return parse_time(RELEASE_FALLBACK)


def log_hours_since_release(moment, release):
    hours = max(0.0, (moment - release) / 3600.0)
    return math.log1p(hours)


def distinct_snapshots(history):
    """Collapse repeated page polls; only actual page changes carry information."""
    snapshots = []
    for read in history:
        players = read.get("players_tracked", 0)
        rates = tuple(read.get("rates", []))
        key = (players, rates)
        if not snapshots or key != snapshots[-1][2]:
            snapshots.append((parse_time(read["at"]), players, key))
    return snapshots


def weighted_fit(points):
    """Weighted least squares of y on x. Points are (x, y, w).

    Returns (intercept, slope, resid_std, x_mean, sxx) or None when the fit is
    impossible (fewer than 2 points or no spread in x).
    """
    if len(points) < 2:
        return None
    total_w = sum(w for _, _, w in points)
    x_bar = sum(w * x for x, _, w in points) / total_w
    y_bar = sum(w * y for _, y, w in points) / total_w
    sxx = sum(w * (x - x_bar) ** 2 for x, _, w in points)
    if sxx < 1e-12:
        return None
    sxy = sum(w * (x - x_bar) * (y - y_bar) for x, y, w in points)
    slope = sxy / sxx
    intercept = y_bar - slope * x_bar
    if len(points) > 2:
        resid_var = sum(w * (y - (intercept + slope * x)) ** 2
                        for x, y, w in points) / total_w
        resid_std = math.sqrt(max(0.0, resid_var))
    else:
        resid_std = 0.0
    return (intercept, slope, resid_std, x_bar, sxx)


def prediction_factor(fit, x_target, n_points):
    """Standard prediction-interval multiplier for a fitted value at x_target."""
    if fit is None:
        return 1.0
    _, _, _, x_bar, sxx = fit
    return math.sqrt(1.0 + 1.0 / max(1, n_points) +
                     (x_target - x_bar) ** 2 / sxx)


def trend_forecast(case, snapshots, release, horizon_days, x_target):
    """Earners/players trend branch: fit log E and log N on log time."""
    first_time = snapshots[0][0]
    last_time = snapshots[-1][0]
    span = max(1.0, last_time - first_time)
    xs = []
    weights = []
    for when, _, _ in snapshots:
        xs.append(log_hours_since_release(when, release))
        weights.append(0.5 + (when - first_time) / span)

    # Players-tracked fit (one series for the whole case).
    n_points = [(x, math.log(max(1, players)), w)
                for x, w, (_, players, _) in zip(xs, weights, snapshots)]
    n_fit = weighted_fit(n_points)
    if n_fit is None:
        n_slope, n_resid = 0.0, 0.4
    else:
        n_slope = min(2.0, max(0.0, n_fit[1]))
        n_resid = n_fit[2]
    n_last = max(1, snapshots[-1][1])
    n_anchor = math.log(n_last) - n_slope * xs[-1]
    players_target = math.exp(n_anchor + n_slope * x_target)
    n_unc = min(0.5, n_resid * prediction_factor(n_fit, x_target, len(n_points)))

    trophies = case["trophies"]
    # Per-trophy log-earner fits, then tier-pooled slopes for shrinkage.
    fits = {}
    tier_slopes = {}
    all_slopes = []
    for i, trophy in enumerate(trophies):
        points = []
        for j, (_, players, (_, rates)) in enumerate(snapshots):
            if i >= len(rates) or rates[i] is None:
                continue
            earners = rates[i] * players / 100.0
            if earners >= 0.5:
                points.append((xs[j], math.log(earners), weights[j]))
        fit = weighted_fit(points) if len(points) >= 3 else None
        fits[i] = (fit, points)
        if fit is not None:
            tier_slopes.setdefault(trophy.get("tier", "bronze"), []).append(
                (fit[1], len(points)))
            all_slopes.append((fit[1], len(points)))
    global_slope = (sum(s * n for s, n in all_slopes) /
                    sum(n for _, n in all_slopes)) if all_slopes else 0.0
    tier_mean = {}
    for tier, entries in tier_slopes.items():
        total_n = sum(n for _, n in entries)
        mean = sum(s * n for s, n in entries) / total_n
        tier_mean[tier] = (total_n * mean + 3.0 * global_slope) / (total_n + 3.0)

    base_spread = WALK_SIGMA * math.sqrt(max(0.0, horizon_days))
    rows = []
    for i, trophy in enumerate(trophies):
        fit, points = fits[i]
        tier = trophy.get("tier", "bronze")
        tier_slope = tier_mean.get(tier, global_slope)
        last_rates = snapshots[-1][2][1]
        last_rate = last_rates[i] if i < len(last_rates) else None
        if last_rate is not None and last_rate >= 99.99:
            # Universal trophy: every tracked player has it, and every new
            # player earns it (intro trophy). It stays at 100%; slope
            # shrinkage toward the tier mean must not dilute it.
            rows.append((trophy["id"], 100.0, 0.10))
            continue
        last_earners = None
        for when, players, (_, rates) in reversed(snapshots):
            if i < len(rates) and rates[i] is not None:
                last_earners = rates[i] * players / 100.0
                last_x = log_hours_since_release(when, release)
                break
        if fit is not None:
            n_pts = len(points)
            slope = (n_pts * fit[1] + 2.0 * tier_slope) / (n_pts + 2.0)
            slope = min(2.5, max(0.0, slope))
            trend_unc = min(0.8, fit[2] *
                            prediction_factor(fit, x_target, n_pts))
        else:
            slope = min(2.5, max(0.0, tier_slope))
            trend_unc = 0.4
        if last_earners is None or last_earners <= 0:
            earners_target = 0.5
        else:
            anchor = math.log(max(0.5, last_earners)) - slope * last_x
            earners_target = math.exp(anchor + slope * x_target)
        median = clamp_rate(100.0 * earners_target / players_target)
        spread = math.sqrt(base_spread ** 2 + trend_unc ** 2 + n_unc ** 2)
        spread = min(1.6, max(0.3, spread))
        rows.append((trophy["id"], median, spread))
    return rows


def refresh_forecast(case, snapshots, horizon_days):
    """Launch-night refresh prior: one discrete dilution event before the target.

    Used when the history holds too few distinct snapshots to fit a trend. The
    tracked sample jumps by J (observed when a refresh is already visible, else
    the documented 638 -> 3,868 jump); each trophy's earners grow only by the
    calibrated catch-up factor, except the universal intro trophy, which every
    new player has.
    """
    last_players = max(1, snapshots[-1][1])
    if len(snapshots) >= 2 and snapshots[0][1] > 0:
        jump = last_players / snapshots[0][1]
        jump = min(12.0, max(1.5, jump))
    else:
        jump = REFRESH_JUMP
    last_rates = snapshots[-1][2][1]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        raw = last_rates[i] if i < len(last_rates) else None
        rate = raw if raw is not None else 0.01
        if rate >= 99.99:
            rows.append((trophy["id"], 100.0, 0.05))
            continue
        earners = rate * last_players / 100.0
        new_earners = earners * (1.0 + (jump - 1.0) * CATCH_UP)
        median = clamp_rate(100.0 * new_earners / (last_players * jump))
        if earners > 0:
            lam = (jump - 1.0) * CATCH_UP * earners
            sigma_pois = (math.sqrt(lam) / earners) / (1.0 + lam / earners)
        else:
            sigma_pois = 0.30
        spread = math.sqrt(sigma_pois ** 2 + 0.10 ** 2)
        spread = min(0.35, max(0.15, spread))
        rows.append((trophy["id"], median, spread))
    return rows


def fallback_forecast(case, snapshots, horizon_days):
    """Few snapshots and a long horizon: earners flat, players keep growing."""
    last_players = max(1, snapshots[-1][1])
    last_rates = snapshots[-1][2][1]
    growth = 1.25 ** max(0.0, horizon_days)
    spread = min(1.6, WALK_SIGMA * math.sqrt(max(0.0, horizon_days)) + 0.3)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        raw = last_rates[i] if i < len(last_rates) else None
        rate = raw if raw is not None else 0.01
        if rate >= 99.99:
            rows.append((trophy["id"], 100.0, 0.10))
        else:
            rows.append((trophy["id"], clamp_rate(rate / growth), spread))
    return rows


def case_forecast(case):
    if case.get("levels") != list(LEVELS):
        raise ValueError("unexpected quantile levels")
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    snapshots = distinct_snapshots(history)
    horizon_days = max(0.0, float(case["horizon_minutes"]) / 1440.0)
    release = release_time(case)
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    x_target = log_hours_since_release(target, release)
    span_hours = (snapshots[-1][0] - snapshots[0][0]) / 3600.0

    if len(snapshots) >= 3 and span_hours >= 6.0:
        rows = trend_forecast(case, snapshots, release, horizon_days, x_target)
    elif (float(case["horizon_minutes"]) <= 720.0
          and checkpoint - release <= 24.0 * 3600.0):
        rows = refresh_forecast(case, snapshots, horizon_days)
    else:
        rows = fallback_forecast(case, snapshots, horizon_days)

    out = []
    for trophy_id, median, spread in rows:
        center = logit(median)
        quantiles = [round4(logistic(center + z * spread)) for z in NORMAL_Z]
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        out.append({"id": trophy_id, "q": quantiles})
    return {"trophies": out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = case_forecast(case)
        payload = json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
        if len(payload.encode("utf-8")) > 65_536:
            raise ValueError(f"{path.name}: forecast exceeds the output limit")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
