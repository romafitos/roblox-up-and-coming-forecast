#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming chart top 10 (make.fun roblox-up-and-coming-forecast-01).

Method (see NOTES.md for the full write-up):
  * Every game on the chart gets a latent log-rank process, mean-reverting
    toward the chart's middle (top games slide down over time: new games push
    the old ones down and games age out at 28 days), with a per-game drift
    deviation estimated from its recent rank trajectory (momentum, shrunk)
    and a diffusion term whose size depends on the game's rank (the top of
    the chart churns hardest).  All parameters are estimated at run time
    from the history in /input, with offline-measured priors as fallback.
  * A seeded Monte Carlo over the window's slots (log-ranks sorted per draw)
    gives P(game at place k), averaged over the window.
  * A reserve is left unnamed at every place for games that join the chart
    after the history ends (estimated from the rate at which new games
    reached the top 10 in the history).
  * Games whose details give a `created` date leave the simulation once
    they turn 28 days old.
Python 3.13, standard library only, deterministic (fixed seed), no network.
Reads $MAKEFUN_INPUT_DIR (else /input) and writes $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import random
from datetime import datetime, timedelta, timezone

SEED = 20261113
N_DRAWS = 4000        # Monte Carlo draws per slot time
N_SLOTS = 12          # evenly spaced slot times across the window
MAX_AGE_DAYS = 28     # games leave the chart 28 days after creation

# Offline-measured priors (from the practice history; the program re-fits
# them from whatever history it is given and falls back to these).
PRIOR_DRIFT_A = 0.024    # log-rank drift per hour at log-rank 0
PRIOR_DRIFT_B = -0.0065  # ... falling with log-rank (mean reversion)
PRIOR_SIGMA0 = 0.115     # overall log-rank diffusion per sqrt(hour)
PRIOR_RESERVE = 0.07     # window-average share of top-10 slots held by newcomers

# Calibration scales, tuned on the two truncated backtest cuts (work/sweep*.py):
# each factor helped on BOTH cuts, and the gains flatten beyond these values.
SIGMA_SCALE = 1.5    # the fitted diffusion under-states how fast the chart moves
KAPPA_SCALE = 2.0    # ... and so does the fitted mean-reversion speed
RESERVE_SCALE = 5.0  # the unnamed reserve also tempers an over-sharp forecast


def find_detail(details, uid):
    """Return the per-game details dict for uid, whatever its container."""
    if not details:
        return None
    try:
        if isinstance(details, dict):
            d = details.get(str(uid)) or details.get(uid)
            return d if isinstance(d, dict) else None
        if isinstance(details, list):
            for d in details:
                if isinstance(d, dict) and d.get("universe_id") == uid:
                    return d
    except Exception:
        return None
    return None


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_input(input_dir):
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    reads = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    reads.sort(key=lambda r: r["at"])
    return task, reads


def chart_of(read):
    """universe_id -> game row of the up-and-coming chart, sorted by rank."""
    rows = sorted(read.get("up_and_coming") or [], key=lambda g: g["rank"])
    return {g["universe_id"]: g for g in rows}


def fit_params(reads):
    """Estimate drift, diffusion and newcomer rate from read pairs."""
    charts = [chart_of(r) for r in reads]
    times = [parse_time(r["at"]) for r in reads]
    pairs = []  # (x0, d, dt_hours) for games present in both reads
    newcomer = []  # (dt_hours, share of top-10 slots held by newcomers)
    for i in range(len(reads)):
        for j in range(i + 1, len(reads)):
            dt = (times[j] - times[i]).total_seconds() / 3600.0
            if dt > 96.0:
                break
            if dt < 1.0:
                continue
            ca, cb = charts[i], charts[j]
            if dt <= 30.0:
                for uid, g in ca.items():
                    if uid in cb:
                        pairs.append((math.log(max(g["rank"], 1)),
                                      math.log(max(cb[uid]["rank"], 1)) - math.log(max(g["rank"], 1)),
                                      dt))
            top10b = sorted(cb.values(), key=lambda g: g["rank"])[:10]
            if top10b:
                newcomer.append((dt, sum(1 for g in top10b if g["universe_id"] not in ca) / len(top10b)))

    # --- drift: d = (a + b*x0) * dt, least squares, dt<=30h pairs ---
    a, b = PRIOR_DRIFT_A, PRIOR_DRIFT_B
    if len(pairs) >= 200:
        # regress d/dt on x0
        sx = sy = sxx = sxy = n = 0.0
        for x0, d, dt in pairs:
            y = d / dt
            sx += x0; sy += y; sxx += x0 * x0; sxy += x0 * y; n += 1
        den = n * sxx - sx * sx
        if den > 0:
            bb = (n * sxy - sx * sy) / den
            aa = (sy - bb * sx) / n
            # sanity clamps: drift at the top is positive (falling), reversion is mild
            a = min(max(aa, 0.0), 0.05)
            b = min(max(bb, -0.02), 0.005)
    kappa = max(-b, 0.002)
    xbar = -a / b if b < -0.0005 else 3.7

    # --- diffusion: sigma(x0) = s0 * (0.196 - 0.030*x0) / 0.115, s0 from short pairs ---
    s0 = PRIOR_SIGMA0
    res = []
    for x0, d, dt in pairs:
        if dt <= 12.0:
            pred = (a + b * x0) * dt
            res.append(((d - pred) ** 2) / dt)
    if len(res) >= 200:
        var = sum(res) / len(res)
        if var > 0:
            s0 = min(max(math.sqrt(var), 0.05), 0.25)

    # --- newcomer rate: share = c * dt^2, through the origin ---
    c = 4.08e-5
    num = den2 = 0.0
    for dt, share in newcomer:
        num += share * dt * dt
        den2 += dt ** 4
    if den2 > 0 and len(newcomer) >= 20:
        cc = num / den2
        if 0 < cc < 5e-4:
            c = cc
    return a, b, kappa, xbar, s0, c


def sigma_of(x0, s0):
    """Rank-dependent diffusion per sqrt(hour): the top churns hardest."""
    g = (0.196 - 0.030 * x0) / 0.115
    return s0 * min(max(g, 0.5), 1.8)


def recent_slope(uid, reads, charts, times, cut_time):
    """Shrunk recent log-rank velocity (per hour) from the game's last reads."""
    pts = []
    for idx in range(len(reads) - 1, -1, -1):
        if times[idx] > cut_time:
            continue
        g = charts[idx].get(uid)
        if g is None:
            continue
        pts.append((times[idx], math.log(max(g["rank"], 1))))
        if len(pts) >= 4 or (pts and (cut_time - pts[-1][0]).total_seconds() > 48 * 3600):
            break
    if len(pts) < 2:
        return 0.0
    t0 = pts[-1][0]
    xs = [(t - t0).total_seconds() / 3600.0 for t, _ in reversed(pts)]
    ys = [y for _, y in reversed(pts)]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    # shrink: only ~20-25% of the measured slope persists (measured slope coef ~0.2)
    return min(max(0.22 * slope, -0.02), 0.02)


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, reads = load_input(input_dir)
    places = int(task.get("places", 10))
    window = task["window"]
    wfrom = parse_time(window["from"])
    wuntil = parse_time(window["until"])
    slots = int(window.get("slots", 72))
    slot_minutes = int(window.get("slot_minutes", 60))
    cut_time = parse_time(task.get("history_until", window["from"]))
    if not reads:
        raise SystemExit("no reads in history.ndjson")

    a, b, kappa, xbar, s0, newcom_c = fit_params(reads)
    charts = [chart_of(r) for r in reads]
    times = [parse_time(r["at"]) for r in reads]

    last = charts[-1]
    last_time = times[-1]
    # candidate games: the chart as it stood at the last read
    cands = []
    for uid, g in last.items():
        x0 = math.log(max(g["rank"], 1))
        v = recent_slope(uid, reads, charts, times, last_time)
        # exit time from details when known (created + 28 days).  The
        # details field's exact shape is not pinned down by the kit
        # (a list of per-game objects, or a dict keyed by universe id),
        # so look it up defensively and give up silently if it is odd.
        exit_t = None
        det = None
        for r in reversed(reads):
            det = find_detail(r.get("details"), uid)
            if det:
                break
        if det:
            try:
                created = det.get("created")
                if created:
                    exit_t = parse_time(created) + timedelta(days=MAX_AGE_DAYS)
            except Exception:
                exit_t = None
        cands.append({
            "uid": uid, "x0": x0, "v": v, "sig": sigma_of(x0, s0), "exit": exit_t,
        })
    n_cand = len(cands)
    if n_cand == 0:
        raise SystemExit("empty chart at last read")

    # slot times (hours after window start), evenly spaced, N_SLOTS of them
    total_hours = (wuntil - wfrom).total_seconds() / 3600.0
    slot_hs = [(i + 0.5) * total_hours / N_SLOTS for i in range(N_SLOTS)]
    # hours after the last read for each slot time
    base_h = (wfrom - last_time).total_seconds() / 3600.0

    rng = random.Random(SEED)
    # freq[place][uid] counts, and rank sums for the top10 ordering
    freq = [dict() for _ in range(places)]
    rank_sum = {c["uid"]: 0.0 for c in cands}
    rank_cnt = {c["uid"]: 0 for c in cands}
    n_sims = 0
    for sh in slot_hs:
        t = max(base_h + sh, 0.0)  # hours after the last read
        decay = math.exp(-KAPPA_SCALE * kappa * t)
        noise_sd_factor = math.sqrt(max((1.0 - decay * decay) / (2.0 * kappa), 0.0)) if kappa > 0 else math.sqrt(t)
        # note: (1-exp(-2kt))/(2k) == sqrt-variance factor for OU; at small kt -> t
        for _ in range(N_DRAWS):
            scored = []
            for c in cands:
                if c["exit"] is not None and (last_time + timedelta(hours=t)) >= c["exit"]:
                    continue
                mean = xbar + (c["x0"] - xbar) * decay + c["v"] * t
                x = mean + c["sig"] * SIGMA_SCALE * noise_sd_factor * rng.gauss(0.0, 1.0)
                scored.append((x, c["uid"]))
            scored.sort(key=lambda p: p[0])
            for place in range(places):
                if place < len(scored):
                    uid = scored[place][1]
                    freq[place][uid] = freq[place].get(uid, 0) + 1
            for pos, (_, uid) in enumerate(scored[:places]):
                rank_sum[uid] += pos + 1
                rank_cnt[uid] += 1
            n_sims += 1

    # window-average newcomer share: mean over slots of c * t^2, tempered up
    # (the reserve is also where the model's overconfidence is corrected)
    mean_t2 = sum((base_h + sh) ** 2 for sh in slot_hs) / len(slot_hs)
    reserve_base = min(max(RESERVE_SCALE * newcom_c * mean_t2, 0.02), 0.40)

    probabilities = []
    for place in range(places):
        # mild ramp: newcomers take lower places more often than #1
        reserve = min(max(reserve_base * (0.7 + 0.06 * (place + 1)), 0.01), 0.30)
        scale = (1.0 - reserve)
        rows = []
        for uid, cnt in freq[place].items():
            p = (cnt / n_sims) * scale
            if p >= 0.001:
                rows.append((uid, p))
        rows.sort(key=lambda r: (-r[1], r[0]))
        rows = rows[:100]
        s = sum(p for _, p in rows)
        if s > scale:  # rounding safety; rescale down
            rows = [(uid, p * scale / s) for uid, p in rows]
        for uid, p in rows:
            probabilities.append({"universe_id": int(uid), "place": place + 1,
                                  "p": round(float(p), 6)})
    # top10: lowest mean simulated rank
    exp_rank = []
    for c in cands:
        uid = c["uid"]
        if rank_cnt[uid]:
            exp_rank.append((rank_sum[uid] / rank_cnt[uid], uid))
    exp_rank.sort()
    top10 = [int(uid) for _, uid in exp_rank[:places]]
    # pad (should not happen) with best remaining candidates
    if len(top10) < places:
        seen = set(top10)
        for c in cands:
            if c["uid"] not in seen:
                top10.append(int(c["uid"]))
                if len(top10) == places:
                    break

    out = {"probabilities": probabilities, "top10": top10}
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")
    print(f"wrote {out_path}: {len(probabilities)} prob rows, top10={top10}")
    print(f"fitted: drift a={a:.4f} b={b:.5f} kappa={kappa:.5f} xbar={xbar:.2f} "
          f"sigma0={s0:.3f} newcomer_c={newcom_c:.2e} reserve_base={reserve_base:.3f}")


if __name__ == "__main__":
    main()
