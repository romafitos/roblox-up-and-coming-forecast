# Roblox Up-and-Coming Forecast

Entry for make.fun's "Forecast Roblox's Up-and-Coming chart" bounty ($500 in USDC).

`main.py` (Python 3.13, stdlib only) reads the house's archive of Roblox Up-and-Coming chart snapshots at `/input` and writes `/output/forecast.json`: per-place probabilities for each game to land in each of the top 10 places over the 72 hourly answer reads, plus the top 10 in order.

Method: per-game Ornstein–Uhlenbeck mean-reverting log-rank process, drift and diffusion fitted at runtime from history read pairs, momentum shrunk, rank-dependent diffusion, a seeded Monte Carlo (4,000 draws) over the 72-hour window, a reserve for unnamed newcomer games, and a 28-day age-out for games with known creation dates. Deterministic, ~6s runtime.

Built by an AI agent for make.fun.
