# bidding-dt

Bridge bidding bot: a small decision-transformer-style model that sees one
hand plus the auction so far and predicts the next call. Supervised
(behavior-cloning) pretraining on ~970k human auctions, then self-play PPO
against a two-table double-dummy team reward (see "Self-play RL" below).

## Data format (`training.txt`)

Two lines per deal:

```
AT7.KT943.A42.K8 J6543.75.K8.AQJT K982.AQ6.T73.952 Q.J82.QJ965.7643
N E-W 1H P 2H P P P
```

- line 1: four hands in **N,E,S,W** order, each `spades.hearts.diamonds.clubs`
- line 2: `dealer vuln call...` with vuln in `None|N-S|E-W|Both`, auction ends `P P P`

## Design

- **Rank equivalence**: all cards below the 9 are equivalent. A hand is a
  32-dim vector: 4 suits x 7 rank classes {A,K,Q,J,T,9,low} counts + suit
  lengths (a sufficient statistic), encoded by a small MLP into a HAND token.
- **Hero rotation**: the auction is rotated so the hero always acts at call
  slots `s % 4 == 0` (0=hero, 1=RHO, 2=partner, 3=LHO). When the hero is not
  the dealer, the frame is front-padded with `(4-o)%4` PAD tokens,
  `o = (hero-dealer)%4`.
- **Sequence**: `[COND, HAND, pad*, calls...]`, max length 29. Learned
  token + absolute-position + role embeddings; vuln (hero frame:
  none/we/they/both) folds into the COND embedding.
- **Targets**: causal next-token cross-entropy at every hero decision slot
  (all four seats of every deal are training views -> ~10M supervised
  decisions). One forward pass supervises all of a hero's decisions.
- **COND token**: return-conditioning placeholder for the offline-RL phase;
  trained as a learned "unconditional" embedding for now. Bridge reward is
  terminal, so a single leading conditioning token is DT-compatible.
- **Actions**: 39 tokens = PAD + P + X + XX + 35 bids ordered by
  (level, CDHSN), so legality is `bid_id > highest_bid_id`. Inference applies
  an exact legality mask (`data/legal.py`, handles balancing doubles etc.).
- **Model**: pre-norm transformer, multi-head self-attention (SDPA, causal),
  SwiGLU FFN. Presets: `tiny` ~0.8M, `small` ~2.2M (default), `base` ~6.4M,
  `large` ~18M params. Trailing PADs cannot affect earlier positions under
  causal attention, so no padding mask is needed.

## Setup

```bash
uv sync --extra cpu --extra dev --extra rl   # this machine (no GPU)
uv sync --extra cuda --extra dev --extra rl  # GPU machine
```

The `rl` extra pulls in [endplay](https://github.com/ThorvaldAagaard/endplay)
(bundles the libdds double-dummy solver); it is imported lazily, so the BC
pipeline works without it. The `plot` extra adds matplotlib for
`bidding_dt.rl.plot` (already present via endplay).

## Usage

```bash
# 1) build the binary cache (~2-3 min, ~150 MB in cache/)
uv run python -m bidding_dt.data.parse --input training.txt --cache cache

# 2) train (GPU box; bf16 autocast is automatic on CUDA)
uv run python -m bidding_dt.train --config configs/small.yaml --out runs/small

# 2') CPU smoke run
uv run python -m bidding_dt.train --preset tiny --max-deals 20000 \
    --batch-size 128 --workers 2 --epochs 1 --out runs/smoke

# 3) evaluate on the test split (top-1/top-3 + breakdowns)
uv run python -m bidding_dt.eval --ckpt runs/small/best.pt

# 4) ask the model for a bid
uv run python -m bidding_dt.bid --ckpt runs/small/best.pt \
    --hand "AT7.KT943.A42.K8" --dealer N --vuln N-S --hero N --auction "P 1D"
# interactive: add --repl   (commands: <calls>, pred, auto, undo, show, quit)
# spot-check vs real auctions:
uv run python -m bidding_dt.bid --ckpt runs/small/best.pt \
    --check-file training.txt --check-deals 20

# 5) serve bots over HTTP for the system_writer "play vs bots" page:
#    /deal /legal /bid /score (JSON; par scoring via the DD solver, tables
#    cached in cache/dd_play.sqlite). Repeat --ckpt to load several named
#    models ([name=]path); the first is the default.
uv run python -m bidding_dt.play_server --ckpt runs/tiny/best.pt --port 8081

# tests
uv run pytest
```

Checkpoints (`runs/*/last.pt|best.pt`) carry the full config; `--resume last`
continues a run. Logs are JSON lines in `runs/*/log.jsonl`.

## Self-play RL (PPO, two-table team reward)

Phase 2: the BC checkpoint is warm-started into an actor-critic (`RLModel` =
BiddingDT trunk + value head) and improved by team-style self-play. Every deal
is played at **two tables** with the same dealer/vuln -- a duplicate team match:

- **league iteration**: the learning team sits N-S at table 0 and E-W at
  table 1; a frozen league opponent holds the other seats at both tables.
  The league pools past snapshots of the learner plus any external frozen
  policies -- other RL runs, supervised (BC) models of any architecture, or
  the random baseline (`--league rl:runs/old/best.pt`,
  `--league bc:runs/small/best.pt`, or `rl.league_members` in the config);
- **self-play iteration**: the learner acts for all four seats at both tables.
  Rollout sampling at temperature `rollout_temp` (>1 flattens the policy)
  supplies enough variance that the two auctions usually diverge -- watch
  `diverge_rate` in the log.

Reward per decision (terminal, IMPs, from the deciding side's view):

```
r_row = reward_scale * ( team_weight * 1[call >= first_diff] * IMPs( score_NS(t0) - score_NS(t1) )
                       + par_weight  * IMPs( score(final contract, DD tricks) - par_score(deal) ) )
```

The **team term** is the IMP swing between the two tables (scores estimated
double-dummy), and it only *trickles down* to calls at/after the first point
where the two auctions differ: if the first five calls were the same at both
tables they contributed nothing to the swing, so only the 6th call onward is
credited. The **par term** is added for every call at one fourth of the
per-IMP team weight (`par_weight: 0.25`, `team_weight: 1.0`). All scores come
from the same 52 cards (libdds via `endplay`, tables cached in sqlite and
deduped across the two tables), so card-lie luck cancels: the team term is
pure within-deal bidding skill, the par term anchors across deals versus
double-dummy-optimal bidding. This pairing also fixes the pathology of a
par-only reward in self-play, where the shared policy sits on both sides of a
zero-sum signal that mostly cancels in the gradient (and can be farmed by
passing deals out).

New agreements emerge because both seats of a partnership share one policy
that only ever sees its own hand + the auction: any coordination must be
carried by the calls themselves. A KL anchor to the frozen BC policy plus an
entropy bonus are annealed to zero, so training starts near human systems and
is free to diverge as it finds better protocols. Opponents are drawn from the
league (past snapshots + external members) to prevent strategy cycling; with
`pfsp_alpha > 0` sampling is tilted toward members the learner is currently
losing to (per-member winrates are tracked from the team swings).

```bash
# benchmark the DD solver first -- it sizes the rollout budget (~5 deals/s
# cold on this 2-core box, tables cached in cache/dd.sqlite)
uv run python -m bidding_dt.dd.bench --deals 200 --cache cache/dd.sqlite

# CPU smoke run (few minutes)
uv run python -m bidding_dt.rl.train_ppo --preset tiny \
    --bc-ckpt runs/smoke/best.pt --config configs/rl_smoke.yaml \
    --out runs/rl_smoke

# GPU training run
uv run python -m bidding_dt.rl.train_ppo --config configs/rl_small.yaml \
    --preset small --bc-ckpt runs/small/best.pt --out runs/rl_small

# further-train an existing RL model against a league of other models:
# --bc-ckpt accepts BC *or* RL checkpoints (auto-detected; the checkpoint's
# architecture is adopted), --league adds frozen opponents (repeatable;
# specs: random | bc:<ckpt> | rl:<ckpt> | <ckpt>)
uv run python -m bidding_dt.rl.train_ppo --config configs/rl_small.yaml \
    --bc-ckpt runs/rl_small/best.pt --out runs/rl_small2 \
    --league bc:runs/small/best.pt --league rl:runs/rl_tiny/best.pt \
    --iters 4000

# resume an interrupted run: --resume last reloads runs/rl_small/last.pt
# (model + optimizer + LR schedule + iteration + best-score) and continues
# from the next iteration; pass a path instead of 'last' to resume from a
# specific checkpoint. Keep --out pointed at the same dir. Training runs to
# rl.iters, so raise --iters to train further than the original target.
# The league (snapshots, external members, per-member winrate stats) is
# persisted under <out>/league and restored on resume (rl.league_persist).
uv run python -m bidding_dt.rl.train_ppo --config configs/rl_small.yaml \
    --preset small --bc-ckpt runs/small/best.pt --out runs/rl_small \
    --resume last --iters 4000

# running plot
uv run python -m bidding_dt.rl.plot runs/rl_small

# strength: team IMPs/board vs fixed opponents (random | self | bc:<path> |
# rl:<path> | <path> -- BC vs RL auto-detected). Each deal is played at TWO
# tables (a duplicate team match):
# the learner sits N-S at one table and E-W at the other, the opponent holds
# the other seats, and the per-board score is the DD IMP swing between the
# tables -- card-lie luck cancels, so no layout averaging is needed.
uv run python -m bidding_dt.rl.eval_rl --ckpt runs/rl_small/best.pt \
    --opp bc:runs/small/best.pt --deals 4000

# emergent-system report: opening/response frequencies per HCP bucket,
# RL vs BC vs human corpus, with total-variation distances
uv run python -m bidding_dt.rl.analyze --ckpt runs/rl_small/best.pt \
    --bc runs/small/best.pt --hands 20000 --out runs/rl_small/system.json

# progress plots from log.jsonl: eval IMPs (+-1 SE), train reward, losses,
# entropy, KL/clipfrac, auction behaviour, schedules, iter cost; several
# runs overlay for comparison. A text summary prints for headless use.
uv run python -m bidding_dt.rl.plot runs/rl_small            # -> runs/rl_small/progress.png
uv run python -m bidding_dt.rl.plot runs/a runs/b --out cmp.png --smooth 25

# bid.py probes RL checkpoints too (trunk is loaded from the dt.* keys)
uv run python -m bidding_dt.bid --ckpt runs/rl_small/best.pt --repl ...
```

In-loop eval (`log.jsonl`): `imps_vs_random`, `imps_vs_bc` are team IMPs/board
on fixed deals -- each deal played at two tables (learner N-S at one, E-W at
the other) against the opponent, scored as the DD swing between tables;
`best.pt` tracks `imps_vs_bc` (or random if no BC anchor). Diagnostics per
iteration: `reward_mean/std`, `team_imps`
(table-0 N-S view of the swing; = learner-team view in league iterations),
`imps_vs_par`, `diverge_rate` and `first_diff_call` (self-play variance
health), `unique_deals` (fresh hands per iteration -- random deals are drawn
anew every iteration, never re-used), passout rate, `league_member` and
`member_winrate` (which league opponent was faced and the learner's EWMA
winrate vs it), KL-to-BC, entropy,
clipfrac. Expect human top-1 agreement to *drop* as conventions diverge --
that is the point; judge by IMPs and the analyze report.
`bidding_dt.rl.plot` renders any run's log.jsonl into an eight-panel
progress figure (and prints a one-line summary for headless use).

## Layout

```
src/bidding_dt/
  config.py            # dataclasses, YAML configs, size presets
  data/vocab.py        # 39-token action space, seat/vuln maps
  data/hands.py        # hand -> 32-dim rank-class encoding
  data/parse.py        # training.txt -> cache/*.npy + splits
  data/dataset.py      # hero-rotated views, targets, collate
  data/legal.py        # exact legality mask (P/X/XX/bids)
  model/transformer.py # MHA + SwiGLU decision transformer (+ hidden())
  dd/scoring.py        # duplicate score + IMP tables (pure python)
  dd/solver.py         # endplay/libdds wrapper, batch solving, sqlite cache
  dd/reward.py         # auction -> contract -> par-diff reward (IMPs)
  dd/bench.py          # solver throughput benchmark
  env/deals.py         # random 52-card deals + vectorized encoding
  env/auction_env.py   # batched auction env (hero-rotated obs, legal masks)
  rl/config.py         # RLConfig / RLTrainConfig (YAML)
  rl/model.py          # RLModel: trunk + value head, BC/RL ckpt loading
  rl/league.py         # opponent league: snapshots + external BC/RL members,
                       # PFSP sampling, disk persistence across resumes
  rl/rollout.py        # two-table team rollouts, gated team/par rewards
  rl/ppo.py            # clipped PPO + value MSE + entropy + KL(pi||pi_BC)
  rl/train_ppo.py      # training entrypoint
  rl/eval_rl.py        # two-table team IMPs/board vs fixed opponents
  rl/analyze.py        # emergent bidding-system extraction vs BC/humans
  rl/plot.py           # log.jsonl -> progress.png, multi-run overlay
  train.py eval.py bid.py
  play_server.py         # HTTP sidecar: bots + par scoring for the website
```

## RL design notes

- **Trajectories** are per-(table-deal, seat) decision chains in the same
  hero-rotated token format as BC; the two tables of a deal are separate
  chains that share the terminal team swing (gated to post-divergence calls).
  gamma=1 (short auctions, terminal reward), GAE lambda=0.95, rewards scaled
  x0.2. Rollout temperature is applied consistently in the PPO logprob
  recompute, so importance ratios stay valid.
- **PPO details audited**: the KL anchor is a properly masked KL over legal
  calls (renormalized on both sides -- an unmasked softmax leaks probability
  onto illegal calls and can report negative "KL"); the uniform baseline
  opponent always samples (greedy argmax over flat logits degenerates to an
  always-pass bot); training deals are freshly sampled every iteration and
  `unique_deals` in the log verifies no re-use.
- **Scoring** is an independent pure-python implementation (standard duplicate
  tables, exhaustively cross-checked vs endplay except for endplay's
  non-standard vul doubled/redoubled undertricks).
- **DDS throughput** is the bottleneck: ~0.2 s/deal per 2 cores here; solve in
  batches (`calc_all_tables`, <=40/call), cache tables by deal, and put DDS on
  CPU workers next to a GPU trainer. If it still bottlenecks at scale, the
  next step is a learned DD-table surrogate calibrated on cached solves.
- **Rank equivalence caveat**: the policy cannot see cards below the 9, but
  rewards use the true 52 cards -- fine for bidding, just don't expect the
  policy to exploit low-card nuance.
- The COND token remains the offline-RL/RTG slot: an alternative route is
  return-conditioned DT on par-diff-labelled human auctions; the PPO route
  above is what `rl/` implements (COND is simply unused there).
