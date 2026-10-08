# bidding-dt

Bridge bidding bot: a small decision-transformer-style model that sees one
hand plus the auction so far and predicts the next call. Supervised
(behavior-cloning) pretraining on ~970k human auctions, then self-play PPO
against a two-table double-dummy team reward (see "Self-play RL" below).
A third model inverts the problem -- from the public auction alone it *samples*
the 13 cards a player can hold (see "Hand inference" below) -- and a fourth
bids from a fixed-length state (auction summary + those samples' encodings)
with an FFN whose encoders are evolved (see "Bidding systems" below).

## Data format (`training.txt`)

Two lines per deal:

```
AT7.KT943.A42.K8 J6543.75.K8.AQJT K982.AQ6.T73.952 Q.J82.QJ965.7643
N E-W 1H P 2H P P P
```

- line 1: four hands in **N,E,S,W** order, each `spades.hearts.diamonds.clubs`
- line 2: `dealer vuln call...` with vuln in `None|N-S|E-W|Both`, auction ends `P P P`

## Design

- **Hand encoding**: a hand is its 52-dim 0/1 card-indicator vector -- every
  rank gets its own slot per suit (suit order S,H,D,C; no rank folding), so
  the encoding is lossless and suit lengths/HCP are derivable from it. A
  small MLP encodes the hero's vector into the single HAND token
  (`data/hands.py`).
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

`uv run` re-syncs the environment on every invocation: running it without the
extras you installed removes them again. Either repeat the extras on the
command (`uv run --extra cpu --extra rl python -m ...`) or activate the venv
first (`source .venv/bin/activate`), which keeps everything installed.

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
#    Card play adds /play/choose /play/result /play/drop: /play/choose picks a
#    bot's card by double-dummy simulation over a pool of hidden-hand candidates
#    consistent with the auction and the play so far (see play_bot.py).
#    Needs torch (cpu/cuda extra) and endplay (rl extra), so repeat the
#    extras here. `runs/` is gitignored: copy a checkpoint to the server.
uv run --extra cpu --extra rl python -m bidding_dt.play_server \
    --ckpt runs/rl_tinyt/best.pt --port 8081
# on a GPU box: --extra cuda instead of --extra cpu
#
# Card-play bot options:
#   --hand-ckpt runs/hand_small/best.pt   read the auction with the VQ hand
#                                         model when building the hidden-hand
#                                         pool (without it a constraint-based
#                                         sampler deals the pool instead)
#   --decl-pool N / --def-pool N          candidate deals the declarer-side /
#                                         a defender bot simulates per decision
#                                         (the declarer pool should be larger)
#   --min-floor N                         the pool minimum never drops below N
#                                         (the minimum shrinks as the hand
#                                         advances; survivors are reused)

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

Both seats of a partnership sample at `rollout_temp` and learn, so a row can
also be punished for its *partner's* exploration noise: when `r_row` is
negative it is scaled by `exp(-D)` in (0, 1], where D is the partner's mean
per-call deviation from greedy play (`logp(greedy call) - logp(call taken)`
under the untempered policy, averaged over the partner's calls in that
auction). A partner that bid exactly as the greedy policy would passes the
full punishment on (D = 0); increasingly unlikely sampled calls shrink it
toward zero. Positive rewards are never scaled and the per-call average keeps
the discount independent of auction length.

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

# multi-core CPU run: cap torch threads so the background DD presolve +
# libdds keep cores, and enable bf16 autocast on AVX512-BF16/AMX CPUs
# (see "PPO on multi-core CPUs" in the RL design notes)
uv run python -m bidding_dt.rl.train_ppo --config configs/rl_small.yaml \
    --preset small --bc-ckpt runs/small/best.pt --device cpu \
    --threads 2 --cpu-bf16 --out runs/rl_small_cpu

# GPU training run
uv run python -m bidding_dt.rl.train_ppo --config configs/rl_small.yaml \
    --preset small --bc-ckpt runs/small/best.pt --out runs/rl_small

# bigger GPU run: the base preset (~6.4M params) with configs/rl_base.yaml,
# which scales up deals_per_iter/minibatch and lowers the LR for the larger
# trunk (warm-start from the matching BC base checkpoint)
uv run python -m bidding_dt.rl.train_ppo --config configs/rl_base.yaml \
    --preset base --bc-ckpt runs/base/best.pt --out runs/rl_base

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

# emergent-system report: the checkpoint bids against itself (all four
# seats, sampled at --temp or deterministic with --greedy) and every auction
# that happens is merged into one prefix tree; each node records how often
# the bot made that bid in that sequence and the min/max HCP and min/max
# length per suit of the hands it held doing it
uv run python -m bidding_dt.rl.analyze --ckpt runs/rl_small/best.pt \
    --deals 4000 --out runs/rl_small/tree.json

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

## Hand inference (VQ encoder-decoder)

`bidding_dt.hand` guesses **one player's 13 cards** from the public information
only: the auction so far (PAD-padded, hero-rotated to the target seat, exactly
the supervised frame) and the vulnerability. Encoder-decoder with a discrete
bottleneck:

- **Encoder**: transformer over `[COND(vuln), KNOWN(excluded cards), calls...]`.
  Trailing batch padding is masked out; the leading frame PADs are *not* padding
  -- their role embedding is what says where the target sits versus the dealer.
- **Bottleneck**: `n_codes` (k=16) learned query slots cross-attend the encoder
  memory and each is vector-quantized against a codebook of `codebook_size`
  (1024) entries, so z is 16 discrete codes (straight-through estimator +
  commitment loss). Codes that keep losing are reseeded from live encoder
  outputs every `revive_every` steps -- a k x V product codebook collapses onto
  a handful of codes otherwise -- and the log's `ppl` is the code-usage
  perplexity to watch that.
- **Decoder**: sees **only** z (never the auction) and draws the 13 cards
  autoregressively in ascending card-id order. Each step is a softmax over the
  52 cards with three things removed: cards already drawn, cards so high that
  the rest of the hand could not fit above them, and cards the caller excluded.
  Every sample is therefore a legal 13-card hand by construction, and the model
  can never hand back an *average* hand.

The discrete bottleneck plus sampling is the point. A multi-2D promises long
hearts **or** long spades; the mean of those two hands is a hand nobody holds.
The codes let the encoder commit to one reading and the decoder samples a whole
hand from it, so repeated draws give the plausible alternatives instead of a
blur. `--codes` additionally samples the codes themselves -- from the near-tied
alternatives, at a scale-free temperature (`--code-temp`: the runner-up is drawn
exp(-1/temp) as often as the winner) -- which widens the spread between reads,
and `train_code_temp > 0` trains the decoder on non-argmin codes as well.

**Masking** (`--mask "AT7.KT943.A42.K8"`, repeatable) removes cards the caller
knows the target cannot hold -- your own hand, and dummy's once it is tabled.
The mask is both a hard decoder constraint *and* an encoder input (the KNOWN
token: knowing 13 dead cards is real information), and training attaches a
random 13-card mask to a fraction of the rows (`train.mask_aug`) so masked
inference is in distribution.

**Training data is generated by the other models** (`hand/gen.py`): random deals
are rolled out through any mix of bidding checkpoints -- BC, RL, the uniform
baseline -- and the resulting (deal, auction) pairs become the training set, so
the guesser learns the conventions those models actually play, including
whatever an RL run invented for itself (`rl/analyze.py` shows how far that
drifts from human bidding). Generation is rollouts only: no DD solving, no
rewards, no endplay. Stores are written in the `cache/` layout, so generated and
human data (`--source cache`) mix freely, and each view samples an auction
*prefix* -- the model works mid-auction, not just on finished ones.

```bash
# generate a reusable store (policies: random | bc:<ckpt> | rl:<ckpt> | <ckpt>,
# optional @weight; N-S and E-W draw independently, so tables mix systems)
uv run python -m bidding_dt.hand.gen --out cache/gen --deals 400000 \
    --policy bc:runs/small/best.pt --policy rl:runs/rl_small/best.pt \
    --policy random@0.1

# train: generates into <out>/gen (unless --gen-deals 0 / --reuse-gen), then
# trains the VQ encoder-decoder on it
uv run python -m bidding_dt.hand.train --preset small --config configs/hand_small.yaml \
    --policy rl:runs/rl_small/best.pt --policy bc:runs/small/best.pt \
    --source cache --out runs/hand_small

# CPU smoke run (a few minutes)
uv run python -m bidding_dt.hand.train --preset tiny --config configs/hand_smoke.yaml \
    --policy bc:runs/smoke/best.pt --out runs/hand_smoke

# ask for hands: what can E hold after "P 1D X 2H", given my own 13 cards?
uv run python -m bidding_dt.hand.sample --ckpt runs/hand_small/best.pt \
    --dealer N --vuln N-S --target E --auction "P 1D X 2H" \
    --mask "AT7.KT943.A42.K8" --n 8 --codes

# spot-check: sampled vs actual hands on a store's val split
uv run python -m bidding_dt.hand.sample --ckpt runs/hand_small/best.pt \
    --source cache/gen --show 5
```

Val metrics in `runs/*/log.jsonl` (`best.pt` tracks `ce`): `ce` teacher-forced
card cross-entropy, `ppl` code-usage perplexity, `acc` greedy card recovery,
`shape` exact suit-shape match rate, `hcp` HCP MAE, `sample_acc`/`best_acc` mean
and best-of-n card recovery over `val_samples` draws, `jaccard` mean pairwise
overlap between draws of the same row (low = still exploring, high = collapsed
onto one hand) and `shapes` distinct suit shapes per row. Train logs also carry
`vq`/`commit` and `revived` (codes reseeded that interval).

Reference points for `ce`: an untrained model sits at log(52) = 3.95 nats/card,
and one that has learned *nothing but the 13-of-52 prior* at ~2.09 -- so only
values below that are real auction information, and there is a long way from
there (an auction is worth a few bits, not the 27 nats that specify a hand).
`hcp` under the ~2.9 MAE of a random hand and a climbing `best_acc` are the same
signal from the sampling side. Greedy `acc` is a weak metric by construction:
the per-card argmax sequence is not a typical hand, so judge by `ce` and the
sampled numbers.

## Bidding systems (fixed-length-state FFN + evolved encoders)

`bidding_dt.system` is a second bidder, built on top of the hand-inference
encoder instead of the decision transformer. Its state is a **fixed-length
vector**, so the decision model is a plain FFN (residual SwiGLU blocks):

| block | dims | source |
| --- | --- | --- |
| own hand | 52 | the usual 0/1 card indicator |
| vulnerability | 4 | hero-relative one-hot (none/we/they/both) |
| dealer | 4 | dealer's seat relative to the actor (self/RHO/pd/LHO) |
| last bid: level, strain | 8 + 6 | one-hot, slot 0 = nobody has bid |
| last bid by | 5 | one-hot relative seat (+ "nobody") |
| double/redouble status | 3 | undoubled / doubled / redoubled |
| doubled by, redoubled by | 5 + 5 | one-hot relative seat (+ "nobody") |
| consecutive passes | 4 | 0, 1, 2, 3+ |
| legal actions | 39 | the exact mask (also applied to the logits) |
| `z_self, z_RHO, z_partner, z_LHO` | 4 x `d_z` | the frozen HandVQ encoder, run on the public auction **in each seat's own frame** |

The call sequence is never read: the last-bid summary carries the current
position, and the four z blocks carry the history -- each is the hand-guesser's
compression of the same public auction from one player's point of view, so what
a player's own bidding says about their own hand arrives as `z_self`. The
current seat is the frame origin (role 0) and needs no feature of its own;
`dealer` is that frame's only positional degree of freedom.

Two nested loops (`system/train.py`):

```
inner (per PPO iteration)      sample system i from the population
                               N-S play encoder_i + adapter_i
                               opponents come from the league
                               two-table team rollouts (DD rewards)
                               PPO-update the shared policy [and adapter]

outer (per generation)         evaluate every system on fixed deals
                               keep the elites, replace the weakest with
                               mutated children:  child = parent + sigma*N(0,1)
                               snapshot the strong systems into the league
```

A *system* is (frozen encoder, adapter): the adapter is the small projection
that turns a seat's z into `d_z` state features, and it is the evolvable part
(`evo.sigma`, absolute by default; `sigma_relative: true` makes it a fraction of
each weight's own std). Children inherit their parent's encoder, so encoders
that keep losing evaluations die out with their adapters. The FFN policy is
shared and trained by PPO; `evo.inner_target` decides whether PPO also trains
the adapters (`policy` | `adapter` | `both`). Snapshots are frozen
single-adapter clones added to the existing league, so later generations face
the systems that already worked.

The policy implements the same `forward_last(tokens, hand, vuln, row_len)`
protocol as `rl/model.py` -- the state is rebuilt from those arguments -- so
`rl/rollout.py` (two-table team matches), `rl/league.py` and `rl/ppo.py` are
reused unchanged, and the PPO logprob recompute sees exactly what the rollout
sampler saw. The encoder's code indices are memoized per auction prefix
(`model.z_cache`), which keeps the recompute off the transformer: a PPO
iteration costs ~0.1 s on CPU once the rollout is done.

```bash
# needs a hand-inference encoder first (see above), plus a BC/RL checkpoint as
# the KL anchor and first league member
uv run python -m bidding_dt.system.train --preset small \
    --config configs/system_small.yaml \
    --encoder runs/hand_small/best.pt --bc-ckpt runs/small/best.pt \
    --league bc:runs/small/best.pt --league rl:runs/rl_small/best.pt \
    --out runs/sys_small

# CPU smoke (2 generations x 2 PPO iterations, 3 systems)
uv run --extra cpu --extra rl python -m bidding_dt.system.train --preset tiny \
    --config configs/system_smoke.yaml --encoder runs/hand_smoke/best.pt \
    --bc-ckpt runs/smoke/best.pt --out runs/sys_smoke

# several encoders -> the population starts with one variant per encoder
uv run python -m bidding_dt.system.train --config configs/system_small.yaml \
    --encoder runs/hand_small/best.pt --encoder runs/hand_base/best.pt \
    --bc-ckpt runs/small/best.pt --out runs/sys_multi
```

The FFN starts with no bidding knowledge, so the KL anchor to the frozen BC
policy doubles as the warm start: keep `rl.kl_beta` high at first and anneal it
to zero (`kl_beta_end`) to hand over to the IMP reward. `log.jsonl` events:
`generation` (team IMPs, reward, decisions, which systems were active, PPO
stats, `z_cache` size), `eval` (per-system IMPs/board + the running
`train_imps`), `snapshot` (systems added to the league), `evolve` (which child
replaced which, its parent, generation and relative noise size, plus the full
ranking). `best.pt` tracks the best evaluated system and stores its adapter
index, so it loads straight back with the matching `--encoder`.

## Layout

```
src/bidding_dt/
  config.py            # dataclasses, YAML configs, size presets
  data/vocab.py        # 39-token action space, seat/vuln maps
  data/hands.py        # hand -> 52-dim 0/1 card-indicator encoding
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
  rl/analyze.py        # self-play auction tree: per-bid HCP/suit-length ranges
  rl/plot.py           # log.jsonl -> progress.png, multi-run overlay
  hand/config.py       # HandVQConfig / train / gen / data sections (YAML)
  hand/model.py        # VQ encoder-decoder: auction -> k codes -> a whole hand
  hand/data.py         # auction stores, (deal, seat, prefix) views, collate
  hand/gen.py          # roll the bidding models -> generated training stores
  hand/train.py        # hand-model training entrypoint + val metrics
  hand/sample.py       # sample hands for a seat, with known-cards masking
  system/state.py      # auction summary + 4 z blocks -> fixed-length state
  system/model.py      # SystemPolicy: SwiGLU FFN over that state (+ z adapters)
  system/population.py # system variants, mutation/selection, league snapshots
  system/config.py     # SystemModelConfig / EvoConfig / SystemConfig (YAML)
  system/train.py      # inner PPO loop + outer evolutionary loop
  train.py eval.py bid.py
  play_server.py         # HTTP sidecar: bots + par scoring for the website
  play_bot.py            # card-play bot: DD simulation over auction-consistent
                         # hidden-hand pools (used by /play/choose)
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
  tables, exhaustively cross-checked vs endplay).
- **DDS throughput** is the bottleneck: ~0.2 s/deal per 2 cores here; solve in
  batches (`calc_all_tables`, <=40/call), cache tables by deal, and put DDS on
  CPU workers next to a GPU trainer. `train_ppo` pipelines the solver against
  the network on a background thread (`dd.reward.warm_cache`; libdds releases
  the GIL and runs its own solver threads): iteration k+1's deals are
  presolved while iteration k runs its PPO update, and the solve is joined
  before scoring, so steady-state wall time is ~max(rollout + update, solve)
  instead of the three summed. The fixed eval deals are prewarmed in small
  chunks the same way, so in-loop evals never stall on solving. If it still
  bottlenecks at scale, the next step is a learned DD-table surrogate
  calibrated on cached solves.
- **PPO on multi-core CPUs** (all `rl.*` config keys / CLI flags):
  - `length_bucket` (default 8): PPO minibatches are length-bucketed -- rows
    are sorted by auction length within random mega-batches of
    `length_bucket * minibatch_size` -- and each minibatch's token block is
    truncated to its longest row (trailing PADs cannot affect earlier
    positions under causal attention, so this is numerically a no-op).
    Measured ~1.5x on the update phase (2500-row buffer, small preset);
    0 = purely random minibatches.
  - `threads` / `--threads`: cap torch intra-op threads (e.g. cores/2) so the
    presolve thread and libdds keep cores; torch's default can also thrash on
    the small minibatch matmuls here.
  - `cpu_bf16` / `--cpu-bf16`: bf16 autocast for the CPU rollout + update
    forwards; ~2x matmul throughput on AVX512-BF16/AMX cores (Zen4+, Intel
    Ice Lake+). *Slower* on older CPUs (e.g. ~2.3x slowdown measured on a
    Skylake i5) -- off by default; benchmark before enabling.
  - The DD sqlite cache persists across runs, so re-training over similar
    deal volume gets progressively cheaper; a warm `cache/dd.sqlite` turns
    reward scoring into pure lookups (resumed smoke iters: ~14s cold -> <2s
    warm).
- The COND token remains the offline-RL/RTG slot: an alternative route is
  return-conditioned DT on par-diff-labelled human auctions; the PPO route
  above is what `rl/` implements (COND is simply unused there).
