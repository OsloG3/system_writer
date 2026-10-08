# System writer

A website to build Bridge bidding systems and practice.

The menu has three sections: **Collections** (`/`, build/edit your bidding
systems), **Bid only** (`/play?m=bid`) and **Play bridge** (`/play?m=play`).

## Bid only (`/play?m=bid`)

Bidding practice, scored in IMPs versus double-dummy par — the cards are **not**
played out. Two tabs:

- **With a bot partner** — you sit South with a bidding-dt bot as your North
  partner against two bot opponents (East/West). Bid the auction; each board is
  scored against double-dummy par.
- **With a partner** — two humans (host South, partner North) against the
  East/West bots. Open a table by naming your partner's account; it appears in
  both players' table lists, so no invite link is needed. Tables are persistent:
  games are stored in `data/game/<id>.json`, keep running while both players are
  offline and survive server restarts. Open as many tables as you like; both
  players review past boards (cards and auction) from the table's history.

## Play bridge (`/play?m=play`)

The full game with bots: you sit South with a bidding-dt bot as your North
partner against two bot opponents. After the auction the whole hand is **played
out card by card** — you click your cards (and dummy's when you declare), while
a card-play bot fills the other seats. The bot picks each card by double-dummy
simulation over a pool of hidden-hand candidates consistent with the auction and
the play so far (see `bidding-dt/src/bidding_dt/play_bot.py`). The board is then
scored on the tricks actually taken versus double-dummy par. Your per-account
history (boards, total and average IMPs/board) is kept in
`data/game/play_stats.json`; recent boards can be reviewed with all four hands,
the full auction and the play.

Both solo sections send `{"mode":"bid"|"play"}` to `POST /api/play/new`; the
board keeps that mode for its whole life.

The bots are served by a small Python sidecar; start it first (in the
`bidding-dt` checkout, sibling or nested — torch and endplay come from the
`cpu`/`cuda` and `rl` extras, so repeat the extras on the command):

```bash
uv run --extra cpu --extra rl python -m bidding_dt.play_server \
    --ckpt runs/rl_tinyt/best.pt --port 8081
# GPU box: --extra cuda instead of --extra cpu.
# runs/ is gitignored, so copy a checkpoint (e.g. runs/rl_tinyt/best.pt)
# to the server.
#
# Card play: add --hand-ckpt runs/hand_small/best.pt to let the play bot read
# the auction with the VQ hand-inference model when it builds its hidden-hand
# pool. Without it the pool is dealt by a constraint-based sampler that still
# honours the auction exclusions, show-outs and the honour-lead rule. Tune the
# pool with --decl-pool/--def-pool (candidate deals per bot) and --min-floor.
```

Then run the site as usual (`go run .`) and open http://localhost:8080/play.

Environment:

- `BIDDING_DT_URL` — sidecar URL (default `http://127.0.0.1:8081`)
- `PLAY_BOTS` — optional per-seat model names for multi-model sidecars,
  e.g. `PLAY_BOTS="N=tiny,E=rl,W=rl"`
- `ATHENEUM_DATA_DIR` — where accounts/trees/play stats live (default `./data`)
