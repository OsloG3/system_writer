# System writer

A website to build Bridge bidding systems and practice.

## Play vs bots (`/play`)

Two ways to play, both scored in IMPs versus double-dummy par:

- **Solo** — you sit South with a bidding-dt bot in every other seat. Your
  per-account history (boards, total and average IMPs/board) is kept in
  `data/play_stats.json`; the recent boards can be reviewed with all four
  hands and the full auction.
- **Partner table** — two humans (host South, partner North) against the
  East/West bots. Tables are persistent: games are stored in
  `data/game_<id>.json`, keep running while both players are offline and
  survive server restarts. Open as many tables as you like with the same
  partner; the partner joins with the 6-character code from the host (or a
  `/play?join=CODE` link) and both players review past boards (cards and
  auction) from the table's history.

The bots are served by a small Python sidecar; start it first:

```bash
cd ../bidding-dt
uv run python -m bidding_dt.play_server --ckpt runs/tiny/best.pt --port 8081
```

Then run the site as usual (`go run .`) and open http://localhost:8080/play.

Environment:

- `BIDDING_DT_URL` — sidecar URL (default `http://127.0.0.1:8081`)
- `PLAY_BOTS` — optional per-seat model names for multi-model sidecars,
  e.g. `PLAY_BOTS="N=tiny,E=rl,W=rl"`
- `ATHENEUM_DATA_DIR` — where accounts/trees/play stats live (default `./data`)
