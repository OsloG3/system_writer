# System writer

A website to build Bridge bidding systems and practice.

## Play vs bots (`/play`)

Sit South at a table with three bots from the sibling `bidding-dt` project
(your partner North is a bot too). When the auction ends, the board is scored
in IMPs versus double-dummy par, and your per-account history (boards, total
and average IMPs/board) is kept in `data/play_stats.json`.

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
