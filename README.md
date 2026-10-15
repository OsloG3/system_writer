# System writer

A website to build Bridge bidding systems and practice.

The menu has four sections: **Collections** (`/`, build/edit your bidding
systems), **Bid only** (`/play?m=bid`), **Play bridge** (`/play?m=play`) and
**Card problems** (`/drills`, author and practise scripted card-play problems).

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

## Card problems (`/drills`)

Author fully-scripted **card-play problems** and practise the community library.
Every rule (auction legality, follow-suit, trick winners, claim bounds,
equivalent-card soundness) is checked locally in `drills.go`, and problems
persist to `data/drills/<id>.json`. Only the optional **bots-finish** ending
calls the bidding-dt sidecar (`BIDDING_DT_URL`), reusing the same `/play/choose`
card-play bot as Play bridge. Two tabs:

- **Create** — build a problem in a five-step wizard: setup (optional title,
  difficulty 1–5 ★, dealer, vulnerability, and whether it tests **Declarer** or
  **Defense**), the four hands from a click-to-assign deck picker, the auction on
  a bidding box (with a comment about the bidding and a **problem comment** that
  should record the defenders' carding agreements), the hero defender for a
  defense problem, then the **card play**. The play is scripted card by card in
  god-mode; every card played by a *hero* seat (declarer + dummy, or the one
  chosen defender) asks for the author's reasoning, pre-filled with
  *"I just decided to play the [card] instead."* Every comment and reasoning box
  expands the suit shorthand `!s !h !d !c` into ♠ ♥ ♦ ♣ live, exactly like the
  system collections.
  - **Equivalent cards** — while scripting a hero card the author may tick any
    of that seat's remaining cards as *also correct*. They stay interchangeable
    for the **whole line**: after the hero swaps one, every equivalent they
    still hold is accepted at each later play of that group. The server proves
    this safe by re-simulating the line under every reordering of the group and
    rejecting it if any trick's winner could move (e.g. cashing a honour that
    was ducked before) — such a choice is not an equivalence.
  - **Multiple lines** — when a hero choice genuinely changes who wins a trick
    (so the play must differ afterwards), the author records separate lines:
    **Branch here** forks the script at the current point and each line then
    runs its own way. Lines may only diverge on a hero card (never on an
    auto-played seat, never on two interchangeable cards), and practice follows
    whichever line matches the cards the hero actually plays.
  - **Endings** — a line ends by playing all 52 cards, by **claiming** at a
    trick boundary (just the declarer's total tricks), or by letting the **bots
    finish**: from the end of the script the hero may play *any* legal card and
    every other seat is played by the card-play bot, one slow card per request
    (paced like the play-a-hand table). The result is always shown as the
    contract and its margin (e.g. **4♠−1**, **3NT=**, **4♥+2**) — no score, no
    IMPs, no par. Authors can edit or delete their own problems.
- **Practice** — filter the library of everyone's problems by role, author and
  difficulty, then play one (each row shows its ending: play-out / claim / bots,
  and how many lines). The bidding is shown up front with the problem comment
  (and, for a bot-ended problem, the **target** result to reach), then the hand
  runs like a play-a-hand table (hero rotated to the bottom, dummy tabled after
  the opening lead). Non-hero seats play themselves from the author's script.
  If the hero picks a card that is neither the authored one nor one of its
  declared equivalents, it is **not** played: the server answers `409` with the
  author's message for the correct card (a wrong pick is counted) and the hero
  chooses again. Messages stay hidden while you play correctly and are all
  revealed in the end-of-hand review. In the bot phase every legal card is
  accepted and the hand is **graded**: finishing at the target result is a
  success, anything else means a mistake was made somewhere during the hand.
  Drills never touch the rankings or `play_stats.json`.

Practice runs are in-memory sessions (`DrillSession`, 6 h TTL, one per user, the
problem snapshotted at the start so an edit cannot desync a run in flight); the
session id is kept in `localStorage` and resumed on reload. The library endpoint
`GET /api/drills` returns metadata only (no hands, script or comments), and a
full problem is readable only by its author via `GET /api/drills/{id}`.

## Rankings

Three separate leaderboards (`GET /api/play/rankings`, shown in both lobbies):
**solo bidding**, **pair bidding** (partner tables, ranked per pair across all
their tables) and **solo play**. Everyone is scored the same way: the average
IMPs per board over their **last 128 boards**, expressed in **milli-IMPs**
(`sum / 128 × 1000`, so players with fewer than 128 boards are averaged as if
the missing boards were 0 IMP/board). To be listed you must have scored at
least **16 boards in the last 3 months**. Per-mode history is capped at those
128 boards (`data/game/play_stats.json` keeps `recent` for bid-only and
`recentPlay` for full play; partner-table boards carry a per-board timestamp).

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
# Card play: add --hand-ckpt runs/hand_cache/best.pt to let the play bot read
# the auction with the VQ hand-inference model when it builds its hidden-hand
# pool. Without it the pool is dealt by a constraint-based sampler that still
# honours the auction exclusions, show-outs and the honour-lead rule. Tune the
# pool with --decl-pool/--def-pool (candidate deals per bot) and --min-floor.
#
# There is no hand checkpoint in the repo yet -- train one first (it reads the
# human auction store in cache/; see bidding-dt/README.md "Hand inference"):
#   uv run --extra cpu python -m bidding_dt.hand.train \
#       --config configs/hand_cache.yaml --out runs/hand_cache
```

Then run the site as usual (`go run .`) and open http://localhost:8080/play.

Static assets are cache-busted automatically: HTML pages are served `no-cache`
(ETag-revalidated) with every `/static/` URL stamped `?v=<version>`, where the
version is a digest of the static directory's file sizes/mtimes (refreshed
within ~3s of an edit, no restart needed). Assets whose `?v=` matches get
`max-age=31536000, immutable`; anything else must revalidate — so a deploy or
edit reaches every browser on its next page load without stale CSS/JS.

On mobile, visitors who have not installed the app see a one-time
"Install Bridge Trainer" banner: the native install prompt where the browser
offers one (`beforeinstallprompt`), or a Share → Add to Home Screen explainer
on iOS. Dismissals are remembered for 30 days; the banner never appears inside
the installed app. The service worker (Web Push) is registered site-wide from
`shared.js`.

Environment:

- `BIDDING_DT_URL` — sidecar URL (default `http://127.0.0.1:8081`)
- `PLAY_BOTS` — optional per-seat model names for multi-model sidecars,
  e.g. `PLAY_BOTS="N=tiny,E=rl,W=rl"`
- `ATHENEUM_DATA_DIR` — where accounts/trees/play stats live (default `./data`)
- `GOOGLE_CLIENT_ID` — OAuth client ID from the Google Cloud console; when
  set, the login dialog shows a "Continue with Google" button. The Google
  ID token is verified server-side against Google's published keys
  (`/api/auth/google`); a verified email links to an existing account, and
  first-time users choose their own username (a suggestion is derived from
  the email). Leave unset to disable Google sign-in. The OAuth
  consent screen asks for a privacy policy URL — use
  `https://<your-domain>/privacy` (served from `static/privacy.html`).
