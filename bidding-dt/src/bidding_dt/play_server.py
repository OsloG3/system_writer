"""HTTP sidecar serving bidding-dt bots for the system_writer "play vs bots" page.

Loads one or more checkpoints once and answers small JSON requests, so the Go
site never shells out per call. Deal generation, legality and par scoring are
the same code paths used in training/eval (data/legal.py, dd/reward.py).

Endpoints (all JSON):

    GET  /health  -> {"ok": true, "models": {name: ckpt_path}}
    POST /deal    -> {"hands": [N,E,S,W 'S.H.D.C'], "dealer": 0..3, "vuln": 0..3}
                     body may pin "dealer"/"vuln"
    POST /legal   -> {"seat": int, "over": bool, "legal": ["P", "1C", ...]}
                     body: {"dealer", "vuln", "calls": ["P", "1H", ...]}
    POST /bid     -> {"seat", "call", "prob", "model", "top": [[call, prob], ...]}
                     body: /legal body + "hands", optional "model", "temp"
    POST /score   -> {"imps", "par_ns", "score_ns", "tricks", "contract"}
                     body: /bid body; IMPs are N-S view vs double-dummy par
    POST /play/choose -> {"seat", "card", "pool", "strat", "timing"}  a bot's
                     card, chosen over the pool of hidden hands consistent with
                     the auction and the play so far: the declarer side plans a
                     strategy on likelihood-weighted, low-card-merged layouts,
                     the other seats rank cards by their weighted double-dummy
                     expectation (see play_bot.py / play_strategy.py; "strat"
                     carries the plan's diagnostics or null, "timing" the
                     decision's per-phase wall-clock seconds and DDS solve
                     count). Body:
                       {"session": stable per-bot key, "dealer", "vuln",
                        "calls": [...], "known": {seat: 'S.H.D.C'},
                        "plays": [[seat, "AS"], ...], "to_act": seat,
                        optional "pool"/"min_floor"/"strategy"/"strat_time",
                        optional "dd_hands": [N,E,S,W 'S.H.D.C']}
                     "dd_hands" is the drills mode: it carries the REAL deal
                     (the card-problems server scripts all four hands). The
                     bot still ranks its options with only the information a
                     normal game gives, but returns the best-ranked card that
                     costs no trick double-dummy on the real deal.
    POST /play/result -> {"contract", "tricks", "dd_tricks", "par_ns",
                          "score_ns", "imps"}  score the *actually played*
                     tricks vs double-dummy par. Body: /score body + "tricks".
    POST /analyze -> {"contract", "target", "dd_tricks", "par_ns", "table",
                      "optimal", "actual_tricks", "plays": [per-card verdict]}
                     god-mode double-dummy breakdown of a played hand: for
                     every card, whether it matched double-dummy best play, the
                     tricks it cost and whether it was contract-critical. Body:
                     /score body + "plays": [[seat, "AS"], ...] in play order.

Run (torch + endplay come from the cpu/cuda and rl extras, so repeat them):

    uv run --extra cpu --extra rl python -m bidding_dt.play_server \
        --ckpt runs/rl_tinyt/best.pt --port 8081
    # on a GPU box use --extra cuda; repeatable names:
    # --ckpt bc=<path> --ckpt rl=<path>
    # add --hand-ckpt runs/hand_small/best.pt to make the card bot read the
    # auction with the VQ hand model (without it, hidden hands are dealt by a
    # constraint-based sampler that still honours the play and the lead rules).
"""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import numpy as np

    from .bid import auction_over, load_model, predict, random_deal, seat_to_act
    from .dd import fastsolve
    from .dd.reward import contract_from_auction, deal_reward
    from .dd.scoring import contract_score, declarer_is_vul, points_to_imps
    from .dd.solver import (TableCache, _PROJ_DENOM_TO_ENDPLAY, dd_table,
                            deal_key, par_score)
    from .play_bot import PlayBot, PlayConfig, PlaySession, card_to_str, str_to_card
except ImportError as e:  # torch / endplay / numpy not installed
    raise SystemExit(
        f"bidding_dt.play_server is missing a dependency ({e.name or e}).\n"
        "The server needs torch and endplay, which live in the cpu/cuda and\n"
        "rl extras. Start it so the extras are kept, e.g.:\n"
        "  uv run --extra cpu --extra rl python -m bidding_dt.play_server "
        "--ckpt <checkpoint> --port 8081\n"
        "  (use --extra cuda instead of --extra cpu on a GPU machine, and\n"
        "   copy a checkpoint first: runs/ is gitignored)"
    ) from e

from .data.hands import encode_hand, str_to_cards
from .data.legal import legal_mask
from .data.parse import MAX_CALLS
from .data.vocab import CALL_PAD, DENOMS, VOCAB_SIZE, call_to_id, id_to_call

MAX_BODY = 64 << 10

# The site allows auctions longer than the model's MAX_CALLS (Go caps them at
# 80 calls, 20 per player). Every endpoint accepts up to this on the wire;
# only /bid actually feeds the auction to the model, and it refuses past
# MAX_CALLS (the Go side auto-passes the bots from there instead of asking).
WIRE_MAX_CALLS = 80


class ServiceError(Exception):
    """Request-level error; `code` becomes the HTTP status."""

    def __init__(self, msg: str, code: int = 400):
        super().__init__(msg)
        self.msg = msg
        self.code = code


class Service:
    """Endpoint logic without any HTTP, so it is directly testable."""

    def __init__(self, ckpts=(), dd_cache=None, temp: float = 0.0,
                 hand_ckpt=None, play_cfg: PlayConfig | None = None):
        self.models: dict[str, tuple] = {}
        self.model_paths: dict[str, str] = {}
        self.default_model: str | None = None
        for name, path in ckpts:
            model, dev = load_model(path)
            self.models[name] = (model, dev)
            self.model_paths[name] = str(path)
            if self.default_model is None:
                self.default_model = name
        self.temp = float(temp)
        self.table_cache = TableCache(dd_cache) if dd_cache else None
        self._sample_lock = threading.Lock()
        # card-play bot: an optional VQ hand model reads the auction for the
        # hidden-hand pool; without one play_bot deals constraint-based hands
        self.hand_model = None
        self.hand_device = "cpu"
        self.hand_ckpt_path: str | None = None
        if hand_ckpt:
            from .hand.model import load_hand_model
            from .train import pick_device
            self.hand_device = pick_device(None)
            self.hand_model = load_hand_model(hand_ckpt, self.hand_device)
            self.hand_ckpt_path = str(hand_ckpt)
        self.play_cfg = play_cfg or PlayConfig()
        self.play_bot = PlayBot(self.hand_model, self.hand_device, self.play_cfg)

    # ---- helpers ----

    def health(self) -> dict:
        return {"ok": True, "models": self.model_paths,
                "default_model": self.default_model,
                "hand_model": self.hand_ckpt_path}

    @staticmethod
    def _seat_int(body: dict, key: str) -> int:
        v = body.get(key)
        if v is None or not isinstance(v, int) or not 0 <= v <= 3:
            raise ServiceError(f"{key} must be an int 0..3 (N=0 E=1 S=2 W=3)")
        return int(v)

    def _auction(self, body: dict):
        dealer = self._seat_int(body, "dealer")
        vuln = body.get("vuln")
        if vuln is None or not isinstance(vuln, int) or not 0 <= vuln <= 3:
            raise ServiceError("vuln must be an int 0..3 (None/N-S/E-W/Both)")
        raw = body.get("calls", [])
        if not isinstance(raw, list) or len(raw) > WIRE_MAX_CALLS:
            raise ServiceError(f"calls must be a list of at most {WIRE_MAX_CALLS} tokens")
        ids = []
        for tok in raw:
            if not isinstance(tok, str):
                raise ServiceError(f"call tokens must be strings, got {tok!r}")
            try:
                cid = call_to_id(tok.upper())
            except ValueError:
                raise ServiceError(f"unknown call: {tok!r}")
            if cid == CALL_PAD:
                raise ServiceError("PAD is not a call")
            ids.append(cid)
        return dealer, int(vuln), ids

    @staticmethod
    def _hand_strings(body: dict) -> list[str]:
        hands = body.get("hands")
        if not isinstance(hands, list) or len(hands) != 4:
            raise ServiceError("hands must be 4 'S.H.D.C' strings (N,E,S,W)")
        for h in hands:
            if not isinstance(h, str):
                raise ServiceError("hands must be strings")
            try:
                encode_hand(h)  # validates 13 legal cards
            except ValueError as e:
                raise ServiceError(str(e))
        return [str(h) for h in hands]

    @staticmethod
    def _encoded(hands: list[str]) -> np.ndarray:
        return np.stack([encode_hand(h) for h in hands])

    # ---- endpoints ----

    def deal(self, body: dict) -> dict:
        rng = np.random.default_rng()
        hands = random_deal(rng)
        dealer = body.get("dealer")
        vuln = body.get("vuln")
        dealer = int(rng.integers(4)) if dealer is None else self._seat_int(body, "dealer")
        if vuln is None:
            vuln = int(rng.integers(4))
        elif not isinstance(vuln, int) or not 0 <= vuln <= 3:
            raise ServiceError("vuln must be an int 0..3")
        return {"hands": list(hands), "dealer": dealer, "vuln": int(vuln)}

    def legal(self, body: dict) -> dict:
        dealer, _vuln, ids = self._auction(body)
        seat = seat_to_act(dealer, ids)
        if auction_over(ids):
            return {"seat": int(seat), "over": True, "legal": []}
        p = (4 - (seat - dealer) % 4) % 4
        mask = legal_mask(np.asarray(ids, dtype=np.int64), role_offset=p)
        return {"seat": int(seat), "over": False,
                "legal": [id_to_call(int(i)) for i in np.flatnonzero(mask)]}

    def bid(self, body: dict) -> dict:
        if not self.models:
            raise ServiceError("no models loaded (start with --ckpt)", 503)
        dealer, vuln, ids = self._auction(body)
        hands = self._hand_strings(body)
        if auction_over(ids):
            raise ServiceError("auction is over")
        if len(ids) >= MAX_CALLS:
            raise ServiceError(f"auction hit the {MAX_CALLS}-call model limit", 409)
        name = body.get("model") or self.default_model
        if name not in self.models:
            raise ServiceError(f"unknown model {name!r}; loaded: {sorted(self.models)}")
        model, dev = self.models[name]
        seat = seat_to_act(dealer, ids)
        temp = float(body.get("temp", self.temp) or 0.0)
        picks = predict(model, dev, self._encoded(hands), seat, dealer, vuln, ids,
                        topk=VOCAB_SIZE)
        if temp > 0 and len(picks) > 1:
            probs = np.array([p for _, p in picks], dtype=np.float64)
            logits = np.log(np.maximum(probs, 1e-12)) / temp
            w = np.exp(logits - logits.max())
            w /= w.sum()
            with self._sample_lock:
                i = int(np.random.default_rng().choice(len(picks), p=w))
        else:
            i = 0
        call, prob = picks[i]
        return {"seat": int(seat), "call": call, "prob": float(prob), "model": name,
                "top": [[c, float(p)] for c, p in picks[:5]]}

    def score(self, body: dict) -> dict:
        dealer, vuln, ids = self._auction(body)
        hands = self._hand_strings(body)
        r = deal_reward(hands, dealer, vuln, ids, cache=self.table_cache)
        contract = None
        if r.contract is not None:
            c = r.contract
            contract = {"level": int(c.level), "denom": DENOMS[c.denom],
                        "declarer": int(c.declarer), "penalty": int(c.penalty)}
        return {"imps": float(r.reward_imps), "par_ns": int(r.par_ns),
                "score_ns": int(r.score_ns),
                "tricks": None if r.tricks is None else int(r.tricks),
                "contract": contract}

    # ---- card play ----

    def _known_hands(self, body: dict) -> dict[int, str]:
        raw = body.get("known")
        if not isinstance(raw, dict) or not raw:
            raise ServiceError("known must be a non-empty {seat: 'S.H.D.C'} map")
        known: dict[int, str] = {}
        for k, h in raw.items():
            try:
                seat = int(k)
            except (TypeError, ValueError):
                raise ServiceError(f"known seat must be an int, got {k!r}")
            if not 0 <= seat <= 3:
                raise ServiceError("known seat must be 0..3")
            if not isinstance(h, str):
                raise ServiceError("known hands must be 'S.H.D.C' strings")
            try:
                encode_hand(h)
            except ValueError as e:
                raise ServiceError(str(e))
            known[seat] = h
        return known

    def _plays(self, body: dict) -> list[tuple[int, int]]:
        raw = body.get("plays", [])
        if not isinstance(raw, list) or len(raw) > 52:
            raise ServiceError("plays must be a list of at most 52 [seat, card]")
        out = []
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ServiceError("each play is [seat, card]")
            seat, card = item
            if not isinstance(seat, int) or not 0 <= seat <= 3:
                raise ServiceError("play seat must be an int 0..3")
            if not isinstance(card, str):
                raise ServiceError("play card must be a string like 'AS'")
            try:
                out.append((seat, str_to_card(card)))
            except ValueError as e:
                raise ServiceError(str(e))
        return out

    @staticmethod
    def _dd_hands(body: dict, known: dict[int, str],
                  plays: list[tuple[int, int]]) -> list[str] | None:
        """Validate the optional drills-mode real deal ([N,E,S,W] hands).

        The real deal must agree with everything the bot already sees: the
        `known` hands and every card actually played. Returns None when the
        request is a plain (non-drill) one.
        """
        raw = body.get("dd_hands")
        if raw is None:
            return None
        if not isinstance(raw, list) or len(raw) != 4:
            raise ServiceError("dd_hands must be 4 'S.H.D.C' strings (N,E,S,W)")
        hands = []
        for h in raw:
            if not isinstance(h, str):
                raise ServiceError("dd_hands must be strings")
            try:
                encode_hand(h)
            except ValueError as e:
                raise ServiceError(str(e))
            hands.append(str(h))
        held = [set(str_to_cards(h).tolist()) for h in hands]
        for seat, h in known.items():
            if held[seat] != set(str_to_cards(h).tolist()):
                raise ServiceError(f"dd_hands[{seat}] disagrees with known[{seat}]")
        for seat, card in plays:
            if card not in held[seat]:
                raise ServiceError("dd_hands disagree with plays: seat "
                                   f"{seat} played a card it does not hold")
            held[seat].discard(card)
        return hands

    def play_choose(self, body: dict) -> dict:
        dealer, vuln, ids = self._auction(body)
        contract = contract_from_auction(dealer, ids)
        if contract is None:
            raise ServiceError("the auction passed out; there is no play")
        known = self._known_hands(body)
        plays = self._plays(body)
        to_act = self._seat_int(body, "to_act")
        if to_act not in known:
            raise ServiceError("to_act must be a seat the bot can see")
        session = body.get("session")
        if not isinstance(session, str) or not session:
            raise ServiceError("session must be a non-empty string")

        cfg = self.play_cfg
        pool = body.get("pool")
        if isinstance(pool, int) and pool > 0:
            cfg = PlayConfig(**{**cfg.__dict__,
                                "decl_pool": pool, "def_pool": pool})
        floor = body.get("min_floor")
        if isinstance(floor, int) and floor >= 0:
            cfg = PlayConfig(**{**cfg.__dict__, "min_floor": floor})
        strategy = body.get("strategy")
        if isinstance(strategy, bool):
            cfg = PlayConfig(**{**cfg.__dict__, "strategy": strategy})
        stime = body.get("strat_time")
        if isinstance(stime, (int, float)) and not isinstance(stime, bool) and stime >= 0:
            cfg = PlayConfig(**{**cfg.__dict__, "strat_time": float(stime)})

        dd_hands = self._dd_hands(body, known, plays)

        def factory():
            return PlaySession(session, dealer, vuln, ids, contract.declarer,
                               contract.denom, known, self.hand_model,
                               self.hand_device, cfg,
                               rng=np.random.default_rng())
        sess = self.play_bot.get_session(session, factory)
        if dd_hands is None:
            card = sess.choose(plays, to_act)
        else:
            # drills mode: rank the options with only the information a real
            # game gives, but never give away a trick on the real deal
            card = sess.choose_drill(plays, to_act,
                                     {s: dd_hands[s] for s in range(4)})
        return {"seat": int(to_act), "card": card_to_str(int(card)),
                "pool": len(sess.pool), "strat": sess.last_search,
                "timing": sess.last_timing}

    def play_result(self, body: dict) -> dict:
        dealer, vuln, ids = self._auction(body)
        hands = self._hand_strings(body)
        tricks = body.get("tricks")
        if not isinstance(tricks, int) or not 0 <= tricks <= 13:
            raise ServiceError("tricks must be an int 0..13 (declarer side)")
        r = deal_reward(hands, dealer, vuln, ids, cache=self.table_cache)
        c = r.contract
        if c is None:
            return {"contract": None, "tricks": None, "dd_tricks": None,
                    "par_ns": int(r.par_ns), "score_ns": 0,
                    "imps": float(points_to_imps(0 - int(r.par_ns)))}
        score = contract_score(c.level, c.denom, c.penalty, tricks,
                               declarer_is_vul(c.declarer, vuln))
        score_ns = score if c.declarer in (0, 2) else -score
        contract = {"level": int(c.level), "denom": DENOMS[c.denom],
                    "declarer": int(c.declarer), "penalty": int(c.penalty)}
        return {"contract": contract, "tricks": int(tricks),
                "dd_tricks": None if r.tricks is None else int(r.tricks),
                "par_ns": int(r.par_ns), "score_ns": int(score_ns),
                "imps": float(points_to_imps(score_ns - int(r.par_ns)))}

    def analyze(self, body: dict) -> dict:
        """Full double-dummy breakdown of a played hand, card by card.

        Unlike /play/choose (which hides two hands and reasons from a pool of
        candidates), this is pure god-mode analysis on the REAL deal: for every
        card actually played, the position *before* that card is solved and the
        played card's double-dummy trick count for the acting side is compared
        against the best any legal card achieves. A card below the best is a
        mistake that cost tricks; one that flips the contract between make and
        defeat (best play vs. the card actually played, both then defended
        optimally) is contract-critical.

        Body: /score body (hands, dealer, vuln, calls) + "plays": [[seat, "AS"],
        ...] in play order (a claim/short hand may stop before 52). Returns the
        per-card verdicts plus the deal's DD table, par and each side's
        makeable game/slam, so a whole session can be aggregated for stats.
        """
        dealer, vuln, ids = self._auction(body)
        hands = self._hand_strings(body)
        contract = contract_from_auction(dealer, ids)
        if contract is None:
            raise ServiceError("the auction passed out; there is no play to analyze")
        plays = self._plays(body)

        declarer = int(contract.declarer)
        dummy = (declarer + 2) % 4
        opener = (declarer + 1) % 4
        denom = int(contract.denom)
        trump_int = _PROJ_DENOM_TO_ENDPLAY[denom]
        target = 6 + int(contract.level)
        all_cards = [set(str_to_cards(h).tolist()) for h in hands]
        seen: set[int] = set()
        for cs in all_cards:
            if len(cs) != 13 or (seen & cs):
                raise ServiceError("each hand must hold 13 distinct cards and "
                                   "no card may appear in two hands")
            seen |= cs
        if len(seen) != 52:
            raise ServiceError("the four hands must be a complete 52-card deal")

        # Deal-level facts: the DD table, par and the played contract's DD
        # trick count. The table is cached, so a bulk run over repeated deals
        # never re-solves one.
        key = deal_key(hands)
        table = self.table_cache.get(key) if self.table_cache else None
        if table is None:
            table = dd_table(hands)
            if self.table_cache:
                self.table_cache.put(key, table)
        par_ns = int(par_score(table, vuln, dealer))
        dd_tricks = int(table[declarer, denom])

        # Pass 1: build every pre-play position (each depends only on the cards
        # already down), so the whole hand solves in one memoised DDS batch.
        played_by_seat: list[set] = [set() for _ in range(4)]
        tricks_decl = 0
        positions: list = []
        ctx: list[dict] = []
        for i, (seat, cid) in enumerate(plays):
            seat, cid = int(seat), int(cid)
            if cid not in all_cards[seat] or cid in played_by_seat[seat]:
                raise ServiceError(
                    f"play {i + 1}: {card_to_str(cid)} is not a card seat "
                    f"{seat} still holds")
            n_done = i // 4
            cur = plays[4 * n_done:i]
            if cur:
                first = int(cur[0][0])
            elif n_done == 0:
                first = opener
            else:
                last = plays[4 * (n_done - 1):4 * n_done]
                first = (int(last[0][0]) + fastsolve.trick_winner(
                    [int(c) for _, c in last], trump_int)) % 4
            to_act = (first + len(cur)) % 4
            remaining = [all_cards[s] - played_by_seat[s] for s in range(4)]
            positions.append((remaining, first, [int(c) for _, c in cur]))
            ctx.append({"seat": seat, "cid": cid, "n_done": n_done,
                        "to_act": to_act, "decl_side": to_act in (declarer, dummy),
                        "tricks_decl": tricks_decl})
            # fold this play into the position that follows it
            played_by_seat[seat].add(cid)
            if (i + 1) % 4 == 0:
                trick = plays[i - 3:i + 1]
                w = (int(trick[0][0]) + fastsolve.trick_winner(
                    [int(c) for _, c in trick], trump_int)) % 4
                if w in (declarer, dummy):
                    tricks_decl += 1

        try:
            solved = fastsolve.solve_positions(positions, trump_int)
        except Exception:
            solved = [{} for _ in positions]

        analyzed: list[dict] = []
        for i, c in enumerate(ctx):
            results = solved[i] or {}
            rec = {"seat": c["seat"], "card": card_to_str(c["cid"]),
                   "to_act": c["to_act"], "decl_side": c["decl_side"]}
            if results:
                remaining_tricks = 13 - c["n_done"]
                decl_side = c["decl_side"]
                base = c["tricks_decl"]

                def to_decl(v, _rt=remaining_tricks, _ds=decl_side, _b=base):
                    return _b + (v if _ds else _rt - v)

                best = max(results.values())
                played_val = results.get(c["cid"])
                rec["best"] = int(best)
                rec["best_decl"] = int(to_decl(best))
                rec["best_cards"] = sorted(card_to_str(x) for x, v
                                           in results.items() if v == best)
                if played_val is not None:
                    pv = int(played_val)
                    rec["played"] = pv
                    rec["played_decl"] = int(to_decl(pv))
                    cost = int(best) - pv
                    rec["cost"] = max(0, cost)
                    rec["mistake"] = bool(cost > 0)
                    rec["critical"] = bool(
                        (rec["best_decl"] >= target) != (rec["played_decl"] >= target))
                else:
                    rec.update({"played": None, "played_decl": None, "cost": 0,
                                "mistake": False, "critical": False})
            else:
                rec.update({"best": None, "best_decl": None, "played": None,
                            "played_decl": None, "cost": 0, "mistake": False,
                            "critical": False, "best_cards": []})
            analyzed.append(rec)

        # actual declarer tricks when the whole hand was played out
        actual = None
        if len(plays) == 52:
            actual = 0
            for t in range(13):
                trick = plays[4 * t:4 * t + 4]
                w = (int(trick[0][0]) + fastsolve.trick_winner(
                    [int(c) for _, c in trick], trump_int)) % 4
                if w in (declarer, dummy):
                    actual += 1

        def side_opt(seats):
            best_tricks, game, small, grand = 0, False, False, False
            for s in seats:
                for d in range(5):
                    t = int(table[s, d])
                    best_tricks = max(best_tricks, t)
                    if t >= (11, 11, 10, 10, 9)[d]:   # game thresholds C D H S N
                        game = True
                    if t >= 12:
                        small = True
                    if t >= 13:
                        grand = True
            return {"best_tricks": best_tricks, "game": game,
                    "small_slam": small, "grand_slam": grand}

        return {
            "contract": {"level": int(contract.level), "denom": DENOMS[denom],
                         "declarer": declarer, "penalty": int(contract.penalty)},
            "target": target, "dd_tricks": dd_tricks, "par_ns": par_ns,
            "table": [[int(table[s, d]) for d in range(5)] for s in range(4)],
            "optimal": {"ns": side_opt([0, 2]), "ew": side_opt([1, 3])},
            "actual_tricks": actual, "n": len(plays), "plays": analyzed,
        }

    def play_drop(self, body: dict) -> dict:
        prefix = body.get("prefix")
        key = body.get("session")
        if isinstance(prefix, str) and prefix:
            self.play_bot.drop_prefix(prefix)
        elif isinstance(key, str) and key:
            self.play_bot.drop(key)
        return {"ok": True}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def svc(self) -> Service:
        return self.server.service  # type: ignore[attr-defined]

    def _send(self, code: int, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        if self.path.split("?")[0] == "/health":
            self._send(200, self.svc.health())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ServiceError("body too large")
            raw = self.rfile.read(n) if n else b"{}"
            body = json.loads(raw or b"{}")
            if not isinstance(body, dict):
                raise ServiceError("body must be a JSON object")
        except ServiceError as e:
            self._send(e.code, {"error": e.msg})
            return
        except (ValueError, json.JSONDecodeError) as e:
            self._send(400, {"error": f"invalid JSON: {e}"})
            return
        endpoints = {"/deal": self.svc.deal, "/legal": self.svc.legal,
                     "/bid": self.svc.bid, "/score": self.svc.score,
                     "/analyze": self.svc.analyze,
                     "/play/choose": self.svc.play_choose,
                     "/play/result": self.svc.play_result,
                     "/play/drop": self.svc.play_drop}
        fn = endpoints.get(self.path.split("?")[0])
        if fn is None:
            self._send(404, {"error": "not found"})
            return
        try:
            self._send(200, fn(body))
        except ServiceError as e:
            self._send(e.code, {"error": e.msg})
        except BrokenPipeError:
            pass
        except Exception as e:  # pragma: no cover - last-resort guard
            self._send(500, {"error": f"internal: {e}"})

    def log_message(self, fmt, *args):
        print(f"[play_server] {self.address_string()} {fmt % args}", flush=True)


def parse_ckpt_specs(specs: list[str]) -> list[tuple[str, str]]:
    """['name=path', 'runs/x/best.pt'] -> [('name','path'), ('x','runs/x/best.pt')]"""
    out = []
    for i, spec in enumerate(specs):
        name, sep, path = spec.partition("=")
        if not sep:
            path = name
            name = Path(path).parent.name or f"model{i}"
        out.append((name, path))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", default=[],
                    help="[name=]checkpoint path, repeatable; first is the default model")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--dd-cache", default="cache/dd_play.sqlite",
                    help="sqlite DD-table cache path ('' to disable)")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="default sampling temperature for /bid (0 = greedy)")
    ap.add_argument("--hand-ckpt", default=None,
                    help="hand-inference (VQ) checkpoint for the card-play bot's "
                         "hidden-hand pool; without it hands are dealt by a "
                         "constraint-based sampler")
    ap.add_argument("--decl-pool", type=int, default=64,
                    help="candidate hidden deals the declarer-side bot simulates")
    ap.add_argument("--def-pool", type=int, default=40,
                    help="candidate hidden deals a defender bot simulates")
    ap.add_argument("--min-floor", type=int, default=6,
                    help="the candidate pool never shrinks below this")
    ap.add_argument("--no-strategy", action="store_true",
                    help="disable the declarer-side strategy search; every seat "
                         "then picks by the weighted per-card double-dummy rank")
    ap.add_argument("--strat-layouts", type=int, default=20,
                    help="merged macro layouts the strategy search evaluates per node")
    ap.add_argument("--strat-time", type=float, default=1.5,
                    help="wall-clock seconds the strategy search may spend per decision")
    ap.add_argument("--play-temp", type=float, default=1.0,
                    help="temperature discounting layouts under which an observed "
                         "hidden play cost tricks double-dummy (0 disables the "
                         "play-quality inference)")
    ap.add_argument("--play-review-time", type=float, default=0.5,
                    help="wall-clock seconds one pool top-up may spend retro-scoring "
                         "fresh candidates against recent hidden plays, newest "
                         "first (0 = review all of them unbudgeted)")
    ap.add_argument("--rank-objective", default="both", choices=("ev", "both"),
                    help="defender (and declarer-fallback) card ranking: pure "
                         "expected double-dummy tricks, or that plus the "
                         "likelihood of beating/making the contract")
    ap.add_argument("--rank-make-bonus", type=float, default=1.0,
                    help="weight of P(beat/make) inside the 'both' rank objective")
    ap.add_argument("--no-tie-prefer-spots", action="store_true",
                    help="pick uniformly among score-tied cards instead of "
                         "preferring spots over (and cheaper over dearer) honours")
    ap.add_argument("--no-batch-topup", action="store_true",
                    help="draw pool top-ups one candidate at a time (up to "
                         "topup_tries decoder calls each) instead of one batched "
                         "call per hidden seat plus constraint repair")
    ap.add_argument("--no-fast-onestep", action="store_true",
                    help="strategy search: score declarer cards with one solve "
                         "per (layout, card) instead of one per layout")
    ap.add_argument("--no-def-smart", action="store_true",
                    help="strategy search: model the deep no-solve defence as "
                         "always-lowest instead of the take/cover/overtake rules")
    ap.add_argument("--no-endgame-skip", action="store_true",
                    help="strategy search: keep searching proved run-out "
                         "endgames instead of returning at the one-step ceiling")
    args = ap.parse_args()

    ckpts = parse_ckpt_specs(args.ckpt)
    play_cfg = PlayConfig(decl_pool=args.decl_pool, def_pool=args.def_pool,
                          min_floor=args.min_floor,
                          strategy=not args.no_strategy,
                          strat_layouts=args.strat_layouts,
                          strat_time=args.strat_time,
                          play_temp=args.play_temp,
                          play_review_time=args.play_review_time,
                          rank_objective=args.rank_objective,
                          rank_make_bonus=args.rank_make_bonus,
                          tie_prefer_spots=not args.no_tie_prefer_spots,
                          batch_topup=not args.no_batch_topup,
                          fast_onestep=not args.no_fast_onestep,
                          def_smart=not args.no_def_smart,
                          endgame_skip=not args.no_endgame_skip)
    svc = Service(ckpts, dd_cache=args.dd_cache or None, temp=args.temp,
                  hand_ckpt=args.hand_ckpt, play_cfg=play_cfg)
    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    httpd.service = svc  # type: ignore[attr-defined]
    print(f"play_server on http://{args.host}:{args.port} "
          f"models={ {n: p for n, p in svc.model_paths.items()} }", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
