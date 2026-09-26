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

Run:

    uv run python -m bidding_dt.play_server --ckpt runs/tiny/best.pt --port 8081
    # multiple named models: --ckpt bc=runs/tiny/best.pt --ckpt rl=runs/rl_tinyt/best.pt
"""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .bid import auction_over, load_model, predict, random_deal, seat_to_act
from .data.hands import encode_hand
from .data.legal import legal_mask
from .data.parse import MAX_CALLS
from .data.vocab import CALL_PAD, DENOMS, VOCAB_SIZE, call_to_id, id_to_call
from .dd.reward import deal_reward
from .dd.solver import TableCache

MAX_BODY = 64 << 10


class ServiceError(Exception):
    """Request-level error; `code` becomes the HTTP status."""

    def __init__(self, msg: str, code: int = 400):
        super().__init__(msg)
        self.msg = msg
        self.code = code


class Service:
    """Endpoint logic without any HTTP, so it is directly testable."""

    def __init__(self, ckpts=(), dd_cache=None, temp: float = 0.0):
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

    # ---- helpers ----

    def health(self) -> dict:
        return {"ok": True, "models": self.model_paths,
                "default_model": self.default_model}

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
        if not isinstance(raw, list) or len(raw) > MAX_CALLS:
            raise ServiceError(f"calls must be a list of at most {MAX_CALLS} tokens")
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
                     "/bid": self.svc.bid, "/score": self.svc.score}
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
    args = ap.parse_args()

    ckpts = parse_ckpt_specs(args.ckpt)
    svc = Service(ckpts, dd_cache=args.dd_cache or None, temp=args.temp)
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
