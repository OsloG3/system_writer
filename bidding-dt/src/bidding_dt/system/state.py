"""Fixed-length decision state: auction summary + four hand-inference codes.

The system policy deliberately does **not** read the call sequence. Everything
it knows is

  * its own 13 cards (the 52-dim indicator, as everywhere else),
  * a compact summary of the *current* auction position -- vulnerability,
    dealer (relative to the actor), the last contract bid (level, strain, who
    made it), double/redouble status and by whom, and the number of
    consecutive passes,
  * the legal-action mask (also applied to the logits at decision time),
  * `z_self, z_RHO, z_partner, z_LHO`: the frozen hand-inference encoder's
    output for all four seats, each computed from the public auction rotated
    into that seat's frame. That is where the rest of the auction history
    lives -- four learned compressions of the same public sequence, one per
    player's likely hand.

Fixed length, so the decision model is a plain FFN (system/model.py).

Everything here is derived from the supervised token layout
`[COND, HAND, pads, calls...]`, which means the same features can be rebuilt
inside `forward_last(tokens, hand, vuln, row_len)` -- the interface
`rl/rollout._act` and `rl/ppo.ppo_update` already use -- so the system policy
drops into the existing rollout and PPO machinery unchanged.

Frame note: the hero (seat to act) is role 0, so "current seat" is the origin
of the frame and carries no information of its own; what matters about it is
where the dealer sits (`p`, the number of leading frame PADs, which *is* the
dealer's seat relative to the actor) and the vulnerability, which the dataset
already encodes hero-relative (none / we / they / both).
"""

from dataclasses import dataclass

import torch

from ..data.vocab import (CALL_PAD, CALL_PASS, CALL_X, CALL_XX, FIRST_BID,
                          VOCAB_SIZE)

# One-hot block widths (also the state layout, in this order):
VULN_DIM = 4          # none / we / they / both (hero-relative, as the dataset)
DEALER_DIM = 4        # dealer's seat relative to the actor: self/RHO/pd/LHO
LEVEL_DIM = 8         # 0 = no bid yet, then 1..7
DENOM_DIM = 6         # 0 = none, then C D H S N
SEAT_DIM = 5          # none / self / RHO / partner / LHO
DBL_DIM = 3           # undoubled / doubled / redoubled
PASS_DIM = 4          # 0, 1, 2, 3+ consecutive passes
LEGAL_DIM = VOCAB_SIZE

SUMMARY_DIM = (VULN_DIM + DEALER_DIM + LEVEL_DIM + DENOM_DIM + SEAT_DIM
               + DBL_DIM + SEAT_DIM + SEAT_DIM + PASS_DIM + LEGAL_DIM)  # 83
N_SEATS = 4
NO_SEAT = -1


@dataclass
class AuctionSummary:
    """The current position, per row, all in the actor-relative frame."""
    p: torch.Tensor          # (B,) leading frame PADs = dealer's relative seat
    n: torch.Tensor          # (B,) calls made so far
    highest: torch.Tensor    # (B,) highest bid id (0 = none; ids are ordered)
    bid_role: torch.Tensor   # (B,) relative seat of the last bidder (-1 = none)
    doubled: torch.Tensor    # (B,) bool
    dbl_role: torch.Tensor   # (B,) relative seat of the doubler (-1 = none)
    redoubled: torch.Tensor  # (B,) bool
    xx_role: torch.Tensor    # (B,) relative seat of the redoubler (-1 = none)
    pass_run: torch.Tensor   # (B,) trailing consecutive passes


def _first_true(mask: torch.Tensor) -> torch.Tensor:
    """Index of the first True per row (0 when the row is all False)."""
    return mask.to(torch.int64).argmax(1)


def _last_where(mask: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    """Largest j where mask holds, else -1."""
    return torch.where(mask, j.expand_as(mask),
                       torch.full_like(mask, -1, dtype=j.dtype)).max(1).values


def summarize(tokens: torch.Tensor, row_len: torch.Tensor) -> AuctionSummary:
    """(B,L) supervised-layout tokens + real row lengths -> AuctionSummary.

    Mirrors data/legal.py and env/auction_env.py: X/XX only count after the
    last bid (intervening passes are irrelevant), and legality is what forces
    the doubler to be an opponent of the bidder and the redoubler a partner.
    """
    calls = tokens[:, 2:]
    B, S = calls.shape
    dev = tokens.device
    if S == 0:
        # nobody has called: every row is a dealer facing an empty auction
        zero = torch.zeros(B, dtype=torch.long, device=dev)
        none = torch.full((B,), NO_SEAT, dtype=torch.long, device=dev)
        no = torch.zeros(B, dtype=torch.bool, device=dev)
        return AuctionSummary(p=zero, n=zero, highest=zero, bid_role=none,
                              doubled=no, dbl_role=none, redoubled=no,
                              xx_role=none, pass_run=zero)
    j = torch.arange(S, device=dev)
    p = _first_true(calls != CALL_PAD)                 # 0 when nobody has called
    n = (row_len - 2 - p).clamp_min(0)
    # region index s holds the call made by relative seat s % 4 (the frame PADs
    # occupy s < p), so roles come straight from the index -- no call-index
    # bookkeeping needed.
    valid = (j[None, :] >= p[:, None]) & (j[None, :] < (row_len - 2)[:, None])

    is_bid = valid & (calls >= FIRST_BID)
    highest = torch.where(is_bid, calls, torch.zeros_like(calls)).max(1).values
    bid_j = _last_where(is_bid, j)
    bid_role = torch.where(highest > 0, bid_j % 4,
                           torch.full_like(p, NO_SEAT))

    after = valid & (j[None, :] > bid_j[:, None])
    is_x = after & (calls == CALL_X)
    is_xx = after & (calls == CALL_XX)
    doubled = is_x.any(1)
    redoubled = is_xx.any(1)
    dbl_j = _last_where(is_x, j)
    xx_j = _last_where(is_xx, j)
    dbl_role = torch.where(doubled, dbl_j % 4, torch.full_like(p, NO_SEAT))
    xx_role = torch.where(redoubled, xx_j % 4, torch.full_like(p, NO_SEAT))

    # trailing consecutive passes: read each row's real calls backwards (a plain
    # flip would start in the row's trailing batch padding) and take the length
    # of the leading run of passes
    back = p[:, None] + n[:, None] - 1 - j[None, :]
    in_row = (j[None, :] < n[:, None]) & (back >= 0)
    rev = torch.gather(calls, 1, back.clamp(min=0, max=max(S - 1, 0)))
    trailing = in_row & (rev == CALL_PASS)
    pass_run = trailing.to(torch.int64).cumprod(1).sum(1)
    return AuctionSummary(p=p, n=n, highest=highest, bid_role=bid_role,
                          doubled=doubled, dbl_role=dbl_role,
                          redoubled=redoubled, xx_role=xx_role,
                          pass_run=pass_run)


def legal_mask(s: AuctionSummary) -> torch.Tensor:
    """(B,39) bool mask rebuilt from the summary (== data/legal.py)."""
    B = s.p.shape[0]
    dev = s.p.device
    mask = torch.zeros(B, VOCAB_SIZE, dtype=torch.bool, device=dev)
    mask[:, CALL_PASS] = True
    cols = torch.arange(VOCAB_SIZE, device=dev)
    mask[:, FIRST_BID:] = cols[None, FIRST_BID:] > s.highest[:, None]
    has_bid = s.highest > 0
    opp = (s.bid_role == 1) | (s.bid_role == 3)
    own = (s.bid_role == 0) | (s.bid_role == 2)
    mask[:, CALL_X] = has_bid & opp & ~s.doubled
    mask[:, CALL_XX] = has_bid & own & s.doubled & ~s.redoubled
    return mask


def _one_hot(idx: torch.Tensor, n: int, none_at_zero: bool = False) -> torch.Tensor:
    """idx (B,) -> (B,n) float one-hot; -1 becomes all zeros (or slot 0)."""
    out = torch.zeros(idx.shape[0], n, device=idx.device, dtype=torch.float32)
    pos = idx if none_at_zero else idx.clamp_min(0)
    out.scatter_(1, pos[:, None], 1.0)
    if not none_at_zero:
        out[idx < 0] = 0.0
    return out


def summary_features(s: AuctionSummary, vuln: torch.Tensor) -> torch.Tensor:
    """(B, SUMMARY_DIM) float block: position summary + legal-action mask."""
    level = torch.where(s.highest > 0, (s.highest - FIRST_BID) // 5 + 1,
                        torch.zeros_like(s.highest))
    denom = torch.where(s.highest > 0, (s.highest - FIRST_BID) % 5 + 1,
                        torch.zeros_like(s.highest))
    status = (s.doubled.to(torch.int64) + s.redoubled.to(torch.int64))
    return torch.cat([
        _one_hot(vuln.to(torch.int64), VULN_DIM, none_at_zero=True),
        _one_hot(s.p, DEALER_DIM, none_at_zero=True),
        _one_hot(level, LEVEL_DIM, none_at_zero=True),
        _one_hot(denom, DENOM_DIM, none_at_zero=True),
        _one_hot(s.bid_role + 1, SEAT_DIM, none_at_zero=True),
        _one_hot(status, DBL_DIM, none_at_zero=True),
        _one_hot(s.dbl_role + 1, SEAT_DIM, none_at_zero=True),
        _one_hot(s.xx_role + 1, SEAT_DIM, none_at_zero=True),
        _one_hot(s.pass_run.clamp(max=PASS_DIM - 1), PASS_DIM, none_at_zero=True),
        legal_mask(s).to(torch.float32),
    ], dim=-1)


def seat_frames(tokens: torch.Tensor, s: AuctionSummary):
    """Hero-frame rows -> the same auction in all four seats' frames.

    Rotating from the hero to relative seat t only changes the number of
    leading frame PADs (p_t = (p - t) % 4); the call list is untouched. Rows
    come back stacked seat-major: [self(B), RHO(B), partner(B), LHO(B)].

    Returns (frames (4B, S+5) long, mask (4B, S+5) bool True=real,
             row_len_t (4B,), vuln_swap (4,) callable-free index table).
    """
    B, L = tokens.shape
    S = L - 2
    dev = tokens.device
    j = torch.arange(S, device=dev)
    calls = torch.gather(tokens[:, 2:], 1,
                         (s.p[:, None] + j).clamp(max=max(S - 1, 0)))
    calls = torch.where(j[None, :] < s.n[:, None], calls, torch.zeros_like(calls))
    width = S + 5                                   # 2 + <=3 pads + S calls
    frames, lens = [], []
    for t in range(N_SEATS):
        pt = (s.p - t) % 4
        frames.append(torch.zeros(B, width, dtype=tokens.dtype, device=dev)
                      .scatter(1, 2 + pt[:, None] + j, calls))
        lens.append(2 + pt + s.n)
    frames = torch.cat(frames, dim=0)               # seat-major: self,RHO,pd,LHO
    lens = torch.cat(lens)
    mask = torch.arange(width, device=dev)[None, :] < lens[:, None]
    return frames, mask, lens


def frame_vuln(vuln: torch.Tensor) -> torch.Tensor:
    """(B,) hero-relative vuln -> (4B,) per-seat-frame vuln (seat-major)."""
    swap = torch.tensor([0, 2, 1, 3], device=vuln.device)
    return torch.cat([vuln, swap[vuln], vuln, swap[vuln]])
