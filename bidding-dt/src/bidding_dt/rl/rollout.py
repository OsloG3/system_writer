"""Self-play rollouts and GAE.

Team-style training: every deal is played at TWO tables with the same
dealer/vuln (a duplicate team match). In league mode the learner team sits
N-S at table 0 and E-W at table 1 while the frozen checkpoint opponent takes
the other seats at both tables; in pure self-play the learner acts for all
four seats at both tables and sampling variance (see `temp`) makes the two
auctions diverge. Every learner-side seat samples at `temp` and is learned
from -- both partners of a team play hot.

Reward (per learner decision, IMPs):

    r_row = team_weight * 1[call_idx >= first_diff] * team_imps(deal)
            + par_weight * par_diff_imps(table auction)

where team_imps = IMPs(score_NS(table0) - score_NS(table1)) -- the IMP
swing between the two tables, scored double-dummy. The team term only
"trickles down" to calls at/after the first point where the two auctions
differ: identical early calls made (or didn't make) no difference to the
swing, so they receive no team credit. The par term applies to every call
and is down-weighted (par_weight = team_weight/4 by config default). Signs:
each row sees both terms from its own side's perspective (N-S positive,
E-W negative; the team term is additionally flipped at table 1, so in
league mode every learner row gets +team_imps when the learner team wins
the comparison).

Because both partners sample, a row can also be credited for its *partner's*
exploration noise, so row rewards are discounted by how far the partner
deviated from greedy play: every recorded call stores

    dev = logp(greedy call) - logp(call taken)    (untempered policy, >= 0)

and each row's reward -- positive or negative -- is scaled by exp(-D) with D
the mean dev over the partner's calls in the same auction (partner_greedy_dev
/ team_row_rewards). exp(-D) lies in (0, 1]: a partner that bid exactly as the
greedy policy would (D = 0) passes the outcome on undiluted, while very
unlikely sampled calls shrink it toward zero. Scaling both signs keeps the
discount symmetric (the old negatives-only form taught losses more weakly
than wins and biased the policy toward entropy), and averaging dev per call
keeps the discount independent of auction length.

When a ConventionConfig is given (rl.conv_weight > 0), every recorded call is
also checked against the enforced bidding system (conventions/: the system
file's pattern entries via patterns.SystemEnforcer, with the built-in rule
checks as the fallback for calls no entry matches; hands evaluated in
Kaplan-Rubens points). The system file is a PURE CONSTRAINT: it only ever
penalizes infringements, there is no positive convention shaping.
The per-call penalty (sum of violated rule weights) is subtracted from the
row reward as conv_weight * penalty *after* the partner-deviation discount,
so a partner's exploration noise never excuses the row's own system break.
The F discipline is per-call: passing partner's forcing bid (bid-(pass)-
pass) charges the partner's pass (f_pass). The GF discipline is terminal:
when a side with a live game force ends the auction declaring an undoubled
partscore or defending an undoubled contract, the side's LAST pass is the
single call charged (gf_pass, rules.terminal_gf_violations).

Every learner decision is recorded in exactly the supervised input format,
so the PPO update reuses the BC representation. Trajectories are per
(env-row, seat): the seat's own decisions form the MDP chain with the row
reward as terminal payoff. gamma=1 by default -- auctions are short and the
reward is terminal, so discounting would only bias early calls.

play_deals() is the single-table variant kept for evaluation (IMPs vs par
against a fixed opponent); play_seat_policies() steps an env with a policy per
(row, seat) and scores nothing, which is how hand/gen.py manufactures training
auctions from a pool of models.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.distributions import Categorical

from ..conventions import (ConventionConfig, HandInfo, SystemEnforcer,
                           check_call, terminal_gf_violations)
from ..conventions.auction import AuctionReader
from ..data.hands import HAND_DIM
from ..data.vocab import VOCAB_SIZE
from ..dd.scoring import points_to_imps
from ..env.auction_env import AuctionBatch
from ..env.deals import Deal


class UniformPolicy:
    """Random-legal baseline opponent (zero logits -> uniform over mask).

    stochastic: always sample -- argmax over flat logits would degenerate
    into an always-pass bot under greedy eval.
    """

    stochastic = True

    def forward_last(self, tokens, hand, vuln, row_len):
        b = tokens.shape[0]
        z = torch.zeros(b, VOCAB_SIZE, device=tokens.device)
        return z, torch.zeros(b, device=tokens.device)


@dataclass
class RolloutBuffer:
    tokens: list = field(default_factory=list)     # (row_len,) int64 each
    hand: list = field(default_factory=list)       # (HAND_DIM,) float32 each
    vuln: list = field(default_factory=list)
    row_len: list = field(default_factory=list)
    mask: list = field(default_factory=list)       # (39,) bool each
    action: list = field(default_factory=list)
    logprob: list = field(default_factory=list)
    value: list = field(default_factory=list)
    deal_idx: list = field(default_factory=list)   # env row (team: 0..2B-1)
    seat: list = field(default_factory=list)
    call_idx: list = field(default_factory=list)   # decision's call position
    dev: list = field(default_factory=list)        # greedy deviation per call
    adv: np.ndarray | None = None
    ret: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.action)

    def add_rows(self, idx, hero, tokens, hand, vuln_cls, row_len, masks,
                 actions, logprobs, values, call_idx, devs=None):
        for k, i in enumerate(idx):
            L = int(row_len[k])
            self.tokens.append(np.ascontiguousarray(tokens[k, :L]))
            self.hand.append(hand[k])
            self.vuln.append(vuln_cls[k])
            self.row_len.append(L)
            self.mask.append(masks[k])
            self.action.append(int(actions[k]))
            self.logprob.append(float(logprobs[k]))
            self.value.append(float(values[k]))
            self.deal_idx.append(int(i))
            self.seat.append(int(hero[k]))
            self.call_idx.append(int(call_idx[k]))
            self.dev.append(float(devs[k]) if devs is not None else 0.0)

    def finish(self, row_reward: np.ndarray, gamma: float = 1.0,
               lam: float = 0.95):
        """Per-decision credit from each row's own (gated) reward.

        The team swing is a single terminal outcome *attributed* to every
        eligible decision (each post-divergence bid "gets the reward"), not a
        sequence of distinct payoffs, so credit is per-decision (bandit): each
        row's return is its own row_reward and the value head is the baseline.

            adv[i] = row_reward[i] - value[i];  ret[i] = row_reward[i]

        Chaining a terminal reward through GAE would be wrong here: it would
        (a) leak the gated team term back onto pre-divergence bids that are
        supposed to get none, and (b) double-count the swing for a seat that
        made several post-divergence calls (the first would accumulate its own
        swing plus the discounted future ones). gamma/lam are accepted for
        signature compatibility but are not used under this attribution.
        """
        n = len(self.deal_idx)
        value = np.asarray(self.value, dtype=np.float64)
        R = np.asarray(row_reward, dtype=np.float64)
        assert R.shape == (n,)
        self.ret = R.astype(np.float32)
        self.adv = (R - value).astype(np.float32)

    def to_torch(self, device):
        """Pad once to max row length; returns dict of tensors on device."""
        n = self.n
        L = max(self.row_len)
        tokens = np.zeros((n, L), dtype=np.int64)
        hand = np.zeros((n, HAND_DIM), dtype=np.float32)
        vuln = np.zeros(n, dtype=np.int64)
        row_len = np.zeros(n, dtype=np.int64)
        mask = np.zeros((n, VOCAB_SIZE), dtype=bool)
        action = np.zeros(n, dtype=np.int64)
        logprob = np.zeros(n, dtype=np.float32)
        for r in range(n):
            tokens[r, : self.row_len[r]] = self.tokens[r]
            hand[r] = self.hand[r]
            vuln[r] = self.vuln[r]
            row_len[r] = self.row_len[r]
            mask[r] = self.mask[r]
            action[r] = self.action[r]
            logprob[r] = self.logprob[r]
        t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
        return {
            "tokens": t(tokens, torch.long),
            "hand": t(hand, torch.float32),
            "vuln": t(vuln, torch.long),
            "row_len": t(row_len, torch.long),
            "mask": t(mask, torch.bool),
            "action": t(action, torch.long),
            "logprob": t(logprob, torch.float32),
            "value": t(np.asarray(self.value, dtype=np.float32), torch.float32),
            "adv": t(self.adv, torch.float32),
            "ret": t(self.ret, torch.float32),
        }


def par_row_rewards(buf: RolloutBuffer, reward_imps_ns: np.ndarray,
                    reward_scale: float) -> np.ndarray:
    """Per-row terminal rewards from per-env-row N-S par-diff IMPs
    (single-table play_deals): the acting side's perspective, scaled."""
    deal_idx = np.asarray(buf.deal_idx)
    seat = np.asarray(buf.seat)
    sign = np.where(seat % 2 == 0, 1.0, -1.0)  # NS positive
    return reward_imps_ns[deal_idx] * sign * reward_scale


def partner_greedy_dev(buf: RolloutBuffer) -> np.ndarray:
    """(n,) mean greedy deviation of each row's partner.

    The partner is seat+2 in the same auction (same env row / deal_idx); its
    deviation is the mean of buf.dev over the partner's own recorded calls --
    dev = logp(greedy call) - logp(call taken) under the untempered policy,
    so 0 means the partner bid exactly as the greedy strategy would and large
    values mean it sampled increasingly unlikely calls. A partner that never
    got to call deviates 0. Rows without recorded devs (legacy buffers) are
    treated as deviation 0 (no scaling).
    """
    n = len(buf.deal_idx)
    if n == 0:
        return np.zeros(0)
    dev = (np.asarray(buf.dev, dtype=np.float64)
           if len(buf.dev) == n else np.zeros(n))
    deal_idx = np.asarray(buf.deal_idx)
    seat = np.asarray(buf.seat)
    key = deal_idx * 4 + seat
    sums = np.zeros(int(key.max()) + 4)
    counts = np.zeros_like(sums)
    np.add.at(sums, key, dev)
    np.add.at(counts, key, 1.0)
    mean = sums / np.maximum(counts, 1.0)
    return mean[deal_idx * 4 + (seat + 2) % 4]


def convention_penalties(buf: RolloutBuffer, env: AuctionBatch,
                         cfg: ConventionConfig | None = None,
                         spec=None,
                         ) -> tuple[np.ndarray, list[list[str]]]:
    """(penalties (n,), violation codes per row) for a rollout.

    Replays each recorded decision's auction prefix (env.calls up to the
    row's call_idx) through the convention reader and weighs the hero's
    actual call against the enforced system. Only uses information the
    model itself sees: hero's hand + public auction + dealer/vuln. Hand
    features (K&R count, lengths) are computed once per (deal, seat).

    `spec` is the parsed system file (conventions/system_file.py): when
    given, its sequence-pattern entries take precedence over the rule-code
    checks (a matched entry with weight > 0 replaces them for that call) and
    add the file-driven force/double discipline (f_pass, to_pass,
    penalty_pull) -- see conventions/patterns.SystemEnforcer. The file is a
    pure constraint: matching entries only ever PENALIZE, nothing rewards a
    clean convention use.

    Two disciplines, two attributions:

    * F (one-round force) is per-call: when the auction goes bid-(pass)-pass
      -- partner's forcing bid, an opponent pass, then the hero's pass --
      the f_pass penalty lands on the PARTNER's pass (the row recorded here),
      never on the forcing bid itself. An intervening opponent bid releases
      the force.
    * GF (game force) is terminal (rules.terminal_gf_violations): each
      COMPLETED auction is judged once per side, and when a side with a live
      game force ends up declaring an undoubled partscore or defending an
      undoubled contract (any level), the gf_pass penalty lands on that
      side's LAST pass -- the single pass that let the auction die.
      Intermediate/forcing passes below game are not penalized, and a
      doubled/redoubled ending is accepted. Violations by seats with no
      recorded row (league opponents) are skipped.
    """
    n = buf.n
    pens = np.zeros(n, dtype=np.float64)
    codes: list[list[str]] = [[] for _ in range(n)]
    if n == 0:
        return pens, codes
    cfg = cfg or ConventionConfig()
    enforcer = SystemEnforcer(spec, cfg) if spec is not None else None
    hands: dict[tuple[int, int], HandInfo] = {}
    readers: dict[int, AuctionReader] = {}
    row_of: dict[tuple[int, int, int], int] = {}
    for i in range(n):
        di = buf.deal_idx[i]
        seat = buf.seat[i]
        row_of[(di, seat, int(buf.call_idx[i]))] = i
        key = (di, seat)
        hi = hands.get(key)
        if hi is None:
            hi = hands[key] = HandInfo(env.deals[di].hands[seat])
        reader = readers.get(di)
        if reader is None:
            reader = readers[di] = AuctionReader(int(env.dealer[di]),
                                                 int(env.vuln[di]))
        # readers are keyed per env row; every row of one auction feeds calls
        # in order, but rows of the same auction arrive interleaved by
        # call_idx, so replay the prefix from scratch when out of order
        ci = int(buf.call_idx[i])
        if reader.n > ci:
            reader = AuctionReader(int(env.dealer[di]), int(env.vuln[di]))
            readers[di] = reader
        while reader.n < ci:
            reader.feed(int(env.calls[di, reader.n]))
        action = int(buf.action[i])
        if enforcer is not None:
            vs = enforcer.check(reader, seat, hi, action)
        else:
            vs = check_call(reader, seat, hi, action, cfg)
        if vs:
            pens[i] = sum(v.weight for v in vs)
            codes[i] = [v.code for v in vs]

    # Terminal GF discipline: judge each completed auction ONCE and charge
    # the GF side's last pass (rules.terminal_gf_violations). With a file,
    # the enforcer's memoized replay state carries both the structural and
    # the file-flagged game forces (patterns.fold_state); without one, a
    # plain full replay of the structural reader is enough.
    for di in sorted(set(buf.deal_idx)):
        nc = int(env.n_calls[di])
        calls = [int(c) for c in env.calls[di, :nc]]
        dealer = int(env.dealer[di])
        if enforcer is not None:
            full = enforcer.state(dealer, calls).reader
        else:
            full = AuctionReader(dealer, int(env.vuln[di]))
            for c in calls:
                full.feed(c)
        for seat, ci, v in terminal_gf_violations(full, cfg):
            i = row_of.get((di, seat, ci))
            if i is None:
                continue          # not a recorded (learner) decision
            pens[i] += v.weight
            codes[i].append(v.code)
    return pens, codes


def team_row_rewards(buf: RolloutBuffer, n_deals: int, team_imps: np.ndarray,
                     diverge: np.ndarray, par_imps_ns: np.ndarray,
                     reward_scale: float, team_weight: float = 1.0,
                     par_weight: float = 0.25,
                     partner_dev: np.ndarray | None = None,
                     conv_pen: np.ndarray | None = None,
                     conv_weight: float = 0.0) -> np.ndarray:
    """Per-row terminal rewards for a two-table team rollout.

    team_imps[j]  = IMPs(score_NS(table0, deal j) - score_NS(table1, deal j))
    diverge[j]    = index of the first call where the two auctions differ
                    (min length when one is a strict prefix of the other);
                    the team term only reaches calls at/after that index.
    par_imps_ns   = (2B,) par-diff IMPs of each table's auction, N-S view.
    partner_dev   = (n,) optional mean greedy deviation of each row's partner
                    (partner_greedy_dev): row rewards -- BOTH signs -- are
                    scaled by exp(-partner_dev) in (0, 1], so an outcome is
                    credited in full only when the partner bid (near-)greedily
                    and attenuates toward zero as the partner's sampled calls
                    get less likely. Symmetric scaling matters: discounting
                    only negatives taught losses ~25-50% more weakly than
                    wins, which made self-play row rewards net-positive and
                    fed an entropy-increasing drift (measured on a BC policy:
                    mean row reward +0.017 -> +0.083 with the asymmetric
                    form). Averaging dev per call keeps the discount
                    independent of auction length.
    conv_pen      = (n,) optional weighted system-convention penalties per
                    row (convention_penalties); reward_scale * conv_weight *
                    conv_pen is subtracted after the partner-dev scaling so the
                    row owns its own system breaks in full. The penalty is
                    scaled by reward_scale, so a file weight of W costs W
                    IMP-equivalents (see the note in the body). The system
                    file is penalty-only: there is no positive convention
                    term in the reward.
    """
    deal_idx = np.asarray(buf.deal_idx)          # env row in [0, 2B)
    seat = np.asarray(buf.seat)
    call_idx = np.asarray(buf.call_idx)
    b = n_deals
    sign = np.where(seat % 2 == 0, 1.0, -1.0)            # row's side, NS +
    table_sign = np.where(deal_idx < b, 1.0, -1.0)       # table0-NS view +
    base = deal_idx % b
    gate = (call_idx >= diverge[base]).astype(np.float64)
    team = team_imps[base] * sign * table_sign * gate
    par = par_imps_ns[deal_idx] * sign
    r = reward_scale * (team_weight * team + par_weight * par)
    if partner_dev is not None:
        scale = np.exp(-np.clip(np.asarray(partner_dev, dtype=np.float64),
                                0.0, None))
        r = r * scale
    # Convention shaping is expressed in the SAME units as the IMP reward: the
    # penalty is scaled by reward_scale too, so a file weight of W costs
    # W * reward_scale reward units == W IMP-equivalents. Without this the raw
    # penalty dwarfed the scaled IMP signal (a weight-1 violation cost 5 IMPs
    # while the IMP reward moved by fractions), which made the policy avoid
    # every convention.
    if conv_pen is not None and conv_weight:
        r = r - reward_scale * conv_weight * np.asarray(conv_pen,
                                                        dtype=np.float64)
    return r


@dataclass
class TeamRollout:
    """Result of play_team_deals: env rows [0,B) = table 0, [B,2B) = table 1."""
    buf: RolloutBuffer | None
    env: AuctionBatch
    n_deals: int
    team_imps: np.ndarray      # (B,) IMPs(score_NS t0 - score_NS t1)
    diverge: np.ndarray        # (B,) first differing call index
    diverged: np.ndarray       # (B,) bool, auctions differ anywhere
    par_imps_ns: np.ndarray    # (2B,) par-diff IMPs per table auction
    score_ns: np.ndarray       # (2B,) achieved score, N-S view
    partner_dev: np.ndarray | None = None   # (n,) mean partner greedy deviation
    conv_pen: np.ndarray | None = None      # (n,) convention penalty per row
    conv_codes: list | None = None          # (n,) violated rule codes per row

    def row_rewards(self, reward_scale: float, team_weight: float = 1.0,
                    par_weight: float = 0.25,
                    conv_weight: float = 0.0) -> np.ndarray:
        return team_row_rewards(self.buf, self.n_deals, self.team_imps,
                                self.diverge, self.par_imps_ns, reward_scale,
                                team_weight, par_weight,
                                partner_dev=self.partner_dev,
                                conv_pen=self.conv_pen,
                                conv_weight=conv_weight)


def _act(mdl, tokens, hand, vuln_cls, row_len, masks, device, greedy,
         temp: float = 1.0, amp: bool = False):
    """One batched decision step for `mdl` (numpy inputs); returns torch
    (actions, logprobs, values, greedy_dev). temp > 1 flattens the
    sampling distribution (logprobs are from the tempered distribution,
    matching PPO recompute). greedy_dev = logp(argmax) - logp(action) under
    the UNTEMPERED masked policy: 0 iff the action is the greedy one, growing
    with how unlikely the sampled action was (used to discount row rewards by
    a partner's exploration noise).
    amp=True runs the forward under bf16 autocast (CPU speedup knob)."""
    with torch.autocast(getattr(device, "type", device),
                        dtype=torch.bfloat16, enabled=amp):
        logits, values = mdl.forward_last(
            torch.as_tensor(tokens, dtype=torch.long, device=device),
            torch.as_tensor(hand, dtype=torch.float32, device=device),
            torch.as_tensor(vuln_cls, dtype=torch.long, device=device),
            torch.as_tensor(row_len, dtype=torch.long, device=device))
    logits = logits.float()
    values = values.float()   # bf16 under amp -> fp32 for numpy/buffer storage
    masks_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
    logits = logits.masked_fill(~masks_t, float("-inf"))
    lp0 = logits.log_softmax(-1)             # untempered, for greedy_dev
    if temp != 1.0:
        logits = logits / temp
    dist = Categorical(logits=logits)
    if greedy and not getattr(mdl, "stochastic", False):
        actions = logits.argmax(-1)
    else:
        actions = dist.sample()
    dev = (lp0.max(-1).values
           - lp0.gather(-1, actions[:, None]).squeeze(-1)).clamp_min(0.0)
    return actions, dist.log_prob(actions), values, dev


@torch.no_grad()
def _run(env: AuctionBatch, model, device, opponent, learner_ns, greedy,
         buf: RolloutBuffer | None, temp: float, amp: bool = False):
    """Step env to completion; record learner decisions into buf (if any).

    Every learner-side seat samples at `temp` and is recorded -- in team mode
    both partners of the learner partnership play hot at their table, and in
    self-play all four seats do at both tables. Each recorded call also stores
    its greedy deviation (buf.dev, see `_act`) so row rewards can be
    discounted by the partner's exploration noise (partner_greedy_dev).
    Opponent seats play their own frozen policy AT THE SAME `temp` and are
    dropped. The rollout is policy-only: the system file is NOT consulted
    here (no exploration overrides, no auxiliary targets) -- it only enters
    afterwards, as per-row penalties in convention_penalties.
    """
    use_opp = opponent is not None and learner_ns is not None
    while env.any_active():
        idx = env.active_idx()
        tokens, hand, vuln_cls, hero, row_len = env.build_inputs(idx)
        masks = env.legal_masks(idx)
        actions = np.zeros(len(idx), dtype=np.int64)

        if use_opp:
            hero_ns = hero % 2 == 0
            is_learn = hero_ns == learner_ns[idx]
        else:
            is_learn = np.ones(len(idx), dtype=bool)

        learn_rows = np.flatnonzero(is_learn)
        if len(learn_rows):
            a, lp, v, dev = _act(model, tokens[learn_rows], hand[learn_rows],
                                 vuln_cls[learn_rows], row_len[learn_rows],
                                 masks[learn_rows], device, greedy, temp, amp)
            a_np = a.cpu().numpy()
            actions[learn_rows] = a_np
            if buf is not None:
                buf.add_rows(idx[learn_rows], hero[learn_rows],
                             tokens[learn_rows], hand[learn_rows],
                             vuln_cls[learn_rows], row_len[learn_rows],
                             masks[learn_rows], a_np,
                             lp.cpu().numpy(), v.cpu().numpy(),
                             env.n_calls[idx[learn_rows]],
                             dev.cpu().numpy())
        opp_rows = np.flatnonzero(~is_learn)
        if len(opp_rows):
            # the frozen opponent samples at the SAME rollout temp as the
            # learner: a temp handicap (learner hot, opponent cold) made
            # identical policies lose ~0.65 IMPs/board with a 12% winrate, so
            # the team swing measured the handicap instead of skill and every
            # league iteration trained on an unfixable negative signal.
            a, _, _, _ = _act(opponent, tokens[opp_rows],
                              hand[opp_rows], vuln_cls[opp_rows],
                              row_len[opp_rows], masks[opp_rows],
                              device, greedy, temp, amp)
            actions[opp_rows] = a.cpu().numpy()

        env.step(idx, actions)


@torch.no_grad()
def play_seat_policies(env: AuctionBatch, policies, seat_policy: np.ndarray,
                       device, greedy: bool = False, temp: float = 1.0,
                       amp: bool = False) -> AuctionBatch:
    """Step `env` to completion with a policy per (row, seat).

    `seat_policy[row, seat]` indexes `policies`, so a table can mix systems
    (BC N-S against RL E-W, self-play, a random seat...). Nothing is recorded
    and no rewards are computed, so this needs no DD solver: it is the
    rollout used to manufacture training data (hand/gen.py), not an RL
    iteration.
    """
    while env.any_active():
        idx = env.active_idx()
        tokens, hand, vuln_cls, hero, row_len = env.build_inputs(idx)
        masks = env.legal_masks(idx)
        actions = np.zeros(len(idx), dtype=np.int64)
        which = seat_policy[idx, hero]
        for pid in np.unique(which):
            rows = np.flatnonzero(which == pid)
            pol = policies[pid]
            a, _, _, _ = _act(pol, tokens[rows], hand[rows],
                              vuln_cls[rows], row_len[rows],
                              masks[rows], device, greedy, temp, amp)
            actions[rows] = a.cpu().numpy()
        env.step(idx, actions)
    return env


def _join_presolve(presolve):
    """Await a background dd.reward.warm_cache future (submitted at iteration
    start) so terminal scoring below is pure cache hits -- the libdds solve
    ran in parallel with the neural rollout instead of after it."""
    if presolve is not None:
        presolve.result()


@torch.no_grad()
def play_deals(model, deals: list[Deal], dealer: np.ndarray, vuln: np.ndarray,
               device, rng: np.random.Generator | None = None, cache=None,
               opponent=None, learner_ns: np.ndarray | None = None,
               greedy: bool = False, record: bool = True,
               temp: float = 1.0, amp: bool = False,
               presolve=None) -> tuple[RolloutBuffer | None, AuctionBatch, np.ndarray]:
    """Single table: play all auctions to completion; returns
    (buffer, env, reward_imps_ns). Used by evaluation harnesses.

    opponent + learner_ns (B,) bool: for deals where learner_ns is set, the
    learner acts for that partnership and `opponent` for the other seats.
    opponent=None (or learner_ns=None) -> pure self-play, all seats recorded.
    presolve: optional warm_cache future to join before scoring.
    """
    env = AuctionBatch(deals, dealer, vuln)
    buf = RolloutBuffer() if record else None
    _run(env, model, device, opponent, learner_ns, greedy, buf, temp, amp)
    _join_presolve(presolve)
    rewards = env.rewards(cache=cache)
    r_ns = np.array([r.reward_imps for r in rewards], dtype=np.float64)
    return buf, env, r_ns


def _divergence(env: AuctionBatch, b: int) -> tuple[np.ndarray, np.ndarray]:
    """(diverge (B,), diverged (B,) bool) between table-0 and table-1 rows."""
    n0, n1 = env.n_calls[:b], env.n_calls[b:]
    c0, c1 = env.calls[:b], env.calls[b:]
    k = np.arange(c0.shape[1])[None, :]
    shared = k < np.minimum(n0, n1)[:, None]
    mism = (c0 != c1) & shared
    any_mism = mism.any(axis=1)
    first_mism = mism.argmax(axis=1)
    diverge = np.where(any_mism, first_mism, np.minimum(n0, n1))
    diverged = any_mism | (n0 != n1)
    return diverge, diverged


@torch.no_grad()
def play_team_deals(model, deals: list[Deal], dealer: np.ndarray,
                    vuln: np.ndarray, device,
                    rng: np.random.Generator | None = None, cache=None,
                    opponent=None, greedy: bool = False, temp: float = 1.0,
                    record: bool = True, amp: bool = False,
                    presolve=None, conv_cfg=None,
                    conv_spec=None) -> TeamRollout:
    """Play every deal at two tables (team match) and score the comparison.

    opponent given (league): the learner team is N-S at table 0 and E-W at
    table 1; `opponent` (frozen checkpoint) holds the other seats at both
    tables. opponent=None (self-play): the learner acts for all four seats
    at both tables -- sampling (temp >= 1) is what makes the two auctions
    diverge; a greedy policy would play identical auctions and get zero
    team signal. Every learner-side seat samples at `temp` and is recorded
    (both partners learn); the frozen opponent samples at the same `temp`,
    so identical policies compare even. roll.partner_dev holds each recorded
    row's mean partner greedy deviation, which row_rewards() uses to
    discount row rewards, both signs (see team_row_rewards).
    amp: bf16 autocast for the rollout forwards (CPU speed knob).
    presolve: optional warm_cache future to join before scoring.
    rng: accepted for API compatibility (the rollout itself is pure policy
    sampling; there are no more file-driven exploration overrides).
    conv_cfg: optional ConventionConfig -- when given (and rows are
    recorded), every learner decision is checked against the enforced
    bidding system and roll.conv_pen/conv_codes carry the weighted per-call
    penalties (applied in row_rewards via conv_weight). The system file is
    penalty-only: no positive convention term is added to the reward.
    conv_spec: optional parsed system file (conventions/system_file.py) --
    its sequence-pattern entries extend/override the rule-code checks and
    drive the F/GF/TO/PENALTY discipline (see convention_penalties).
    """
    b = len(deals)
    dealer2 = np.concatenate([dealer, dealer]).astype(np.int64)
    vuln2 = np.concatenate([vuln, vuln]).astype(np.int64)
    env = AuctionBatch(list(deals) + list(deals), dealer2, vuln2)
    learner_ns = (np.concatenate([np.ones(b, bool), np.zeros(b, bool)])
                  if opponent is not None else None)
    buf = RolloutBuffer() if record else None
    _run(env, model, device, opponent, learner_ns, greedy, buf, temp, amp)

    _join_presolve(presolve)
    rewards = env.rewards(cache=cache)
    score_ns = np.array([r.score_ns for r in rewards], dtype=np.int64)
    par_imps = np.array([r.reward_imps for r in rewards], dtype=np.float64)
    team_imps = np.array([points_to_imps(int(score_ns[j] - score_ns[b + j]))
                          for j in range(b)], dtype=np.float64)
    diverge, diverged = _divergence(env, b)
    partner_dev = partner_greedy_dev(buf) if buf is not None else None
    conv_pen = conv_codes = None
    if buf is not None and conv_cfg is not None:
        conv_pen, conv_codes = convention_penalties(buf, env, conv_cfg,
                                                    conv_spec)
    return TeamRollout(buf=buf, env=env, n_deals=b, team_imps=team_imps,
                       diverge=diverge, diverged=diverged,
                       par_imps_ns=par_imps, score_ns=score_ns,
                       partner_dev=partner_dev,
                       conv_pen=conv_pen, conv_codes=conv_codes)
