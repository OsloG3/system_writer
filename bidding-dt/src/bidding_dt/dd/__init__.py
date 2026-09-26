"""Double-dummy scoring and reward package.

- scoring.py: pure-python duplicate score + IMP conversion (no deps)
- solver.py:  DDS wrapper (endplay/libdds, lazy import) + sqlite table cache
- reward.py:  auction -> contract -> par-diff reward in IMPs

The RL environment deals with full 52-card deals; the policy keeps using the
card-indicator encoding. DD solving is the throughput bottleneck (~ms/deal
per core), so tables are cached by deal and solved in batches.
"""

from .reward import DealReward, contract_from_auction, deal_reward, deal_rewards
from .scoring import PENALTY_DOUBLE, PENALTY_NONE, PENALTY_REDOUBLE
from .scoring import contract_score, points_to_imps

__all__ = [
    "DealReward",
    "PENALTY_NONE",
    "PENALTY_DOUBLE",
    "PENALTY_REDOUBLE",
    "contract_from_auction",
    "contract_score",
    "deal_reward",
    "deal_rewards",
    "points_to_imps",
]
