"""Bidding systems: a fixed-length-state FFN policy evolved over hand encoders.

- state.py:      public auction summary + four z blocks -> fixed-length state
- model.py:      SystemPolicy (SwiGLU FFN + value head) and the z adapters
- population.py: system variants, mutation/selection, league snapshots
- config.py:     SystemModelConfig / EvoConfig / SystemConfig (YAML)
- train.py:      inner PPO loop, outer evolutionary loop

The policy speaks the same `forward_last(tokens, hand, vuln, row_len)` protocol
as rl/model.py, so rl/rollout.py, rl/league.py and rl/ppo.py are reused as-is.
"""
