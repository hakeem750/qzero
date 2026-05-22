"""
MCTS Node with virtual loss support for parallel search.

IMPROVEMENT over blueprint:
  Virtual loss: when a thread selects a path to a leaf, it temporarily
  adds a penalty (virtual_loss) to all nodes on that path before the
  real backup.  This discourages other threads from selecting the same
  path, spreading exploration across the tree — critical for batched
  leaf collection where multiple leaves are selected before any backup.

  Without virtual loss, batched MCTS collapses to near-serial behaviour
  because all workers race to the same best leaf.
"""
from __future__ import annotations

import math
from typing import Dict, Optional
import numpy as np

from env.state import QuoridorState


class Node:
    __slots__ = (
        "prior",
        "visit_count",
        "value_sum",
        "virtual_loss",     # IMPROVEMENT: virtual loss counter
        "children",
        "state",
        "expanded",
        "to_play",
        "action_from_parent",
    )

    def __init__(
        self,
        prior: float,
        state: Optional[QuoridorState] = None,
        to_play: int = 1,
        action_from_parent: int = -1,
    ) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss = 0       # IMPROVEMENT
        self.children: Dict[int, "Node"] = {}
        self.state = state
        self.expanded = False
        self.to_play = to_play
        self.action_from_parent = action_from_parent

    # ------------------------------------------------------------------
    @property
    def q_value(self) -> float:
        """Mean action value, including virtual loss penalty."""
        n = self.visit_count + self.virtual_loss
        if n <= 0:
            return 0.0
        return (self.value_sum - self.virtual_loss) / n

    def puct_score(self, parent_visit: int, c_puct: float) -> float:
        """Q + U (PUCT exploration bonus)."""
        parent_visit = max(0, parent_visit)
        child_visit = max(0, self.visit_count + self.virtual_loss)
        u = c_puct * self.prior * math.sqrt(parent_visit) / (1 + child_visit)
        return self.q_value + u

    # ------------------------------------------------------------------
    # Virtual loss (IMPROVEMENT)
    # ------------------------------------------------------------------
    def add_virtual_loss(self, vl: int = 1) -> None:
        self.virtual_loss += vl

    def revert_virtual_loss(self, vl: int = 1) -> None:
        self.virtual_loss = max(0, self.virtual_loss - vl)

    # ------------------------------------------------------------------
    def backup(self, value: float) -> None:
        """Increment visit count and accumulate value."""
        self.visit_count += 1
        self.value_sum  += value
