from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional
from .types import MemoryEpisode


class EbbinghausDecayModel:
    """
    Implements a modified Ebbinghaus forgetting curve with spacing and access reinforcement:
        R(t) = exp( - delta_hours / (half_life_hours * (1 + ln(1 + access_count))) )
    """

    def __init__(self, default_half_life_hours: float = 72.0, min_retention_threshold: float = 0.15):
        self.default_half_life_hours = default_half_life_hours
        self.min_retention_threshold = min_retention_threshold

    def compute_retention(
        self,
        episode: MemoryEpisode,
        current_time: Optional[datetime] = None,
    ) -> float:
        now = current_time or datetime.now(timezone.utc)
        delta_seconds = max(0.0, (now - episode.created_at).total_seconds())
        delta_hours = delta_seconds / 3600.0

        # High importance increases stability / half-life
        stability_multiplier = 0.5 + (episode.importance * 1.5)
        # Frequent access reinforces memory retention (spacing effect)
        reinforcement = 1.0 + math.log(1.0 + episode.access_count)

        effective_half_life = self.default_half_life_hours * stability_multiplier * reinforcement
        retention = math.exp(-delta_hours / effective_half_life)
        return max(0.0, min(1.0, retention))

    def should_prune(
        self,
        episode: MemoryEpisode,
        current_time: Optional[datetime] = None,
    ) -> bool:
        retention = self.compute_retention(episode, current_time)
        return retention < self.min_retention_threshold
