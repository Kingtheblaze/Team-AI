"""
Temporal Decay Functions for Hierarchical Adaptive Prototypes.
"""
import math
from datetime import datetime
from typing import Optional
from app.utils.time_utils import now_utc


class TemporalDecay:
    """
    Computes time-decay weights based on exponential half-life models.
    λ = exp( - ln(2) * Δt / T_half )
    """

    def __init__(self, half_life_seconds: float = 300.0):
        self.half_life_seconds = max(1.0, half_life_seconds)
        self._decay_constant = math.log(2.0) / self.half_life_seconds

    def decay_factor(self, event_time: datetime, reference_time: Optional[datetime] = None) -> float:
        """
        Compute decay multiplier in range (0.0, 1.0].
        Events occurring right now have factor 1.0; events older by half_life have factor 0.5.
        """
        ref = reference_time or now_utc()
        delta_sec = max(0.0, (ref - event_time).total_seconds())
        factor = math.exp(-self._decay_constant * delta_sec)
        return min(1.0, max(1e-6, factor))

    def compute_salience_score(
        self,
        count: int,
        last_updated: datetime,
        base_importance: float = 1.0,
        reference_time: Optional[datetime] = None,
    ) -> float:
        """
        Compute current dynamic salience score for a prototype or cluster.
        Combines heavy-hitter frequency with exponential temporal decay.
        """
        decay = self.decay_factor(last_updated, reference_time)
        frequency_weight = math.log1p(count)
        return (frequency_weight * base_importance) * decay
