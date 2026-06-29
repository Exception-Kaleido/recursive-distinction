from __future__ import annotations

from typing import List

from .boundary import BoundaryGate
from .types import Action, SensoryItem, StepResult, Trace


class TraceField:
    def __init__(self, gate: BoundaryGate, capacity: int = 500) -> None:
        self.gate = gate
        self.capacity = capacity
        self.traces: List[Trace] = []
        self.step_index = 0

    def record(self, item: SensoryItem, action: Action, result: StepResult) -> Trace:
        self.step_index += 1
        return self._store(
            item,
            action,
            result.resistance,
            result.delta_viability,
            source_step=self.step_index,
            commit_step=self.step_index,
            temporal_delay=0,
            temporal_eligibility=1.0,
        )

    def _store(
        self,
        item: SensoryItem,
        action: Action,
        resistance: float,
        delta_viability: float,
        *,
        source_step: int,
        commit_step: int,
        temporal_delay: int,
        temporal_eligibility: float,
    ) -> Trace:
        features = item.features()
        features["resistance"] = resistance
        features["delta_viability"] = delta_viability
        trace = Trace(
            features=item.features(),
            action=action,
            resistance=resistance,
            delta_viability=delta_viability,
            source_step=source_step,
            commit_step=commit_step,
            temporal_delay=temporal_delay,
            temporal_eligibility=temporal_eligibility,
        )
        trace.selected_features = self.gate.select(features)
        self.gate.update_from_reentry(trace, self.traces)
        self.traces.append(trace)
        if len(self.traces) > self.capacity:
            self.traces.pop(0)
        return trace


class TemporalTraceField(TraceField):
    """Binds delayed outcomes to short-lived action-context eligibility traces."""

    def __init__(
        self,
        gate: BoundaryGate,
        capacity: int = 500,
        min_delay: int = 3,
        max_delay: int = 5,
    ) -> None:
        super().__init__(gate, capacity)
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.eligible: List[tuple[int, SensoryItem, Action, float]] = []

    def record_temporal(
        self, item: SensoryItem, action: Action, result: StepResult
    ) -> List[Trace]:
        self.step_index += 1
        committed: List[Trace] = []
        if result.resistance >= 0.5:
            committed.append(
                self._commit(
                    item,
                    action,
                    result.resistance,
                    -0.05,
                    self.step_index,
                )
            )
        else:
            self.eligible.append((self.step_index, item, action, result.resistance))

        outcome_effect = result.delta_viability + 0.05
        if abs(outcome_effect) > 0.5:
            candidates = [
                entry
                for entry in self.eligible
                if self.min_delay
                <= self.step_index - entry[0]
                <= self.max_delay
            ]
            if candidates:
                origin = max(candidates, key=lambda entry: self._salience(entry[1]))
                self.eligible.remove(origin)
                committed.append(
                    self._commit(
                        origin[1],
                        origin[2],
                        origin[3],
                        outcome_effect - 0.05,
                        origin[0],
                    )
                )

        expired = [
            entry
            for entry in self.eligible
            if self.step_index - entry[0] > self.max_delay
        ]
        for entry in expired:
            self.eligible.remove(entry)
            committed.append(
                self._commit(
                    entry[1],
                    entry[2],
                    entry[3],
                    -0.05,
                    entry[0],
                )
            )
        return committed

    def _commit(
        self,
        item: SensoryItem,
        action: Action,
        resistance: float,
        delta_viability: float,
        source_step: int,
    ) -> Trace:
        delay = self.step_index - source_step
        eligibility = max(
            0.0,
            1.0 - max(0, delay - self.min_delay) / max(1, self.max_delay),
        )
        return self._store(
            item,
            action,
            resistance,
            delta_viability,
            source_step=source_step,
            commit_step=self.step_index,
            temporal_delay=delay,
            temporal_eligibility=eligibility,
        )

    @staticmethod
    def _salience(item: SensoryItem) -> float:
        empty = (0.92, 0.92, 0.92)
        return sum(
            abs(value - background)
            for value, background in zip(
                (item.color_r, item.color_g, item.color_b), empty
            )
        )
