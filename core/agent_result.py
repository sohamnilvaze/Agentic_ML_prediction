"""
Standard response returned by every agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from core.constants import AgentStatus


@dataclass
class AgentResult:
    """
    Standard output of every agent.
    """

    status: AgentStatus

    reasoning: str

    next_stage: str

    execution_time: float = 0.0

    recommendations: List[str] = field(default_factory=list)

    logs: List[str] = field(default_factory=list)

    human_comments: str = ""

    def add_log(self, message: str) -> None:

        self.logs.append(message)

    def add_recommendation(self, recommendation: str) -> None:

        self.recommendations.append(recommendation)