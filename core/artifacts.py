"""
Data Artifact definition.

Represents any artifact generated during the workflow:
- Dataset
- Report
- Model
- Metrics
- JSON metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

from core.utils import current_timestamp


@dataclass
class DataArtifact:
    """
    Generic artifact produced by any agent.
    """

    name: str
    artifact_type: str
    path: Path
    version: str = "v1"

    created_at: str = field(default_factory=current_timestamp)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact into serializable dictionary."""

        return {
            "name": self.name,
            "artifact_type": self.artifact_type,
            "path": str(self.path),
            "version": self.version,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }