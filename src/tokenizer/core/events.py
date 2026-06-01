from dataclasses import dataclass
from typing import Literal, Optional


EventKind = Literal["progress", "status", "artifact", "done", "error"]


@dataclass(frozen=True)
class JobEvent:
    kind: EventKind
    message: str = ""
    total_windows: Optional[int] = None
    completed_windows: Optional[int] = None
    eta_seconds: Optional[float] = None
    artifact_path: Optional[str] = None

    def validate(self) -> None:
        if self.kind not in {"progress", "status", "artifact", "done", "error"}:
            raise ValueError(f"unsupported event kind: {self.kind}")
        if self.kind == "progress":
            if self.total_windows is None or self.completed_windows is None:
                raise ValueError("progress events require total_windows and completed_windows")
            if self.total_windows < 0 or self.completed_windows < 0:
                raise ValueError("window counts must be non-negative")
            if self.completed_windows > self.total_windows:
                raise ValueError("completed_windows cannot exceed total_windows")
