import dataclasses
from typing import Dict, List


@dataclasses.dataclass()
class Record:
    step: int
    epoch: int
    metrics: Dict[str, Dict[str, float]]
