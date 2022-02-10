import numbers
from typing import Dict


def format_dict(d: Dict[str, numbers.Number]):
    return " ".join(
        [
            f"{k}=" + f"{v:.6f}" if isinstance(v, numbers.Number) else f"{v}"
            for k, v in d.items()
        ]
    )
