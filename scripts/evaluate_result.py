#!/usr/bin/env python3
"""Re-evaluate a saved experiment directory (delegates to `pgtwin evaluate`).

    python scripts/evaluate_result.py --experiment outputs/push_box_001
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pgtwin.cli.evaluate import main as evaluate_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(evaluate_main(sys.argv[1:]))
