from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure package-style imports like scheduler_pubmed.src.* work in test runs.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Default test mode should skip startup migrations unless a test opts in.
os.environ.setdefault("PUBMED_SCHEDULER_RUN_MIGRATIONS_ON_STARTUP", "false")
