"""Put `src/` on the path so `pytest` works from the repo root.

The README documents a bare `pytest`, but the package is not installed and the
tests import `pixarch` directly, so without this they fail at collection with
ModuleNotFoundError unless PYTHONPATH is set by hand.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
