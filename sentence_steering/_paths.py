"""Central paths for the sentence-steering project code (now living under mats/).

The steering *vectors* are trained inside the forked repo (steering-thinking-llms); all the
downstream code -- fetching rollouts, validating vectors, sweeping, single-sentence steering,
Phase 2 generation -- lives here in mats/sentence_steering/. Importing this module wires the
two together, so a script works regardless of cwd or whether the repo was `pip install -e .`d:

  - puts steering-thinking-llms on sys.path  -> `import utils` / `import messages` resolve
  - sets MEAN_VECTORS_DIR                     -> utils.load_model_and_vectors finds the .pt
  - loads mats/.env if python-dotenv is present
  - exposes the standard locations as constants
"""
import os
import sys
from pathlib import Path

MATS = Path(__file__).resolve().parent.parent
STEERING_REPO = MATS / "steering-thinking-llms"
VECTORS_DIR = STEERING_REPO / "train-steering-vectors" / "results" / "vars"
THOUGHT_BRANCHES = MATS / "thought-branches" / "blackmail"
AGENTIC_MISALIGNMENT = MATS / "thought-branches" / "agentic-misalignment"
DATA_DIR = MATS / "data" / "blackmail"
RESULTS_DIR = MATS / "results" / "vars"
DOTENV = MATS / ".env"

# import utils / messages from the steering repo regardless of install state or cwd
if str(STEERING_REPO) not in sys.path:
    sys.path.insert(0, str(STEERING_REPO))

# tell the ported utils where the trained mean-vector .pt files live (absolute, not
# cwd-relative). utils falls back to its in-repo default when this is unset.
os.environ.setdefault("MEAN_VECTORS_DIR", str(VECTORS_DIR))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import dotenv
    if DOTENV.exists():
        dotenv.load_dotenv(DOTENV)
except Exception:
    pass
