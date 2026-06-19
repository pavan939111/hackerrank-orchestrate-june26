import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths - relative to repo root
REPO_ROOT = Path(__file__).parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CLAIMS_CSV = DATASET_DIR / "claims.csv"
SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQ_CSV = DATASET_DIR / "evidence_requirements.csv"
IMAGES_DIR = DATASET_DIR / "images"
OUTPUT_CSV = REPO_ROOT / "output.csv"

# API keys - load all GEMINI_API_KEY_1 through GEMINI_API_KEY_10
GEMINI_API_KEYS = []
for i in range(1, 11):
    key = os.environ.get(f"GEMINI_API_KEY_{i}") or os.environ.get(f"gemini_{i}")
    if key:
        GEMINI_API_KEYS.append(key.strip())
# Fallback to single key
if not GEMINI_API_KEYS:
    single = os.environ.get("GEMINI_API_KEY") or os.environ.get("gemini")
    if single:
        GEMINI_API_KEYS.append(single.strip())

# Model config
GEMINI_MODEL = "gemini-3.1-flash-lite"
MAX_AGENT_ITERATIONS = 6
CONFIDENCE_THRESHOLD = 0.4
IMAGE_MAX_SIZE = 1024  # max longest side in pixels
