"""Evaluation entry point per AGENTS.md §6.1.

Delegates to the root evaluation harness at evaluation/eval_main.py.
Run from repo root:  python code/evaluation/main.py
Or from here:        python main.py
"""

import sys
import os

# Add the root evaluation/ directory to path
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_eval_dir = os.path.join(_repo_root, "evaluation")
sys.path.insert(0, _eval_dir)

from eval_main import run_evaluation

if __name__ == "__main__":
    run_evaluation()
