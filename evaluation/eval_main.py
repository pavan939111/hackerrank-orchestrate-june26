"""Run evaluation on the 20 labeled sample claims."""

import sys
import os
import csv
import time

# Add code/ to path so we can import the pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from config import SAMPLE_CLAIMS_CSV, DATASET_DIR, IMAGES_DIR
from ingest import load_all_data, parse_image_paths
from cv_checks import run_pre_checks
from orchestrator import process_claim
from adjudicator import adjudicate
from validator import validate_row
from metrics import per_column_report, print_report, confusion_matrix

def load_expected(csv_path) -> list[dict]:
    """Load the labeled sample_claims.csv as list of dicts."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def run_evaluation():
    print("Loading data...")
    claims, evidence_reqs, user_history = load_all_data()
    expected_rows = load_expected(SAMPLE_CLAIMS_CSV)
    
    # Load sample claims (user_001 through user_020)
    sample_claims_raw = []
    with open(SAMPLE_CLAIMS_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sample_claims_raw.append(row)
    
    # We need ClaimRow objects — rebuild them from sample CSV
    from schemas import ClaimRow
    sample_claims = []
    for row in sample_claims_raw:
        sample_claims.append(ClaimRow(
            user_id=row["user_id"],
            image_paths=row["image_paths"],
            user_claim=row["user_claim"],
            claim_object=row["claim_object"]
        ))
    
    print(f"Running pipeline on {len(sample_claims)} sample claims...")
    start = time.time()
    
    predicted_rows = []
    for i, claim in enumerate(sample_claims):
        print(f"  [{i+1}/{len(sample_claims)}] {claim.user_id}...", end=" ")
        
        # Pre-checks
        images = parse_image_paths(claim.image_paths)
        pre_checks = []
        for img_id, img_path in images:
            # img_path is already absolute path resolved by parse_image_paths
            pre_checks.append(run_pre_checks(img_id, img_path))
        
        # Agent loop
        observation = process_claim(claim, evidence_reqs, user_history, pre_checks)
        
        # Adjudicator
        row = adjudicate(observation)
        validated = validate_row(row, claim.claim_object)
        predicted_rows.append(validated)
        
        print(f"-> {validated['claim_status']}")
    
    elapsed = time.time() - start
    print(f"\nPipeline completed in {elapsed:.1f}s")
    
    # Compare
    report = per_column_report(predicted_rows, expected_rows)
    print_report(report)
    
    # Confusion matrix for claim_status
    pred_statuses = [r["claim_status"] for r in predicted_rows]
    exp_statuses = [r["claim_status"] for r in expected_rows]
    labels = ["supported", "contradicted", "not_enough_information"]
    cm = confusion_matrix(pred_statuses, exp_statuses, labels)
    
    print("\nClaim Status Confusion Matrix:")
    print(f"{'':>25} | {'supported':>12} | {'contradicted':>12} | {'not_enough':>12}")
    print("-" * 70)
    for exp_label in labels:
        row_vals = [str(cm[exp_label].get(p, 0)) for p in labels]
        display = "not_enough" if exp_label == "not_enough_information" else exp_label
        print(f"expected {display:>15} | {'  |  '.join(f'{v:>10}' for v in row_vals)}")
    
    # Worst rows
    print("\n--- WORST PERFORMING ROWS (investigate these) ---")
    for col in ["claim_status", "issue_type", "object_part"]:
        if report[col].get("mismatches"):
            for m in report[col]["mismatches"]:
                print(f"  {col}: {m['user_id']} predicted={m['predicted']} expected={m['expected']}")
    
    # Special Hinglish check print
    print("\n--- SPECIAL MULTILINGUAL / HINGLISH PARSE CHECK (user_002) ---")
    user_002_pred = next((r for r in predicted_rows if r["user_id"] == "user_002"), None)
    user_002_exp = next((r for r in expected_rows if r["user_id"] == "user_002"), None)
    if user_002_pred and user_002_exp:
        print(f"user_002 Claim Status: Predicted={user_002_pred['claim_status']} Expected={user_002_exp['claim_status']}")
        print(f"user_002 Issue Type:  Predicted={user_002_pred['issue_type']} Expected={user_002_exp['issue_type']}")
        print(f"user_002 Object Part:  Predicted={user_002_pred['object_part']} Expected={user_002_exp['object_part']}")
    else:
        print("user_002 not found in dataset")
        
    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    import json
    with open(results_path, "w") as f:
        json.dump({
            "elapsed_seconds": elapsed,
            "report": {k: {kk: vv for kk, vv in v.items() if kk != "mismatches" and kk != "per_row"}
                       for k, v in report.items()},
            "claim_status_accuracy": report["claim_status"]["accuracy"],
            "total_claims": len(sample_claims),
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    run_evaluation()
