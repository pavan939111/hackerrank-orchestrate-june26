"""Compare two pipeline configurations on sample claims."""

import sys
import os
import csv
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from config import SAMPLE_CLAIMS_CSV, DATASET_DIR
from ingest import load_all_data, parse_image_paths
from cv_checks import run_pre_checks, resize_for_vlm
from orchestrator import process_claim
from adjudicator import adjudicate
from validator import validate_row
from vlm_client import call_gemini_vision
from tools import inspect_images as inspect_images_batched
from metrics import per_column_report
from schemas import ClaimRow

def inspect_images_per_image(image_paths, claim_object, claimed_family, claimed_part,
                              minimum_image_evidence, history_summary):
    """Config B: one VLM call per image, then merge results."""
    all_results = []
    for img_id, img_path in image_paths:
        # Call inspect_images with a single image at a time
        single_result = inspect_images_batched(
            [(img_id, img_path)],
            claim_object, claimed_family, claimed_part,
            minimum_image_evidence, history_summary
        )
        per_img = single_result.get("per_image", [])
        all_results.extend(per_img)
    
    return {"per_image": all_results}

def run_config(sample_claims, evidence_reqs, user_history, expected_rows, config_name, use_per_image=False):
    """Run a single config on sample claims and return results."""
    print(f"\n{'='*50}")
    print(f"Running Config {config_name}...")
    print(f"{'='*50}")
    
    start = time.time()
    predicted = []
    total_calls = 0
    
    for i, claim in enumerate(sample_claims):
        print(f"  [{i+1}/{len(sample_claims)}] {claim.user_id}...", end=" ")
        
        images = parse_image_paths(claim.image_paths)
        pre_checks = [run_pre_checks(img_id, path) for img_id, path in images]
        
        if use_per_image:
            # Modified pipeline: per-image calls
            from tools import parse_claim, get_evidence_requirement, get_user_history
            parsed = parse_claim(claim.user_claim, claim.claim_object)
            requirement = get_evidence_requirement(claim.claim_object, parsed["claimed_family"], evidence_reqs)
            history = get_user_history(claim.user_id, user_history)
            inspection = inspect_images_per_image(
                images, claim.claim_object, parsed["claimed_family"],
                parsed["claimed_part"], requirement["minimum_image_evidence"],
                history["history_summary"]
            )
            observation = {
                "claim": claim, "parsed": parsed, "requirement": requirement,
                "history": history, "inspection": inspection,
                "pre_checks": pre_checks, "image_ids": [img_id for img_id, _ in images]
            }
            total_calls += 1 + len(images)  # 1 parse + N inspect
        else:
            observation = process_claim(claim, evidence_reqs, user_history, pre_checks)
            total_calls += 2  # 1 parse + 1 batched inspect
        
        row = adjudicate(observation)
        validated = validate_row(row, claim.claim_object)
        predicted.append(validated)
        print(f"-> {validated['claim_status']}")
    
    elapsed = time.time() - start
    report = per_column_report(predicted, expected_rows)
    
    return {
        "config": config_name,
        "elapsed": elapsed,
        "total_calls": total_calls,
        "claim_status_accuracy": report["claim_status"]["accuracy"],
        "issue_type_accuracy": report["issue_type"]["accuracy"],
        "object_part_accuracy": report["object_part"]["accuracy"],
        "evidence_met_accuracy": report["evidence_standard_met"]["accuracy"],
        "valid_image_accuracy": report["valid_image"]["accuracy"],
        "severity_accuracy": report["severity"]["accuracy"],
        "full_report": report,
    }

def main():
    claims_data, evidence_reqs, user_history = load_all_data()
    
    expected_rows = []
    sample_claims = []
    with open(SAMPLE_CLAIMS_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            expected_rows.append(row)
            sample_claims.append(ClaimRow(
                user_id=row["user_id"],
                image_paths=row["image_paths"],
                user_claim=row["user_claim"],
                claim_object=row["claim_object"]
            ))
    
    # Config A: batched (default)
    result_a = run_config(sample_claims, evidence_reqs, user_history, expected_rows,
                          "A (batched)", use_per_image=False)
    
    # Config B: per-image
    result_b = run_config(sample_claims, evidence_reqs, user_history, expected_rows,
                          "B (per-image)", use_per_image=True)
    
    # Print comparison
    print(f"\n{'='*70}")
    print("CONFIGURATION COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Config A (batched)':>18} {'Config B (per-img)':>18} {'Delta':>10}")
    print("-" * 76)
    
    metrics = [
        ("claim_status_accuracy", "Claim Status Acc"),
        ("issue_type_accuracy", "Issue Type Acc"),
        ("severity_accuracy", "Severity Acc"),
        ("valid_image_accuracy", "Valid Image Acc"),
        ("elapsed", "Time (seconds)"),
        ("total_calls", "API Calls"),
    ]
    
    for key, label in metrics:
        a_val = result_a[key]
        b_val = result_b[key]
        delta = b_val - a_val
        if isinstance(a_val, float) and key != "elapsed":
            print(f"{label:<30} {a_val:>17.1%} {b_val:>17.1%} {delta:>+9.1%}")
        else:
            a_str = f"{a_val:.1f}" if isinstance(a_val, float) else str(a_val)
            b_str = f"{b_val:.1f}" if isinstance(b_val, float) else str(b_val)
            d_str = f"{delta:+.1f}" if isinstance(delta, float) else f"{delta:+d}"
            print(f"{label:<30} {a_str:>18} {b_str:>18} {d_str:>10}")
    
    winner = "A (batched)" if result_a["claim_status_accuracy"] >= result_b["claim_status_accuracy"] else "B (per-image)"
    print(f"\nSelected config: {winner}")
    
    # Save comparison
    comparison_path = os.path.join(os.path.dirname(__file__), "comparison_results.json")
    with open(comparison_path, "w") as f:
        json.dump({"config_a": {k: v for k, v in result_a.items() if k != "full_report"},
                    "config_b": {k: v for k, v in result_b.items() if k != "full_report"},
                    "winner": winner}, f, indent=2)
    print(f"Comparison saved to {comparison_path}")

if __name__ == "__main__":
    main()
