import argparse
import json
from datetime import datetime
from config import OUTPUT_CSV, DATASET_DIR, REPO_ROOT
from ingest import load_all_data, parse_image_paths
from cv_checks import run_pre_checks
from orchestrator import process_claim
from adjudicator import adjudicate
from validator import validate_row, write_output_csv

class PipelineLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.entries = []
        self.start_time = datetime.now().isoformat()
    
    def log_claim(self, user_id: str, stage: str, data: dict):
        self.entries.append({
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "stage": stage,
            "data": data
        })
    
    def save(self):
        output = {
            "run_start": self.start_time,
            "run_end": datetime.now().isoformat(),
            "total_claims": len(set(e["user_id"] for e in self.entries)),
            "entries": self.entries
        }
        with open(self.log_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

def process_claim_stub(claim, evidence_reqs, user_history) -> dict:
    """Stub processor — returns valid but placeholder output."""
    images = parse_image_paths(claim.image_paths)
    image_ids = [img_id for img_id, _ in images]
    
    return {
        "user_id": claim.user_id,
        "image_paths": claim.image_paths,
        "user_claim": claim.user_claim,
        "claim_object": claim.claim_object,
        "evidence_standard_met": "true",
        "evidence_standard_met_reason": "Placeholder — VLM not yet connected.",
        "risk_flags": "none",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "Placeholder — VLM not yet connected.",
        "supporting_image_ids": ";".join(image_ids) if image_ids else "none",
        "valid_image": "true",
        "severity": "unknown"
    }

def stub_adjudicate(obs: dict) -> dict:
    """Temporary: maps observations to output. Phase 3 replaces with real adjudicator."""
    claim = obs["claim"]
    parsed = obs["parsed"]
    inspection = obs["inspection"]
    per_image = inspection.get("per_image", [])
    
    best = max(per_image, key=lambda x: x.get("confidence", 0)) if per_image else {}
    
    return {
        "user_id": claim.user_id,
        "image_paths": claim.image_paths,
        "user_claim": claim.user_claim,
        "claim_object": claim.claim_object,
        "evidence_standard_met": "true",
        "evidence_standard_met_reason": "Placeholder pending adjudicator.",
        "risk_flags": "none",
        "issue_type": best.get("issue_type_detected", "unknown"),
        "object_part": best.get("object_part_detected", "unknown"),
        "claim_status": "not_enough_information",
        "claim_status_justification": "Pending adjudicator implementation.",
        "supporting_image_ids": ";".join(obs["image_ids"]) if obs["image_ids"] else "none",
        "valid_image": "true",
        "severity": "unknown"
    }

def main():
    parser = argparse.ArgumentParser(description="Process damage claims verification.")
    parser.add_argument("--stub", action="store_true", help="Run with placeholder stub processor.")
    parser.add_argument("--sample", action="store_true", help="Run on sample_claims.csv instead of claims.csv.")
    args = parser.parse_args()

    import time
    start_time = time.time()
    print(f"Pipeline started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    from ingest import load_claims, load_evidence_requirements, load_user_history
    from config import EVIDENCE_REQ_CSV, USER_HISTORY_CSV, SAMPLE_CLAIMS_CSV, CLAIMS_CSV
    
    claims_path = SAMPLE_CLAIMS_CSV if args.sample else CLAIMS_CSV
    claims = load_claims(str(claims_path))
    evidence_reqs = load_evidence_requirements(str(EVIDENCE_REQ_CSV))
    user_history = load_user_history(str(USER_HISTORY_CSV))
    
    logger = PipelineLogger(str(REPO_ROOT / "pipeline_run.json"))
    
    results = []
    for i, claim in enumerate(claims):
        print(f"[{i+1}/{len(claims)}] Processing {claim.user_id}...")

        try:
            if args.stub:
                row = process_claim_stub(claim, evidence_reqs, user_history)
                logger.log_claim(claim.user_id, "adjudication", row)
            else:
                # Run pre-checks
                images = parse_image_paths(claim.image_paths)
                pre_checks = [run_pre_checks(img_id, path) for img_id, path in images]
                logger.log_claim(claim.user_id, "pre_checks", {"pre_checks": pre_checks})

                # Agent loop
                observation = process_claim(claim, evidence_reqs, user_history, pre_checks)
                logger.log_claim(claim.user_id, "parse_claim", observation["parsed"])
                logger.log_claim(claim.user_id, "inspection", observation["inspection"])

                # Real adjudicator integration
                row = adjudicate(observation)
                logger.log_claim(claim.user_id, "adjudication", row)

            validated = validate_row(row, claim.claim_object)

        except Exception as e:
            print(f"  [ERROR] {claim.user_id} failed: {e} — inserting safe default row")
            images_safe = parse_image_paths(claim.image_paths)
            safe_row = {
                "user_id": claim.user_id,
                "image_paths": claim.image_paths,
                "user_claim": claim.user_claim,
                "claim_object": claim.claim_object,
                "evidence_standard_met": "false",
                "evidence_standard_met_reason": "Processing error — could not evaluate this claim.",
                "risk_flags": "manual_review_required",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": "An error occurred during processing. Manual review required.",
                "supporting_image_ids": "none",
                "valid_image": "false",
                "severity": "unknown",
            }
            validated = validate_row(safe_row, claim.claim_object)
            logger.log_claim(claim.user_id, "error", {"error": str(e)})

        logger.log_claim(claim.user_id, "verdict", {
            "claim_status": validated["claim_status"],
            "severity": validated["severity"],
            "claim_status_justification": validated["claim_status_justification"]
        })
        results.append(validated)
    
    write_output_csv(results, OUTPUT_CSV)
    logger.save()
    
    end_time = time.time()
    duration = end_time - start_time
    
    from collections import Counter
    statuses = Counter(r["claim_status"] for r in results)
    
    print(f"Pipeline ended at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total duration: {duration:.2f} seconds")
    print(f"Claims processed: {len(results)}")
    print(f"Status distribution: {dict(statuses)}")
    print(f"Written {len(results)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

