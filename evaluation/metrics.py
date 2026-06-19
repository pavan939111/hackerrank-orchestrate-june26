"""Metrics for comparing predicted vs expected claim review outputs."""

from collections import Counter

def accuracy(predicted: list[str], expected: list[str]) -> float:
    """Simple accuracy: fraction of exact matches."""
    if not expected:
        return 0.0
    correct = sum(1 for p, e in zip(predicted, expected) if p.strip().lower() == e.strip().lower())
    return correct / len(expected)

def confusion_matrix(predicted: list[str], expected: list[str], labels: list[str]) -> dict:
    """Build confusion matrix as dict of dicts: matrix[expected][predicted] = count."""
    matrix = {e: {p: 0 for p in labels} for e in labels}
    for p, e in zip(predicted, expected):
        p_clean = p.strip().lower()
        e_clean = e.strip().lower()
        if e_clean in matrix and p_clean in matrix[e_clean]:
            matrix[e_clean][p_clean] += 1
    return matrix

def set_f1(predicted_str: str, expected_str: str, delimiter: str = ";") -> dict:
    """F1 score for set-valued fields like risk_flags and supporting_image_ids."""
    pred_set = set(x.strip().lower() for x in predicted_str.split(delimiter) if x.strip())
    exp_set = set(x.strip().lower() for x in expected_str.split(delimiter) if x.strip())
    
    if not pred_set and not exp_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set or not exp_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    tp = len(pred_set & exp_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(exp_set) if exp_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}

def per_column_report(predicted_rows: list[dict], expected_rows: list[dict]) -> dict:
    """Compare predicted vs expected across all scoreable columns."""
    
    EXACT_MATCH_COLS = [
        "claim_status", "issue_type", "object_part",
        "evidence_standard_met", "valid_image", "severity"
    ]
    SET_COLS = ["risk_flags", "supporting_image_ids"]
    
    report = {}
    
    for col in EXACT_MATCH_COLS:
        pred = [r.get(col, "").strip().lower() for r in predicted_rows]
        exp = [r.get(col, "").strip().lower() for r in expected_rows]
        acc = accuracy(pred, exp)
        report[col] = {
            "accuracy": acc,
            "correct": sum(1 for p, e in zip(pred, exp) if p == e),
            "total": len(exp),
            "mismatches": [
                {"row": i, "user_id": expected_rows[i].get("user_id", "?"),
                 "predicted": pred[i], "expected": exp[i]}
                for i in range(len(exp)) if pred[i] != exp[i]
            ]
        }
    
    for col in SET_COLS:
        pred = [r.get(col, "none") for r in predicted_rows]
        exp = [r.get(col, "none") for r in expected_rows]
        f1_scores = [set_f1(p, e) for p, e in zip(pred, exp)]
        avg_f1 = sum(s["f1"] for s in f1_scores) / len(f1_scores) if f1_scores else 0.0
        report[col] = {
            "avg_f1": avg_f1,
            "per_row": [
                {"row": i, "user_id": expected_rows[i].get("user_id", "?"),
                 "predicted": pred[i], "expected": exp[i],
                 "f1": f1_scores[i]["f1"]}
                for i in range(len(exp)) if f1_scores[i]["f1"] < 1.0
            ]
        }
    
    return report

def print_report(report: dict):
    """Pretty print the evaluation report to console."""
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    
    for col, data in report.items():
        if "accuracy" in data:
            status = "PASS" if data["accuracy"] >= 0.8 else "WARN" if data["accuracy"] >= 0.6 else "FAIL"
            print(f"\n[{status}] {col}: {data['accuracy']:.1%} ({data['correct']}/{data['total']})")
            if data["mismatches"]:
                for m in data["mismatches"][:5]:
                    print(f"       {m['user_id']}: predicted={m['predicted']} expected={m['expected']}")
        elif "avg_f1" in data:
            status = "PASS" if data["avg_f1"] >= 0.8 else "WARN" if data["avg_f1"] >= 0.6 else "FAIL"
            print(f"\n[{status}] {col}: avg_f1={data['avg_f1']:.2f}")
            if data["per_row"]:
                for m in data["per_row"][:5]:
                    print(f"       {m['user_id']}: pred={m['predicted']} exp={m['expected']} f1={m['f1']:.2f}")
    
    print("\n" + "=" * 70)
