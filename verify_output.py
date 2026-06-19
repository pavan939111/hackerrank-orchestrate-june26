# Save as verify_output.py in repo root, run it
import csv
import sys
import os

# Add code/ to path to resolve imports avoiding name conflict with built-in code module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))

from schemas import OUTPUT_COLUMNS, CLAIM_STATUSES, ISSUE_TYPES, SEVERITIES, RISK_FLAGS

with open("output.csv", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

print(f"Rows: {len(rows)} (expected 44)")
assert len(rows) == 44, f"FAIL: got {len(rows)} rows"

# Column order
with open("output.csv", newline="", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    header = next(reader)
assert header == OUTPUT_COLUMNS, f"FAIL: column mismatch\nGot:      {header}\nExpected: {OUTPUT_COLUMNS}"
print("Column order: PASS")

# Status distribution
from collections import Counter
statuses = Counter(r["claim_status"] for r in rows)
print(f"Status distribution: {dict(statuses)}")
assert len(statuses) > 1, "FAIL: all same status"

# No empty fields
for i, row in enumerate(rows):
    for col in OUTPUT_COLUMNS:
        val = row.get(col, "").strip()
        assert val, f"FAIL: row {i} ({row['user_id']}) has empty {col}"
print("No empty fields: PASS")

# Valid enums
for row in rows:
    assert row["claim_status"] in CLAIM_STATUSES, f"Bad status: {row['claim_status']} in {row['user_id']}"
    assert row["issue_type"] in ISSUE_TYPES + ["unknown"], f"Bad issue: {row['issue_type']} in {row['user_id']}"
    assert row["severity"] in SEVERITIES, f"Bad severity: {row['severity']} in {row['user_id']}"
    assert row["evidence_standard_met"] in ("true", "false"), f"Bad evidence_met: {row['evidence_standard_met']}"
    assert row["valid_image"] in ("true", "false"), f"Bad valid_image: {row['valid_image']}"
    flags = row["risk_flags"].split(";")
    for f in flags:
        assert f.strip() in RISK_FLAGS, f"Bad flag: {f} in {row['user_id']}"
print("Enum validation: PASS")

# Justification not empty and references image IDs for supported claims
for row in rows:
    j = row["claim_status_justification"]
    assert len(j) > 20, f"FAIL: justification too short for {row['user_id']}: {j}"
    if row["claim_status"] == "supported":
        ids = row["supporting_image_ids"]
        assert ids != "none", f"FAIL: supported but no supporting_image_ids for {row['user_id']}"
print("Justification quality: PASS")

print("\n=== ALL CHECKS PASSED ===")
