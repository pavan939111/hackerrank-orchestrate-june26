import csv
from typing import List, Dict, Any
from schemas import (
    snap_to_enum,
    CLAIM_STATUSES,
    ISSUE_TYPES,
    OBJECT_PARTS,
    RISK_FLAGS,
    SEVERITIES,
    OUTPUT_COLUMNS
)

def to_bool_str(val: Any) -> str:
    """Normalize boolean-like values to lowercase 'true' or 'false'."""
    if isinstance(val, bool):
        return "true" if val else "false"
    val_str = str(val).strip().lower()
    if val_str in ("true", "1", "yes", "y"):
        return "true"
    return "false"

def validate_row(row: Dict[str, Any], claim_object: str) -> Dict[str, Any]:
    """Validate and clean values of a row to ensure strict schema adherence."""
    # 1. Snap claim_status to CLAIM_STATUSES
    claim_status = snap_to_enum(row.get("claim_status", "not_enough_information"), CLAIM_STATUSES)
    
    # 2. Snap issue_type to ISSUE_TYPES
    issue_type = snap_to_enum(row.get("issue_type", "unknown"), ISSUE_TYPES)
    
    # 3. Snap object_part to OBJECT_PARTS[claim_object]
    allowed_parts = OBJECT_PARTS.get(claim_object, ["unknown"])
    object_part = snap_to_enum(row.get("object_part", "unknown"), allowed_parts)
    
    # 4. Snap severity to SEVERITIES
    severity = snap_to_enum(row.get("severity", "unknown"), SEVERITIES)
    
    # 5. Ensure evidence_standard_met and valid_image are lowercase "true"/"false" strings
    evidence_standard_met = to_bool_str(row.get("evidence_standard_met", "true"))
    valid_image = to_bool_str(row.get("valid_image", "true"))
    
    # 6. Parse risk_flags by semicolon, snap each to RISK_FLAGS, rejoin with semicolons, default to "none"
    raw_flags = row.get("risk_flags", "none")
    if not raw_flags:
        risk_flags = "none"
    else:
        flags = [f.strip() for f in str(raw_flags).split(";") if f.strip()]
        snapped = [snap_to_enum(f, RISK_FLAGS) for f in flags]
        unique = []
        for f in snapped:
            if f not in unique:
                unique.append(f)
        if len(unique) > 1 and "none" in unique:
            unique = [f for f in unique if f != "none"]
        risk_flags = ";".join(unique) if unique else "none"
        
    # 7. Ensure supporting_image_ids is semicolon-separated IDs or "none"
    raw_ids = row.get("supporting_image_ids", "none")
    if not raw_ids:
        supporting_image_ids = "none"
    else:
        ids = [i.strip() for i in str(raw_ids).split(";") if i.strip()]
        supporting_image_ids = ";".join(ids) if ids else "none"
        
    # 8. Ensure claim_status_justification is non-empty (if empty, set to "Insufficient evidence for assessment.")
    justification = str(row.get("claim_status_justification", "")).strip()
    if not justification:
        justification = "Insufficient evidence for assessment."
        
    # Build final dict in exact OUTPUT_COLUMNS order
    return {
        "user_id": row.get("user_id", ""),
        "image_paths": row.get("image_paths", ""),
        "user_claim": row.get("user_claim", ""),
        "claim_object": row.get("claim_object", ""),
        "evidence_standard_met": evidence_standard_met,
        "evidence_standard_met_reason": row.get("evidence_standard_met_reason", ""),
        "risk_flags": risk_flags,
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": justification,
        "supporting_image_ids": supporting_image_ids,
        "valid_image": valid_image,
        "severity": severity
    }

def write_output_csv(rows: List[Dict[str, Any]], output_path: str):
    """Write rows to CSV with exact column order and proper quoting matching input formats."""
    with open(output_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for r in rows:
            ordered_row = {col: r.get(col, '') for col in OUTPUT_COLUMNS}
            writer.writerow(ordered_row)
