import difflib
from dataclasses import dataclass
from typing import List, Dict, Any

CLAIM_STATUSES = ["supported", "contradicted", "not_enough_information"]

ISSUE_TYPES = ["dent", "scratch", "crack", "glass_shatter", "broken_part", 
               "missing_part", "torn_packaging", "crushed_packaging", 
               "water_damage", "stain", "none", "unknown"]

CAR_PARTS = ["front_bumper", "rear_bumper", "door", "hood", "windshield",
             "side_mirror", "headlight", "taillight", "fender", 
             "quarter_panel", "body", "unknown"]

LAPTOP_PARTS = ["screen", "keyboard", "trackpad", "hinge", "lid", 
                "corner", "port", "base", "body", "unknown"]

PACKAGE_PARTS = ["box", "package_corner", "package_side", "seal", 
                 "label", "contents", "item", "unknown"]

OBJECT_PARTS = {"car": CAR_PARTS, "laptop": LAPTOP_PARTS, "package": PACKAGE_PARTS}

RISK_FLAGS = ["none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
              "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
              "claim_mismatch", "possible_manipulation", "non_original_image",
              "text_instruction_present", "user_history_risk", "manual_review_required"]

SEVERITIES = ["none", "low", "medium", "high", "unknown"]

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids",
    "valid_image", "severity"
]

def snap_to_enum(value: Any, allowed: List[str]) -> str:
    """Returns the closest match from the allowed list using difflib.get_close_matches.
    
    If no match with cutoff=0.4 is found, fallback to 'unknown'.
    """
    if value is None:
        return "unknown"
    
    # If it is a list or tuple, extract the first element or convert it
    if isinstance(value, (list, tuple)):
        if len(value) > 0:
            value = value[0]
        else:
            return "unknown"
            
    val_str = str(value).strip().lower()
    if not val_str:
        return "unknown"
    if val_str == "none":
        if "none" in allowed:
            return "none"
        return "unknown"
        
    # Exact match check first
    if val_str in allowed:
        return val_str
        
    matches = difflib.get_close_matches(val_str, allowed, n=1, cutoff=0.4)
    if matches:
        return matches[0]
    return "unknown"

@dataclass
class ClaimRow:
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str

@dataclass
class VerdictRow:
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: str
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: str
    severity: str
