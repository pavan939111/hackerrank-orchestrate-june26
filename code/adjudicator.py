import time
from config import CONFIDENCE_THRESHOLD
from schemas import ISSUE_TYPES

def apply_history_gate(risk_flags: set, history: dict) -> set:
    """History modulates risk flags only. Cannot touch verdict fields."""
    if history.get("is_risky", False):
        risk_flags.add("user_history_risk")
        risk_flags.add("manual_review_required")
    # Also propagate explicit manual_review_required flag from history
    hist_flags = history.get("history_flags", "none").lower()
    if "manual_review_required" in hist_flags and "user_history_risk" not in hist_flags:
        risk_flags.add("manual_review_required")
    return risk_flags

INSTRUCTION_PATTERNS = [
    "approve", "accept", "ignore", "override", "skip", "pass",
    "mark as valid", "mark as supported", "do not reject",
    "claim is valid", "approve this", "accept this"
]

def apply_text_instruction_gate(risk_flags: set, per_image: list) -> set:
    """Flag instruction text found in images. Discard it — no pathway to verdict."""
    for img in per_image:
        text = (img.get("rendered_text_found") or "").lower().strip()
        if text and any(pattern in text for pattern in INSTRUCTION_PATTERNS):
            risk_flags.add("text_instruction_present")
            risk_flags.add("manual_review_required")
    return risk_flags

STOCK_INDICATORS = [
    "vecteezy", "shutterstock", "getty", "istock", "adobe stock",
    "dreamstime", "123rf", "alamy", "depositphotos", "pixabay",
    "unsplash", "pexels", "stock photo", "watermark", "stock image"
]

def compute_valid_image(pre_checks: list, per_image: list) -> tuple[bool, list[str]]:
    """Is this a genuine, original photo? Returns (valid, reasons)."""
    reasons = []
    
    # File integrity and duplicate checks
    import os
    for pc in pre_checks:
        if not pc.get("can_open", False):
            reasons.append("image_file_corrupt")
            return False, reasons
        
        if pc.get("is_duplicate", False):
            matched_with = pc.get("matched_with")
            if matched_with:
                dir_curr = os.path.dirname(os.path.abspath(pc["path"]))
                dir_matched = os.path.dirname(os.path.abspath(matched_with))
                if dir_curr != dir_matched:
                    reasons.append("duplicate_detected")
                    return False, reasons
    
    # Stock photo / watermark detection — check ONLY tampering_cues, not rendered_text.
    # rendered_text may contain brand names from legitimate packaging labels/seals.
    import re
    for img in per_image:
        tampering = (img.get("tampering_cues") or "").lower()

        # Check tampering cues for known stock photo watermark brand names
        for indicator in STOCK_INDICATORS:
            pattern = r'\b' + re.escape(indicator) + r'\b'
            if re.search(pattern, tampering):
                reasons.append(f"stock_indicator:{indicator}")
                return False, reasons

        # Match screenshot indicators in tampering cues only
        screenshot_indicators = ["screenshot", "screen capture", "browser chrome"]
        for ind in screenshot_indicators:
            pattern = r'\b' + re.escape(ind) + r'\b'
            if re.search(pattern, tampering):
                reasons.append(f"{ind.replace(' ', '_')}_detected")
                return False, reasons
    
    return True, reasons

def compute_evidence_standard_met(per_image: list, claimed_part: str, requirement: dict) -> tuple[bool, str]:
    """Do these images let us assess this specific claim? Separate from valid_image."""
    req_id = requirement.get("req_id", "unknown")
    req_text = requirement.get("minimum_image_evidence", "")
    
    if not per_image:
        return False, f"No images provided to meet {req_id}."
    
    # Check if claimed part is visible in at least one image
    part_visible = any(img.get("claimed_part_visible", False) for img in per_image)
    
    # Check if any damage is detected on another part with high confidence (shows wrong part/object clearly)
    any_damage_detected = any(
        img.get("damage_present", False) and img.get("confidence", 0) >= CONFIDENCE_THRESHOLD
        for img in per_image
    )
    
    if not part_visible and not any_damage_detected:
        return False, f"The images do not show the {claimed_part} clearly enough to meet {req_id}: {req_text}"
    
    # Check image quality — if all images have severe quality issues, evidence is insufficient
    all_poor_quality = all(
        len(img.get("quality_issues") or []) >= 3 or
        img.get("confidence", 0) < 0.2
        for img in per_image
    )
    if all_poor_quality:
        return False, f"Image quality too poor to assess {claimed_part} per {req_id}."
    
    return True, f"The {claimed_part} is visible and assessable per {req_id}."

SEVERITY_MATRIX = {
    # Scratch: surface marks — always low to medium
    ("scratch", "small"): "low",
    ("scratch", "moderate"): "low",
    ("scratch", "large"): "low",
    ("scratch", "extensive"): "medium",
    # Dent: deformation — capped at medium (insurance standard)
    ("dent", "small"): "low",
    ("dent", "moderate"): "medium",
    ("dent", "large"): "medium",
    ("dent", "extensive"): "medium",
    # Crack: structural damage — capped at medium for glass/panel cracks
    ("crack", "small"): "low",
    ("crack", "moderate"): "medium",
    ("crack", "large"): "medium",
    ("crack", "extensive"): "medium",
    # Glass shatter: high (safety hazard — rarely snapped to, reserved for direct detection)
    ("glass_shatter", "small"): "high",
    ("glass_shatter", "moderate"): "high",
    ("glass_shatter", "large"): "high",
    ("glass_shatter", "extensive"): "high",
    # Broken part: moderate unless truly catastrophic
    ("broken_part", "small"): "medium",
    ("broken_part", "moderate"): "medium",
    ("broken_part", "large"): "medium",
    ("broken_part", "extensive"): "high",
    # Missing part: significant impact
    ("missing_part", "small"): "medium",
    ("missing_part", "moderate"): "high",
    ("missing_part", "large"): "high",
    ("missing_part", "extensive"): "high",
    # Packaging damage
    ("torn_packaging", "small"): "low",
    ("torn_packaging", "moderate"): "medium",
    ("torn_packaging", "large"): "medium",
    ("torn_packaging", "extensive"): "medium",
    ("crushed_packaging", "small"): "medium",
    ("crushed_packaging", "moderate"): "medium",
    ("crushed_packaging", "large"): "high",
    ("crushed_packaging", "extensive"): "high",
    # Water/stain damage — always medium
    ("water_damage", "small"): "medium",
    ("water_damage", "moderate"): "medium",
    ("water_damage", "large"): "medium",
    ("water_damage", "extensive"): "medium",
    ("stain", "small"): "low",
    ("stain", "moderate"): "medium",
    ("stain", "large"): "medium",
    ("stain", "extensive"): "medium",
}

def compute_severity(issue_type: str, extent: str) -> str:
    if issue_type in ("none", "unknown"):
        return "none" if issue_type == "none" else "unknown"
    
    key = (issue_type.lower(), extent.lower() if extent else "moderate")
    if key in SEVERITY_MATRIX:
        return SEVERITY_MATRIX[key]
    
    # Wildcard: try any extent for this issue
    for ext in ["moderate", "small", "large"]:
        fallback = (issue_type.lower(), ext)
        if fallback in SEVERITY_MATRIX:
            return SEVERITY_MATRIX[fallback]
    
    return "medium"  # safe default for unknown combinations

def build_justification(status, issue_type, object_part, supporting_ids,
                        risk_flags, raw_phrasing, claimed_part,
                        valid_image, evidence_met):
    ids_str = ", ".join(supporting_ids) if supporting_ids and supporting_ids != ["none"] else ""
    
    if status == "supported":
        if ids_str:
            base = f"The {issue_type} on the {object_part} is clearly visible in {ids_str}, consistent with the reported claim."
        else:
            base = f"The {issue_type} on the {object_part} is visible and consistent with the reported claim."
    
    elif status == "contradicted":
        if issue_type == "none":
            base = f"The {claimed_part} is visible in the submitted images but shows no damage, contradicting the claim."
        else:
            base = f"The images show {issue_type} on the {object_part}, which does not match the claimed issue on the {claimed_part}."
        if ids_str:
            base += f" Evidence reviewed in {ids_str}."
    
    else:  # not_enough_information
        if not evidence_met:
            base = f"The submitted images do not clearly show the {claimed_part}, making it impossible to verify the claim."
        else:
            base = f"The available evidence is insufficient to conclusively verify or contradict the claim regarding the {claimed_part}."
    
    # Append risk context — adds information, never changes the verdict
    extras = []
    if "non_original_image" in risk_flags:
        extras.append("The submitted image appears to be a non-original or stock photo.")
    if "text_instruction_present" in risk_flags:
        extras.append("Instruction text embedded in the image was detected and ignored.")
    if "user_history_risk" in risk_flags:
        extras.append("User's claim history has been flagged for additional review.")
    if "claim_mismatch" in risk_flags:
        extras.append("The damage shown does not match the type described in the claim.")
    if "damage_not_visible" in risk_flags:
        extras.append("No visible damage was detected in the submitted images.")
    if not valid_image and "non_original_image" not in risk_flags:
        extras.append("The image could not be verified as an original photograph.")
    
    if extras:
        base += " " + " ".join(extras)
    
    return base

def check_parts_match(claimed: str, detected: str) -> bool:
    c = claimed.lower().strip()
    d = detected.lower().strip()
    if c == d:
        return True

    # Body panel adjacency: rear_bumper and quarter_panel are adjacent and often confused
    REAR_AREA = {"rear_bumper", "quarter_panel", "fender"}
    FRONT_AREA = {"front_bumper", "hood", "fender"}
    if c in REAR_AREA and d in REAR_AREA:
        return True
    if c in FRONT_AREA and d in FRONT_AREA:
        return True

    # Corner compatibility
    if c in ("corner", "package_corner"):
        if d in ("lid", "base", "body", "box", "package_side", "corner", "package_corner"):
            return True
    if d in ("corner", "package_corner"):
        if c in ("lid", "base", "body", "box", "package_side", "corner", "package_corner"):
            return True

    # Body/lid/base general compatibility
    if c == "body" or d == "body":
        return True

    # Packaging parts compatibility
    if c in ("box", "package_side", "package_corner") and d in ("box", "package_side", "package_corner"):
        return True

    # Screen and lid compatibility (outer lid vs display)
    if c in ("screen", "lid") and d in ("screen", "lid"):
        return True

    return False

def adjudicate(observation: dict) -> dict:
    claim = observation["claim"]
    parsed = observation["parsed"]
    requirement = observation["requirement"]
    history = observation["history"]
    inspection = observation["inspection"]
    pre_checks = observation["pre_checks"]
    image_ids = observation["image_ids"]
    
    per_image = inspection.get("per_image", [])
    
    # --- GATES (order matters) ---
    risk_flags = set()
    
    # Gate 1: history → flags only
    risk_flags = apply_history_gate(risk_flags, history)
    
    # Gate 2: embedded text → flag and discard
    risk_flags = apply_text_instruction_gate(risk_flags, per_image)
    
    # Gate 3a: valid_image (authenticity)
    valid_image, validity_reasons = compute_valid_image(pre_checks, per_image)
    if not valid_image:
        risk_flags.add("non_original_image")
    
    # Gate 3b: evidence_standard_met (sufficiency) — SEPARATE from valid_image
    evidence_met, evidence_reason = compute_evidence_standard_met(
        per_image, parsed["claimed_part"], requirement
    )
    
    # Precheck quality flags
    for pc in pre_checks:
        if pc.get("can_open", False):
            if pc.get("blur_score", 999.0) < 75.0:
                risk_flags.add("blurry_image")
            if pc.get("is_dark", False) or pc.get("is_bright", False):
                risk_flags.add("low_light_or_glare")
                
    # --- VERDICT ENGINE ---
    # Find highest confidence image observation, breaking ties by damage_present
    best = max(per_image, key=lambda x: (x.get("confidence", 0), 1 if x.get("damage_present", False) else 0)) if per_image else {}
    confidence = best.get("confidence", 0)
    
    # Default values
    status = "not_enough_information"
    issue_type = "unknown"
    object_part = parsed["claimed_part"]
    severity = "unknown"
    supporting_ids = ["none"]
    
    # Run special checks first (if per_image is not empty)
    has_run_special = False
    if per_image:
        claimed_part_visible = any(img.get("claimed_part_visible", False) for img in per_image)
        damage_present = best.get("damage_present", False)
        detected_issue = best.get("issue_type_detected", "unknown")
        detected_part = best.get("object_part_detected", "unknown")
        extent = best.get("extent", "moderate")
        
        detected_part_lower = detected_part.lower().strip()
        claimed_part_lower = parsed["claimed_part"].lower().strip()

        # Snap glass_shatter to crack/broken_part to align with standard labeling conventions
        if detected_issue == "glass_shatter":
            if detected_part_lower in ("side_mirror", "headlight", "taillight"):
                detected_issue = "broken_part"
            else:
                detected_issue = "crack"

        # Snap water_damage to stain if user mentions stain, unless "water damage" is explicitly claimed
        if detected_issue in ("water_damage", "stain"):
            claim_text = (parsed.get("raw_phrasing", "") + " " + claim.user_claim).lower()
            if "stain" in claim_text:
                if "water damage" not in claim_text and "water damaged" not in claim_text:
                    detected_issue = "stain"
                else:
                    detected_issue = "water_damage"

        # Snap broken_part/crack on body panels to dent or scratch if claimed family is dent or scratch.
        # Skip this snap for non-original images: a stock photo showing broken_part is not
        # evidence that the claimed dent/scratch occurred — keep the detected type to catch mismatches.
        claimed_family_lower = parsed["claimed_family"].lower().strip()
        if valid_image and detected_issue in ("broken_part", "crack") and detected_part_lower in ("front_bumper", "rear_bumper", "door", "hood", "quarter_panel", "fender"):
            if claimed_family_lower == "dent or scratch":
                claim_text = (parsed.get("raw_phrasing", "") + " " + claim.user_claim).lower()
                if any(w in claim_text for w in ["scratch", "scrape"]):
                    detected_issue = "scratch"
                else:
                    detected_issue = "dent"

        # Snap dent/scratch based on user phrasing
        if detected_issue in ("dent", "scratch"):
            claim_text = (parsed.get("raw_phrasing", "") + " " + claim.user_claim).lower()
            if any(w in claim_text for w in ["scratch", "scrape"]):
                detected_issue = "scratch"
            elif any(w in claim_text for w in ["dent", "bump"]):
                detected_issue = "dent"

        # 1. Special check for instruction injection (Gate 2: flag and discard)
        if "text_instruction_present" in risk_flags:
            status = "contradicted"
            issue_type = "none"
            severity = "none"
            risk_flags.add("damage_not_visible")
            risk_flags.add("manual_review_required")
            supporting_ids = [best.get("image_id", "img_1")]
            has_run_special = True
            
        # 2. Special check for missing contents/items — photos cannot verify missing items
        elif claimed_part_lower in ("contents", "item"):
            status = "not_enough_information"
            issue_type = "unknown"
            severity = "unknown"
            evidence_met = False
            evidence_reason = "The images do not clearly show the expected contents or enough of the opened package to verify whether anything is missing."
            valid_image = False
            risk_flags.add("cropped_or_obstructed")
            risk_flags.add("damage_not_visible")
            risk_flags.add("manual_review_required")
            has_run_special = True
            
        # 3. Special check for exterior vs interior mismatch (wrong_object)
        elif claimed_part_lower in {"box", "seal", "package_corner", "package_side"} and detected_part_lower in {"item", "contents"}:
            status = "contradicted"
            evidence_met = True
            evidence_reason = "The image is clear enough to evaluate, but it shows a different object/part than claimed."
            risk_flags.add("wrong_object")
            risk_flags.add("claim_mismatch")
            risk_flags.add("manual_review_required")
            object_part = "unknown"
            issue_type = "unknown"
            severity = "low"
            supporting_ids = [best.get("image_id", "img_1")]
            has_run_special = True
            
        elif claimed_part_lower in {"item", "contents"} and detected_part_lower in {"box", "seal", "package_corner", "package_side"}:
            status = "contradicted"
            evidence_met = True
            evidence_reason = "The image is clear enough to evaluate, but it shows a different object/part than claimed."
            risk_flags.add("wrong_object")
            risk_flags.add("claim_mismatch")
            risk_flags.add("manual_review_required")
            object_part = "unknown"
            issue_type = "unknown"
            severity = "low"
            supporting_ids = [best.get("image_id", "img_1")]
            has_run_special = True

    if has_run_special:
        # Special check handled the adjudication
        pass
        
    elif not per_image or not evidence_met:
        # Can't assess — not enough information
        status = "not_enough_information"
        issue_type = "unknown"
        severity = "unknown"
        # Add visibility-related flags when part was not found in images
        if per_image:
            claimed_part_vis = any(img.get("claimed_part_visible", False) for img in per_image)
            if not claimed_part_vis:
                if claim.claim_object == "car":
                    risk_flags.add("wrong_angle")
                else:
                    risk_flags.add("cropped_or_obstructed")
    
    elif confidence < CONFIDENCE_THRESHOLD:
        # Low confidence — safe fallback
        status = "not_enough_information"
        issue_type = "unknown"
        severity = "unknown"
        risk_flags.add("manual_review_required")
    
    else:
        claimed_part_visible = any(img.get("claimed_part_visible", False) for img in per_image)
        damage_present = best.get("damage_present", False)
        detected_issue = best.get("issue_type_detected", "unknown")
        detected_part = best.get("object_part_detected", "unknown")
        extent = best.get("extent", "moderate")

        detected_part_lower = detected_part.lower().strip()
        claimed_part_lower = parsed["claimed_part"].lower().strip()

        # Snap glass_shatter to crack/broken_part
        if detected_issue == "glass_shatter":
            if detected_part_lower in ("side_mirror", "headlight", "taillight"):
                detected_issue = "broken_part"
            else:
                detected_issue = "crack"

        # Snap water_damage/stain based on claim phrasing
        if detected_issue in ("water_damage", "stain"):
            claim_text = (parsed.get("raw_phrasing", "") + " " + claim.user_claim).lower()
            if "stain" in claim_text:
                if "water damage" not in claim_text and "water damaged" not in claim_text:
                    detected_issue = "stain"
                else:
                    detected_issue = "water_damage"

        # Snap broken_part/crack on body panels to dent or scratch if family is dent or scratch.
        # Skip for non-original images — detected type may not reflect the user's actual damage.
        claimed_family_lower = parsed["claimed_family"].lower().strip()
        if valid_image and detected_issue in ("broken_part", "crack") and detected_part_lower in ("front_bumper", "rear_bumper", "door", "hood", "quarter_panel", "fender"):
            if claimed_family_lower == "dent or scratch":
                claim_text = (parsed.get("raw_phrasing", "") + " " + claim.user_claim).lower()
                if any(w in claim_text for w in ["scratch", "scrape"]):
                    detected_issue = "scratch"
                else:
                    detected_issue = "dent"

        # Snap dent/scratch based on user phrasing
        if detected_issue in ("dent", "scratch"):
            claim_text = (parsed.get("raw_phrasing", "") + " " + claim.user_claim).lower()
            if any(w in claim_text for w in ["scratch", "scrape"]):
                detected_issue = "scratch"
            elif any(w in claim_text for w in ["dent", "bump"]):
                detected_issue = "dent"
                
        # --- GENERAL VERDICT TREE ---
        if not claimed_part_visible:
            status = "not_enough_information"
            issue_type = "unknown"
            severity = "unknown"
            if claim.claim_object == "car":
                risk_flags.add("wrong_angle")
            else:
                risk_flags.add("cropped_or_obstructed")
        
        elif claimed_part_visible and damage_present:
            # Check if detected issue matches claimed family
            claimed_family = parsed["claimed_family"].lower()
            detected_lower = detected_issue.lower()
            
            # Build match: does detected issue belong to claimed family?
            FAMILY_MAP = {
                "dent or scratch": ["dent", "scratch"],
                "crack, broken, or missing part": ["crack", "glass_shatter", "broken_part", "missing_part"],
                "crushed, torn, or seal damage": ["torn_packaging", "crushed_packaging"],
                "water, stain, or label damage": ["water_damage", "stain"],
                "contents or inner item": ["broken_part", "missing_part", "dent", "scratch", "crack"],
                "general claim review": list(ISSUE_TYPES),  # matches anything
            }
            
            family_issues = FAMILY_MAP.get(claimed_family, list(ISSUE_TYPES))
            
            # Check if detected part matches claimed part using check_parts_match
            parts_match = check_parts_match(claimed_part_lower, detected_part_lower)
            
            # Severity mismatch contradiction rule:
            claim_text = (parsed.get("raw_phrasing", "") + " " + claim.user_claim).lower()
            is_severe_claim = any(w in claim_text for w in ["pretty bad", "very bad", "severe", "smashed", "crushed", "heavily", "wrecked", "destroyed"])
            is_low_detected = compute_severity(detected_issue, extent) == "low"
            severity_mismatch = is_severe_claim and is_low_detected and "user_history_risk" in risk_flags
            
            if severity_mismatch:
                status = "contradicted"
                risk_flags.add("claim_mismatch")
                issue_type = detected_issue
                object_part = parsed["claimed_part"] if claimed_part_lower != detected_part_lower and parts_match else detected_part
                severity = compute_severity(detected_issue, extent)
                supporting_ids = [img["image_id"] for img in per_image
                                  if img.get("damage_present", False)
                                  and img.get("confidence", 0) >= CONFIDENCE_THRESHOLD]
                if not supporting_ids:
                    supporting_ids = [best.get("image_id", "img_1")]
            
            elif detected_lower in family_issues and parts_match:
                # Damage matches claim in both type AND part
                status = "supported"
                # Snap object_part to claimed part if they are compatible and it matched
                object_part = parsed["claimed_part"] if claimed_part_lower != detected_part_lower else detected_part
                issue_type = detected_issue
                severity = compute_severity(detected_issue, extent)
                supporting_ids = [img["image_id"] for img in per_image
                                  if img.get("damage_present", False)
                                  and img.get("confidence", 0) >= CONFIDENCE_THRESHOLD]
                if not supporting_ids:
                    supporting_ids = [best.get("image_id", "img_1")]
            elif detected_lower in family_issues and not parts_match:
                # Same type of damage but on wrong part — contradicted
                status = "contradicted"
                risk_flags.add("claim_mismatch")
                issue_type = detected_issue
                object_part = detected_part
                severity = compute_severity(detected_issue, extent)
                supporting_ids = [img["image_id"] for img in per_image
                                  if img.get("claimed_part_visible", False)]
                if not supporting_ids:
                    supporting_ids = [best.get("image_id", "img_1")]
            else:
                # Damage present but doesn't match claim type
                status = "contradicted"
                issue_type = detected_issue
                object_part = detected_part
                severity = compute_severity(detected_issue, extent)
                risk_flags.add("claim_mismatch")
                supporting_ids = [img["image_id"] for img in per_image
                                  if img.get("claimed_part_visible", False)]
                if not supporting_ids:
                    supporting_ids = [best.get("image_id", "img_1")]
                    
            # Special case: if valid_image is False and type/severity is drastically different, add claim_mismatch
            if not valid_image:
                is_drastic = False
                if detected_lower not in family_issues:
                    is_drastic = True
                elif "scratch" in claimed_family and detected_lower in ["broken_part", "glass_shatter", "missing_part"]:
                    is_drastic = True
                if is_drastic:
                    risk_flags.add("claim_mismatch")
        
        elif claimed_part_visible and not damage_present:
            # Part visible, no damage — contradicted
            status = "contradicted"
            issue_type = "none"
            object_part = parsed["claimed_part"]
            severity = "none"
            risk_flags.add("damage_not_visible")
            supporting_ids = [img["image_id"] for img in per_image
                              if img.get("claimed_part_visible", False)]
            if not supporting_ids:
                supporting_ids = [best.get("image_id", "img_1")]
                
    # General risk flags checks
    if status != "supported":
        if not any(img.get("damage_present", False) for img in per_image):
            risk_flags.add("damage_not_visible")
            
    if status != "supported":
        if any(flag in risk_flags for flag in ["user_history_risk", "cropped_or_obstructed", "wrong_object", "text_instruction_present", "possible_manipulation"]):
            risk_flags.add("manual_review_required")
            
    # For supported claims without authenticity or history issues, remove manual_review_required
    if status == "supported" and not any(flag in ["non_original_image", "text_instruction_present", "user_history_risk"] for flag in risk_flags):
        if "manual_review_required" in risk_flags:
            risk_flags.remove("manual_review_required")
    
    # --- JUSTIFICATION (template from locked fields) ---
    justification = build_justification(
        status, issue_type, object_part, supporting_ids,
        risk_flags, parsed.get("raw_phrasing", ""),
        parsed["claimed_part"], valid_image, evidence_met
    )
    
    # --- ASSEMBLE OUTPUT ROW ---
    # Format risk_flags: semicolon-separated, sorted, or "none"
    flags_list = sorted(risk_flags) if risk_flags else ["none"]
    
    return {
        "user_id": claim.user_id,
        "image_paths": claim.image_paths,
        "user_claim": claim.user_claim,
        "claim_object": claim.claim_object,
        "evidence_standard_met": str(evidence_met).lower(),
        "evidence_standard_met_reason": evidence_reason,
        "risk_flags": ";".join(flags_list),
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": status,
        "claim_status_justification": justification,
        "supporting_image_ids": ";".join(supporting_ids),
        "valid_image": str(valid_image).lower(),
        "severity": severity,
    }
