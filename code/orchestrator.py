from ingest import parse_image_paths
from tools import parse_claim, get_evidence_requirement, get_user_history, inspect_images

def process_claim(claim, evidence_reqs: dict, user_history: dict, pre_checks: list) -> dict:
    """Agent loop: calls 4 tools in sequence, returns observation bundle.
    
    No verdict, no policy decision — just observations.
    """
    # Tool 1: parse the claim text
    parsed = parse_claim(claim.user_claim, claim.claim_object)
    print(f"  [Tool 1] parsed claim: family={parsed.get('claimed_family')}, part={parsed.get('claimed_part')}")
    
    # Tool 2: look up evidence requirement (pure Python)
    requirement = get_evidence_requirement(
        claim.claim_object, parsed["claimed_family"], evidence_reqs
    )
    print(f"  [Tool 2] requirement: id={requirement.get('req_id')}")
    
    # Tool 3: look up user history (pure Python)
    history = get_user_history(claim.user_id, user_history)
    print(f"  [Tool 3] user history: risky={history.get('is_risky')}")
    
    # Tool 4: inspect images (THE one VLM vision call)
    images = parse_image_paths(claim.image_paths)
    inspection = inspect_images(
        images,
        claim.claim_object,
        parsed["claimed_family"],
        parsed["claimed_part"],
        requirement["minimum_image_evidence"],
        history["history_summary"]
    )
    print(f"  [Tool 4] inspected images count={len(inspection.get('per_image', []))}")
    
    # Return the observation bundle — no verdict here
    return {
        "claim": claim,
        "parsed": parsed,
        "requirement": requirement,
        "history": history,
        "inspection": inspection,
        "pre_checks": pre_checks,
        "image_ids": [img_id for img_id, _ in images]
    }

