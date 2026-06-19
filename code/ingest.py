import csv
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
from schemas import ClaimRow
from config import DATASET_DIR, CLAIMS_CSV, EVIDENCE_REQ_CSV, USER_HISTORY_CSV

class EvidenceRequirements(dict):
    """Subclass of dict to hold fallback all-rules separately."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallbacks: Dict[str, List[Dict[str, str]]] = {}

def load_claims(csv_path: str) -> List[ClaimRow]:
    """Read claims CSV and return list of ClaimRow dataclasses.
    
    Handles Windows line endings safely.
    """
    claims = []
    with open(csv_path, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            claims.append(ClaimRow(
                user_id=row['user_id'],
                image_paths=row['image_paths'],
                user_claim=row['user_claim'],
                claim_object=row['claim_object']
            ))
    return claims

def load_evidence_requirements(csv_path: str) -> EvidenceRequirements:
    """Read evidence_requirements.csv into a dictionary keyed by (claim_object, applies_to).
    
    Also build a fallback lookup: for each claim_object, store the "all" rules separately.
    """
    reqs = EvidenceRequirements()
    with open(csv_path, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj = row['claim_object']
            applies = row['applies_to']
            reqs[(obj, applies)] = row
            
            # If claim_object is "all", save it in fallbacks for all objects
            if obj == "all":
                if "all" not in reqs.fallbacks:
                    reqs.fallbacks["all"] = []
                reqs.fallbacks["all"].append(row)
    return reqs

def load_user_history(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """Read user_history.csv into a dict keyed by user_id.
    
    Each value is a dict with all columns plus a computed is_risky boolean.
    """
    history = {}
    with open(csv_path, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = row['user_id']
            
            # Compute is_risky
            rejected_str = row.get('rejected_claim', '0')
            rejected_claim = int(rejected_str) if rejected_str.isdigit() else 0
            history_flags = row.get('history_flags', '')
            is_risky = (rejected_claim >= 2) or ('risk' in history_flags)
            
            user_data = dict(row)
            user_data['is_risky'] = is_risky
            history[user_id] = user_data
    return history

def parse_image_paths(image_paths_str: str) -> List[Tuple[str, str]]:
    """Split semicolon-separated paths, extract image ID, and return list of (image_id, full_path) tuples.
    
    Resolves prefixes relative to DATASET_DIR.
    """
    if not image_paths_str or image_paths_str.lower() == "none":
        return []
        
    results = []
    paths = image_paths_str.split(';')
    for p in paths:
        p = p.strip()
        if not p:
            continue
            
        filename = os.path.basename(p)
        image_id, _ = os.path.splitext(filename)
        
        # If paths contain a duplicate dataset/ prefix, normalize it
        parts = Path(p).parts
        if parts and parts[0] == "dataset":
            rel_path = Path(*parts[1:])
        else:
            rel_path = Path(p)
            
        full_path = str(DATASET_DIR / rel_path)
        results.append((image_id, full_path))
    return results

def load_all_data() -> Tuple[List[ClaimRow], EvidenceRequirements, Dict[str, Dict[str, Any]]]:
    """Convenience function that calls all loaders and returns (claims, evidence_reqs, user_history)."""
    claims = load_claims(str(CLAIMS_CSV))
    evidence_reqs = load_evidence_requirements(str(EVIDENCE_REQ_CSV))
    user_history = load_user_history(str(USER_HISTORY_CSV))
    return claims, evidence_reqs, user_history
