import time
import os
import json
import base64
from google import genai
from google.genai import types
from config import GEMINI_API_KEYS, GEMINI_MODEL

class KeyRotator:
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.index = 0
        self.health = {k: {"calls": 0, "errors": 0, "cooldown_until": 0.0} for k in keys}
    
    def get_next_key(self) -> str:
        if not self.keys:
            raise ValueError("No Gemini API keys found. Please set GEMINI_API_KEY_1 to 10 or GEMINI_API_KEY in .env / environment.")
        now = time.time()
        for _ in range(len(self.keys)):
            key = self.keys[self.index % len(self.keys)]
            self.index += 1
            if self.health[key]["cooldown_until"] <= now:
                self.health[key]["calls"] += 1
                return key
        earliest = min(h["cooldown_until"] for h in self.health.values())
        wait = max(0, earliest - now + 0.5)
        print(f"All keys cooling down, waiting {wait:.1f}s...")
        time.sleep(wait)
        return self.get_next_key()
    
    def report_error(self, key: str):
        h = self.health[key]
        h["errors"] += 1
        backoff = min(2 ** h["errors"], 60)
        h["cooldown_until"] = time.time() + backoff
        print(f"Key ...{key[-6:] if len(key) >= 6 else key} hit 429/error, cooldown {backoff}s")
    
    def report_success(self, key: str):
        self.health[key]["errors"] = 0

rotator = KeyRotator(GEMINI_API_KEYS)

def clean_json_response(text: str) -> str:
    """Strip markdown fences (e.g. ```json ... ```) to isolate clean JSON text."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def call_gemini_text(prompt: str, system_prompt: str = "") -> str:
    """Text-only call for parse_claim.
    
    Uses key rotator, retries up to 3 times on failure.
    """
    last_err = None
    for attempt in range(len(rotator.keys)):
        key = rotator.get_next_key()
        try:
            client = genai.Client(api_key=key)
            config = types.GenerateContentConfig(
                system_instruction=system_prompt if system_prompt else None,
                temperature=0.1,
                response_mime_type="application/json"
            )
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config
            )
            rotator.report_success(key)
            text = response.text if response.text else ""
            cleaned = clean_json_response(text)
            
            # Print a short log line
            key_suffix = key[-4:] if len(key) >= 4 else key
            print(f"[key ...{key_suffix}] call_gemini_text -> {len(cleaned)} chars")
            
            # Baseline rate guard
            time.sleep(1.0)
            return cleaned
        except Exception as e:
            last_err = e
            rotator.report_error(key)
    raise last_err

def call_gemini_vision(prompt: str, image_bytes_list: list[tuple[str, bytes]], system_prompt: str = "") -> str:
    """Multimodal call for inspect_images.
    
    Takes a list of (image_id, jpeg_bytes) tuples.
    """
    last_err = None
    for attempt in range(len(rotator.keys)):
        key = rotator.get_next_key()
        try:
            client = genai.Client(api_key=key)
            
            # Label each image in the prompt text itself
            labels = []
            parts = []
            for idx, (img_id, img_bytes) in enumerate(image_bytes_list):
                labels.append(f"Image {idx+1} ({img_id})")
            
            labeled_prompt = prompt
            if labels:
                labeled_prompt = prompt + "\n\nImages:\n" + ", ".join(labels)
                
            parts.append(labeled_prompt)
            for img_id, img_bytes in image_bytes_list:
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
                
            config = types.GenerateContentConfig(
                system_instruction=system_instruction if (system_instruction := system_prompt) else None,
                temperature=0.1,
                response_mime_type="application/json"
            )
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=parts,
                config=config
            )
            rotator.report_success(key)
            text = response.text if response.text else ""
            cleaned = clean_json_response(text)
            
            # Print a short log line
            key_suffix = key[-4:] if len(key) >= 4 else key
            print(f"[key ...{key_suffix}] call_gemini_vision -> {len(cleaned)} chars")
            
            # Baseline rate guard
            time.sleep(1.0)
            return cleaned
        except Exception as e:
            last_err = e
            rotator.report_error(key)
    raise last_err

