import os
import io
import numpy as np
from PIL import Image, ImageFilter
import imagehash

def check_file_integrity(image_path: str) -> dict:
    """Try to open the image with PIL.
    
    Return {"can_open": bool, "width": int, "height": int, "mode": str, "filesize": int}.
    """
    filesize = 0
    try:
        if os.path.exists(image_path):
            filesize = os.path.getsize(image_path)
    except Exception:
        pass

    try:
        with Image.open(image_path) as img:
            # force loading image data to ensure integrity
            img.verify()
        
        # open again for properties because verify() closes/invalidates the image object
        with Image.open(image_path) as img:
            return {
                "can_open": True,
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "filesize": filesize
            }
    except Exception:
        return {
            "can_open": False,
            "width": 0,
            "height": 0,
            "mode": "",
            "filesize": filesize
        }

def check_blur(image_path: str) -> float:
    """Convert image to grayscale, compute Laplacian variance.
    
    Higher = sharper. Return the variance score.
    """
    try:
        with Image.open(image_path) as img:
            img_gray = img.convert('L')
            # 3x3 Laplacian kernel
            laplacian_kernel = ImageFilter.Kernel(
                (3, 3), 
                [0, 1, 0, 1, -4, 1, 0, 1, 0], 
                scale=1, 
                offset=128
            )
            lap_img = img_gray.filter(laplacian_kernel)
            arr = np.array(lap_img, dtype=np.float64)
            # The variance of the Laplacian filter values
            return float(np.var(arr))
    except Exception:
        return 0.0

def check_brightness(image_path: str) -> dict:
    """Compute brightness histogram.
    
    Return {"mean_brightness": float, "is_dark": bool, "is_bright": bool}
    where dark < 50 and bright > 200 on 0-255 scale.
    """
    try:
        with Image.open(image_path) as img:
            img_gray = img.convert('L')
            arr = np.array(img_gray, dtype=np.float64)
            mean_brightness = float(np.mean(arr))
            is_dark = mean_brightness < 50.0
            is_bright = mean_brightness > 200.0
            return {
                "mean_brightness": mean_brightness,
                "is_dark": is_dark,
                "is_bright": is_bright
            }
    except Exception:
        return {
            "mean_brightness": 0.0,
            "is_dark": False,
            "is_bright": False
        }

def resize_for_vlm(image_path: str, max_size: int = 1024) -> bytes:
    """Resize image so longest side is max_size.
    
    Return as JPEG bytes (for sending to Gemini API). Keep aspect ratio.
    """
    try:
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
                
            width, height = img.size
            if max(width, height) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            out_bytes = io.BytesIO()
            img.save(out_bytes, format="JPEG")
            return out_bytes.getvalue()
    except Exception:
        return b""

def run_pre_checks(image_id: str, image_path: str) -> dict:
    """Run all checks, return combined dict."""
    integrity = check_file_integrity(image_path)
    if not integrity["can_open"]:
        return {
            "image_id": image_id,
            "path": image_path,
            "can_open": False,
            "blur_score": 0.0,
            "brightness": 0.0,
            "is_dark": False,
            "is_bright": False,
            "width": 0,
            "height": 0,
            "is_duplicate": False,
            "phash": None,
            "matched_with": None
        }
        
    blur_score = check_blur(image_path)
    brightness_dict = check_brightness(image_path)
    dup_dict = check_duplicate(image_path)
    
    return {
        "image_id": image_id,
        "path": image_path,
        "can_open": True,
        "blur_score": blur_score,
        "brightness": brightness_dict["mean_brightness"],
        "is_dark": brightness_dict["is_dark"],
        "is_bright": brightness_dict["is_bright"],
        "width": integrity["width"],
        "height": integrity["height"],
        "is_duplicate": dup_dict["is_duplicate"],
        "phash": dup_dict["phash"],
        "matched_with": dup_dict["matched_with"]
    }

def compute_phash(image_path: str):
    """Compute perceptual hash for duplicate/stock detection."""
    try:
        img = Image.open(image_path)
        return str(imagehash.phash(img))
    except Exception:
        return None

_phash_registry = {}

def check_duplicate(image_path: str, threshold: int = 10) -> dict:
    """Check if this image is a near-duplicate of one already seen."""
    phash = compute_phash(image_path)
    if phash is None:
        return {"is_duplicate": False, "phash": None, "matched_with": None}
    
    for registered_path, registered_hash in _phash_registry.items():
        if registered_path == image_path:
            continue
        try:
            distance = imagehash.hex_to_hash(phash) - imagehash.hex_to_hash(registered_hash)
            if distance <= threshold:
                return {"is_duplicate": True, "phash": phash, "matched_with": registered_path}
        except Exception:
            continue
    
    _phash_registry[image_path] = phash
    return {"is_duplicate": False, "phash": phash, "matched_with": None}
