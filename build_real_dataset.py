import hashlib
import json
import os
import re
import urllib.parse
import urllib.request

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TARGET_PER_CATEGORY = 150
ALLOWED = {".jpg", ".jpeg", ".png", ".webp"}
SEARCH_TERMS = {
    "biodegradable": [
        "food waste", "compost", "vegetable waste", "fruit waste", "garden waste",
        "leaves waste", "wood waste", "coffee grounds", "eggshell waste",
        "banana peel waste", "orange peel compost", "grass clippings", "sawdust waste",
        "seafood waste", "bread waste", "organic kitchen waste", "compost bin",
    ],
    "recyclable": [
        "recycling paper", "cardboard recycling", "plastic bottle recycling",
        "glass bottle recycling", "aluminum can recycling", "metal recycling",
        "newspaper recycling", "plastic packaging recycling",
        "steel can recycling", "tin can recycling", "beverage carton recycling",
        "scrap metal recycling", "glass jar recycling", "recycled textile",
        "recycling bin paper plastic", "PET bottle recycling",
    ],
    "residual": [
        "non recyclable waste", "residual waste", "mixed waste", "used tissue waste",
        "candy wrapper waste", "dirty plastic waste", "disposable mask waste",
        "styrofoam waste", "sanitary waste",
        "broken ceramic waste", "broken glass waste", "disposable gloves waste",
        "food packaging waste", "mixed household garbage", "landfill waste",
    ],
    "special": [
        "electronic waste", "battery waste", "hazardous waste", "e waste recycling",
        "mobile phone recycling", "computer recycling", "fluorescent lamp waste",
        "paint can hazardous waste", "aerosol can hazardous waste",
        "printer cartridge recycling", "power bank recycling", "hard drive recycling",
        "circuit board recycling", "chemical container waste", "thermometer hazardous waste",
    ],
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def clean_category(folder):
    hashes = set()
    removed = 0
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            continue
        if filename.startswith("aug_"):
            os.remove(path)
            removed += 1
            continue
        extension = os.path.splitext(filename)[1].lower()
        if extension not in ALLOWED:
            continue
        with open(path, "rb") as file:
            digest = sha256(file.read())
        if digest in hashes:
            os.remove(path)
            removed += 1
        else:
            hashes.add(digest)
    return hashes, removed


def search_commons(term, continue_token=None):
    parameters = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": term,
        "gsrnamespace": "6",
        "gsrlimit": "50",
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": "900",
    }
    if continue_token:
        parameters["gsroffset"] = continue_token
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(parameters)
    request = urllib.request.Request(url, headers={"User-Agent": "CleanSpot dataset builder/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def download_image(url):
    request = urllib.request.Request(url, headers={"User-Agent": "CleanSpot dataset builder/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            return None
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[0] < 80 or image.shape[1] < 80:
        return None
    return data


def safe_name(title):
    name = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    return name[:70] or "commons_image"


def build_category(category):
    folder = os.path.join(DATASET_DIR, category)
    os.makedirs(folder, exist_ok=True)
    known_hashes, removed = clean_category(folder)
    metadata_path = os.path.join(folder, "sources.jsonl")
    sources = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                    sources[record["sha256"]] = record
                except json.JSONDecodeError:
                    continue

    saved = 0
    for term in SEARCH_TERMS[category]:
        if len(known_hashes) >= TARGET_PER_CATEGORY:
            break
        response = search_commons(term)
        for page in response.get("query", {}).get("pages", {}).values():
            if len(known_hashes) >= TARGET_PER_CATEGORY:
                break
            imageinfo = page.get("imageinfo", [{}])[0]
            image_url = imageinfo.get("thumburl") or imageinfo.get("url")
            if not image_url:
                continue
            try:
                data = download_image(image_url)
            except Exception:
                continue
            if data is None:
                continue
            digest = sha256(data)
            if digest in known_hashes:
                continue
            filename = f"commons_{digest[:16]}.jpg"
            with open(os.path.join(folder, filename), "wb") as file:
                file.write(data)
            known_hashes.add(digest)
            sources[digest] = {
                "sha256": digest,
                "file": filename,
                "source_url": imageinfo.get("descriptionurl", image_url),
                "image_url": image_url,
                "title": page.get("title", ""),
                "license": imageinfo.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "unknown"),
                "search_term": term,
            }
            saved += 1

    with open(metadata_path, "w", encoding="utf-8") as file:
        for record in sources.values():
            if os.path.exists(os.path.join(folder, record["file"])):
                file.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(f"{category}: removed {removed}, saved {saved}, total unique images {len(known_hashes)}")


for category in SEARCH_TERMS:
    build_category(category)
