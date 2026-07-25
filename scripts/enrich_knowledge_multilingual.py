"""
Script to enrich raw JSON knowledge files in data/ with multilingual arabic_name and aliases fields.
Business knowledge belongs in data files, not Python code.
"""

import json
import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path("data")

MULTILINGUAL_METADATA = {
    "Classic Credit Card": {
        "arabic_name": "بطاقة الكلاسيك الائتمانية",
        "aliases": ["بطاقة الكلاسيك", "الكلاسيك", "كلاسيك", "Classic Card", "بطاقة كلاسيك بنك مصر"],
    },
    "Gold Credit Cards": {
        "arabic_name": "بطاقة الجولد الائتمانية",
        "aliases": ["بطاقة الجولد", "الجولد", "جولد", "بطاقة ذهبية", "Gold Card", "بطاقة جولد بنك مصر"],
    },
    "Titanium Credit Card": {
        "arabic_name": "بطاقة التيتانيوم الائتمانية",
        "aliases": ["بطاقة التيتانيوم", "التيتانيوم", "تيتانيوم", "Titanium Card", "بطاقة تيتانيوم بنك مصر"],
    },
    "Platinum Visa - Master Credit Card": {
        "arabic_name": "بطاقة فيزا وماستركارد البلاتينية",
        "aliases": ["بطاقة البلاتينيوم", "البلاتينيوم", "بلاتينيوم", "بطاقة فيزا بلاتينيوم", "بطاقة ماستركارد بلاتينيوم", "بطاقة البلاتينيوم بنك مصر", "Platinum Card"],
    },
    "Visa Infinite": {
        "arabic_name": "بطاقة فيزا انفينيت",
        "aliases": ["بطاقة الانفينيت", "الانفينيت", "انفينيت", "Visa Infinite Card"],
    },
    "Visa Signature": {
        "arabic_name": "بطاقة فيزا سيجنتشر",
        "aliases": ["بطاقة السيجنتشر", "السيجنتشر", "سيجنتشر", "Visa Signature Card"],
    },
    "World Credit Card": {
        "arabic_name": "بطاقة الورلد الائتمانية",
        "aliases": ["بطاقة الورلد", "الورلد", "ورلد", "World Card"],
    },
    "World Elite Credit Card": {
        "arabic_name": "بطاقة الورلد اليت الائتمانية",
        "aliases": ["بطاقة الورلد اليت", "الورلد اليت", "ورلد اليت", "World Elite Card"],
    },
    "Al-Araby Card": {
        "arabic_name": "بطاقة العربي",
        "aliases": ["بطاقة العربي", "العربي", "Al-Araby Card"],
    },
    "Asatha MasterCard": {
        "arabic_name": "بطاقة أساطة ماستركارد",
        "aliases": ["بطاقة أساطة", "أساطة", "اساطة", "Asatha Card"],
    },
    "Card from Banque Misr secured against any other credit card": {
        "arabic_name": "بطاقة بنك مصر بضمان بطاقة ائتمان اخرى",
        "aliases": ["بطاقة بضمان بطاقة اخرى", "بطاقة مسبقة الدفع"],
    },
    "Visa Infinite Private": {
        "arabic_name": "بطاقة فيزا انفينيت الخاصة",
        "aliases": ["انفينيت برايفيت", "الخاصة", "Visa Infinite Private Card"],
    },
}


def enrich_all() -> None:
    for json_path in sorted(DATA_DIR.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        name = data.get("name")
        if name in MULTILINGUAL_METADATA:
            meta = MULTILINGUAL_METADATA[name]
            data["arabic_name"] = meta["arabic_name"]
            data["aliases"] = meta["aliases"]

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"Updated {json_path.name} with arabic_name='{meta['arabic_name']}' and {len(meta['aliases'])} aliases.")


if __name__ == "__main__":
    enrich_all()
