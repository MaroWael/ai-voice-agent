"""
Query Optimization — Dictionaries

Pure data module. Contains no logic — only configurable lookup tables.

All three dictionaries are imported by RuleBasedQueryOptimizer and can be
extended without touching any logic.

PHRASE_NORMALIZATIONS:
    Maps common conversational phrases (Arabic and English) to their canonical
    retrieval form. Applied before keyword expansion.

BANKING_KEYWORD_EXPANSIONS:
    Maps a canonical term to a list of synonyms/aliases. Each expansion is
    appended after the original token to widen retrieval coverage.
    Entries are bilingual — Arabic and English synonyms coexist.

PRODUCT_ALIASES:
    Maps informal or colloquial product references to the canonical name
    fragment used in the knowledge base. Applied last so keyword expansion
    can first normalize the text.
"""

# ---------------------------------------------------------------------------
# Conversational phrase normalizations
# Maps a phrase that commonly appears in colloquial queries to a cleaner,
# canonical form that retrieval models handle better.
# ---------------------------------------------------------------------------
PHRASE_NORMALIZATIONS: dict[str, str] = {
    # Arabic openers / filler phrases
    "ممكن تقولي": "",
    "ممكن": "",
    "عايز اعرف": "",
    "عاوز اعرف": "",
    "محتاج اعرف": "",
    "بس عايز اعرف": "",
    "ايه هو": "",
    "إيه هو": "",
    "ايه هي": "",
    "إيه هي": "",
    "فيه ايه": "",
    "فيه إيه": "",
    "قولي عن": "",
    "قولي على": "",
    "وضح لي": "",
    "شرح لي": "",
    "ازاي": "كيف",
    "إزاي": "كيف",
    "ازيك": "",
    "ايه": "ما",
    "إيه": "ما",
    # English openers
    "can you tell me": "",
    "could you tell me": "",
    "i want to know": "",
    "i need to know": "",
    "what is the": "",
    "what are the": "",
    "tell me about": "",
    "explain": "",
    "how do i": "",
    "how can i": "",
    "is there": "",
    "are there": "",
}

# ---------------------------------------------------------------------------
# Banking keyword expansions
# Key: the root retrieval term (Arabic or English).
# Value: list of synonyms appended after the key in the optimized query.
# Bilingual entries intentionally mixed — the embedding model (BGE-M3) is
# multilingual and benefits from seeing both forms.
# ---------------------------------------------------------------------------
BANKING_KEYWORD_EXPANSIONS: dict[str, list[str]] = {
    # Fees & charges
    "fees": ["رسوم", "charges", "costs", "تكاليف", "مصاريف"],
    "رسوم": ["fees", "charges", "تكاليف", "مصاريف"],
    "مصاريف": ["fees", "رسوم", "charges"],
    "تكاليف": ["fees", "رسوم", "charges"],
    # Benefits & rewards
    "benefits": ["مزايا", "فوائد", "rewards", "privileges", "امتيازات"],
    "مزايا": ["benefits", "فوائد", "rewards", "امتيازات"],
    "فوائد": ["benefits", "مزايا", "rewards"],
    "امتيازات": ["privileges", "benefits", "مزايا"],
    # Credit limit
    "credit limit": ["حد الائتمان", "الحد الائتماني", "limit", "حد البطاقة"],
    "حد الائتمان": ["credit limit", "limit", "الحد الائتماني"],
    "الحد الائتماني": ["credit limit", "حد الائتمان", "limit"],
    "حد البطاقة": ["credit limit", "حد الائتمان"],
    # Installments
    "installments": ["أقساط", "تقسيط", "payment plan", "tenor"],
    "أقساط": ["installments", "تقسيط", "payment plan"],
    "تقسيط": ["installments", "أقساط", "payment plan"],
    # Eligibility / requirements
    "eligibility": ["شروط", "متطلبات", "requirements", "conditions", "criteria"],
    "شروط": ["eligibility", "requirements", "conditions", "متطلبات"],
    "متطلبات": ["requirements", "شروط", "eligibility"],
    # Cash withdrawals
    "cash withdrawal": ["سحب نقدي", "ATM", "سحب"],
    "سحب نقدي": ["cash withdrawal", "ATM", "سحب"],
    "سحب": ["cash withdrawal", "سحب نقدي"],
    # Interest rate
    "interest rate": ["فائدة", "سعر الفائدة", "rate"],
    "فائدة": ["interest rate", "سعر الفائدة", "rate"],
    "سعر الفائدة": ["interest rate", "فائدة"],
    # Renewal / issuance
    "renewal": ["تجديد", "renew"],
    "تجديد": ["renewal", "renew"],
    "issuance": ["إصدار", "issue", "استخراج"],
    "إصدار": ["issuance", "issue"],
    # International usage
    "international": ["دولي", "خارج مصر", "abroad", "overseas"],
    "دولي": ["international", "خارج مصر", "abroad"],
    # Rewards / points
    "rewards": ["نقاط", "مكافآت", "points", "cashback"],
    "نقاط": ["points", "rewards", "مكافآت"],
    "مكافآت": ["rewards", "نقاط", "points"],
    "points": ["نقاط", "rewards", "مكافآت"],
    # Lounge access
    "lounge": ["صالة مطار", "airport lounge", "lounge access"],
    "صالة مطار": ["lounge", "airport lounge"],
    # Penalty
    "penalty": ["غرامة", "عقوبة", "fine"],
    "غرامة": ["penalty", "fine"],
}

# ---------------------------------------------------------------------------
# Product aliases
# Maps informal, colloquial, or abbreviated references to the canonical
# product-name fragment used in the knowledge base documents.
# Applied during optimization so retrieval can find the correct product.
# ---------------------------------------------------------------------------
PRODUCT_ALIASES: dict[str, str] = {
    # Platinum
    "البلاتينيوم": "Platinum",
    "بلاتينيوم": "Platinum",
    "platinum card": "Platinum",
    "بطاقة البلاتينيوم": "Platinum",
    # Gold
    "الجولد": "Gold",
    "جولد": "Gold",
    "gold card": "Gold",
    "بطاقة الجولد": "Gold",
    # Classic
    "الكلاسيك": "Classic",
    "كلاسيك": "Classic",
    "classic card": "Classic",
    # Titanium
    "التيتانيوم": "Titanium",
    "تيتانيوم": "Titanium",
    "titanium card": "Titanium",
    # Visa Infinite
    "الانفينيت": "Infinite",
    "انفينيت": "Infinite",
    "infinite card": "Infinite",
    "visa infinite": "Visa Infinite",
    # Visa Signature
    "السيجنتشر": "Signature",
    "سيجنتشر": "Signature",
    "signature card": "Signature",
    # World
    "الورلد": "World",
    "ورلد": "World",
    "world card": "World",
    # World Elite
    "الورلد اليت": "World Elite",
    "ورلد اليت": "World Elite",
    "world elite card": "World Elite",
    # Al-Araby
    "العربي": "Al-Araby",
    "بطاقة العربي": "Al-Araby",
    "al araby": "Al-Araby",
    # Asatha
    "أساطة": "Asatha",
    "اساطة": "Asatha",
    "asatha": "Asatha",
}
