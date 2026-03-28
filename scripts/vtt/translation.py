"""
vtt/translation.py
──────────────────
Sarvam AI integration:
  - Auto image-type detection from OCR text
  - Telugu OCR normalization (character-level error correction)
  - Telugu → Tamil translation (3 word-type rules, single API call)
"""

import re
import json
import time
import requests


IMAGE_TYPE_DESCRIPTIONS = {
    'signboard' : 'an outdoor signboard for a building, institution or business',
    'newspaper' : 'a newspaper or magazine article',
    'road_sign' : 'a road direction or location sign',
    'poster'    : 'a poster, banner or advertisement',
    'document'  : 'a formal document or notice',
}

_SARVAM_URL = 'https://api.sarvam.ai/v1/chat/completions'


def _headers(api_key: str) -> dict:
    return {
        'api-subscription-key': api_key,
        'Content-Type': 'application/json',
    }

def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from sarvam-m output."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# ── Image type detection ──────────────────────────────────────────────────────

def detect_image_type(processed_areas: list[dict],
                      api_key: str) -> str:
    """
    Automatically detects image type from OCR text using sarvam-m.
    Falls back to 'signboard' if detection fails.
    """
    all_text = ' '.join(a.get('full_text', '') for a in processed_areas).strip()
    if not all_text:
        return 'signboard'

    prompt = (
        "You are given text extracted by OCR from an image. "
        "Based on the text content, classify the image into exactly ONE of these types:\n"
        "  newspaper  — article or magazine with paragraph text\n"
        "  signboard  — building, institution or business sign\n"
        "  road_sign  — direction or location sign\n"
        "  poster     — advertisement or banner\n"
        "  document   — formal notice or letter\n\n"
        "Rules:\n"
        "  - Reply with ONLY the single word label, nothing else\n"
        "  - If unsure, reply: signboard\n\n"
        f"OCR text: {all_text[:500]}"
    )
    payload = {
        'model': 'sarvam-m',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 10,
        'temperature': 0.0,
    }
    try:
        r = requests.post(_SARVAM_URL, headers=_headers(api_key),
                          json=payload, timeout=20)
        r.raise_for_status()
        raw = _strip_think(r.json()['choices'][0]['message']['content']).lower()
        if raw in IMAGE_TYPE_DESCRIPTIONS:
            return raw
    except Exception:
        pass
    return 'signboard'


# ── Telugu OCR normalization ──────────────────────────────────────────────────

def normalize_telugu_ocr(raw_text: str,
                          api_key: str,
                          retries: int = 3) -> str:
    """
    Fix character-level OCR errors in Telugu text using sarvam-m.
    Strict prompt: ONLY fixes misrecognised characters, never rewrites.
    """
    if not raw_text or not raw_text.strip():
        return raw_text

    prompt = (
        "You are a Telugu spell-checker for OCR output. "
        "Your ONLY job is to fix character-level OCR recognition errors.\n\n"
        "WHAT TO FIX (character mistakes only):\n"
        "  - A digit replacing a Telugu character: '0' instead of 'ొ', '6' instead of 'గ'\n"
        "  - An ASCII symbol replacing a Telugu character: '@' instead of 'అ'\n"
        "  - A clearly broken vowel sign (matra)\n"
        "  - A conjunct consonant split incorrectly\n\n"
        "WHAT NOT TO DO:\n"
        "  - Do NOT rewrite, rephrase or paraphrase\n"
        "  - Do NOT change word order or add/remove words\n"
        "  - Do NOT change proper nouns or place names\n"
        "  - Do NOT change English words in Telugu script\n"
        "  - Do NOT translate\n"
        "  - If unsure about any word, leave it exactly as-is\n\n"
        "OUTPUT: The corrected Telugu text only. No explanation.\n\n"
        f"OCR text: {raw_text}"
    )
    payload = {
        'model': 'sarvam-m',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 512,
        'temperature': 0.0,
    }
    for attempt in range(retries):
        try:
            r = requests.post(_SARVAM_URL, headers=_headers(api_key),
                              json=payload, timeout=30)
            r.raise_for_status()
            out = _strip_think(r.json()['choices'][0]['message']['content'])
            return out
        except Exception as e:
            print(f'  [normalize] attempt {attempt+1}: {e}')
            time.sleep(2 ** attempt)
    return raw_text


# ── Telugu → Tamil translation ────────────────────────────────────────────────

def translate_areas(corrected_texts: list[str],
                    image_type: str,
                    api_key: str,
                    retries: int = 3) -> list[str]:
    """
    Translate all areas in ONE API call for full document context.

    Three word-type rules enforced:
      Rule 1 — Native Telugu words       → TRANSLATE to Tamil
      Rule 2 — English loanwords         → RESTORE to English
      Rule 3 — Proper nouns / places     → TRANSLITERATE to Tamil script
    """
    def clean(t: str) -> str:
        return _strip_think(t) if t else ''

    indexed = [(i, clean(t)) for i, t in enumerate(corrected_texts)
               if t and clean(t).strip()]
    if not indexed:
        return [''] * len(corrected_texts)

    img_desc = IMAGE_TYPE_DESCRIPTIONS.get(image_type, image_type)
    numbered = '\n'.join(f'Line {n+1}: {t}' for n, (_, t) in enumerate(indexed))

    prompt = (
        f"You are an expert Telugu to Tamil translator.\n"
        f"The text below was extracted from {img_desc}.\n\n"
        f"THREE RULES — follow strictly for every word:\n"
        f"  Rule 1 — Native Telugu words: TRANSLATE to Tamil\n"
        f"    e.g. కొంతమందికి→சிலருக்கு  అవకాశం→வாய்ப்பு  పెద్ద→பெரிய\n"
        f"  Rule 2 — English words in Telugu script: RESTORE to English\n"
        f"    e.g. పోలీస్→Police  డాక్టర్→Doctor  చెకప్→Checkup  గిఫ్ట్→Gift\n"
        f"  Rule 3 — Proper nouns / place names: TRANSLITERATE to Tamil script\n"
        f"    e.g. రజనీకాంత్→ரஜினிகாந்த்  కృష్ణంపాలెం→கிருஷ்ணம்பாலெம்\n\n"
        f"All lines are from the SAME image — use full context.\n\n"
        f"TEXT:\n{numbered}\n\n"
        f"Return ONLY valid JSON. No explanation. No markdown:\n"
        f'{{ "translations": [ {{"line": 1, "tamil": "..."}}, ... ] }}'
    )
    payload = {
        'model': 'sarvam-m',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 2048,
        'temperature': 0.2,
    }

    results = {}
    for attempt in range(retries):
        try:
            r = requests.post(_SARVAM_URL, headers=_headers(api_key),
                              json=payload, timeout=60)
            r.raise_for_status()
            content = _strip_think(r.json()['choices'][0]['message']['content'])
            # Strip markdown fences if present
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'^```\s*',     '', content)
            content = re.sub(r'\s*```$',     '', content).strip()
            brace = content.find('{')
            if brace > 0:
                content = content[brace:]
            parsed = json.loads(content)
            for item in parsed['translations']:
                n = item['line'] - 1
                if 0 <= n < len(indexed):
                    results[indexed[n][0]] = item['tamil']
            break
        except Exception as e:
            print(f'  [translate] attempt {attempt+1}: {e}')
            if attempt == 0:
                print(f'  Response snippet: {locals().get("content","none")[:200]}')
            time.sleep(2 ** attempt)

    return [results.get(i, '') for i in range(len(corrected_texts))]
