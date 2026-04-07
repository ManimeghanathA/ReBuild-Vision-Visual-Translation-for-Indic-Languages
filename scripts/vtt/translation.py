from groq import Groq
import re
import json
import time

# ── Setup ─────────────────────────────────────────────────────────────────────
# Using Groq free tier instead of Gemini (Gemini free tier is region-blocked in India).
# Get your free API key at: https://console.groq.com
# Model: llama-3.3-70b-versatile — strong multilingual support for Indian languages.
MODEL_NAME = 'llama-3.3-70b-versatile'

# Cache the client instead of creating a new one on every call.
_client_cache: dict = {}

def _get_client(api_key: str) -> Groq:
    if api_key not in _client_cache:
        _client_cache[api_key] = Groq(api_key=api_key)
    return _client_cache[api_key]

IMAGE_TYPE_DESCRIPTIONS = {
    'signboard' : 'an outdoor signboard for a building, institution or business',
    'newspaper' : 'a newspaper or magazine article',
    'road_sign' : 'a road direction or location sign',
    'poster'    : 'a poster, banner or advertisement',
    'document'  : 'a formal document or notice',
}

def _get_groq_response(client: Groq, prompt: str, json_mode: bool = False) -> str:
    kwargs = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 2048,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(3):
        try:
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content
            if text:
                return text.strip()
            return ""
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(2 ** attempt)
    return ""

def _clean_json_string(raw_string: str) -> str:
    return re.sub(r'```json\s?|\s?```', '', raw_string).strip()

# ── Image type detection ──────────────────────────────────────────────────────

def detect_image_type(processed_areas: list[dict], api_key: str) -> str:
    client = _get_client(api_key)
    all_text = ' '.join(a.get('full_text', '') for a in processed_areas).strip()
    if not all_text:
        return 'signboard'

    prompt = (
        "SYSTEM: You are a document classifier.\n"
        "TASK: Classify the following OCR text into exactly ONE category.\n"
        "CATEGORIES: [newspaper, signboard, road_sign, poster, document]\n"
        "CONSTRAINT: Output ONLY the category name. No punctuation, no explanation.\n"
        f"TEXT: {all_text[:500]}"
    )

    try:
        raw = _get_groq_response(client, prompt).lower()
        if raw in IMAGE_TYPE_DESCRIPTIONS:
            return raw
    except Exception:
        pass
    return 'signboard'

# ── Telugu OCR normalization ──────────────────────────────────────────────────

def normalize_telugu_ocr(raw_text: str, api_key: str, retries: int = 3) -> str:
    if not raw_text or not raw_text.strip():
        return raw_text

    client = _get_client(api_key)

    prompt = f"""
SYSTEM: You are an expert Telugu Linguist specializing in OCR error correction.
TASK: Correct the following distorted Telugu text extracted from street signs and movie posters.

CORRECTION RULES:
1. CHARACTER FIXES: Replace Replace the garbage like '@,0' in the text with similar looking telugu letter and it should also be appropriate with its neighbouring letters and should give meaning on whole and correct the telugu according to the context.
2. WORD MERGING and correction: If words are split by random spaces (e.g., 'సి ని మా'), merge them ('సినిమా') and correct that telugu according to the context - make it meaningful telugu
3. CONJUNCTS: Fix split or missing 'othulu' and 'gamakalu' (e.g., 'క్ ష' -> 'క్ష') and misread letter from OCR. Fix grammar: Reattach missing matras (vowel signs) based on context.
4. NO AUTHORING: Do not rewrite the sentence. Do not add new meaning.
5. NO THINKING: Do not explain your logic. Do not include <think> tags.
6. Fix word breaks: Merge syllables that were split by spaces (e.g., 'సి ని మా' -> 'సినిమా').
7. NO DIALOGUE: Do not explain your changes. No <think> tags.
8. OUTPUT ONLY the corrected Telugu text.

INPUT TEXT:
{raw_text}

CORRECTED TELUGU:"""

    for attempt in range(retries):
        try:
            return _get_groq_response(client, prompt)
        except Exception as e:
            print(f"Normalize Error: {e}")
            time.sleep(2 ** attempt)
    return raw_text

# ── Telugu → Tamil translation ────────────────────────────────────────────────

def translate_areas(corrected_texts: list[str], image_type: str, api_key: str, retries: int = 3) -> list[str]:
    client = _get_client(api_key)
    indexed = [(i, t.strip()) for i, t in enumerate(corrected_texts) if t and t.strip()]
    if not indexed:
        return [''] * len(corrected_texts)

    img_desc = IMAGE_TYPE_DESCRIPTIONS.get(image_type, image_type)
    numbered_input = '\n'.join(f"L{n+1}: {t}" for n, (_, t) in enumerate(indexed))

    prompt = f"""
SYSTEM: You are a professional translator from Telugu to Tamil.
The input text is from a {img_desc} and may contain residual OCR errors and noise.

TRANSLATION RULES:
- RULE 0: If a line is unreadable garbage or just random characters, try correcting it with respect to the context if not correctable return an empty string "". Look at the Telugu, if that telugu make sense translate else, correct it to appropriate and nearlest telugu word it can form and translate accoring to follwoing rules.
- RULE 1: Translate native Telugu to natural Tamil.
- RULE 2: Transliterate English loanwords (Checkup, Police, Doctor) into TAMIL SCRIPT.
- RULE 3: Transliterate Proper Nouns (Rajinikanth, Hyderabad) into TAMIL SCRIPT.
- RULE 4: Some sentences requires both translation and transliteration, act smartly and provide result as accurate as possible.
- RULE 5: NO THINKING: Output ONLY JSON. Do not describe your process.

OUTPUT FORMAT (JSON):
{{
  "translations": [
    {{ "line_id": 1, "tamil": "..." }},
    {{ "line_id": 2, "tamil": "..." }}
  ]
}}

INPUT:
{numbered_input}
"""

    results = {}
    for attempt in range(retries):
        try:
            raw_json = _get_groq_response(client, prompt, json_mode=True)
            cleaned_json = _clean_json_string(raw_json)
            parsed = json.loads(cleaned_json)

            for item in parsed.get('translations', []):
                line_idx = item['line_id'] - 1
                if 0 <= line_idx < len(indexed):
                    orig_idx = indexed[line_idx][0]
                    results[orig_idx] = item.get('tamil', '')
            break
        except Exception as e:
            print(f"Translation Error: {e}")
            time.sleep(2 ** attempt)

    return [results.get(i, '') for i in range(len(corrected_texts))]
