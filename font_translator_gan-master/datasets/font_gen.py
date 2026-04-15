import os
import random
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# CONFIG
# -----------------------------

TELUGU_FONT_DIR = "./Noto_Sans_Telugu/static"
TAMIL_FONT_DIR = "./Noto_Sans_Tamil/static"

OUTPUT_ROOT = "./font"

IMG_SIZE = 64
FONT_SIZE = 48

# Tamil = content + target
TAMIL_BASE = list("கசடதபமயரலவ")
TAMIL_VOWEL_SIGNS = ["", "ா", "ி", "ீ", "ு", "ூ", "ெ", "ே", "ை", "ொ", "ோ", "ௌ"]

TAMIL_CHARS = [c + v for c in TAMIL_BASE for v in TAMIL_VOWEL_SIGNS]
TELUGU_BASE = list("కగచటతపమయరలవశసహ")
TELUGU_VOWELS = ["", "ా", "ి", "ీ", "ు", "ూ", "ె", "ే", "ై", "ొ", "ో", "ౌ"]

TELUGU_CHARS = [c + v for c in TELUGU_BASE for v in TELUGU_VOWELS]

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# -----------------------------
# HELPERS
# -----------------------------

def extract_style_name(filename):
    name = filename.replace(".ttf", "")

    # Remove prefixes
    name = name.replace("NotoSansTelugu_", "")
    name = name.replace("NotoSansTamil_", "")
    name = name.replace("NotoSansTelugu-", "")
    name = name.replace("NotoSansTamil-", "")

    # Normalize
    name = name.replace("-", "_")

    return name


def render_char(char, font_path):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=255)
    draw = ImageDraw.Draw(img)

    font_size = FONT_SIZE

    while font_size > 5:
        font = ImageFont.truetype(font_path, font_size)

        bbox = draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        # Check if it fits
        if w <= IMG_SIZE * 0.9 and h <= IMG_SIZE * 0.9:
            break

        font_size -= 2  # shrink step

    # Center it
    x = (IMG_SIZE - w) // 2 - bbox[0]
    y = (IMG_SIZE - h) // 2 - bbox[1]

    draw.text((x, y), char, fill=0, font=font)

    return img


# -----------------------------
# STEP 1: LOAD & MATCH FONTS
# -----------------------------

telugu_fonts = {}
tamil_fonts = {}

for f in os.listdir(TELUGU_FONT_DIR):
    if f.endswith(".ttf"):
        style = extract_style_name(f)
        telugu_fonts[style] = os.path.join(TELUGU_FONT_DIR, f)

for f in os.listdir(TAMIL_FONT_DIR):
    if f.endswith(".ttf"):
        style = extract_style_name(f)
        tamil_fonts[style] = os.path.join(TAMIL_FONT_DIR, f)

common_styles = sorted(list(set(telugu_fonts) & set(tamil_fonts)))

print(f"✅ Found {len(common_styles)} matched styles")

# -----------------------------
# STEP 2: SPLIT STYLES
# -----------------------------

random.shuffle(common_styles)
split_idx = int(TRAIN_RATIO * len(common_styles))

train_styles = common_styles[:split_idx]
test_style_styles = common_styles[split_idx:]

# -----------------------------
# STEP 3: SPLIT CHARACTERS
# -----------------------------

random.shuffle(TAMIL_CHARS)
split_idx = int(TRAIN_RATIO * len(TAMIL_CHARS))

train_chars = TAMIL_CHARS[:split_idx]
test_content_chars = TAMIL_CHARS[split_idx:]

# -----------------------------
# CORE GENERATOR
# -----------------------------

def generate_dataset(split_name, styles, tamil_chars, telugu_chars):
    base_path = os.path.join(OUTPUT_ROOT, split_name)

    print(f"\n📁 Generating: {split_name}")

    for style in styles:
        telugu_font = telugu_fonts[style]
        tamil_font = tamil_fonts[style]

        telugu_dir = os.path.join(base_path, "telugu", style)
        tamil_dir = os.path.join(base_path, "tamil", style)

        os.makedirs(telugu_dir, exist_ok=True)
        os.makedirs(tamil_dir, exist_ok=True)

        # Telugu (style images)
        for char in telugu_chars:
            img = render_char(char, telugu_font)
            img.save(os.path.join(telugu_dir, f"{char}.png"))

        # Tamil (target images)
        for char in tamil_chars:
            img = render_char(char, tamil_font)
            img.save(os.path.join(tamil_dir, f"{char}.png"))

    # Source (Tamil content)
    source_dir = os.path.join(base_path, "source")
    os.makedirs(source_dir, exist_ok=True)

    # Use neutral font (Regular)
    base_font = tamil_fonts.get("Regular", list(tamil_fonts.values())[0])

    for char in tamil_chars:
        img = render_char(char, base_font)
        img.save(os.path.join(source_dir, f"{char}.png"))

    print(f"✅ Done: {split_name}")


# -----------------------------
# STEP 4: GENERATE SPLITS
# -----------------------------

# TRAIN
generate_dataset(
    "train",
    train_styles,
    train_chars,
    TELUGU_CHARS
)

# TEST UNKNOWN STYLE (new fonts)
generate_dataset(
    "test_unknown_style",
    test_style_styles,
    train_chars,
    TELUGU_CHARS
)

# TEST UNKNOWN CONTENT (new characters)
generate_dataset(
    "test_unknown_content",
    train_styles,
    test_content_chars,
    TELUGU_CHARS
)

print("\n🎉 ALL DATASETS GENERATED SUCCESSFULLY!")