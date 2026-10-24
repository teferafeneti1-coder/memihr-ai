"""
Adulis — Amharic <-> Afan Oromo learning app
=============================================
Named for the ancient Red Sea port that moved goods in and out of Aksum;
this app moves sentences back and forth between Amharic and Afan Oromo.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000

Configure the model location with the MODEL_PATH environment variable, or
edit DEFAULT_MODEL_PATH below. It must point at a local HuggingFace-format
checkpoint folder (config.json, model.safetensors, tokenizer files, etc.)
"""

import os
import re
import random
import unicodedata
from pathlib import Path

from flask import Flask, jsonify, render_template, request

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = "/home/feneti/egate/egate/oromifa-amharic & amharic-oromifa/archive"
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)

AM_DATA_PATH = os.environ.get(
    "AM_DATA_PATH", os.path.join(os.path.dirname(__file__), "data", "amharic.txt")
)
OM_DATA_PATH = os.environ.get(
    "OM_DATA_PATH", os.path.join(os.path.dirname(__file__), "data", "afanoromo.txt")
)

LANG_CODE = {"am": "amh_Ethi", "om": "gaz_Latn"}
LESSON_SIZE = 15  # consecutive sentence pairs per lesson (IDs run topically)
MAX_INPUT_CHARS = 500

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model — loaded once at startup
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[adulis] loading model from: {MODEL_PATH}")
print(f"[adulis] device: {device}")

# Wrap MODEL_PATH in Path so huggingface_hub treats it as a local directory
# rather than running it through the repo-ID string validator (which rejects
# paths that contain special characters like '&').
_model_path = Path(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(_model_path).to(device)
model.eval()
print("[adulis] model ready")


def translate(text: str, direction: str, max_length: int = 96, num_beams: int = 5) -> str:
    src, tgt = direction.split("2")
    tokenizer.src_lang = LANG_CODE[src]
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANG_CODE[tgt]),
            max_length=max_length,
            num_beams=num_beams,
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# Training data — loaded once, used for lessons + quizzes (not for inference)
# ---------------------------------------------------------------------------


def normalize_quotes(text):
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def clean_trailing(text):
    text = text.strip()
    text = re.sub(r"\s*\.\s*\.\s*$", ".", text)
    text = re.sub(r"\.{2,}$", ".", text)
    return text


def normalize_amharic(text):
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "ሃ": "ሀ", "ሐ": "ሀ", "ሓ": "ሀ", "ኀ": "ሀ", "ኃ": "ሀ", "ኸ": "ሀ",
        "ሠ": "ሰ", "ሧ": "ሶ", "ዐ": "አ", "ዓ": "አ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = normalize_quotes(text)
    text = re.sub(r"\s+", " ", text)
    return clean_trailing(text)


def normalize_oromo(text):
    text = unicodedata.normalize("NFC", text)
    text = normalize_quotes(text)
    text = re.sub(r"\s+", " ", text)
    return clean_trailing(text)


def load_id_text(path):
    data = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or "|" not in line:
                continue
            sid, text = line.split("|", 1)
            data[sid.strip()] = text.strip()
    return data


def classify(am, om):
    """Rough content tag used to vary lesson/quiz presentation."""
    if re.search(r"\d", am) or re.search(r"\d", om):
        return "math"
    if am.strip().endswith("?") or om.strip().endswith("?"):
        return "question"
    return "statement"


def build_corpus():
    am_map = load_id_text(AM_DATA_PATH)
    om_map = load_id_text(OM_DATA_PATH)
    common_ids = sorted(set(am_map) & set(om_map))

    pairs = []
    for sid in common_ids:
        am = normalize_amharic(am_map[sid])
        om = normalize_oromo(om_map[sid])
        if am and om:
            pairs.append({"id": sid, "am": am, "om": om, "type": classify(am, om)})
    return pairs


CORPUS = build_corpus()
print(f"[adulis] loaded {len(CORPUS)} sentence pairs for lessons/quizzes")

LESSONS = [
    CORPUS[i : i + LESSON_SIZE] for i in range(0, len(CORPUS), LESSON_SIZE)
]

# ---------------------------------------------------------------------------
# Word bank — curated vocabulary for children's word-level mode
# Each entry: am (Amharic), om (Afan Oromo), category, emoji hint
# Extracted from corpus sentences + a small hand-curated set of basic vocab.
# ---------------------------------------------------------------------------

WORD_BANK = [
    # ── Numbers ────────────────────────────────────────────────────────────
    {"am": "አንድ",    "om": "tokko",   "cat": "numbers", "emoji": "1️⃣"},
    {"am": "ሁለት",    "om": "lama",    "cat": "numbers", "emoji": "2️⃣"},
    {"am": "ሶስት",    "om": "sadii",   "cat": "numbers", "emoji": "3️⃣"},
    {"am": "አራት",    "om": "afur",    "cat": "numbers", "emoji": "4️⃣"},
    {"am": "አምስት",   "om": "shan",    "cat": "numbers", "emoji": "5️⃣"},
    {"am": "ስድስት",   "om": "ja'a",    "cat": "numbers", "emoji": "6️⃣"},
    {"am": "ሰባት",    "om": "torba",   "cat": "numbers", "emoji": "7️⃣"},
    {"am": "ስምንት",   "om": "saddeet", "cat": "numbers", "emoji": "8️⃣"},
    {"am": "ዘጠኝ",    "om": "sagal",   "cat": "numbers", "emoji": "9️⃣"},
    {"am": "አስር",    "om": "kudhan",  "cat": "numbers", "emoji": "🔟"},

    # ── Body ───────────────────────────────────────────────────────────────
    {"am": "እጅ",     "om": "harka",   "cat": "body",    "emoji": "✋"},
    {"am": "እግር",    "om": "miila",   "cat": "body",    "emoji": "🦶"},
    {"am": "ዓይን",    "om": "ija",     "cat": "body",    "emoji": "👁️"},
    {"am": "አፍ",     "om": "afaan",   "cat": "body",    "emoji": "👄"},
    {"am": "ጆሮ",     "om": "gurra",   "cat": "body",    "emoji": "👂"},
    {"am": "አፍንጫ",   "om": "funyaan", "cat": "body",    "emoji": "👃"},
    {"am": "ራስ",     "om": "mataa",   "cat": "body",    "emoji": "🧠"},
    {"am": "ጥርስ",    "om": "ilkaan",  "cat": "body",    "emoji": "🦷"},
    {"am": "ጣት",     "om": "quba",    "cat": "body",    "emoji": "☝️"},

    # ── Family ─────────────────────────────────────────────────────────────
    {"am": "አባት",    "om": "abbaa",   "cat": "family",  "emoji": "👨"},
    {"am": "እናት",    "om": "haadha",  "cat": "family",  "emoji": "👩"},
    {"am": "ወንድም",   "om": "obbolaa", "cat": "family",  "emoji": "👦"},
    {"am": "እህት",    "om": "obboleettii", "cat": "family", "emoji": "👧"},
    {"am": "ልጅ",     "om": "ijoollee","cat": "family",  "emoji": "🧒"},
    {"am": "ቤተሰብ",   "om": "maatii",  "cat": "family",  "emoji": "👨‍👩‍👧‍👦"},

    # ── Animals ────────────────────────────────────────────────────────────
    {"am": "ዝሆን",    "om": "arba",    "cat": "animals", "emoji": "🐘"},
    {"am": "አንበሳ",   "om": "leenca",  "cat": "animals", "emoji": "🦁"},
    {"am": "ፈረስ",    "om": "farda",   "cat": "animals", "emoji": "🐴"},
    {"am": "ከብት",    "om": "loon",    "cat": "animals", "emoji": "🐄"},
    {"am": "ውሻ",     "om": "saree",   "cat": "animals", "emoji": "🐕"},
    {"am": "ድመት",    "om": "adurree", "cat": "animals", "emoji": "🐱"},
    {"am": "ወፍ",     "om": "simbirra","cat": "animals", "emoji": "🐦"},
    {"am": "አሳ",     "om": "qurxummii","cat": "animals", "emoji": "🐟"},
    {"am": "ፍየል",    "om": "re'ee",   "cat": "animals", "emoji": "🐐"},
    {"am": "በግ",      "om": "hoolaa",  "cat": "animals", "emoji": "🐑"},

    # ── Nature ─────────────────────────────────────────────────────────────
    {"am": "ውሃ",     "om": "bishaan", "cat": "nature",  "emoji": "💧"},
    {"am": "ፀሐይ",    "om": "aduu",    "cat": "nature",  "emoji": "☀️"},
    {"am": "ዛፍ",     "om": "muka",    "cat": "nature",  "emoji": "🌳"},
    {"am": "ወንዝ",    "om": "laga",    "cat": "nature",  "emoji": "🏞️"},
    {"am": "ባሕር",    "om": "galaana", "cat": "nature",  "emoji": "🌊"},
    {"am": "ሰማይ",    "om": "samii",   "cat": "nature",  "emoji": "🌤️"},
    {"am": "ምድር",    "om": "lafaa",   "cat": "nature",  "emoji": "🌍"},
    {"am": "ተራራ",    "om": "gaara",   "cat": "nature",  "emoji": "⛰️"},
    {"am": "ሳር",     "om": "marga",   "cat": "nature",  "emoji": "🌿"},
    {"am": "አበባ",    "om": "daraaraa","cat": "nature",  "emoji": "🌸"},
    {"am": "ዝናብ",    "om": "rooba",   "cat": "nature",  "emoji": "🌧️"},
    {"am": "ድንጋይ",   "om": "dhagaa",  "cat": "nature",  "emoji": "🪨"},
    {"am": "እሳት",    "om": "ibiddaa", "cat": "nature",  "emoji": "🔥"},

    # ── Food ───────────────────────────────────────────────────────────────
    {"am": "ምግብ",    "om": "nyaata",  "cat": "food",    "emoji": "🍽️"},
    {"am": "ዳቦ",     "om": "buddeena","cat": "food",    "emoji": "🍞"},
    {"am": "ወተት",    "om": "aannan", "cat": "food",    "emoji": "🥛"},
    {"am": "ፍሬ",     "om": "firiiwwan","cat": "food",   "emoji": "🍎"},
    {"am": "ስጋ",     "om": "foon",    "cat": "food",    "emoji": "🥩"},
    {"am": "ቡና",     "om": "buna",    "cat": "food",    "emoji": "☕"},
    {"am": "ማር",     "om": "damma",   "cat": "food",    "emoji": "🍯"},
    {"am": "ጨው",     "om": "soogidda","cat": "food",    "emoji": "🧂"},

    # ── Colors ─────────────────────────────────────────────────────────────
    {"am": "ቀይ",     "om": "diimaa",  "cat": "colors",  "emoji": "🔴"},
    {"am": "ነጭ",     "om": "adii",    "cat": "colors",  "emoji": "⚪"},
    {"am": "ጥቁር",    "om": "gurracha","cat": "colors",  "emoji": "⚫"},
    {"am": "አረንጓዴ",  "om": "magariisa","cat": "colors", "emoji": "🟢"},
    {"am": "ሰማያዊ",   "om": "cuquliisa","cat": "colors", "emoji": "🔵"},
    {"am": "ቢጫ",     "om": "keelloo", "cat": "colors",  "emoji": "🟡"},

    # ── School ─────────────────────────────────────────────────────────────
    {"am": "ትምህርት",  "om": "barnoota","cat": "school",  "emoji": "📚"},
    {"am": "ተማሪ",    "om": "barataa", "cat": "school",  "emoji": "🧑‍🎓"},
    {"am": "መምህር",   "om": "barsiisaa","cat": "school",  "emoji": "👨‍🏫"},
    {"am": "ደብተር",   "om": "kitaaba", "cat": "school",  "emoji": "📓"},
    {"am": "እስኪብሪቶ", "om": "pensila", "cat": "school",  "emoji": "✏️"},
    {"am": "ፊደል",    "om": "qubee",   "cat": "school",  "emoji": "🔤"},
    {"am": "ቤት",     "om": "mana",    "cat": "school",  "emoji": "🏠"},
    {"am": "ጥያቄ",    "om": "gaaffii", "cat": "school",  "emoji": "❓"},
    {"am": "መልስ",    "om": "deebii",  "cat": "school",  "emoji": "💬"},

    # ── Aksum/history (domain-specific, slightly older kids) ───────────────
    {"am": "ወርቅ",    "om": "warqee",  "cat": "aksum",   "emoji": "🥇"},
    {"am": "ወደብ",    "om": "buufata", "cat": "aksum",   "emoji": "⚓"},
    {"am": "ንጉሥ",    "om": "mootii",  "cat": "aksum",   "emoji": "👑"},
    {"am": "ሳንቲም",   "om": "saantima","cat": "aksum",   "emoji": "🪙"},
    {"am": "ሀገር",    "om": "biyya",   "cat": "aksum",   "emoji": "🗺️"},
    {"am": "ታሪክ",    "om": "seenaa",  "cat": "aksum",   "emoji": "📜"},
    {"am": "ንግድ",    "om": "daldala", "cat": "aksum",   "emoji": "🤝"},
    {"am": "ሠራዊት",   "om": "waraana", "cat": "aksum",   "emoji": "⚔️"},
    {"am": "ሐውልት",   "om": "yaadannoo","cat": "aksum",  "emoji": "🗿"},
    {"am": "ባህል",    "om": "aadaa",   "cat": "aksum",   "emoji": "🎭"},
]

WORD_CATEGORIES = sorted({w["cat"] for w in WORD_BANK})


def lesson_title(idx, chunk):
    first, last = chunk[0]["id"], chunk[-1]["id"]
    math_count = sum(1 for p in chunk if p["type"] == "math")
    tag = " · word problems" if math_count >= 3 else ""
    return f"Lesson {idx + 1} ({first}\u2013{last}){tag}"


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes — translation
# ---------------------------------------------------------------------------


@app.route("/api/translate", methods=["POST"])
def api_translate():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    direction = payload.get("direction")

    if not text:
        return jsonify({"error": "Enter some text to translate."}), 400
    if len(text) > MAX_INPUT_CHARS:
        return jsonify({"error": f"Keep input under {MAX_INPUT_CHARS} characters."}), 400
    if direction not in ("am2om", "om2am"):
        return jsonify({"error": "Direction must be am2om or om2am."}), 400

    try:
        result = translate(text, direction)
    except Exception as exc:  # surface a readable message instead of a 500 traceback
        return jsonify({"error": f"Translation failed: {exc}"}), 500

    return jsonify({"translation": result})


# ---------------------------------------------------------------------------
# Routes — lessons
# ---------------------------------------------------------------------------


@app.route("/api/lessons")
def api_lessons():
    return jsonify(
        [
            {"id": i, "title": lesson_title(i, chunk), "count": len(chunk)}
            for i, chunk in enumerate(LESSONS)
        ]
    )


@app.route("/api/lessons/<int:lesson_id>")
def api_lesson_detail(lesson_id):
    if lesson_id < 0 or lesson_id >= len(LESSONS):
        return jsonify({"error": "No such lesson."}), 404
    chunk = LESSONS[lesson_id]
    return jsonify(
        {
            "id": lesson_id,
            "title": lesson_title(lesson_id, chunk),
            "pairs": [{"am": p["am"], "om": p["om"], "type": p["type"]} for p in chunk],
        }
    )


# ---------------------------------------------------------------------------
# Routes — quiz
# ---------------------------------------------------------------------------


@app.route("/api/quiz")
def api_quiz():
    """Build a small multiple-choice quiz from the training corpus.

    Query params:
      lesson: lesson id, or "mix" for a random sample across all lessons
      count:  number of questions (default 5, max 15)
    """
    lesson_param = request.args.get("lesson", "mix")
    count = min(int(request.args.get("count", 5)), 15)

    if lesson_param == "mix":
        pool = CORPUS
    else:
        try:
            lesson_id = int(lesson_param)
            pool = LESSONS[lesson_id]
        except (ValueError, IndexError):
            return jsonify({"error": "No such lesson."}), 404

    if len(pool) < 4:
        return jsonify({"error": "Not enough sentences in this lesson for a quiz."}), 400

    sample_size = min(count, len(pool))
    chosen = random.sample(pool, sample_size)

    questions = []
    for i, pair in enumerate(chosen):
        # alternate direction so both are practiced
        direction = "am2om" if i % 2 == 0 else "om2am"
        src_key, tgt_key = ("am", "om") if direction == "am2om" else ("om", "am")

        correct = pair[tgt_key]
        distractor_pool = [p[tgt_key] for p in pool if p["id"] != pair["id"]]
        distractors = random.sample(distractor_pool, min(3, len(distractor_pool)))

        options = distractors + [correct]
        random.shuffle(options)

        questions.append(
            {
                "id": pair["id"],
                "direction": direction,
                "prompt": pair[src_key],
                "options": options,
                "correct_index": options.index(correct),
            }
        )

    return jsonify({"questions": questions})


@app.route("/api/lessons/<int:lesson_id>/translate", methods=["POST"])
def api_lesson_translate(lesson_id):
    """Translate every sentence in a lesson with the live model.

    POST body (JSON):
      direction: "am2om" (default) or "om2am"

    Returns the same structure as /api/lessons/<id> but with an extra
    "model_translation" field on every pair — the model's live output for
    the source side — and the reference translation (the other language's
    stored text) so the UI can show both side-by-side.
    """
    if lesson_id < 0 or lesson_id >= len(LESSONS):
        return jsonify({"error": "No such lesson."}), 404

    payload   = request.get_json(silent=True) or {}
    direction = payload.get("direction", "am2om")
    if direction not in ("am2om", "om2am"):
        return jsonify({"error": "Direction must be am2om or om2am."}), 400

    src_key, tgt_key = ("am", "om") if direction == "am2om" else ("om", "am")
    chunk = LESSONS[lesson_id]

    results = []
    for pair in chunk:
        src_text = pair[src_key]
        try:
            model_out = translate(src_text, direction)
        except Exception as exc:
            model_out = f"[error: {exc}]"
        results.append({
            "id":                pair["id"],
            "type":              pair["type"],
            "source":            src_text,
            "reference":         pair[tgt_key],   # stored ground-truth
            "model_translation": model_out,        # live model output
            "direction":         direction,
        })

    return jsonify({
        "id":        lesson_id,
        "title":     lesson_title(lesson_id, chunk),
        "direction": direction,
        "pairs":     results,
    })


@app.route("/api/translate/word", methods=["POST"])
def api_translate_word():
    """Translate a single word or short phrase with the live model.

    POST body (JSON):
      text:      the word/phrase to translate
      direction: "am2om" or "om2am"

    Used by the kids flip-card and quiz confirmation panels.
    """
    payload   = request.get_json(silent=True) or {}
    text      = (payload.get("text") or "").strip()
    direction = payload.get("direction", "am2om")

    if not text:
        return jsonify({"error": "No text provided."}), 400
    if direction not in ("am2om", "om2am"):
        return jsonify({"error": "Direction must be am2om or om2am."}), 400

    try:
        result = translate(text, direction)
    except Exception as exc:
        return jsonify({"error": f"Translation failed: {exc}"}), 500

    return jsonify({"translation": result, "text": text, "direction": direction})


@app.route("/api/words")
def api_words():
    """Return word bank, optionally filtered by category.

    Query params:
      category: one of the category slugs (numbers, body, family, animals,
                nature, food, colors, school, aksum) — omit for all words.
    """
    cat = request.args.get("category", "").strip().lower()
    words = WORD_BANK if not cat else [w for w in WORD_BANK if w["cat"] == cat]
    return jsonify({
        "categories": WORD_CATEGORIES,
        "words": words,
    })


@app.route("/api/words/quiz")
def api_words_quiz():
    """Build a word-matching quiz.

    Query params:
      category: category slug or 'mix'  (default: mix)
      count:    number of questions      (default 6, max 12)
      direction: am2om | om2am | mix     (default: mix)
    """
    cat       = request.args.get("category", "mix").strip().lower()
    count     = min(int(request.args.get("count", 6)), 12)
    direction = request.args.get("direction", "mix").strip().lower()

    pool = WORD_BANK if cat == "mix" else [w for w in WORD_BANK if w["cat"] == cat]
    if len(pool) < 4:
        return jsonify({"error": "Not enough words in this category."}), 400

    chosen = random.sample(pool, min(count, len(pool)))
    questions = []
    for i, word in enumerate(chosen):
        if direction == "mix":
            d = "am2om" if i % 2 == 0 else "om2am"
        else:
            d = direction

        src_key, tgt_key = ("am", "om") if d == "am2om" else ("om", "am")
        correct = word[tgt_key]

        distractor_pool = [w[tgt_key] for w in pool if w[src_key] != word[src_key]]
        distractors = random.sample(distractor_pool, min(3, len(distractor_pool)))
        options = distractors + [correct]
        random.shuffle(options)

        questions.append({
            "prompt":        word[src_key],
            "emoji":         word["emoji"],
            "cat":           word["cat"],
            "direction":     d,
            "options":       options,
            "correct_index": options.index(correct),
        })

    return jsonify({"questions": questions})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
