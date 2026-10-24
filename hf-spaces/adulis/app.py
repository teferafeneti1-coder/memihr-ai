"""
app.py — Adulis Space wrapper for Hugging Face
Amharic <-> Afan Oromo translation + lessons + quiz + word bank.

Model is downloaded from HF Hub on first run if MODEL_PATH is not set.
HF Hub repo: fenetii/adulis-nllb
"""

import os
import re
import random
import unicodedata
from pathlib import Path

from flask import Flask, jsonify, render_template, request

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── Config ─────────────────────────────────────────────────────────────────
# On HF Spaces, model is downloaded from Hub automatically.
# Locally, set MODEL_PATH env var to your checkpoint folder.
HF_MODEL_REPO = "fenetii/adulis-nllb"
MODEL_PATH    = os.environ.get("MODEL_PATH", HF_MODEL_REPO)

AM_DATA_PATH = os.environ.get(
    "AM_DATA_PATH", os.path.join(os.path.dirname(__file__), "data", "amharic.txt")
)
OM_DATA_PATH = os.environ.get(
    "OM_DATA_PATH", os.path.join(os.path.dirname(__file__), "data", "afanoromo.txt")
)

LANG_CODE  = {"am": "amh_Ethi", "om": "gaz_Latn"}
LESSON_SIZE = 15
MAX_INPUT_CHARS = 500
PORT = int(os.environ.get("PORT", 7860))

app = Flask(__name__)

# ── Load model ─────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[adulis] loading model: {MODEL_PATH}")
print(f"[adulis] device: {device}")

_model_path = Path(MODEL_PATH) if os.path.isdir(MODEL_PATH) else MODEL_PATH
tokenizer = AutoTokenizer.from_pretrained(_model_path)
model     = AutoModelForSeq2SeqLM.from_pretrained(_model_path).to(device)
model.eval()
print("[adulis] model ready ✅")


def translate(text, direction, max_length=96, num_beams=5):
    src, tgt = direction.split("2")
    tokenizer.src_lang = LANG_CODE[src]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANG_CODE[tgt]),
            max_length=max_length,
            num_beams=num_beams,
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


# ── Corpus helpers (identical to original) ─────────────────────────────────
def normalize_quotes(text):
    return text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")

def clean_trailing(text):
    text = text.strip()
    text = re.sub(r"\s*\.\s*\.\s*$", ".", text)
    text = re.sub(r"\.{2,}$", ".", text)
    return text

def normalize_amharic(text):
    text = unicodedata.normalize("NFC", text)
    for k, v in {"ሃ":"ሀ","ሐ":"ሀ","ሓ":"ሀ","ኀ":"ሀ","ኃ":"ሀ","ኸ":"ሀ","ሠ":"ሰ","ሧ":"ሶ","ዐ":"አ","ዓ":"አ"}.items():
        text = text.replace(k, v)
    return clean_trailing(re.sub(r"\s+", " ", normalize_quotes(text)))

def normalize_oromo(text):
    return clean_trailing(re.sub(r"\s+", " ", normalize_quotes(unicodedata.normalize("NFC", text))))

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
    if re.search(r"\d", am) or re.search(r"\d", om):
        return "math"
    if am.strip().endswith("?") or om.strip().endswith("?"):
        return "question"
    return "statement"

def build_corpus():
    am_map = load_id_text(AM_DATA_PATH)
    om_map = load_id_text(OM_DATA_PATH)
    pairs  = []
    for sid in sorted(set(am_map) & set(om_map)):
        am = normalize_amharic(am_map[sid])
        om = normalize_oromo(om_map[sid])
        if am and om:
            pairs.append({"id": sid, "am": am, "om": om, "type": classify(am, om)})
    return pairs

CORPUS  = build_corpus()
LESSONS = [CORPUS[i:i+LESSON_SIZE] for i in range(0, len(CORPUS), LESSON_SIZE)]
print(f"[adulis] {len(CORPUS)} sentence pairs loaded")

# ── Word bank (same as original) ───────────────────────────────────────────
WORD_BANK = [
    {"am":"አንድ","om":"tokko","cat":"numbers","emoji":"1️⃣"},
    {"am":"ሁለት","om":"lama","cat":"numbers","emoji":"2️⃣"},
    {"am":"ሶስት","om":"sadii","cat":"numbers","emoji":"3️⃣"},
    {"am":"አራት","om":"afur","cat":"numbers","emoji":"4️⃣"},
    {"am":"አምስት","om":"shan","cat":"numbers","emoji":"5️⃣"},
    {"am":"እጅ","om":"harka","cat":"body","emoji":"✋"},
    {"am":"እግር","om":"miila","cat":"body","emoji":"🦶"},
    {"am":"ዓይን","om":"ija","cat":"body","emoji":"👁️"},
    {"am":"አፍ","om":"afaan","cat":"body","emoji":"👄"},
    {"am":"አባት","om":"abbaa","cat":"family","emoji":"👨"},
    {"am":"እናት","om":"haadha","cat":"family","emoji":"👩"},
    {"am":"ልጅ","om":"ijoollee","cat":"family","emoji":"🧒"},
    {"am":"ዝሆን","om":"arba","cat":"animals","emoji":"🐘"},
    {"am":"አንበሳ","om":"leenca","cat":"animals","emoji":"🦁"},
    {"am":"ፈረስ","om":"farda","cat":"animals","emoji":"🐴"},
    {"am":"ውሃ","om":"bishaan","cat":"nature","emoji":"💧"},
    {"am":"ፀሐይ","om":"aduu","cat":"nature","emoji":"☀️"},
    {"am":"ዛፍ","om":"muka","cat":"nature","emoji":"🌳"},
    {"am":"ምግብ","om":"nyaata","cat":"food","emoji":"🍽️"},
    {"am":"ዳቦ","om":"buddeena","cat":"food","emoji":"🍞"},
    {"am":"ቡና","om":"buna","cat":"food","emoji":"☕"},
    {"am":"ቀይ","om":"diimaa","cat":"colors","emoji":"🔴"},
    {"am":"ነጭ","om":"adii","cat":"colors","emoji":"⚪"},
    {"am":"ጥቁር","om":"gurracha","cat":"colors","emoji":"⚫"},
    {"am":"ትምህርት","om":"barnoota","cat":"school","emoji":"📚"},
    {"am":"ተማሪ","om":"barataa","cat":"school","emoji":"🧑‍🎓"},
    {"am":"መምህር","om":"barsiisaa","cat":"school","emoji":"👨‍🏫"},
]
WORD_CATEGORIES = sorted({w["cat"] for w in WORD_BANK})

def lesson_title(idx, chunk):
    first, last = chunk[0]["id"], chunk[-1]["id"]
    tag = " · word problems" if sum(1 for p in chunk if p["type"]=="math") >= 3 else ""
    return f"Lesson {idx+1} ({first}–{last}){tag}"

# ── Routes (identical to original) ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/translate", methods=["POST"])
def api_translate():
    payload   = request.get_json(silent=True) or {}
    text      = (payload.get("text") or "").strip()
    direction = payload.get("direction")
    if not text:
        return jsonify({"error": "Enter some text to translate."}), 400
    if len(text) > MAX_INPUT_CHARS:
        return jsonify({"error": f"Keep input under {MAX_INPUT_CHARS} characters."}), 400
    if direction not in ("am2om", "om2am"):
        return jsonify({"error": "Direction must be am2om or om2am."}), 400
    try:
        return jsonify({"translation": translate(text, direction)})
    except Exception as exc:
        return jsonify({"error": f"Translation failed: {exc}"}), 500

@app.route("/api/lessons")
def api_lessons():
    return jsonify([{"id": i, "title": lesson_title(i, c), "count": len(c)}
                    for i, c in enumerate(LESSONS)])

@app.route("/api/lessons/<int:lesson_id>")
def api_lesson_detail(lesson_id):
    if lesson_id < 0 or lesson_id >= len(LESSONS):
        return jsonify({"error": "No such lesson."}), 404
    chunk = LESSONS[lesson_id]
    return jsonify({"id": lesson_id, "title": lesson_title(lesson_id, chunk),
                    "pairs": [{"am": p["am"], "om": p["om"], "type": p["type"]} for p in chunk]})

@app.route("/api/quiz")
def api_quiz():
    lesson_param = request.args.get("lesson", "mix")
    count = min(int(request.args.get("count", 5)), 15)
    pool  = CORPUS if lesson_param == "mix" else LESSONS[int(lesson_param)]
    if len(pool) < 4:
        return jsonify({"error": "Not enough sentences."}), 400
    chosen    = random.sample(pool, min(count, len(pool)))
    questions = []
    for i, pair in enumerate(chosen):
        direction = "am2om" if i % 2 == 0 else "om2am"
        src_key, tgt_key = ("am", "om") if direction == "am2om" else ("om", "am")
        correct = pair[tgt_key]
        distractor_pool = [p[tgt_key] for p in pool if p["id"] != pair["id"]]
        distractors = random.sample(distractor_pool, min(3, len(distractor_pool)))
        options = distractors + [correct]
        random.shuffle(options)
        questions.append({"id": pair["id"], "direction": direction,
                          "prompt": pair[src_key], "options": options,
                          "correct_index": options.index(correct)})
    return jsonify({"questions": questions})

@app.route("/api/translate/word", methods=["POST"])
def api_translate_word():
    payload   = request.get_json(silent=True) or {}
    text      = (payload.get("text") or "").strip()
    direction = payload.get("direction", "am2om")
    if not text:
        return jsonify({"error": "No text provided."}), 400
    try:
        return jsonify({"translation": translate(text, direction), "text": text, "direction": direction})
    except Exception as exc:
        return jsonify({"error": f"Translation failed: {exc}"}), 500

@app.route("/api/words")
def api_words():
    cat   = request.args.get("category", "").strip().lower()
    words = WORD_BANK if not cat else [w for w in WORD_BANK if w["cat"] == cat]
    return jsonify({"categories": WORD_CATEGORIES, "words": words})

@app.route("/api/words/quiz")
def api_words_quiz():
    cat       = request.args.get("category", "mix").strip().lower()
    count     = min(int(request.args.get("count", 6)), 12)
    direction = request.args.get("direction", "mix").strip().lower()
    pool      = WORD_BANK if cat == "mix" else [w for w in WORD_BANK if w["cat"] == cat]
    if len(pool) < 4:
        return jsonify({"error": "Not enough words."}), 400
    chosen    = random.sample(pool, min(count, len(pool)))
    questions = []
    for i, word in enumerate(chosen):
        d = "am2om" if direction == "mix" and i%2==0 else ("om2am" if direction=="mix" else direction)
        src_key, tgt_key = ("am","om") if d=="am2om" else ("om","am")
        correct = word[tgt_key]
        distractor_pool = [w[tgt_key] for w in pool if w[src_key] != word[src_key]]
        distractors = random.sample(distractor_pool, min(3, len(distractor_pool)))
        options = distractors + [correct]
        random.shuffle(options)
        questions.append({"prompt": word[src_key], "emoji": word["emoji"],
                          "cat": word["cat"], "direction": d,
                          "options": options, "correct_index": options.index(correct)})
    return jsonify({"questions": questions})

@app.route("/api/lessons/<int:lesson_id>/translate", methods=["POST"])
def api_lesson_translate(lesson_id):
    if lesson_id < 0 or lesson_id >= len(LESSONS):
        return jsonify({"error": "No such lesson."}), 404
    payload   = request.get_json(silent=True) or {}
    direction = payload.get("direction", "am2om")
    if direction not in ("am2om","om2am"):
        return jsonify({"error": "Direction must be am2om or om2am."}), 400
    src_key, tgt_key = ("am","om") if direction=="am2om" else ("om","am")
    results = []
    for pair in LESSONS[lesson_id]:
        try:    model_out = translate(pair[src_key], direction)
        except Exception as exc: model_out = f"[error: {exc}]"
        results.append({"id": pair["id"], "type": pair["type"],
                         "source": pair[src_key], "reference": pair[tgt_key],
                         "model_translation": model_out, "direction": direction})
    return jsonify({"id": lesson_id, "title": lesson_title(lesson_id, LESSONS[lesson_id]),
                    "direction": direction, "pairs": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
