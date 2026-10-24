"""
Amharic ↔ English AI Teaching Assistant
Flask backend — all translation done by the fine-tuned NLLB-200 model.

Startup is instant. Translations are computed lazily:
  - The requested lesson category is translated on first access, then cached.
  - A background thread quietly pre-warms all other categories after the server
    is up, so by the time a student opens a second category it is already ready.
  - Quiz questions use the same cache; daily word is translated live (one call).
"""

import random
import datetime
import threading
from flask import Flask, request, render_template, jsonify, session

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_DIR = "/home/feneti/egate/egate/Amharic-English-finetuned/archive"
SECRET_KEY = "amharic-teach-secret-2024"

# ──────────────────────────────────────────────
# LESSON CORPUS  (Amharic only — English filled lazily by the model)
# ──────────────────────────────────────────────
LESSONS_RAW = {
    "greetings": [
        {"am": "ሰላም",           "phonetic": "se-lam",              "sentence": "ሰላም፣ ስሜ ሌሊሳ ነው።"},
        {"am": "አመሰግናለሁ",      "phonetic": "ame-seg-na-le-hu",    "sentence": "ላደረጉልኝ ነገር አመሰግናለሁ።"},
        {"am": "ይቅርታ",         "phonetic": "yi-kir-ta",           "sentence": "ይቅርታ፣ ዘግይቻለሁ።"},
        {"am": "እንኳን ደህና መጡ",  "phonetic": "in-kwan deh-na met-u","sentence": "እንኳን ደህና መጡ ወደ ቤታችን።"},
        {"am": "እንዴት ነህ?",     "phonetic": "in-det neh?",         "sentence": "ዛሬ እንዴት ነህ?"},
        {"am": "ደህና ነኝ",       "phonetic": "deh-na neny",         "sentence": "ደህና ነኝ፣ አመሰግናለሁ።"},
    ],
    "family": [
        {"am": "እናት",   "phonetic": "i-nat",     "sentence": "እናቴ ቡና ታፈላለች።"},
        {"am": "አባት",   "phonetic": "a-bat",     "sentence": "አባቴ ሥራ ይሠራል።"},
        {"am": "ወንድም", "phonetic": "wen-dim",   "sentence": "ወንድሜ ትምህርት ቤት ይሄዳል።"},
        {"am": "እህት",   "phonetic": "i-hit",     "sentence": "እህቴ ትዘምራለች።"},
        {"am": "ቤተሰብ", "phonetic": "be-te-seb", "sentence": "ቤተሰቤ ትልቅ ነው።"},
        {"am": "አያት",   "phonetic": "a-yat",     "sentence": "አያቴ ታሪክ ይነግረኛል።"},
    ],
    "nature": [
        {"am": "ፀሐይ",  "phonetic": "tse-hay",  "sentence": "ፀሐዩ ዛሬ ደማቅ ነው።"},
        {"am": "ዝናብ",  "phonetic": "zi-nab",   "sentence": "ዝናቡ ምድሪቱን አረሰወ።"},
        {"am": "ዛፍ",   "phonetic": "zaf",      "sentence": "ዛፉ ረጅም ነው።"},
        {"am": "አበባ",  "phonetic": "a-be-ba",  "sentence": "አበቦቹ ቀይ ናቸው።"},
        {"am": "ወንዝ",  "phonetic": "wenz",     "sentence": "ወንዙ ፈጣን ነው።"},
        {"am": "ተራራ", "phonetic": "te-ra-ra", "sentence": "ተራራው ከፍ ያለ ነው።"},
    ],
    "school": [
        {"am": "ትምህርት ቤት", "phonetic": "tim-hirt-bet", "sentence": "ትምህርት ቤቴ ቅርብ ነው።"},
        {"am": "መጽሐፍ",     "phonetic": "mets-haf",     "sentence": "ይህ መጽሐፍ አስደሳች ነው።"},
        {"am": "መምህር",     "phonetic": "mem-hir",      "sentence": "መምህሬ ደግ ሰው ናቸው።"},
        {"am": "ጓደኛ",      "phonetic": "gwa-de-nya",   "sentence": "ጓደኛዬ ብልህ ነው።"},
        {"am": "ጥያቄ",      "phonetic": "ti-ya-ke",     "sentence": "አንድ ጥያቄ አለኝ።"},
        {"am": "መልስ",      "phonetic": "mels",         "sentence": "መልሱ ትክክል ነው።"},
    ],
    "food": [
        {"am": "እንጀራ", "phonetic": "in-je-ra", "sentence": "እንጀራው ሙቅ ነው።"},
        {"am": "ውሃ",   "phonetic": "wi-ha",    "sentence": "ቀዝቃዛ ውሃ ጠጣሁ།"},
        {"am": "ምግብ",  "phonetic": "mi-gib",   "sentence": "ምግቡ ጣፋጭ ነው።"},
        {"am": "ቡና",   "phonetic": "bu-na",    "sentence": "ቡናው ሞቃት ነው።"},
        {"am": "ዳቦ",   "phonetic": "da-bo",    "sentence": "ዳቦ በጠዋት በላሁ።"},
        {"am": "ፍሬ",   "phonetic": "fi-re",    "sentence": "ፍሬው ጣፋጭ ነው።"},
    ],
}

# ──────────────────────────────────────────────
# MODEL  (load once, fast)
# ──────────────────────────────────────────────
print("Loading model…")
tok = AutoTokenizer.from_pretrained(MODEL_DIR, src_lang="amh_Ethi", tgt_lang="eng_Latn")
model_nllb = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
model_nllb.eval()
print("Model ready.")

# Lock so background thread and request threads don't translate simultaneously
_translate_lock = threading.Lock()


def translate(text: str, max_length: int = 128) -> str:
    with _translate_lock:
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
        with torch.no_grad():
            out = model_nllb.generate(
                **inputs,
                forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"),
                max_length=max_length,
                num_beams=2,
            )
        return tok.batch_decode(out, skip_special_tokens=True)[0]


# ──────────────────────────────────────────────
# LAZY TRANSLATION CACHE
# LESSONS[cat] is None until first request for that category.
# Once translated it stays cached for the lifetime of the process.
# ──────────────────────────────────────────────
LESSONS: dict[str, list[dict] | None] = {cat: None for cat in LESSONS_RAW}
_cache_lock = threading.Lock()


def _translate_category(category: str) -> list[dict]:
    """Translate a category and store it in the cache. Thread-safe."""
    with _cache_lock:
        # Double-check after acquiring lock
        if LESSONS[category] is not None:
            return LESSONS[category]
        print(f"Translating category: {category}…")
        enriched = []
        for w in LESSONS_RAW[category]:
            enriched.append({
                "am":          w["am"],
                "phonetic":    w["phonetic"],
                "sentence":    w["sentence"],
                "en":          translate(w["am"]),
                "sentence_en": translate(w["sentence"]),
            })
        LESSONS[category] = enriched
        print(f"  {category} ready ({len(enriched)} words).")
        return enriched


def _get_all_words() -> list[dict]:
    """Return flat word list from whatever categories are already cached."""
    words = []
    for cat, data in LESSONS.items():
        if data:
            words.extend(data)
    return words


def _background_prewarm():
    """Silently translate all categories in the background after server starts."""
    for cat in LESSONS_RAW:
        if LESSONS[cat] is None:
            _translate_category(cat)
    print("Background pre-warm complete — all categories cached.")


# ──────────────────────────────────────────────
# DAILY WORD  — translated live (single call, fast)
# ──────────────────────────────────────────────
def get_daily_word() -> dict:
    all_raw = [w for words in LESSONS_RAW.values() for w in words]
    day_index = datetime.date.today().toordinal() % len(all_raw)
    w = all_raw[day_index]
    return {
        "am":       w["am"],
        "phonetic": w["phonetic"],
        "en":       translate(w["am"]),
    }


# ──────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = SECRET_KEY


# ── Main page ──────────────────────────────────
@app.route("/")
def index():
    return render_template(
        "index.html",
        lessons=list(LESSONS_RAW.keys()),
        daily_word=get_daily_word(),
    )


# ── Set / get student name ─────────────────────
@app.route("/api/set-name", methods=["POST"])
def set_name():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name required"}), 400
    session["student_name"] = name
    session.setdefault("stars", 0)
    session.setdefault("streak", 0)
    session.setdefault("session_score", 0)
    return jsonify({
        "name":          name,
        "stars":         session["stars"],
        "streak":        session["streak"],
        "session_score": session["session_score"],
    })


@app.route("/api/progress")
def get_progress():
    return jsonify({
        "name":          session.get("student_name", ""),
        "stars":         session.get("stars", 0),
        "streak":        session.get("streak", 0),
        "session_score": session.get("session_score", 0),
    })


# ── Translate (free-form) ──────────────────────
@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > 300:
        return jsonify({"error": "Text too long (max 300 characters)"}), 400
    result = translate(text)
    return jsonify({"translation": result, "source": text})


# ── Lesson: lazy-translate on first access ─────
@app.route("/api/lesson/<category>")
def api_lesson(category):
    if category not in LESSONS_RAW:
        return jsonify({"error": "Unknown category"}), 404
    # Translate now if not yet cached (blocks this request, ~10-20s first time)
    words = LESSONS[category] or _translate_category(category)
    return jsonify({"category": category, "words": words})


# ── Quiz: generate a question ─────────────────
@app.route("/api/quiz/question", methods=["POST"])
def api_quiz_question():
    data = request.get_json(force=True) or {}
    seen = set(data.get("seen", []))

    # Use whatever is cached; fall back to raw Amharic if nothing translated yet
    cached = _get_all_words()
    if not cached:
        # Nothing translated yet — translate greetings on the spot as seed
        cached = _translate_category("greetings")

    pool = [w for w in cached if w["am"] not in seen]
    if not pool:
        return jsonify({"done": True})

    correct = random.choice(pool)
    distractors = random.sample(
        [w for w in cached if w["am"] != correct["am"]],
        k=min(3, len(cached) - 1),
    )
    options = [w["am"] for w in distractors] + [correct["am"]]
    random.shuffle(options)

    return jsonify({
        "question_en": correct["en"],
        "correct_am":  correct["am"],
        "phonetic":    correct["phonetic"],
        "options":     options,
        "done":        False,
    })


# ── Quiz: check answer + update score ─────────
@app.route("/api/quiz/check", methods=["POST"])
def api_quiz_check():
    data = request.get_json(force=True)
    chosen  = (data.get("chosen")  or "").strip()
    correct = (data.get("correct") or "").strip()
    is_correct = chosen == correct

    if is_correct:
        session["stars"]         = session.get("stars", 0) + 1
        session["streak"]        = session.get("streak", 0) + 1
        session["session_score"] = session.get("session_score", 0) + 10
    else:
        session["streak"] = 0

    return jsonify({
        "correct":       is_correct,
        "stars":         session.get("stars", 0),
        "streak":        session.get("streak", 0),
        "session_score": session.get("session_score", 0),
    })


if __name__ == "__main__":
    # Kick off background pre-warm AFTER the server is already serving
    t = threading.Thread(target=_background_prewarm, daemon=True)
    t.start()
    app.run(host="127.0.0.1", port=5000, debug=False)
