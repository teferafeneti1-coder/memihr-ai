"""
app.py — Adulis Gradio Space
Amharic <-> Afan Oromo translation + lessons + quiz + word bank.
Model: fenetii/adulis-nllb (downloaded from HF Hub on first run)
"""

import os, re, random, unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import gradio as gr

# ── Model ─────────────────────────────────────────────────────────────────────
HF_MODEL_REPO = "fenetii/adulis-nllb"
MODEL_PATH    = os.environ.get("MODEL_PATH", HF_MODEL_REPO)
LANG_CODE     = {"am": "amh_Ethi", "om": "gaz_Latn"}
MAX_CHARS     = 500

print(f"[adulis] loading model: {MODEL_PATH}")
device    = "cuda" if torch.cuda.is_available() else "cpu"
_path     = Path(MODEL_PATH) if os.path.isdir(MODEL_PATH) else MODEL_PATH
tokenizer = AutoTokenizer.from_pretrained(_path)
model     = AutoModelForSeq2SeqLM.from_pretrained(_path).to(device)
model.eval()
print(f"[adulis] model ready ✅  device={device}")


def translate(text: str, direction: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if len(text) > MAX_CHARS:
        return f"⚠️ Input too long (max {MAX_CHARS} characters)"
    if direction not in ("am2om", "om2am"):
        return "⚠️ Invalid direction"
    src, tgt = direction.split("2")
    tokenizer.src_lang = LANG_CODE[src]
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANG_CODE[tgt]),
            max_length=128, num_beams=5,
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


# ── Corpus ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"

def _norm_quotes(t):
    return t.replace("\u201c",'"').replace("\u201d",'"').replace("\u2018","'").replace("\u2019","'")

def _clean(t):
    t = t.strip()
    t = re.sub(r"\s*\.\s*\.\s*$",".",t)
    t = re.sub(r"\.{2,}$",".",t)
    return t

def _norm_am(t):
    t = unicodedata.normalize("NFC", t)
    for k,v in {"ሃ":"ሀ","ሐ":"ሀ","ሓ":"ሀ","ኀ":"ሀ","ኃ":"ሀ","ኸ":"ሀ","ሠ":"ሰ","ዐ":"አ","ዓ":"አ"}.items():
        t = t.replace(k,v)
    return _clean(re.sub(r"\s+"," ",_norm_quotes(t)))

def _norm_om(t):
    return _clean(re.sub(r"\s+"," ",_norm_quotes(unicodedata.normalize("NFC",t))))

def _load_corpus():
    am_map, om_map = {}, {}
    for line in (DATA_DIR/"amharic.txt").read_text(encoding="utf-8").splitlines():
        if "|" in line:
            sid, txt = line.split("|",1)
            am_map[sid.strip()] = txt.strip()
    for line in (DATA_DIR/"afanoromo.txt").read_text(encoding="utf-8").splitlines():
        if "|" in line:
            sid, txt = line.split("|",1)
            om_map[sid.strip()] = txt.strip()
    pairs = []
    for sid in sorted(set(am_map) & set(om_map)):
        am = _norm_am(am_map[sid])
        om = _norm_om(om_map[sid])
        if am and om:
            pairs.append({"id": sid, "am": am, "om": om})
    return pairs

CORPUS  = _load_corpus()
LESSONS = [CORPUS[i:i+15] for i in range(0, len(CORPUS), 15)]
print(f"[adulis] {len(CORPUS)} sentence pairs, {len(LESSONS)} lessons")

# ── Word bank ─────────────────────────────────────────────────────────────────
WORD_BANK = [
    ("አንድ","tokko","1️⃣","numbers"), ("ሁለት","lama","2️⃣","numbers"),
    ("ሶስት","sadii","3️⃣","numbers"), ("አራት","afur","4️⃣","numbers"),
    ("አምስት","shan","5️⃣","numbers"), ("ስድስት","ja'a","6️⃣","numbers"),
    ("ሰባት","torba","7️⃣","numbers"), ("ስምንት","saddeet","8️⃣","numbers"),
    ("ዘጠኝ","sagal","9️⃣","numbers"), ("አስር","kudhan","🔟","numbers"),
    ("እጅ","harka","✋","body"),     ("እግር","miila","🦶","body"),
    ("ዓይን","ija","👁️","body"),      ("አፍ","afaan","👄","body"),
    ("ጆሮ","gurra","👂","body"),     ("ራስ","mataa","🧠","body"),
    ("አባት","abbaa","👨","family"),  ("እናት","haadha","👩","family"),
    ("ወንድም","obbolaa","👦","family"),("እህት","obboleettii","👧","family"),
    ("ልጅ","ijoollee","🧒","family"),("ቤተሰብ","maatii","👨‍👩‍👧‍👦","family"),
    ("ዝሆን","arba","🐘","animals"),  ("አንበሳ","leenca","🦁","animals"),
    ("ፈረስ","farda","🐴","animals"), ("ከብት","loon","🐄","animals"),
    ("ውሻ","saree","🐕","animals"),  ("ድመት","adurree","🐱","animals"),
    ("ወፍ","simbirra","🐦","animals"),("አሳ","qurxummii","🐟","animals"),
    ("ውሃ","bishaan","💧","nature"), ("ፀሐይ","aduu","☀️","nature"),
    ("ዛፍ","muka","🌳","nature"),    ("ወንዝ","laga","🏞️","nature"),
    ("ሰማይ","samii","🌤️","nature"), ("ምድር","lafaa","🌍","nature"),
    ("ተራራ","gaara","⛰️","nature"), ("ዝናብ","rooba","🌧️","nature"),
    ("ምግብ","nyaata","🍽️","food"),  ("ዳቦ","buddeena","🍞","food"),
    ("ወተት","aannan","🥛","food"),   ("ቡና","buna","☕","food"),
    ("ማር","damma","🍯","food"),     ("ጨው","soogidda","🧂","food"),
    ("ቀይ","diimaa","🔴","colors"),  ("ነጭ","adii","⚪","colors"),
    ("ጥቁር","gurracha","⚫","colors"),("አረንጓዴ","magariisa","🟢","colors"),
    ("ሰማያዊ","cuquliisa","🔵","colors"),("ቢጫ","keelloo","🟡","colors"),
    ("ትምህርት","barnoota","📚","school"),("ተማሪ","barataa","🧑‍🎓","school"),
    ("መምህር","barsiisaa","👨‍🏫","school"),("ደብተር","kitaaba","📓","school"),
    ("ፊደል","qubee","🔤","school"),  ("ቤት","mana","🏠","school"),
    ("ሀገር","biyya","🗺️","aksum"),   ("ታሪክ","seenaa","📜","aksum"),
    ("ንጉሥ","mootii","👑","aksum"),  ("ንግድ","daldala","🤝","aksum"),
    ("ወርቅ","warqee","🥇","aksum"),  ("ባህል","aadaa","🎭","aksum"),
]
CATEGORIES = sorted(set(w[3] for w in WORD_BANK))


# ── Gradio functions ──────────────────────────────────────────────────────────

def do_translate(text, direction_label):
    direction = "am2om" if "Amharic → Afan Oromo" in direction_label else "om2am"
    if not text.strip():
        return "⚠️ Please enter some text."
    result = translate(text, direction)
    return result


def get_lesson(lesson_idx):
    if lesson_idx < 0 or lesson_idx >= len(LESSONS):
        return "Invalid lesson"
    chunk = LESSONS[lesson_idx]
    lines = [f"**Lesson {lesson_idx+1}** — {len(chunk)} sentence pairs\n"]
    lines.append("| # | አማርኛ (Amharic) | Afan Oromo |")
    lines.append("|---|----------------|------------|")
    for i, pair in enumerate(chunk, 1):
        lines.append(f"| {i} | {pair['am']} | {pair['om']} |")
    return "\n".join(lines)


def build_quiz():
    chosen = random.sample(CORPUS, min(5, len(CORPUS)))
    questions = []
    for i, pair in enumerate(chosen):
        direction = "am2om" if i % 2 == 0 else "om2am"
        src_key, tgt_key = ("am","om") if direction=="am2om" else ("om","am")
        correct = pair[tgt_key]
        others  = [p[tgt_key] for p in CORPUS if p["id"] != pair["id"]]
        distractors = random.sample(others, min(3, len(others)))
        options = distractors + [correct]
        random.shuffle(options)
        src_lang = "አማርኛ" if src_key=="am" else "Afan Oromo"
        tgt_lang = "Afan Oromo" if tgt_key=="om" else "አማርኛ"
        questions.append({
            "prompt": pair[src_key],
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "options": options,
            "correct": correct,
        })
    return questions

_quiz_state = {"questions": [], "index": 0, "score": 0}

def start_quiz():
    _quiz_state["questions"] = build_quiz()
    _quiz_state["index"]  = 0
    _quiz_state["score"]  = 0
    return show_question()

def show_question():
    qs = _quiz_state["questions"]
    idx = _quiz_state["index"]
    if idx >= len(qs):
        score = _quiz_state["score"]
        total = len(qs)
        emoji = "🏆" if score == total else "👍" if score >= total//2 else "📚"
        return (
            f"## {emoji} Quiz Complete!\n\n"
            f"**Score: {score} / {total}**\n\n"
            f"{'Excellent! You know your Amharic and Afan Oromo!' if score==total else 'Keep practicing — you are getting better!'}\n\n"
            f"Click **Start Quiz** to try again.",
            gr.update(choices=[], visible=False),
            gr.update(visible=False),
        )
    q = qs[idx]
    prompt_md = (
        f"### Question {idx+1} of {len(qs)}\n\n"
        f"**Translate from {q['src_lang']} to {q['tgt_lang']}:**\n\n"
        f"> {q['prompt']}"
    )
    return (
        prompt_md,
        gr.update(choices=q["options"], value=None, visible=True),
        gr.update(visible=True),
    )

def check_answer(choice):
    qs  = _quiz_state["questions"]
    idx = _quiz_state["index"]
    if idx >= len(qs) or not choice:
        return "Please select an answer.", *show_question()
    q = qs[idx]
    correct = q["correct"]
    if choice == correct:
        _quiz_state["score"] += 1
        feedback = f"✅ **Correct!**"
    else:
        feedback = f"❌ **Wrong.** Correct answer: **{correct}**"
    _quiz_state["index"] += 1
    prompt_md, radio_update, btn_update = show_question()
    return feedback, prompt_md, radio_update, btn_update


def get_wordbank(category):
    words = [w for w in WORD_BANK if w[3] == category]
    lines = [f"## {category.title()} Vocabulary\n"]
    lines.append("| Emoji | አማርኛ (Amharic) | Afan Oromo |")
    lines.append("|-------|----------------|------------|")
    for am, om, emoji, _ in words:
        lines.append(f"| {emoji} | {am} | {om} |")
    return "\n".join(lines)


# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Adulis — Amharic ↔ Afan Oromo",
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="yellow"),
    css="""
        .header { text-align: center; padding: 16px 0 8px; }
        .header h1 { font-size: 2em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 8px; }
    """
) as demo:

    gr.HTML("""
        <div class="header">
            <h1>🌍 Adulis</h1>
            <p class="subtitle">
                Amharic (አማርኛ) ↔ Afan Oromo Translation<br/>
                <small>Fine-tuned NLLB model · Part of <b>memihr ai</b> · INSA Ethiopia AI Summer Camp</small>
            </p>
        </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Translate ────────────────────────────────────────────────
        with gr.Tab("🔄 Translate"):
            gr.Markdown("### Live Translation — powered by fine-tuned NLLB model")
            direction = gr.Radio(
                choices=["Amharic → Afan Oromo (አማርኛ → Afan Oromo)",
                         "Afan Oromo → Amharic (Afan Oromo → አማርኛ)"],
                value="Amharic → Afan Oromo (አማርኛ → Afan Oromo)",
                label="Direction",
            )
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Input",
                        placeholder="Type here… / እዚህ ይጻፉ… / Asitti barreessi…",
                        lines=4, max_lines=8,
                    )
                with gr.Column():
                    output_text = gr.Textbox(
                        label="Translation",
                        lines=4, max_lines=8,
                        interactive=False,
                    )
            translate_btn = gr.Button("🔄 Translate", variant="primary")
            gr.Examples(
                examples=[
                    ["ሰላም፣ እንደምን አደርክ?", "Amharic → Afan Oromo (አማርኛ → Afan Oromo)"],
                    ["Maqaan kee hoo maal?", "Afan Oromo → Amharic (Afan Oromo → አማርኛ)"],
                    ["ኢትዮጵያ ቆንጆ ሀገር ናት።", "Amharic → Afan Oromo (አማርኛ → Afan Oromo)"],
                    ["Barnoota kee akkamiin deemaa jira?", "Afan Oromo → Amharic (Afan Oromo → አማርኛ)"],
                    ["ውሃ ጠጣ።", "Amharic → Afan Oromo (አማርኛ → Afan Oromo)"],
                ],
                inputs=[input_text, direction],
            )
            translate_btn.click(do_translate, [input_text, direction], output_text)
            input_text.submit(do_translate, [input_text, direction], output_text)

        # ── Tab 2: Lessons ──────────────────────────────────────────────────
        with gr.Tab("📚 Lessons"):
            gr.Markdown(f"### Browse {len(LESSONS)} lessons from the bilingual corpus")
            lesson_slider = gr.Slider(
                minimum=0, maximum=len(LESSONS)-1,
                value=0, step=1, label=f"Lesson (0 – {len(LESSONS)-1})",
            )
            lesson_output = gr.Markdown(get_lesson(0))
            lesson_slider.change(get_lesson, lesson_slider, lesson_output)

        # ── Tab 3: Quiz ─────────────────────────────────────────────────────
        with gr.Tab("🎯 Quiz"):
            gr.Markdown("### Test your Amharic ↔ Afan Oromo knowledge")
            start_btn    = gr.Button("🎯 Start Quiz", variant="primary")
            question_md  = gr.Markdown("")
            answer_radio = gr.Radio(choices=[], label="Choose your answer:", visible=False)
            submit_btn   = gr.Button("Submit Answer", visible=False)
            feedback_md  = gr.Markdown("")

            start_btn.click(
                start_quiz, [],
                [question_md, answer_radio, submit_btn]
            )
            submit_btn.click(
                check_answer, [answer_radio],
                [feedback_md, question_md, answer_radio, submit_btn]
            )

        # ── Tab 4: Word Bank ────────────────────────────────────────────────
        with gr.Tab("🔤 Word Bank"):
            gr.Markdown("### Vocabulary by category — Amharic ↔ Afan Oromo")
            cat_dropdown = gr.Dropdown(
                choices=CATEGORIES,
                value=CATEGORIES[0],
                label="Category",
            )
            wordbank_output = gr.Markdown(get_wordbank(CATEGORIES[0]))
            cat_dropdown.change(get_wordbank, cat_dropdown, wordbank_output)

        # ── Tab 5: About ────────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
## About Adulis

**Adulis** is named after the ancient Red Sea port that moved goods between Aksum and the world.
This app moves sentences between **Amharic** and **Afan Oromo** — two of Ethiopia's most spoken languages.

### Model
- Base: Meta's **NLLB-200** (No Language Left Behind)
- Fine-tuned on an Amharic ↔ Afan Oromo parallel corpus
- Languages: `amh_Ethi` (Amharic) ↔ `gaz_Latn` (Afan Oromo)
- Corpus: **{len(CORPUS)} sentence pairs** across **{len(LESSONS)} lessons**
- Model Hub: [fenetii/adulis-nllb](https://huggingface.co/fenetii/adulis-nllb)

### memihr ai Suite
| App | Description |
|-----|-------------|
| 🌍 **Adulis** (this) | Amharic ↔ Afan Oromo translation |
| ✍️ **Fidel CNN** | Ethiopic handwriting recognition (287 characters) |
| 🤟 **Sign Language** | Real-time ASL recognition via webcam |
| 📖 **English-Amharic** | English → Amharic from-scratch Transformer |
| 🎓 **Teaching Assistant** | Amharic-English lessons with fine-tuned NLLB |

Built for the **INSA Ethiopia AI Summer Camp** 🇪🇹
            """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
