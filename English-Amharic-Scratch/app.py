"""
English -> Amharic translation server.

Loads the from-scratch Transformer (matching the architecture trained in
english_to_amharic_from_scratch_colab.ipynb) and serves a small Flask app
with a single translation endpoint + a simple HTML page.

Expected folder layout (relative to this file):

    app.py
    templates/
        index.html
    model/
        model.pt
        en_spm.model
        am_spm.model
        config.json

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import json
import math
import os

import sentencepiece as spm
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PT_PATH = os.path.join(MODEL_DIR, "model.pt")
EN_SPM_PATH = os.path.join(MODEL_DIR, "en_spm.model")
AM_SPM_PATH = os.path.join(MODEL_DIR, "am_spm.model")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

# --------------------------------------------------------------------------
# Model definition — must match the notebook exactly, or the saved
# state_dict won't load.
# --------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=128,
    ):
        super().__init__()
        self.d_model = d_model

        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_ID)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_key_padding_mask):
        src_emb = self.pos_enc(self.src_tok_emb(src) * math.sqrt(self.d_model))
        return self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        tgt_emb = self.pos_enc(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))
        return self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask):
        tgt_mask = generate_square_subsequent_mask(tgt.size(1), tgt.device)
        memory = self.encode(src, src_key_padding_mask)
        out = self.decode(tgt, memory, tgt_mask, tgt_key_padding_mask, src_key_padding_mask)
        return self.generator(out)


def generate_square_subsequent_mask(size, device):
    return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)


def make_padding_mask(seq, pad_id=PAD_ID):
    return seq == pad_id


# --------------------------------------------------------------------------
# Load config, tokenizers, and model once at startup
# --------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

D_MODEL = CONFIG.get("d_model", 256)
MAX_LEN = CONFIG.get("max_len", 128)

en_sp = spm.SentencePieceProcessor(model_file=EN_SPM_PATH)
am_sp = spm.SentencePieceProcessor(model_file=AM_SPM_PATH)

model = Seq2SeqTransformer(
    src_vocab_size=en_sp.get_piece_size(),
    tgt_vocab_size=am_sp.get_piece_size(),
    d_model=D_MODEL,
    max_len=MAX_LEN,
).to(DEVICE)

state_dict = torch.load(MODEL_PT_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

print(f"Loaded model on {DEVICE}. EN vocab={en_sp.get_piece_size()}, "
      f"AM vocab={am_sp.get_piece_size()}, test_bleu={CONFIG.get('test_bleu')}")


# --------------------------------------------------------------------------
# Greedy decoding — identical logic to the notebook's translate()
# --------------------------------------------------------------------------
def translate(sentence: str, max_len: int = MAX_LEN) -> str:
    sentence = sentence.strip()
    if not sentence:
        return ""

    src_ids = en_sp.encode(sentence, out_type=int)[: max_len - 1] + [EOS_ID]
    src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
    src_pad_mask = make_padding_mask(src)

    with torch.no_grad():
        memory = model.encode(src, src_pad_mask)
        ys = torch.tensor([[BOS_ID]], dtype=torch.long, device=DEVICE)
        for _ in range(max_len - 1):
            tgt_mask = generate_square_subsequent_mask(ys.size(1), DEVICE)
            out = model.decode(ys, memory, tgt_mask, None, src_pad_mask)
            logits = model.generator(out[:, -1, :])
            next_token = logits.argmax(dim=-1).item()
            ys = torch.cat([ys, torch.tensor([[next_token]], device=DEVICE)], dim=1)
            if next_token == EOS_ID:
                break

    out_ids = ys[0].tolist()[1:]
    if EOS_ID in out_ids:
        out_ids = out_ids[: out_ids.index(EOS_ID)]
    return am_sp.decode(out_ids)


# --------------------------------------------------------------------------
# Bilingual word bank — every entry is {"en": "...", "am": "..."}
# Categories use both English and Amharic labels for the UI.
# --------------------------------------------------------------------------
WORD_BANK = {
    "🐾 Animals / እንስሳት": [
        {"en": "cat",      "am": "ድመት"},
        {"en": "dog",      "am": "ውሻ"},
        {"en": "bird",     "am": "ወፍ"},
        {"en": "fish",     "am": "ዓሳ"},
        {"en": "lion",     "am": "አንበሳ"},
        {"en": "elephant", "am": "ዝሆን"},
        {"en": "monkey",   "am": "ዝንጀሮ"},
        {"en": "rabbit",   "am": "ጥንቸል"},
        {"en": "horse",    "am": "ፈረስ"},
        {"en": "cow",      "am": "ላም"},
        {"en": "sheep",    "am": "በግ"},
        {"en": "frog",     "am": "እንቁራሪት"},
    ],
    "🍎 Food / ምግብ": [
        {"en": "bread",    "am": "ዳቦ"},
        {"en": "milk",     "am": "ወተት"},
        {"en": "water",    "am": "ውሃ"},
        {"en": "apple",    "am": "አፕል"},
        {"en": "banana",   "am": "ሙዝ"},
        {"en": "egg",      "am": "እንቁላል"},
        {"en": "rice",     "am": "ሩዝ"},
        {"en": "meat",     "am": "ሥጋ"},
        {"en": "orange",   "am": "ብርቱካን"},
        {"en": "mango",    "am": "ማንጎ"},
        {"en": "honey",    "am": "ማር"},
        {"en": "butter",   "am": "ቅቤ"},
    ],
    "👨‍👩‍👧 Family / ቤተሰብ": [
        {"en": "mother",      "am": "እናት"},
        {"en": "father",      "am": "አባት"},
        {"en": "sister",      "am": "እህት"},
        {"en": "brother",     "am": "ወንድም"},
        {"en": "baby",        "am": "ሕፃን"},
        {"en": "grandmother", "am": "አያት"},
        {"en": "grandfather", "am": "አያት"},
        {"en": "family",      "am": "ቤተሰብ"},
        {"en": "child",       "am": "ልጅ"},
        {"en": "friend",      "am": "ጓደኛ"},
    ],
    "🌈 Colors / ቀለማት": [
        {"en": "red",    "am": "ቀይ"},
        {"en": "blue",   "am": "ሰማያዊ"},
        {"en": "green",  "am": "አረንጓዴ"},
        {"en": "yellow", "am": "ቢጫ"},
        {"en": "white",  "am": "ነጭ"},
        {"en": "black",  "am": "ጥቁር"},
        {"en": "orange", "am": "ብርቱካናማ"},
        {"en": "purple", "am": "ወይናማ"},
        {"en": "pink",   "am": "ሮዝ"},
        {"en": "brown",  "am": "ቡናማ"},
    ],
    "📚 School / ትምህርት ቤት": [
        {"en": "book",    "am": "መጽሐፍ"},
        {"en": "pen",     "am": "እስክርቢቶ"},
        {"en": "teacher", "am": "መምህር"},
        {"en": "student", "am": "ተማሪ"},
        {"en": "school",  "am": "ትምህርት ቤት"},
        {"en": "learn",   "am": "ተማር"},
        {"en": "read",    "am": "አንብብ"},
        {"en": "write",   "am": "ጻፍ"},
        {"en": "numbers", "am": "ቁጥሮች"},
        {"en": "letters", "am": "ፊደሎች"},
    ],
    "🌍 Places / ቦታዎች": [
        {"en": "home",     "am": "ቤት"},
        {"en": "market",   "am": "ገበያ"},
        {"en": "church",   "am": "ቤተ ክርስቲያን"},
        {"en": "hospital", "am": "ሆስፒታል"},
        {"en": "road",     "am": "መንገድ"},
        {"en": "city",     "am": "ከተማ"},
        {"en": "village",  "am": "መንደር"},
        {"en": "river",    "am": "ወንዝ"},
        {"en": "mountain", "am": "ተራራ"},
        {"en": "field",    "am": "ሜዳ"},
    ],
    "😊 Feelings / ስሜቶች": [
        {"en": "happy",   "am": "ደስተኛ"},
        {"en": "sad",     "am": "አዝኗል"},
        {"en": "angry",   "am": "የተናደደ"},
        {"en": "tired",   "am": "ደክሟል"},
        {"en": "hungry",  "am": "ረሃብ"},
        {"en": "scared",  "am": "ፈርቷል"},
        {"en": "excited", "am": "ተደሰተ"},
        {"en": "love",    "am": "ፍቅር"},
        {"en": "good",    "am": "ጥሩ"},
        {"en": "bad",     "am": "መጥፎ"},
    ],
    "🗣️ Greetings / ሰላምታ": [
        {"en": "hello",      "am": "ሰላም"},
        {"en": "goodbye",    "am": "ቸር ይሁንልህ"},
        {"en": "thank you",  "am": "አመሰግናለሁ"},
        {"en": "please",     "am": "እባክህ"},
        {"en": "sorry",      "am": "ይቅርታ"},
        {"en": "yes",        "am": "አዎ"},
        {"en": "no",         "am": "አይ"},
        {"en": "help",       "am": "እርዳታ"},
        {"en": "welcome",    "am": "እንኳን ደህና መጣህ"},
        {"en": "good morning","am": "እንደምን አደርክ"},
    ],
    "🏃 Actions / ድርጊቶች": [
        {"en": "run",   "am": "ሮጥ"},
        {"en": "walk",  "am": "ሂድ"},
        {"en": "eat",   "am": "ብላ"},
        {"en": "drink", "am": "ጠጣ"},
        {"en": "sleep", "am": "ተኛ"},
        {"en": "play",  "am": "ጫወት"},
        {"en": "sing",  "am": "ዘምር"},
        {"en": "dance", "am": "ዘፍን"},
        {"en": "cook",  "am": "አብሥል"},
        {"en": "pray",  "am": "ጸልይ"},
    ],
    "🌦️ Nature / ተፈጥሮ": [
        {"en": "sun",   "am": "ፀሐይ"},
        {"en": "moon",  "am": "ጨረቃ"},
        {"en": "star",  "am": "ኮከብ"},
        {"en": "rain",  "am": "ዝናብ"},
        {"en": "wind",  "am": "ነፋስ"},
        {"en": "fire",  "am": "እሳት"},
        {"en": "water", "am": "ውሃ"},
        {"en": "tree",  "am": "ዛፍ"},
        {"en": "flower","am": "አበባ"},
        {"en": "sky",   "am": "ሰማይ"},
    ],
}

# Flat list of all pairs — used by /quiz and /dailyword
ALL_PAIRS = [pair for pairs in WORD_BANK.values() for pair in pairs]

import random as _random
import datetime as _datetime


def _daily_seed() -> int:
    """Deterministic seed based on today's date so everyone gets the same daily word."""
    today = _datetime.date.today()
    return today.year * 10000 + today.month * 100 + today.day


# Complexity guard thresholds
MAX_WORDS = 15          # more than this is too advanced
MAX_WORD_LENGTH = 14    # very long single words are usually complex / technical
MAX_CHARS = 200         # hard character ceiling


def complexity_error(text: str):
    """Return a bilingual friendly error string if the input is too complex, else None."""
    words = text.split()
    if len(words) > MAX_WORDS:
        return (
            f"ዓረፍተ ነገሩ ረጅም ነው! 😅  That's a long sentence! "
            f"Try {MAX_WORDS} words or fewer / {MAX_WORDS} ቃላት ወይም ያነሰ ሞክር።"
        )
    long_words = [w for w in words if len(w.strip(".,!?;:\"'")) > MAX_WORD_LENGTH]
    if long_words:
        return (
            "ያ ቃል ውስብስብ ይመስላል! 🤔  That word looks tricky! "
            "Try everyday words from the word bank / ከቃላት ባንኩ ቀላል ቃላትን ሞክር።"
        )
    if len(text) > MAX_CHARS:
        return (
            f"ብዙ ፊደላት! ✏️  Too many characters! "
            f"Keep it under {MAX_CHARS} / ከ{MAX_CHARS} ፊደሎች በታች ያድርጉ።"
        )
    return None


# --------------------------------------------------------------------------
# Flask app
# --------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/wordbank")
def wordbank():
    """Return the word bank as JSON so the frontend can render it."""
    return jsonify(WORD_BANK)


@app.route("/dailyword")
def daily_word():
    """Return a deterministic word-of-the-day (same for everyone on the same date)."""
    rng = _random.Random(_daily_seed())
    pair = rng.choice(ALL_PAIRS)
    return jsonify(pair)


@app.route("/quiz")
def quiz():
    """
    Return a multiple-choice quiz question.
    Response: { question_en, question_am, correct, choices: [str, str, str, str] }
    All choices are Amharic words; exactly one is correct.
    Optional query param: ?category=<category name>
    """
    category = request.args.get("category", "").strip()
    if category and category in WORD_BANK:
        pool = WORD_BANK[category]
    else:
        pool = ALL_PAIRS

    if len(pool) < 4:
        pool = ALL_PAIRS  # fallback to full list if category is tiny

    correct_pair = _random.choice(pool)
    # Pick 3 wrong answers from the rest of the pool
    wrong_pool = [p for p in ALL_PAIRS if p["am"] != correct_pair["am"]]
    wrong_choices = _random.sample(wrong_pool, min(3, len(wrong_pool)))
    choices = [w["am"] for w in wrong_choices] + [correct_pair["am"]]
    _random.shuffle(choices)

    return jsonify({
        "question_en": correct_pair["en"],
        "question_am": correct_pair["am"],   # used to reveal answer
        "correct":     correct_pair["am"],
        "choices":     choices,
    })


@app.route("/translate", methods=["POST"])
def translate_endpoint():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please type something first! 😊"}), 400

    err = complexity_error(text.strip())
    if err:
        return jsonify({"error": err}), 400

    translation = translate(text)
    return jsonify({"source": text, "translation": translation})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
