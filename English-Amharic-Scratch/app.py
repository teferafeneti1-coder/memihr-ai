"""
English -> Amharic translation server.
Uses pure NumPy inference — no PyTorch at runtime.
Weights are loaded from model/weights.npz (exported by export_numpy.py).
"""

import json
import math
import os

import numpy as np
import sentencepiece as spm
from flask import Flask, jsonify, render_template, request

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

WEIGHTS_PATH = os.path.join(MODEL_DIR, "weights.npz")
EN_SPM_PATH  = os.path.join(MODEL_DIR, "en_spm.model")
AM_SPM_PATH  = os.path.join(MODEL_DIR, "am_spm.model")
CONFIG_PATH  = os.path.join(MODEL_DIR, "config.json")

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

# --------------------------------------------------------------------------
# Pure-NumPy Transformer inference
# --------------------------------------------------------------------------

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    return w * (x - mean) / np.sqrt(var + eps) + b

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

def relu(x):
    return np.maximum(0, x)

def multihead_attn(q, k, v, in_proj_w, in_proj_b, out_proj_w, out_proj_b,
                   nhead, attn_mask=None, key_padding_mask=None):
    """
    q: (T, d), k: (S, d), v: (S, d)  — batch-size-1, squeezed
    in_proj_w: (3*d, d)  projects Q, K, V all at once
    """
    d = q.shape[-1]
    head_dim = d // nhead

    # Project each separately using slices of in_proj_w / in_proj_b
    q_p = q @ in_proj_w[:d].T      + in_proj_b[:d]       # (T, d)
    k_p = k @ in_proj_w[d:2*d].T   + in_proj_b[d:2*d]    # (S, d)
    v_p = v @ in_proj_w[2*d:].T    + in_proj_b[2*d:]      # (S, d)

    T, S = q.shape[0], k.shape[0]

    # Split heads: (nhead, seq, head_dim)
    def split_heads(x, seq):
        return x.reshape(seq, nhead, head_dim).transpose(1, 0, 2)

    Q = split_heads(q_p, T)   # (nhead, T, head_dim)
    K = split_heads(k_p, S)   # (nhead, S, head_dim)
    V = split_heads(v_p, S)   # (nhead, S, head_dim)

    scale  = math.sqrt(head_dim)
    scores = Q @ K.transpose(0, 2, 1) / scale   # (nhead, T, S)

    if attn_mask is not None:
        scores = scores + attn_mask               # broadcast over heads

    if key_padding_mask is not None:
        # mask shape: (S,) — True means ignore
        scores = scores + key_padding_mask[None, None, :] * -1e9

    attn_w = softmax(scores)                     # (nhead, T, S)
    ctx    = attn_w @ V                          # (nhead, T, head_dim)
    ctx    = ctx.transpose(1, 0, 2).reshape(T, d)

    return ctx @ out_proj_w.T + out_proj_b


def encoder_layer(x, src_pad_mask, w, nhead):
    # Self-attention
    residual = x
    attn_out = multihead_attn(
        x, x, x,
        w["self_attn.in_proj_weight"], w["self_attn.in_proj_bias"],
        w["self_attn.out_proj.weight"], w["self_attn.out_proj.bias"],
        nhead, key_padding_mask=src_pad_mask,
    )
    x = layer_norm(residual + attn_out, w["norm1.weight"], w["norm1.bias"])

    # FFN
    residual = x
    ffn = relu(x @ w["linear1.weight"].T + w["linear1.bias"])
    ffn = ffn @ w["linear2.weight"].T + w["linear2.bias"]
    x   = layer_norm(residual + ffn, w["norm2.weight"], w["norm2.bias"])
    return x


def decoder_layer(tgt, memory, tgt_mask, src_pad_mask, w, nhead):
    # Self-attention (masked)
    residual = tgt
    sa = multihead_attn(
        tgt, tgt, tgt,
        w["self_attn.in_proj_weight"], w["self_attn.in_proj_bias"],
        w["self_attn.out_proj.weight"], w["self_attn.out_proj.bias"],
        nhead, attn_mask=tgt_mask,
    )
    tgt = layer_norm(residual + sa, w["norm1.weight"], w["norm1.bias"])

    # Cross-attention
    residual = tgt
    ca = multihead_attn(
        tgt, memory, memory,
        w["multihead_attn.in_proj_weight"], w["multihead_attn.in_proj_bias"],
        w["multihead_attn.out_proj.weight"], w["multihead_attn.out_proj.bias"],
        nhead, key_padding_mask=src_pad_mask,
    )
    tgt = layer_norm(residual + ca, w["norm2.weight"], w["norm2.bias"])

    # FFN
    residual = tgt
    ffn = relu(tgt @ w["linear1.weight"].T + w["linear1.bias"])
    ffn = ffn @ w["linear2.weight"].T + w["linear2.bias"]
    tgt = layer_norm(residual + ffn, w["norm3.weight"], w["norm3.bias"])
    return tgt


class NumpyTransformer:
    def __init__(self, weights_path, config):
        raw = np.load(weights_path)
        self.W   = {k: raw[k].astype(np.float32) for k in raw.files}
        self.d   = config["d_model"]
        self.max_len = config["max_len"]
        self.nhead   = 8
        self.n_enc   = 4
        self.n_dec   = 4

        # positional encoding: (1, max_len, d)
        self.pe = self.W["pos_enc.pe"]  # (1, 128, 256)

        # embeddings
        self.src_emb = self.W["src_tok_emb.weight"]  # (vocab, d)
        self.tgt_emb = self.W["tgt_tok_emb.weight"]
        self.gen_w   = self.W["generator.weight"]    # (tgt_vocab, d)
        self.gen_b   = self.W["generator.bias"]

        # layer weights per index
        self.enc_layers = [
            {k.split(f"encoder.layers.{i}.")[1]: v
             for k, v in self.W.items() if k.startswith(f"encoder.layers.{i}.")}
            for i in range(self.n_enc)
        ]
        self.dec_layers = [
            {k.split(f"decoder.layers.{i}.")[1]: v
             for k, v in self.W.items() if k.startswith(f"decoder.layers.{i}.")}
            for i in range(self.n_dec)
        ]

    def _pos_embed(self, ids, emb_matrix):
        seq_len = len(ids)
        x = emb_matrix[ids] * math.sqrt(self.d)            # (T, d)
        x = x + self.pe[0, :seq_len, :]                    # add PE
        return x.astype(np.float32)

    def encode(self, src_ids, src_pad_mask):
        x = self._pos_embed(src_ids, self.src_emb)
        for lw in self.enc_layers:
            x = encoder_layer(x, src_pad_mask, lw, self.nhead)
        return x  # (S, d)

    def decode_step(self, tgt_ids, memory, src_pad_mask):
        T = len(tgt_ids)
        # causal mask: (T, T) upper-tri = -inf
        tgt_mask = np.full((T, T), -1e9, dtype=np.float32)
        tgt_mask = np.triu(tgt_mask, k=1)

        x = self._pos_embed(tgt_ids, self.tgt_emb)
        for lw in self.dec_layers:
            x = decoder_layer(x, memory, tgt_mask, src_pad_mask, lw, self.nhead)

        logits = x[-1] @ self.gen_w.T + self.gen_b   # (tgt_vocab,)
        return logits


# --------------------------------------------------------------------------
# Load config, tokenizers, model
# --------------------------------------------------------------------------
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = json.load(f)

en_sp = spm.SentencePieceProcessor(model_file=EN_SPM_PATH)
am_sp = spm.SentencePieceProcessor(model_file=AM_SPM_PATH)

model = NumpyTransformer(WEIGHTS_PATH, CONFIG)
print(f"✅ Model loaded (numpy). EN vocab={en_sp.get_piece_size()}, "
      f"AM vocab={am_sp.get_piece_size()}, BLEU={CONFIG.get('test_bleu'):.2f}")
print("🚀 Server ready.")


# --------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------
def translate(sentence: str, max_len: int = 128) -> str:
    sentence = sentence.strip()
    if not sentence:
        return ""

    src_ids     = en_sp.encode(sentence, out_type=int)[:max_len - 1] + [EOS_ID]
    src_pad_mask = np.array([i == PAD_ID for i in src_ids], dtype=np.float32)

    memory = model.encode(src_ids, src_pad_mask)

    tgt_ids = [BOS_ID]
    for _ in range(max_len - 1):
        logits     = model.decode_step(tgt_ids, memory, src_pad_mask)
        next_token = int(np.argmax(logits))
        tgt_ids.append(next_token)
        if next_token == EOS_ID:
            break

    out_ids = tgt_ids[1:]
    if EOS_ID in out_ids:
        out_ids = out_ids[:out_ids.index(EOS_ID)]
    return am_sp.decode(out_ids)


# --------------------------------------------------------------------------
# Word bank, quiz, daily word  (unchanged)
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
        {"en": "hello",       "am": "ሰላም"},
        {"en": "goodbye",     "am": "ቸር ይሁንልህ"},
        {"en": "thank you",   "am": "አመሰግናለሁ"},
        {"en": "please",      "am": "እባክህ"},
        {"en": "sorry",       "am": "ይቅርታ"},
        {"en": "yes",         "am": "አዎ"},
        {"en": "no",          "am": "አይ"},
        {"en": "help",        "am": "እርዳታ"},
        {"en": "welcome",     "am": "እንኳን ደህና መጣህ"},
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
        {"en": "sun",    "am": "ፀሐይ"},
        {"en": "moon",   "am": "ጨረቃ"},
        {"en": "star",   "am": "ኮከብ"},
        {"en": "rain",   "am": "ዝናብ"},
        {"en": "wind",   "am": "ነፋስ"},
        {"en": "fire",   "am": "እሳት"},
        {"en": "water",  "am": "ውሃ"},
        {"en": "tree",   "am": "ዛፍ"},
        {"en": "flower", "am": "አበባ"},
        {"en": "sky",    "am": "ሰማይ"},
    ],
}

ALL_PAIRS = [p for pairs in WORD_BANK.values() for p in pairs]

import random as _random
import datetime as _datetime


def _daily_seed():
    today = _datetime.date.today()
    return today.year * 10000 + today.month * 100 + today.day


MAX_WORDS      = 15
MAX_WORD_LENGTH = 14
MAX_CHARS      = 200


def complexity_error(text):
    words = text.split()
    if len(words) > MAX_WORDS:
        return (f"ዓረፍተ ነገሩ ረጅም ነው! 😅  That's a long sentence! "
                f"Try {MAX_WORDS} words or fewer / {MAX_WORDS} ቃላት ወይም ያነሰ ሞክር።")
    long_words = [w for w in words if len(w.strip(".,!?;:\"'")) > MAX_WORD_LENGTH]
    if long_words:
        return ("ያ ቃል ውስብስብ ይመስላል! 🤔  That word looks tricky! "
                "Try everyday words / ቀላል ቃላትን ሞክር።")
    if len(text) > MAX_CHARS:
        return (f"ብዙ ፊደላት! ✏️  Too many characters! Keep it under {MAX_CHARS}.")
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
    return jsonify(WORD_BANK)

@app.route("/dailyword")
def daily_word():
    rng  = _random.Random(_daily_seed())
    pair = rng.choice(ALL_PAIRS)
    return jsonify(pair)

@app.route("/quiz")
def quiz():
    category = request.args.get("category", "").strip()
    pool = WORD_BANK.get(category, ALL_PAIRS) if category else ALL_PAIRS
    if len(pool) < 4:
        pool = ALL_PAIRS
    correct_pair  = _random.choice(pool)
    wrong_pool    = [p for p in ALL_PAIRS if p["am"] != correct_pair["am"]]
    wrong_choices = _random.sample(wrong_pool, min(3, len(wrong_pool)))
    choices = [w["am"] for w in wrong_choices] + [correct_pair["am"]]
    _random.shuffle(choices)
    return jsonify({
        "question_en": correct_pair["en"],
        "question_am": correct_pair["am"],
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
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
