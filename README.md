# 🧠 memihr ai — Ethiopian AI Learning Suite

An AI-powered learning platform for Ethiopian languages and sign language.
Built for the **INSA Ethiopia AI Summer Camp 2026** 🇪🇹

---

## 🌐 Live Model

| Model | Link |
|-------|------|
| Adulis NLLB (Amharic ↔ Afan Oromo) | [huggingface.co/fenetii/adulis-nllb](https://huggingface.co/fenetii/adulis-nllb) |

---

## 📦 Projects

### 1. ✍️ Fidel CNN Model Tester — port 5001
Draw any handwritten Ethiopic (Fidel) character on a canvas — a CNN trained from scratch recognizes it from 287 classes. Supports user corrections and in-browser fine-tuning.

- **Model**: Custom CNN (from scratch), 3× conv blocks, 287 classes
- **Model size**: ~4 MB (included in repo)
- **Location**: `fidelaifinal/`

### 2. 🤟 Sign Language Translator — port 8000
Real-time ASL word recognition from the webcam using MediaPipe hand landmarks + a custom MLP classifier trained in PyTorch.

- **Model**: MediaPipe Hands + custom MLP (63 landmark features → ASL word classes)
- **Model size**: ~1.4 MB (included in repo)
- **Location**: `egate/signlanguagetranslate/`

### 3. 📖 English-Amharic Scratch — port 5004
English → Amharic translation with a Transformer built from scratch (no pretrained weights). Includes word bank, daily word, and quiz.

- **Model**: Custom Transformer (from scratch), BLEU 9.54
- **Model size**: ~52 MB (included in repo)
- **Location**: `English-Amharic-Scratch/`

### 4. 🌍 Adulis (Amharic ↔ Afan Oromo) — port 5003
Bidirectional Amharic ↔ Afan Oromo translation, flashcards, quizzes and word bank using a fine-tuned NLLB model.

- **Model**: Meta NLLB-200 fine-tuned on Amharic ↔ Afan Oromo corpus
- **Model size**: 2.4 GB — **download from HF Hub** (see below)
- **Location**: `oromifa-amharic & amharic-oromifa/adulis/`

### 5. 🎓 Amharic-English Teaching Assistant — port 5002
Amharic → English lessons, quizzes and free translation with a fine-tuned NLLB model and a background pre-warm system.

- **Model**: Meta NLLB-200 fine-tuned on Amharic-English corpus
- **Model size**: 2.4 GB — **download separately** (see below)
- **Location**: `Amharic-English-finetuned/`

---

## 🚀 How to Run (Full Suite)

### 1. Clone the repo
```bash
git clone https://github.com/fenetii/memihr-ai.git
cd memihr-ai
```

### 2. Set up Python environment
```bash
python -m venv myenv
source myenv/bin/activate
pip install flask fastapi uvicorn torch transformers sentencepiece psutil requests pillow numpy
```

### 3. Download the large NLLB models

**Adulis model** (2.4 GB):
```bash
# Option A — using huggingface_hub
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('fenetii/adulis-nllb', local_dir='oromifa-amharic & amharic-oromifa/archive')"

# Option B — using hf CLI
hf download fenetii/adulis-nllb --local-dir "oromifa-amharic & amharic-oromifa/archive"
```

**Amharic-English model** (2.4 GB):
> ⚠️ This model is not yet publicly hosted. Contact the author or skip this project.
> Place the model files in `Amharic-English-finetuned/archive/`

### 4. Start the dashboard
```bash
cd egate/memihr-ai
./run.sh
```

Then open **http://127.0.0.1:9000** in your browser.

The dashboard automatically starts all projects in order (light apps first, heavy apps last). The memory guard will skip heavy projects if RAM is insufficient.

### 5. Or run any project individually
```bash
# Fidel CNN
cd fidelaifinal && python app.py

# Sign Language
cd egate/signlanguagetranslate/backend && python app.py

# English-Amharic Scratch
cd English-Amharic-Scratch && python app.py

# Adulis (set MODEL_PATH first)
MODEL_PATH="oromifa-amharic & amharic-oromifa/archive" python app.py

# Amharic-English
cd Amharic-English-finetuned && python app.py
```

---

## 🤖 AI Models Summary

| Project | Model Type | Trained | Size |
|---------|-----------|---------|------|
| Fidel CNN | CNN from scratch | Custom dataset, 287 Ethiopic chars | 4 MB |
| Sign Language | MLP on MediaPipe landmarks | ASL gestures | 1.4 MB |
| English-Amharic Scratch | Transformer from scratch | BLEU 9.54 | 52 MB |
| Adulis | NLLB-200 fine-tuned | Amharic ↔ Afan Oromo corpus | 2.4 GB |
| Amharic-English | NLLB-200 fine-tuned | Amharic-English corpus | 2.4 GB |

**All models run locally on CPU — zero cloud API cost.**

---

## 🏗️ Architecture

```
memihr-ai/
  egate/
    memihr-ai/          ← Dashboard (FastAPI, port 9000)
    signlanguagetranslate/ ← Sign Language (FastAPI, port 8000)
  fidelaifinal/         ← Fidel CNN (Flask, port 5001)
  English-Amharic-Scratch/ ← Scratch Transformer (Flask, port 5004)
  oromifa-amharic & amharic-oromifa/
    adulis/             ← Adulis (Flask, port 5003)
  Amharic-English-finetuned/ ← Teaching Assistant (Flask, port 5002)
  hf-spaces/            ← Hugging Face deployment files
```

The **memihr-ai dashboard** starts all projects via a subprocess launcher that:
- Overrides each Flask app's port without editing project files
- Disables debug/reloader (prevents double model loading)
- Gives each app a unique session cookie name
- Applies a memory guard before starting heavy models

---

## 📋 Requirements

- Python 3.10+
- 8+ GB RAM recommended (16 GB for all 5 projects simultaneously)
- CPU only (no GPU required)
- Linux / macOS

---

## 📁 What's NOT in this repo

| File | Why excluded | Where to get it |
|------|-------------|-----------------|
| `oromifa-amharic*/archive/` | 2.4 GB NLLB model | [huggingface.co/fenetii/adulis-nllb](https://huggingface.co/fenetii/adulis-nllb) |
| `Amharic-English-finetuned/archive/` | 2.4 GB NLLB model | Contact author |
| `fidelaifinal/corrections/` | Runtime output | Generated when you use the app |
| `*/models/mlp_versions/` | Trained checkpoints | Generated when you train |

---

## 👤 Author

**Feneti Tefera**
INSA Ethiopia AI Summer Camp 2026
GitHub: [@fenetii](https://github.com/fenetii)
HuggingFace: [@fenetii](https://huggingface.co/fenetii)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
