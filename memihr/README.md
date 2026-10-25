# memihr — AI Projects

A collection of five AI projects built around Ethiopian languages, sign language translation, and script recognition. All five are live and publicly accessible.

---

## Live Deployments

| Project | URL | Description |
|---|---|---|
| 🤟 Sign Language Translator | [sign-language-yi6z.onrender.com](https://sign-language-yi6z.onrender.com) | Real-time ASL word recognition in the browser |
| 🔤 Fidel CNN | [fidel-cnn.onrender.com](https://fidel-cnn.onrender.com) | Ethiopic (Ge'ez) script character recognition |
| 🌐 Amharic→English (Fine-tuned) | [fenetii/nllb-amharic-english-finetuned](https://huggingface.co/fenetii/nllb-amharic-english-finetuned) | Fine-tuned NLLB-600M translation model on Hugging Face |
| 🇪🇹 English→Amharic (Scratch) | [english-amharic-scratch-memhirai.onrender.com](https://english-amharic-scratch-memhirai.onrender.com) | From-scratch Transformer translation app |
| 🎬 Demo Video | [youtu.be/mt9Ag7MOpyg](https://youtu.be/mt9Ag7MOpyg) | Full walkthrough of all projects |

---

## Projects

### 1. Sign Language Translator
Real-time ASL word recognition running entirely in the browser. Uses MediaPipe hand and pose landmarks, a pre-trained MLP, and a live kNN classifier. No server-side image processing — the camera never leaves the browser.

- **Live:** [sign-language-yi6z.onrender.com](https://sign-language-yi6z.onrender.com)
- **Stack:** FastAPI backend · MediaPipe · PyTorch MLP · kNN fusion · vanilla JS
- **Features:** add custom gestures, retrain MLP live, parity self-test page
- **Folder:** `signlanguagetranslate/`

---

### 2. Fidel CNN
Convolutional neural network that recognizes Ethiopic (Ge'ez / Fidel) script characters. Upload or draw a character and get the predicted label with confidence score.

- **Live:** [fidel-cnn.onrender.com](https://fidel-cnn.onrender.com)
- **Stack:** Flask · PyTorch CNN · vanilla JS
- **Folder:** `fidelaifinal/` (outside this workspace)

---

### 3. Amharic → English (Fine-tuned NLLB)
`facebook/nllb-200-distilled-600M` fine-tuned on ~65k Amharic–English sentence pairs across six domains (Bible, JW daily quotes, news, legal, history). Trained on Kaggle with 2 × Tesla T4 GPUs using DDP.

- **Model:** [huggingface.co/fenetii/nllb-amharic-english-finetuned](https://huggingface.co/fenetii/nllb-amharic-english-finetuned)
- **Stack:** HuggingFace Transformers · NLLB-200 · Seq2SeqTrainer · DDP
- **Direction:** Amharic (`amh_Ethi`) → English (`eng_Latn`)
- **Folder:** `Amharic-English-finetuned/`

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("fenetii/nllb-amharic-english-finetuned", src_lang="amh_Ethi")
model     = AutoModelForSeq2SeqLM.from_pretrained("fenetii/nllb-amharic-english-finetuned")

inputs     = tokenizer("ሰላም ዓለም", return_tensors="pt")
translated = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"))
print(tokenizer.decode(translated[0], skip_special_tokens=True))
```

---

### 4. English → Amharic (From Scratch)
Encoder-decoder Transformer trained from random initialization — no pretrained weights, no pretrained tokenizer. Two SentencePiece BPE vocabularies (8k tokens each) trained directly on the corpus. Built to demonstrate a full NMT pipeline from the ground up. Features a language learning UI with lessons, quizzes, and sentence translation.

- **Live:** [english-amharic-scratch-memhirai.onrender.com](https://english-amharic-scratch-memhirai.onrender.com)
- **Stack:** Flask · PyTorch Transformer · SentencePiece BPE · gunicorn · Render
- **Direction:** English → Amharic
- **Trained on:** Google Colab T4
- **Folder:** `English-Amharic-Scratch/`

---

### 5. Adulis — Amharic ↔ Afan Oromo
Translation assistant between Amharic and Afan Oromo using a fine-tuned NLLB model.

- **Stack:** Flask · HuggingFace Transformers · NLLB
- **Folder:** `oromifa-amharic & amharic-oromifa/adulis/` (outside this workspace)

---

## memihr-ai Dashboard

A single-page dashboard that runs and manages all five projects locally. Handles process lifecycle, memory guarding, staggered startup, and log streaming.

```bash
cd memihr-ai
./run.sh
# open http://127.0.0.1:9000
```

See [`memihr-ai/README.md`](memihr-ai/README.md) for full dashboard documentation.

---

## Demo Video

Full walkthrough of all five projects — sign language recognition, Fidel CNN, both translation apps, and the language learning UI.

[![Demo Video](https://img.youtube.com/vi/mt9Ag7MOpyg/0.jpg)](https://youtu.be/mt9Ag7MOpyg)

---

## Notebooks

Training notebooks are in `ipynb for all/`:

| Notebook | Description |
|---|---|
| `amharic-englishfinetuned.ipynb` | NLLB fine-tuning on Kaggle (2× T4, DDP) |
| `english_to_amharic_from_scratch_colab(1)(1).ipynb` | Full NMT pipeline from scratch on Colab |
| `afanoromo-amharic(1).ipynb` | Afan Oromo ↔ Amharic training |
| `cnnfidel.ipynb` | Fidel CNN training |
| `finetuned.ipynb` | Earlier fine-tuning experiments |
| `signlanguagetransator.ipynb` | Sign language model experiments |
