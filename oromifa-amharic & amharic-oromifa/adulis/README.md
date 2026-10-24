# Adulis — Amharic ⇄ Afan Oromo learning app

A local Flask app built around your fine-tuned NLLB-200 checkpoint. Three tabs:

- **Translate** — bidirectional Amharic ⇄ Afan Oromo, calls your model directly.
- **Lessons** — the 11.8k training sentences, grouped into browsable chunks
  (flashcard style, Amharic + Afan Oromo side by side). Word-problem-heavy
  chunks are flagged.
- **Quiz** — multiple-choice questions auto-generated from the same sentences:
  read a sentence in one language, pick the sentence that means the same
  thing in the other. Direction alternates question to question.

## Setup

```bash
cd adulis
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Point it at your model

By default `app.py` looks at:

```
/home/feneti/oromifa-amharic & amharic-oromifa/archive
```

If your checkpoint lives somewhere else, override it with an environment
variable instead of editing the code:

```bash
export MODEL_PATH="/home/feneti/oromifa-amharic & amharic-oromifa/archive"
```

(On Windows/PowerShell: `$env:MODEL_PATH = "..."`)

That folder needs to contain the full HuggingFace save — `config.json`,
`model.safetensors`, `tokenizer.json`, `sentencepiece.bpe.model`,
`special_tokens_map.json`, `tokenizer_config.json`, `generation_config.json`.
The optimizer/scheduler/rng/trainer-state files in your screenshot aren't
needed at inference time — they're training checkpoint bookkeeping, not
required for `AutoModelForSeq2SeqLM.from_pretrained(...)` to load, but they're
harmless to leave in the folder.

## Run

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in a browser.

## Notes

- The Lessons and Quiz tabs read `data/amharic.txt` and `data/afanoromo.txt`
  (already included here — your original 11,800 pairs) at startup. They use
  the same ID-join + normalization logic your training notebook used, so the
  text matches what the model was actually trained on.
- Lessons are simply consecutive runs of 15 sentence pairs — the IDs in your
  corpus already run topically (e.g. a block of ~15–20 about Aksum's trade),
  so this keeps each lesson coherent without needing extra topic-modeling.
- The quiz has no server-side session/anti-cheat — it's a personal study
  tool, not a graded test, so the correct answer is sent to the browser
  along with the question and revealed after you answer.
- First translation request after starting the server will be a little slow
  while the model does its first forward pass; subsequent ones are faster.
- If you don't have a GPU locally, it still runs on CPU — just slower per
  translation (a few seconds instead of under one).
