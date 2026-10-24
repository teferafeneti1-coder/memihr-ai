---
title: Adulis — Amharic ↔ Afan Oromo Translator
emoji: 🌍
colorFrom: green
colorTo: yellow
sdk: docker
pinned: true
license: mit
---

# 🌍 Adulis — Amharic ↔ Afan Oromo Translator

Named after the ancient Red Sea port that connected Aksum to the world — **Adulis** translates between **Amharic (አማርኛ)** and **Afan Oromo**, two of Ethiopia's most widely spoken languages.

## Features
- 🔄 **Bidirectional translation** — Amharic → Afan Oromo and Afan Oromo → Amharic
- 📚 **Structured lessons** — sentence pairs from a curated bilingual corpus
- 🃏 **Flashcards** — flip-card vocabulary practice
- 🎯 **Quiz mode** — multiple-choice questions from lesson content
- 🔤 **Word bank** — categorized vocabulary (numbers, body, family, animals, food, colors, school)

## Model
- **Base**: Meta's NLLB-200 (No Language Left Behind)
- **Fine-tuned** on an Amharic ↔ Afan Oromo parallel corpus
- **Languages**: `amh_Ethi` (Amharic) ↔ `gaz_Latn` (Afan Oromo)
- Model hosted at: [fenetii/adulis-nllb](https://huggingface.co/fenetii/adulis-nllb)
- **First load**: ~2–3 minutes (downloading 2.4 GB model)

## Part of memihr ai
This app is part of the **memihr ai** suite — an AI learning platform for Ethiopian languages, built for the INSA Ethiopia AI Summer Camp.

| App | Description |
|-----|-------------|
| ✍️ **Fidel CNN** | Ethiopic handwriting recognition |
| 🌍 **Adulis** (this) | Amharic ↔ Afan Oromo translation |
| 🤟 **Sign Language** | Real-time ASL recognition |

## Run locally
```bash
git clone https://huggingface.co/spaces/fenetii/adulis
cd adulis
pip install -r requirements.txt
# Download model from HF Hub or set MODEL_PATH to your local checkpoint
python app.py
# Open http://localhost:7860
```
