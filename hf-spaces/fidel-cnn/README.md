---
title: Fidel CNN — Ethiopic Script Recognizer
emoji: ✍️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: true
license: mit
---

# ✍️ Fidel CNN — Ethiopic Handwriting Recognizer

Draw any handwritten **Ethiopic (Fidel) character** on the canvas and a CNN recognizes it instantly from 287 classes.

## Features
- 🖊️ **Free draw mode** — draw any Fidel character, see top-5 predictions with confidence scores
- 🔄 **Guided test mode** — cycles through all 287 characters one by one
- 🧠 **Learn from mistakes** — flag wrong predictions, collect corrections, fine-tune the model in-browser
- ⚡ CPU-only, runs on any machine

## Model
- Custom CNN trained from scratch on handwritten Ethiopic characters
- Architecture: 3× Conv blocks (32→64→128 filters) + classifier head
- 287 output classes covering the full Ethiopic syllabary
- Model size: ~7.9 MB
- Fine-tunable at runtime from user corrections (last conv block + classifier only)

## Part of memihr ai
This app is part of the **memihr ai** suite — an AI learning platform for Ethiopian languages and sign language, built for the INSA Ethiopia AI Summer Camp.

| App | Description |
|-----|-------------|
| ✍️ **Fidel CNN** (this) | Ethiopic handwriting recognition |
| 🌍 **Adulis** | Amharic ↔ Afan Oromo translation |
| 🤟 **Sign Language** | Real-time ASL recognition |

## Run locally
```bash
git clone https://huggingface.co/spaces/fenetii/fidel-cnn
cd fidel-cnn
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```
