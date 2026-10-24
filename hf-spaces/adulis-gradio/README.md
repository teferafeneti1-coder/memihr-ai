---
title: Adulis — Amharic ↔ Afan Oromo Translator
emoji: 🌍
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
---

# 🌍 Adulis — Amharic ↔ Afan Oromo Translator

Named after the ancient Red Sea port that connected Aksum to the world — **Adulis** translates between **Amharic (አማርኛ)** and **Afan Oromo**, two of Ethiopia's most widely spoken languages.

## Features
- 🔄 **Live translation** — Amharic → Afan Oromo and Afan Oromo → Amharic
- 📚 **Lessons** — browse bilingual sentence pairs from a curated corpus
- 🎯 **Quiz** — multiple-choice questions to test your knowledge
- 🔤 **Word bank** — vocabulary by category (numbers, family, animals, food, colors…)

## Model
- **Base**: Meta's NLLB-200 (No Language Left Behind)
- **Fine-tuned** on Amharic ↔ Afan Oromo parallel corpus
- Hosted at: [fenetii/adulis-nllb](https://huggingface.co/fenetii/adulis-nllb)
- First load takes ~2–3 minutes (downloading model)

## Part of memihr ai
Built for the **INSA Ethiopia AI Summer Camp** — an AI learning platform for Ethiopian languages and sign language.
