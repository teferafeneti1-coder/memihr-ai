---
title: Sign Language Translator
emoji: 🤟
colorFrom: blue
colorTo: cyan
sdk: docker
pinned: true
license: mit
---

# 🤟 Sign Language Translator

Real-time **ASL (American Sign Language)** word recognition from your webcam using MediaPipe hand landmarks + a custom MLP classifier.

## Features
- 📷 **Live webcam recognition** — shows predicted ASL sign in real time
- 🧠 **MLP classifier** — trained on hand landmark coordinates from MediaPipe
- 📖 **Word learning mode** — learn new ASL signs with guided practice
- 🎯 **Custom gesture training** — record and train your own signs in the browser

## Model
- **MediaPipe Hands** — extracts 21 hand landmarks (x, y, z) from webcam frames
- **Custom MLP** — classifies 63 landmark features into ASL word classes
- Model size: ~1.4 MB (bundled in repo)
- Runs entirely in the browser (MediaPipe) + lightweight backend (MLP inference)

## ⚠️ Camera Permission Required
This app needs access to your webcam. When prompted, click **Allow**.

## Part of memihr ai
This app is part of the **memihr ai** suite — an AI learning platform for Ethiopian languages and sign language, built for the INSA Ethiopia AI Summer Camp.

| App | Description |
|-----|-------------|
| ✍️ **Fidel CNN** | Ethiopic handwriting recognition |
| 🌍 **Adulis** | Amharic ↔ Afan Oromo translation |
| 🤟 **Sign Language** (this) | Real-time ASL recognition |

## Run locally
```bash
git clone https://huggingface.co/spaces/fenetii/sign-language
cd sign-language
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```
