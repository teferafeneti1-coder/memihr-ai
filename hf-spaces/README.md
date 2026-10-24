# memihr ai — Hugging Face Deployment

AI learning platform for Ethiopian languages and sign language.
Built for the **INSA Ethiopia AI Summer Camp**.

---

## Live Spaces

| App | Link | Description |
|-----|------|-------------|
| ✍️ Fidel CNN | [spaces/fenetii/fidel-cnn](https://huggingface.co/spaces/fenetii/fidel-cnn) | Draw Ethiopic Fidel characters — CNN recognizes them |
| 🌍 Adulis | [spaces/fenetii/adulis](https://huggingface.co/spaces/fenetii/adulis) | Amharic ↔ Afan Oromo translation with fine-tuned NLLB |
| 🤟 Sign Language | [spaces/fenetii/sign-language](https://huggingface.co/spaces/fenetii/sign-language) | Real-time ASL recognition via webcam + MediaPipe |

## Model Hub

| Model | Link | Size |
|-------|------|------|
| Adulis NLLB | [fenetii/adulis-nllb](https://huggingface.co/fenetii/adulis-nllb) | ~2.4 GB |

---

## How to Upload

### 1. Install huggingface_hub and login
```bash
source /home/feneti/ai-engineering/myenv/bin/activate
pip install huggingface_hub
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens
```

### 2. Run the push script
```bash
cd /home/feneti/egate/egate/hf-spaces
bash push_to_hf.sh
```

That's it. The script:
- Uploads the Adulis NLLB model (2.4 GB) to Model Hub
- Uploads all 3 Spaces with correct files
- Prints the live links when done

---

## Run the full suite locally (all 5 apps)

```bash
cd /home/feneti/egate/egate/egate/memihr-ai
./run.sh
# Open http://127.0.0.1:9000
```

---

## AI Models Used

| App | Model | Type |
|-----|-------|------|
| Fidel CNN | Custom CNN (from scratch) | 287-class Ethiopic character recognition |
| Adulis | Meta NLLB-200 (fine-tuned) | Amharic ↔ Afan Oromo translation |
| Sign Language | MediaPipe + custom MLP | ASL hand landmark classification |
| English-Amharic | Meta NLLB-200 (fine-tuned) | Amharic → English translation |
| English-Amharic Scratch | Custom Transformer (from scratch) | English → Amharic (BLEU 9.54) |

All models run **locally on CPU** — zero cloud API cost.
