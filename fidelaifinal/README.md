# Fidel CNN — Model Tester

A local web app to test your fine-tuned Fidel CNN by drawing characters
with the mouse/touch, in two modes:

- **Free draw** — draw anything, see the top-5 predicted characters with confidence.
- **Guided test** — cycles through all 287 characters in random order; you draw
  each one, it tells you right/wrong, and tracks running accuracy plus a list
  of missed characters.

## Setup

1. **Export labels from your notebook.** `CHARS_FULL` (the class-index →
   character mapping) is specific to your dataset and isn't saved by the
   training notebook by default. Run the snippet in
   `export_labels_snippet.py` in your notebook (Kaggle) to produce
   `chars.json`, and download it alongside your model checkpoint.

2. **Place the files:**
   ```
   fidel_test_app/
     models/
       fidel_cnn_inference1.pt   ← your exported model
       chars.json                ← from step 1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run:**
   ```bash
   python3 app.py
   ```
   Then open http://localhost:5000 in your browser.

## Notes on accuracy

- The app preprocesses your canvas drawing the same way the notebook's
  validation set was preprocessed (grayscale → invert → resize to 25×25 →
  normalize mean/std 0.5, no augmentation at inference). If real-world
  drawing accuracy still feels noticeably below your notebook's reported
  val_acc, check the "Missed so far" list during a guided test — it often
  reveals patterns (e.g. specific character families that are visually
  similar, or stroke thickness the model wasn't trained on).
- If you get an error on startup about a class-count mismatch between
  `chars.json` and the model, it means the two files came from different
  training runs — re-export both together from the same notebook run.
