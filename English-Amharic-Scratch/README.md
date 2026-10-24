# English → Amharic — local Flask deployment

## 1. Folder layout

Put these files in `/home/feneti/English-Amharic-Scratch` like this:

```
English-Amharic-Scratch/
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
└── model/
    ├── model.pt
    ├── en_spm.model
    ├── am_spm.model
    └── config.json
```

Your uploaded files were named `model.pt`, `en_spm.model`, `am_spm.model`,
and `config(1).json` — move them into `model/` and rename
`config(1).json` to `config.json`.

From your terminal (adjust source paths as needed):

```bash
cd /home/feneti/English-Amharic-Scratch
mkdir -p model templates
mv ~/Downloads/model.pt ~/Downloads/en_spm.model ~/Downloads/am_spm.model model/
mv ~/Downloads/config\(1\).json model/config.json
```

## 2. Install dependencies

You said your virtualenv is already activated in that folder, so just:

```bash
pip install -r requirements.txt
```

This installs Flask, PyTorch (CPU build by default — fine for inference),
and SentencePiece.

## 3. Run the server

```bash
python app.py
```

You should see something like:

```
Loaded model on cpu. EN vocab=8000, AM vocab=8000, test_bleu=...
 * Running on http://127.0.0.1:5000
```

Open **http://127.0.0.1:5000** in your browser, type an English sentence,
click Translate.

## Notes

- `app.py` rebuilds the exact `Seq2SeqTransformer` architecture from the
  training notebook and loads `model.pt`'s `state_dict` into it — the
  class definitions must stay in sync with what produced the checkpoint,
  which they already do here.
- Decoding is greedy (argmax at each step), same as the notebook's
  `translate()` function — good enough for a demo, but beam search would
  give better quality if you want to extend it later.
- If `config.json` doesn't have `d_model`/`max_len` keys for some reason,
  the app falls back to the notebook's defaults (256 / 128).
- `GET /health` returns a small JSON status check if you want to confirm
  the server + model loaded correctly (e.g. for a supervisor/systemd unit).
- For anything beyond local testing (real users, concurrent requests),
  swap `app.run(debug=True)` for a production WSGI server like `gunicorn`
  or `waitress`, and turn `debug` off.
