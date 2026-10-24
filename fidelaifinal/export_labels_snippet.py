# ── Add this cell at the end of your fine-tuning notebook (after CHARS_FULL ──
# is built in Step 1, or right before/after the export cell in Step 7) ────────
# It saves the class-index -> character mapping the model actually expects,
# so the testing app can label predictions correctly.

import json

with open('/kaggle/working/fidel-cnn-finetuned/chars.json', 'w', encoding='utf-8') as f:
    json.dump(CHARS_FULL, f, ensure_ascii=False, indent=2)

print('Saved chars.json with', len(CHARS_FULL), 'characters')

# Then download BOTH files from Kaggle → Output:
#   fidel_cnn_inference1.pt
#   chars.json
# and place them in your local app's models/ folder:
#   fidel_test_app/models/fidel_cnn_inference1.pt
#   fidel_test_app/models/chars.json
