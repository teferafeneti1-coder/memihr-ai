"""
Generate frontend/tests/parity_case.json for the JS parity self-test.
Run from workspace root:
    source /home/feneti/ai-engineering-camp/myenv/bin/activate
    python scripts/make_parity_case.py
"""
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PKG  = ROOT / "asl_landmark_package"
OUT  = ROOT / "frontend" / "tests" / "parity_case.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

FEATURE_DIM = 163
POSE_IDX    = [0,11,12,13,14,15,16,17,18,19,20,21,22]

# ── reference feature extractor (verbatim from spec) ─────────────────────────
def make_features(hands_px, pose_px):
    hs  = sorted(hands_px, key=lambda h: h[0,0])[:2]
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    scales = []
    for i, h in enumerate(hs):
        sc = float(np.linalg.norm(h[9,:2] - h[0,:2])) + 1e-6
        vec[i*63:(i+1)*63] = ((h - h[0]) / sc).reshape(-1)
        scales.append(sc)
    if len(hs) == 2:
        vec[126:129] = (hs[1][0] - hs[0][0]) / max(scales)
    if pose_px is not None:
        nose, ls, rs = pose_px[0], pose_px[11], pose_px[12]
        sw = float(np.linalg.norm(ls - rs))
        if sw > 10:
            mid = (ls + rs) / 2
            vec[137:163] = ((pose_px[POSE_IDX] - mid) / sw).reshape(-1)
            for i, h in enumerate(hs):
                w = h[0,:2]
                vec[129+i*4:129+i*4+4] = [
                    (w[0]-nose[0])/sw, (w[1]-nose[1])/sw,
                    (w[0]-mid[0])/sw,  (w[1]-mid[1])/sw,
                ]
    return vec

# ── MLP forward pass ──────────────────────────────────────────────────────────
def mlp_forward(feat, mlp):
    x = (feat - np.array(mlp["mean"], np.float32)) / np.array(mlp["std"], np.float32)
    for i, layer in enumerate(mlp["layers"]):
        W = np.array(layer["W"], np.float32)   # (out, in)
        b = np.array(layer["b"], np.float32)
        x = x @ W.T + b
        if i < len(mlp["layers"]) - 1:
            x = np.maximum(0, x)               # ReLU
    # softmax
    x = x - x.max()
    ex = np.exp(x)
    return (ex / ex.sum()).tolist()

# ── random realistic hand (21 landmarks, pixel space) ─────────────────────────
rng = np.random.default_rng(42)

def rand_hand(cx=320, cy=300, scale=80):
    """Realistic-ish hand: 21 points clustered around (cx,cy)."""
    h = np.zeros((21,3), np.float32)
    # wrist at origin cluster
    h[0] = [cx, cy, 0]
    for j in range(1, 21):
        h[j] = h[0] + rng.uniform(-scale, scale, 3)
        h[j,2] *= 0.3   # z much smaller
    return h

def rand_pose(W=1280, H=720):
    """33 pose landmarks (x,y) in pixel space."""
    p = np.zeros((33,2), np.float32)
    # nose
    p[0] = [W/2, H*0.2]
    # left shoulder (11), right shoulder (12)
    p[11] = [W/2 - 100, H*0.4]
    p[12] = [W/2 + 100, H*0.4]
    for j in range(33):
        if j not in [0,11,12]:
            p[j] = p[11] + rng.uniform(-150, 150, 2)
    return p

# ── load MLP ──────────────────────────────────────────────────────────────────
mlp = json.loads((PKG / "asl_landmark_mlp.json").read_text())
assert mlp["feature_dim"] == FEATURE_DIM, "Unexpected feature_dim"

# ── build test cases ──────────────────────────────────────────────────────────
cases = []

# Case 0: 1 hand, with pose
h0 = rand_hand(300, 320)
p0 = rand_pose()
f0 = make_features([h0], p0)
cases.append({
    "name": "1_hand_with_pose",
    "hands_px": [h0.tolist()],
    "pose_px":  p0.tolist(),
    "features": f0.tolist(),
    "probs":    mlp_forward(f0, mlp),
})

# Case 1: 2 hands, with pose
h1a = rand_hand(200, 300)
h1b = rand_hand(500, 280)
p1  = rand_pose()
f1  = make_features([h1a, h1b], p1)
cases.append({
    "name": "2_hands_with_pose",
    "hands_px": [h1a.tolist(), h1b.tolist()],
    "pose_px":  p1.tolist(),
    "features": f1.tolist(),
    "probs":    mlp_forward(f1, mlp),
})

# Case 2: 2 hands, no pose
h2a = rand_hand(250, 310)
h2b = rand_hand(480, 295)
f2  = make_features([h2a, h2b], None)
cases.append({
    "name": "2_hands_no_pose",
    "hands_px": [h2a.tolist(), h2b.tolist()],
    "pose_px":  None,
    "features": f2.tolist(),
    "probs":    mlp_forward(f2, mlp),
})

# Case 3: no hands, pose only
f3 = make_features([], rand_pose())
p3 = rand_pose()
f3 = make_features([], p3)
cases.append({
    "name": "no_hands_pose_only",
    "hands_px": [],
    "pose_px":  p3.tolist(),
    "features": f3.tolist(),
    "probs":    mlp_forward(f3, mlp),
})

# Case 4: 1 hand, no pose
h4 = rand_hand(400, 350)
f4 = make_features([h4], None)
cases.append({
    "name": "1_hand_no_pose",
    "hands_px": [h4.tolist()],
    "pose_px":  None,
    "features": f4.tolist(),
    "probs":    mlp_forward(f4, mlp),
})

output = {
    "feature_dim": FEATURE_DIM,
    "mlp_classes": mlp["classes"],
    "cases": cases,
}
OUT.write_text(json.dumps(output, indent=2))
print(f"Wrote {len(cases)} cases to {OUT}")
