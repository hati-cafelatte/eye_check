import json
import numpy as np

# --- 調整するパラメータ ---
EAR_THRESHOLD = 0.22   # この値以上で目が開いていると判定
# -------------------------

_L = [362, 385, 387, 263, 373, 380]
_R = [33,  160, 158, 133, 153, 144]

def _ear(pts, idx):
    p = [pts[i] for i in idx]
    def d(a, b): return float(np.linalg.norm(np.array(a) - np.array(b)))
    denom = 2.0 * d(p[0], p[3])
    return (d(p[1], p[5]) + d(p[2], p[4])) / denom if denom > 0 else 0.0

def process(lm_json):
    pts = json.loads(lm_json)
    ear = (_ear(pts, _L) + _ear(pts, _R)) / 2.0
    return json.dumps({
        "ear":  round(float(ear), 3),
        "open": bool(ear >= EAR_THRESHOLD),
    })
