import json, time
import numpy as np

# --- 調整するパラメータ ---
EAR_THRESHOLD       = 0.22   # この値以上で目が開いていると判定

PITCH_DIP_THRESHOLD = 0.07   # ベースラインからこの値以上うつむいたら「dip」とみなす
ROWING_DIP_COUNT    = 4      # この回数以上 dip→復帰 を繰り返したら rowing
ROWING_WINDOW_SEC   = 30.0   # rowing 判定のカウントウィンドウ（秒）
BASELINE_FRAMES     = 60     # 起動直後の何フレームを基準姿勢として使うか
# -------------------------

_L = [362, 385, 387, 263, 373, 380]
_R = [33,  160, 158, 133, 153, 144]

_closed_start     = None
_dip_start        = None
_dip_events       = []
_pitch_baseline   = None
_baseline_samples = []

def _ear(pts, idx):
    p = [pts[i] for i in idx]
    def d(a, b): return float(np.linalg.norm(np.array(a) - np.array(b)))
    denom = 2.0 * d(p[0], p[3])
    return (d(p[1], p[5]) + d(p[2], p[4])) / denom if denom > 0 else 0.0

def _pitch(pts):
    nose     = pts[1]
    eye_mid  = [(pts[33][0] + pts[263][0]) / 2, (pts[33][1] + pts[263][1]) / 2]
    chin     = pts[152]
    forehead = pts[10]
    face_h   = chin[1] - forehead[1]
    if face_h <= 0:
        return 0.5
    return (nose[1] - eye_mid[1]) / face_h

def process(lm_json):
    global _closed_start, _dip_start, _dip_events, _pitch_baseline, _baseline_samples

    now = time.time()
    pts = json.loads(lm_json)

    ear   = (_ear(pts, _L) + _ear(pts, _R)) / 2.0
    open_ = ear >= EAR_THRESHOLD

    if not open_:
        if _closed_start is None:
            _closed_start = now
        closed_sec = now - _closed_start
    else:
        _closed_start = None
        closed_sec    = 0.0

    p = _pitch(pts)

    if _pitch_baseline is None:
        _baseline_samples.append(p)
        if len(_baseline_samples) >= BASELINE_FRAMES:
            _pitch_baseline = float(np.median(_baseline_samples))
        rowing = False
    else:
        dip = (p - _pitch_baseline) > PITCH_DIP_THRESHOLD

        if dip and _dip_start is None:
            _dip_start = now
        elif not dip and _dip_start is not None:
            _dip_events.append(now)
            _dip_start = None

        _dip_events = [t for t in _dip_events if now - t < ROWING_WINDOW_SEC]
        rowing = len(_dip_events) >= ROWING_DIP_COUNT

    return json.dumps({
        "ear":        round(float(ear), 3),
        "open":       bool(open_),
        "closed_sec": round(float(closed_sec), 1),
        "rowing":     bool(rowing),
        "pitch":      round(float(p), 3),
    })
