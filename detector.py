import json, time, math
import numpy as np

# --- 調整するパラメータ ---
EAR_THRESHOLD      = 0.22   # この値以上で目が開いていると判定

PERCLOS_WINDOW     = 60.0   # PERCLOS を計算する時間ウィンドウ（秒）
PERCLOS_THRESHOLD  = 0.30   # この割合以上閉じていたら drowsy

MAR_HALF_THRESHOLD = 0.30   # この値以上 MAR_THRESHOLD 未満で口が半開き
MAR_THRESHOLD      = 0.55   # この値以上で口が開いている（欠伸検知）
YAWN_COOLDOWN      = 5.0    # 欠伸を連続カウントしない間隔（秒）

TILT_THRESHOLD     = 12.0   # 頭の左右傾き（度）がこれ以上で tilt

PITCH_DIP          = 0.07   # ベースラインからこの値以上うつむいたら dip
ROWING_DIP_COUNT   = 2      # この回数以上 dip→復帰 を繰り返したら rowing
ROWING_WINDOW      = 30.0   # rowing カウントウィンドウ（秒）
ROWING_RESET       = 10.0   # 最後の dip からこの秒数前を向き続けたら rowing リセット
BASELINE_FRAMES    = 60     # 起動時の基準姿勢サンプル数
# -------------------------

_EYE_L = [362, 385, 387, 263, 373, 380]
_EYE_R = [33,  160, 158, 133, 153, 144]

_closed_start      = None
_ear_log           = []
_last_yawn         = 0.0
_yawn_count        = 0
_dip_start         = None
_dip_events        = []
_last_dip_time     = None
_baseline_samples  = []
_pitch_base        = None

def _dist(a, b):
    return math.dist(a, b)

def _ear(pts, idx):
    p = [pts[i] for i in idx]
    denom = 2.0 * _dist(p[0], p[3])
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / denom if denom > 0 else 0.0

def _mar(pts):
    vert  = _dist(pts[13], pts[14])
    horiz = _dist(pts[78], pts[308])
    return vert / horiz if horiz > 0 else 0.0

def _pitch(pts):
    nose    = pts[1]
    eye_mid = [(pts[33][0]+pts[263][0])/2, (pts[33][1]+pts[263][1])/2]
    face_h  = pts[152][1] - pts[10][1]
    return (nose[1] - eye_mid[1]) / face_h if face_h > 0 else 0.5

def _tilt(pts):
    dx = pts[263][0] - pts[33][0]
    dy = pts[263][1] - pts[33][1]
    return math.degrees(math.atan2(dy, dx))

def process(lm_json):
    global _closed_start, _ear_log, _last_yawn, _yawn_count
    global _dip_start, _dip_events, _last_dip_time, _baseline_samples, _pitch_base

    now = time.time()
    pts = json.loads(lm_json)

    ear   = (_ear(pts, _EYE_L) + _ear(pts, _EYE_R)) / 2.0
    open_ = ear >= EAR_THRESHOLD

    if not open_:
        if _closed_start is None:
            _closed_start = now
        closed_sec = now - _closed_start
    else:
        _closed_start = None
        closed_sec    = 0.0

    _ear_log.append((now, not open_))
    _ear_log = [(t, c) for t, c in _ear_log if now - t < PERCLOS_WINDOW]
    perclos  = sum(c for _, c in _ear_log) / len(_ear_log) if _ear_log else 0.0

    mar       = _mar(pts)
    yawn      = mar >= MAR_THRESHOLD
    half_open = MAR_HALF_THRESHOLD <= mar < MAR_THRESHOLD
    if yawn and (now - _last_yawn) > YAWN_COOLDOWN:
        _yawn_count += 1
        _last_yawn   = now

    tilt      = _tilt(pts)
    head_tilt = abs(tilt) > TILT_THRESHOLD

    p = _pitch(pts)

    if _pitch_base is None:
        _baseline_samples.append(p)
        if len(_baseline_samples) >= BASELINE_FRAMES:
            _pitch_base = float(np.median(_baseline_samples))
        rowing = False
    else:
        dip = (p - _pitch_base) > PITCH_DIP

        if dip and _dip_start is None:
            _dip_start = now
        elif not dip and _dip_start is not None:
            _dip_events.append(now)
            _last_dip_time = now
            _dip_start     = None

        _dip_events = [t for t in _dip_events if now - t < ROWING_WINDOW]

        if _last_dip_time and (now - _last_dip_time) > ROWING_RESET:
            _dip_events.clear()
            _last_dip_time = None

        rowing = len(_dip_events) >= ROWING_DIP_COUNT

    return json.dumps({
        "ear":        round(float(ear), 3),
        "open":       bool(open_),
        "closed_sec": round(float(closed_sec), 1),
        "perclos":    round(float(perclos), 3),
        "yawn":       bool(yawn),
        "half_open":  bool(half_open),
        "yawn_count": int(_yawn_count),
        "head_tilt":  bool(head_tilt),
        "tilt_deg":   round(float(tilt), 1),
        "rowing":     bool(rowing),
        "pitch":      round(float(p), 3),
    })
