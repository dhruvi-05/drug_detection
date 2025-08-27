# predict.py
import argparse
import json
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib

# thresholds (same as train.py)
EARLY_WINDOW_DAYS = 3
EARLY_MSG_THRESHOLD = 50
SIM_SWAP_THRESHOLD = 2
ACCOUNT_REPORTS_THRESHOLD = 5
UNIQUE_CONTACTS_THRESHOLD = 100
IMG_GIF_RATIO_THRESHOLD = 5.0
PROFILE_UPDATES_THRESHOLD = 5
SHORT_MSG_LEN = 10
SHORT_MSG_RATIO_THRESHOLD = 0.6
LONG_MSG_LEN = 200
LONG_MSG_RATIO_THRESHOLD = 0.3
GROUP_REP_COUNT_HIGH = 4
GROUP_REP_RATIO_THRESHOLD = 0.5
NOCTURNAL_START, NOCTURNAL_END = 0, 5
NOCTURNAL_RATIO_THRESHOLD = 0.3
SAME_TIME_RATIO_THRESHOLD = 0.5

MODEL_PATH = "suspicion_model.pkl"

# ---------------- Helpers
def safe_int(x, default=0):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return int(x)
        s = str(x).strip()
        if s == "" or s.lower() in ("none", "null"):
            return default
        return int(float(s))
    except Exception:
        return default

def parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        fmts = ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
        for f in fmts:
            try:
                return datetime.strptime(s, f)
            except Exception:
                pass
    return None

def to_ist(dt):
    if dt is None:
        return None
    return dt + timedelta(hours=5, minutes=30)

def minute_of_day(dt):
    return dt.hour * 60 + dt.minute if dt else None

def ratio(n, d):
    return float(n) / max(1.0, float(d))

# ---------------- Feature extraction (aligned with train.py)
def compute_base_and_flags(user):
    meta = user.get("User_Header_Metadata", {}) or {}
    uid = meta.get("user_id") or meta.get("user_id_hash") or meta.get("userId") or meta.get("id") or None

    sim_swaps = safe_int(meta.get("sim_swaps") or meta.get("sim_swap_count") or 0)
    total_reports = safe_int(meta.get("total_report_count") or meta.get("reports") or 0)
    unique_contacts = safe_int(meta.get("unique_contacts") or 0)
    total_text = safe_int(meta.get("total_text_messages_sent") or meta.get("total_text") or 0)
    total_img = safe_int(meta.get("total_img_sent") or 0)
    total_gif = safe_int(meta.get("total_gif_and_sticker_sent") or 0)
    profile_updates = safe_int(meta.get("no_of_profile_img_updates") or 0)

    gchats = user.get("GroupChats", []) or []
    pchats = user.get("PersonalChats", []) or []

    msgs = []
    for m in gchats:
        ts = parse_dt(m.get("timestamp") or m.get("time") or "")
        dt = to_ist(ts) if ts else None
        msgs.append({
            "is_group": True,
            "length": safe_int(m.get("msg_length") or m.get("message_length") or 0),
            "dt": dt,
            "minute": minute_of_day(dt) if dt else None,
            "group_id": m.get("group_id"),
            "group_name": m.get("group_name"),
            "group_reported_count": safe_int(m.get("group_reported_count") or 0),
            "msg_type": (m.get("msg_type") or "").lower(),
            "sender_id": m.get("sender_id")
        })
    for m in pchats:
        ts = parse_dt(m.get("timestamp") or m.get("time") or "")
        dt = to_ist(ts) if ts else None
        msgs.append({
            "is_group": False,
            "length": safe_int(m.get("msg_length") or m.get("message_length") or 0),
            "dt": dt,
            "minute": minute_of_day(dt) if dt else None,
            "group_id": None,
            "group_name": None,
            "group_reported_count": 0,
            "msg_type": (m.get("msg_type") or "text").lower(),
            "sender_id": m.get("sender_id") or meta.get("user_id")
        })

    total_msgs = len(msgs)
    lengths = [m["length"] for m in msgs if m["length"] is not None]
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    median_len = float(np.median(lengths)) if lengths else 0.0
    short_ratio = ratio(sum(1 for L in lengths if L < SHORT_MSG_LEN), total_msgs)
    long_ratio = ratio(sum(1 for L in lengths if L > LONG_MSG_LEN), total_msgs)
    std_len = float(np.std(lengths)) if lengths else 0.0

    days = {m["dt"].date() for m in msgs if m["dt"]}
    active_days = len(days) if days else 1
    msg_rate = ratio(total_msgs, active_days)

    nocturnal_count = sum(1 for m in msgs if m["dt"] and NOCTURNAL_START <= m["dt"].hour < NOCTURNAL_END)
    nocturnal_ratio = ratio(nocturnal_count, total_msgs)

    minutes = [m["minute"] for m in msgs if m["minute"] is not None]
    mode_share = ratio(Counter(minutes).most_common(1)[0][1], len(minutes)) if minutes else 0.0

    group_msgs = [m for m in msgs if m["is_group"]]
    group_high_rep = sum(1 for m in group_msgs if m.get("group_reported_count", 0) >= GROUP_REP_COUNT_HIGH)
    group_high_rep_ratio = ratio(group_high_rep, len(group_msgs))

    names_by_gid = defaultdict(set)
    for m in group_msgs:
        gid = m.get("group_id")
        if gid:
            names_by_gid[gid].add(m.get("group_name"))
    group_name_changes = sum(1 for k in names_by_gid if len(names_by_gid[k]) > 1)

    creation_raw = meta.get("account_creation_date") or None
    creation_dt = parse_dt(creation_raw) if creation_raw else None
    if creation_dt:
        creation_dt = to_ist(creation_dt)
    first_msg_dt = min([m["dt"] for m in msgs if m["dt"]], default=None)
    start_anchor = creation_dt or first_msg_dt
    early_count = 0
    if start_anchor:
        window_end = start_anchor + timedelta(days=EARLY_WINDOW_DAYS)
        early_count = sum(1 for m in msgs if m["dt"] and start_anchor <= m["dt"] <= window_end)

    image_gif_ratio = ratio(total_img + total_gif, total_text)
    reports_per_day = ratio(total_reports, active_days)

    flags = {
        "flag_too_many_chats_early": int(early_count > EARLY_MSG_THRESHOLD),
        "flag_many_sim_swaps": int(sim_swaps > SIM_SWAP_THRESHOLD),
        "flag_many_reports": int(total_reports > ACCOUNT_REPORTS_THRESHOLD),
        "flag_many_unique_contacts": int(unique_contacts > UNIQUE_CONTACTS_THRESHOLD),
        "flag_image_gif_ratio_high": int(image_gif_ratio > IMG_GIF_RATIO_THRESHOLD),
        "flag_profile_updates_high": int(profile_updates > PROFILE_UPDATES_THRESHOLD),
        "flag_short_msgs": int(short_ratio > SHORT_MSG_RATIO_THRESHOLD),
        "flag_long_msgs": int(long_ratio > LONG_MSG_RATIO_THRESHOLD),
        "flag_group_report_high": int(group_high_rep_ratio > GROUP_REP_RATIO_THRESHOLD),
        "flag_group_name_changes": int(group_name_changes >= 3),
        "flag_nocturnal": int(nocturnal_ratio > NOCTURNAL_RATIO_THRESHOLD),
        "flag_same_time_daily": int(mode_share > SAME_TIME_RATIO_THRESHOLD),
    }

    numerics = {
        "user_id": uid,
        "sim_swaps": sim_swaps,
        "total_reports": total_reports,
        "unique_contacts": unique_contacts,
        "avg_msg_len": avg_len,
        "median_msg_len": median_len,
        "short_msg_ratio": short_ratio,
        "long_msg_ratio": long_ratio,
        "std_msg_len": std_len,
        "nocturnal_ratio": nocturnal_ratio,
        "active_days": active_days,
        "msg_count": total_msgs,
        "msg_rate": msg_rate,
        "image_gif_ratio": image_gif_ratio,
        "group_high_rep_ratio": group_high_rep_ratio,
        "group_name_change_count": group_name_changes,
        "early_msg_count": early_count,
        "reports_per_day": reports_per_day
    }

    combined = {}
    combined.update(numerics)
    combined.update(flags)
    combined["flag_count"] = int(sum(flags.values()))
    combined["_flags_list"] = [k for k, v in flags.items() if v]
    return combined

# ---------------- Load JSON
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- Map features robustly
def map_features_for_row(feature_cols, computed):
    row = {}
    for col in feature_cols:
        row[col] = computed.get(col, 0)
    return row

# ---------------- Prediction
def predict_and_rank(json_path, model_path, top_k=5, rule_threshold=6, alpha=0.6):
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict) and "model" in bundle and "feature_cols" in bundle:
        model = bundle["model"]
        feature_cols = bundle["feature_cols"]
    else:
        raise RuntimeError("Model saved without feature_cols. Please retrain with updated train.py")

    raw = load_json(json_path)
    if isinstance(raw, dict) and "groups" in raw:
        users = []
        for g in raw["groups"]:
            users.extend(g.get("user_metrics", []) or [])
    elif isinstance(raw, list):
        users = raw
    else:
        users = list(raw.values()) if isinstance(raw, dict) else []

    rows = []
    mapped = []
    for u in users:
        comp = compute_base_and_flags(u)
        rows.append(comp)
        mapped.append(map_features_for_row(feature_cols, comp))

    X = pd.DataFrame(mapped, columns=feature_cols).fillna(0).astype(float)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.predict(X).astype(float)

    out = pd.DataFrame({
        "user_id": [r.get("user_id") for r in rows],
        "flag_count": [r.get("flag_count", 0) for r in rows],
        "predicted_proba": proba
    })

    max_flags = max(1, int(out["flag_count"].max()))
    out["final_score"] = out["predicted_proba"] + alpha * (out["flag_count"] / float(max_flags))
    out["suspicious_priority"] = out["flag_count"] >= rule_threshold

    out = out.sort_values(by=["suspicious_priority", "final_score", "predicted_proba"],
                          ascending=[False, False, False]).reset_index(drop=True)

    top = out[["user_id", "flag_count", "predicted_proba"]].head(top_k)
    top["predicted_proba"] = top["predicted_proba"].round(6)

    print(f"\n🚨 Top {top_k} Most Suspicious Users 🚨")
    print(top.to_string(index=False))

    out.to_csv("predictions.csv", index=False)
    print("\nSaved predictions.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", "-f", required=True, help="testing JSON file")
    ap.add_argument("--model", "-m", default=MODEL_PATH)
    ap.add_argument("--topk", "-k", type=int, default=5)
    ap.add_argument("--rule-threshold", "-r", type=int, default=6, help="flag_count threshold")
    ap.add_argument("--alpha", type=float, default=0.6, help="weight of flags in hybrid score")
    args = ap.parse_args()
    predict_and_rank(args.file, args.model, top_k=args.topk, rule_threshold=args.rule_threshold, alpha=args.alpha)
