import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib

# thresholds
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
    return dt + timedelta(hours=5, minutes=30) if dt else None

def minute_of_day(dt):
    return dt.hour * 60 + dt.minute if dt else None

def ratio(n, d):
    return float(n) / max(1.0, float(d))

def compute_base_and_flags(user):
    meta = user.get("User_Header_Metadata", {}) or {}
    uid = meta.get("user_id") or meta.get("userId") or meta.get("id")

    last_known_ip = meta.get("last_known_ip")
    last_online = meta.get("last_online")
    current_mobile_no = meta.get("current_mobile_no")
    last_device_logged = meta.get("last_device_logged")

    sim_swaps = safe_int(meta.get("sim_swaps") or meta.get("sim_swap_count") or 0)
    total_reports = safe_int(meta.get("total_report_count") or meta.get("reports") or 0)
    unique_contacts = safe_int(meta.get("unique_contacts") or 0)
    total_text = safe_int(meta.get("total_text_messages_sent") or 0)
    total_img = safe_int(meta.get("total_img_sent") or 0)
    total_gif = safe_int(meta.get("total_gif_and_sticker_sent") or 0)
    profile_updates = safe_int(meta.get("no_of_profile_img_updates") or 0)

    gchats = user.get("GroupChats", []) or []
    pchats = user.get("PersonalChats", []) or []

    msgs = []
    for m in gchats + pchats:
        ts = parse_dt(m.get("timestamp") or m.get("time") or "")
        dt = to_ist(ts) if ts else None
        msgs.append({
            "is_group": "group_id" in m,
            "length": safe_int(m.get("msg_length") or m.get("message_length") or 0),
            "dt": dt,
            "minute": minute_of_day(dt) if dt else None,
            "group_id": m.get("group_id"),
            "group_reported_count": safe_int(m.get("group_reported_count") or 0),
            "msg_type": (m.get("msg_type") or "text").lower(),
            "sender_id": m.get("sender_id") or meta.get("user_id")
        })

    total_msgs = len(msgs)
    lengths = [m["length"] for m in msgs if m["length"] is not None]
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    median_len = float(np.median(lengths)) if lengths else 0.0
    short_ratio = ratio(sum(1 for L in lengths if L < SHORT_MSG_LEN), total_msgs)
    long_ratio = ratio(sum(1 for L in lengths if L > LONG_MSG_LEN), total_msgs)

    days = {m["dt"].date() for m in msgs if m["dt"]}
    active_days = len(days)

    nocturnal_count = sum(1 for m in msgs if m["dt"] and NOCTURNAL_START <= m["dt"].hour < NOCTURNAL_END)
    nocturnal_ratio = ratio(nocturnal_count, total_msgs)

    minutes = [m["minute"] for m in msgs if m["minute"] is not None]
    mode_share = ratio(Counter(minutes).most_common(1)[0][1], len(minutes)) if minutes else 0.0

    group_msgs = [m for m in msgs if m["is_group"]]
    group_high_rep = sum(1 for m in group_msgs if m.get("group_reported_count", 0) >= GROUP_REP_COUNT_HIGH)
    group_high_rep_ratio = ratio(group_high_rep, len(group_msgs))

    uniq_ips = {m.get("sender_ip") for m in pchats if m.get("sender_ip")}
    location_hops = len(uniq_ips)

    image_gif_ratio = ratio(total_img + total_gif, total_text)

    flags = {
        "flag_many_sim_swaps": int(sim_swaps > SIM_SWAP_THRESHOLD),
        "flag_many_reports": int(total_reports > ACCOUNT_REPORTS_THRESHOLD),
        "flag_many_unique_contacts": int(unique_contacts > UNIQUE_CONTACTS_THRESHOLD),
        "flag_image_gif_ratio_high": int(image_gif_ratio > IMG_GIF_RATIO_THRESHOLD),
        "flag_profile_updates_high": int(profile_updates > PROFILE_UPDATES_THRESHOLD),
        "flag_short_msgs": int(short_ratio > SHORT_MSG_RATIO_THRESHOLD),
        "flag_long_msgs": int(long_ratio > LONG_MSG_RATIO_THRESHOLD),
        "flag_group_report_high": int(group_high_rep_ratio > GROUP_REP_RATIO_THRESHOLD),
        "flag_nocturnal": int(nocturnal_ratio > NOCTURNAL_RATIO_THRESHOLD),
        "flag_same_time_daily": int(mode_share > SAME_TIME_RATIO_THRESHOLD),
    }

    combined = dict(
        user_id=uid,
        sim_swaps=sim_swaps,
        total_reports=total_reports,
        unique_contacts=unique_contacts,
        avg_msg_len=avg_len,
        median_msg_len=median_len,
        short_msg_ratio=short_ratio,
        long_msg_ratio=long_ratio,
        nocturnal_ratio=nocturnal_ratio,
        active_days=active_days,
        msg_count=total_msgs,
        location_hops=location_hops,
        image_gif_ratio=image_gif_ratio,
        group_high_rep_ratio=group_high_rep_ratio,
        flag_count=sum(flags.values()),
        last_known_ip=last_known_ip,
        last_online=last_online,
        current_mobile_no=current_mobile_no,
        last_device_logged=last_device_logged,
    )
    combined.update(flags)
    return combined

def map_features_for_row(feature_cols, computed):
    return {col: computed.get(col, 0) for col in feature_cols}

def predict_and_rank(data, model_path, top_k=5, rule_threshold=6, alpha=0.6):
    bundle = joblib.load(model_path)
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    feature_cols = bundle.get("feature_cols") if isinstance(bundle, dict) else list(getattr(model, "feature_names_in_", []))

    users = []
    if isinstance(data, dict) and "groups" in data:
        for g in data["groups"]:
            users.extend(g.get("user_metrics", []))
    elif isinstance(data, list):
        users = data

    rows, mapped = [], []
    for u in users:
        comp = compute_base_and_flags(u)
        rows.append(comp)
        mapped.append(map_features_for_row(feature_cols, comp))

    if not mapped:
        return {"error": "No users found"}

    X = pd.DataFrame(mapped, columns=feature_cols).fillna(0).astype(float)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X).astype(float)

    out = pd.DataFrame({
        "user_id": [r.get("user_id") for r in rows],
        "flag_count": [r.get("flag_count", 0) for r in rows],
        "predicted_proba": proba,
        "last_known_ip": [r.get("last_known_ip") for r in rows],
        "last_online": [r.get("last_online") for r in rows],
        "current_mobile_no": [r.get("current_mobile_no") for r in rows],
        "last_device_logged": [r.get("last_device_logged") for r in rows],
    })

    max_flags = max(1, out["flag_count"].max())
    out["final_score"] = out["predicted_proba"] + alpha * (out["flag_count"] / max_flags)
    out["suspicious_priority"] = out["flag_count"] >= rule_threshold

    out = out.sort_values(by=["suspicious_priority", "final_score"], ascending=[False, False])
    top = out[["user_id", "flag_count", "predicted_proba", "last_known_ip", "last_online", "current_mobile_no", "last_device_logged"]].head(top_k)
    top["predicted_proba"] = top["predicted_proba"].round(6)

    return {"top_k": top_k, "results": top.to_dict(orient="records")}
if _name_ == "_main_":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", "-f", help="JSON file input")
    ap.add_argument("--model", "-m", default=MODEL_PATH)
    ap.add_argument("--topk", "-k", type=int, default=5)
    args = ap.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        raw_data = json.loads(sys.stdin.read())

    result = predict_and_rank(raw_data, args.model, top_k=args.topk)
    print(json.dumps(result, indent=2))
