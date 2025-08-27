# train.py
import os
import json
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
import joblib
from tqdm import tqdm
import math
import random

# ---------------- Config / thresholds (same as predict.py)
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

MODEL_OUT = "suspicion_model.pkl"
FEATURES_OUT = "features_labeled.csv"
RANDOM_STATE = 42

# ---------------- Helpers
def safe_int(x, default=0):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.integer, np.floating)):
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
        # try common formats
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

# ---------------- Feature extraction (same logic as predict)
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

    days = {m["dt"].date() for m in msgs if m["dt"]}
    active_days = len(days)

    nocturnal_count = sum(1 for m in msgs if m["dt"] and NOCTURNAL_START <= m["dt"].hour < NOCTURNAL_END)
    nocturnal_ratio = ratio(nocturnal_count, total_msgs)

    minutes = [m["minute"] for m in msgs if m["minute"] is not None]
    mode_share = 0.0
    if minutes:
        mode_share = ratio(Counter(minutes).most_common(1)[0][1], len(minutes))

    group_msgs = [m for m in msgs if m["is_group"]]
    group_high_rep = sum(1 for m in group_msgs if m.get("group_reported_count", 0) >= GROUP_REP_COUNT_HIGH)
    group_high_rep_ratio = ratio(group_high_rep, len(group_msgs))

    names_by_gid = defaultdict(set)
    for m in group_msgs:
        gid = m.get("group_id")
        if gid:
            names_by_gid[gid].add(m.get("group_name"))
    group_name_changes = sum(1 for k in names_by_gid if len([n for n in names_by_gid[k] if n]) > 1)

    creation_raw = meta.get("account_creation_date") or meta.get("creation_date") or None
    creation_dt = parse_dt(creation_raw) if creation_raw else None
    if creation_dt:
        creation_dt = to_ist(creation_dt)
    first_msg_dt = min([m["dt"] for m in msgs if m["dt"]], default=None)
    start_anchor = None
    if creation_dt and first_msg_dt:
        start_anchor = min(creation_dt, first_msg_dt)
    elif creation_dt:
        start_anchor = creation_dt
    elif first_msg_dt:
        start_anchor = first_msg_dt
    early_count = 0
    if start_anchor:
        window_end = start_anchor + timedelta(days=EARLY_WINDOW_DAYS)
        early_count = sum(1 for m in msgs if m["dt"] and start_anchor <= m["dt"] <= window_end)

    uniq_ips = set()
    for m in pchats:
        ip = m.get("sender_ip") or m.get("sender_ip_hash")
        if ip:
            uniq_ips.add(ip)
    location_hops = len(uniq_ips)

    image_gif_ratio = ratio(total_img + total_gif, total_text)

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
        "nocturnal_ratio": nocturnal_ratio,
        "active_days": active_days,
        "msg_count": total_msgs,
        "degree_centrality": 0.0,
        "betweenness": 0.0,
        "location_hops": location_hops,
        "image_gif_ratio": image_gif_ratio,
        "group_high_rep_ratio": group_high_rep_ratio,
        "group_name_change_count": group_name_changes,
        "early_msg_count": early_count
    }

    combined = {}
    combined.update(numerics)
    combined.update({k: int(v) for k, v in flags.items()})
    combined["flag_count"] = int(sum(flags.values()))
    combined["_flags_list"] = [k for k, v in flags.items() if v]
    return combined

# ---------------- Load JSON file(s)
def load_json_records(path):
    """
    Accept either:
     - a single JSON file that is a list of user objects
     - a directory containing many JSON files (each a single user or array)
    """
    if os.path.isdir(path):
        records = []
        for fname in os.listdir(path):
            if not fname.lower().endswith(".json"):
                continue
            fp = os.path.join(path, fname)
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
                if isinstance(d, list):
                    records.extend(d)
                elif isinstance(d, dict):
                    # check if wrapped under "users" or "groups"
                    if "users" in d:
                        records.extend(d["users"])
                    elif "groups" in d:
                        for g in d["groups"]:
                            records.extend(g.get("user_metrics", []) or [])
                    else:
                        records.append(d)
        return records
    else:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
            if isinstance(d, list):
                return d
            elif isinstance(d, dict):
                if "users" in d:
                    return d["users"]
                if "groups" in d:
                    recs = []
                    for g in d["groups"]:
                        recs.extend(g.get("user_metrics", []) or [])
                    return recs
                # fallback to values
                return list(d.values())
        return []

# ---------------- Main training pipeline
def main(suspicious_path, normal_path, out_model=MODEL_OUT):
    # load
    sus = load_json_records(suspicious_path)
    norm = load_json_records(normal_path)
    print(f"Loaded suspicious={len(sus)}, normal={len(norm)}")

    rows = []
    labels = []
    for u in tqdm(sus, desc="suspicious"):
        rows.append(compute_base_and_flags(u))
        labels.append(1)
    for u in tqdm(norm, desc="normal"):
        rows.append(compute_base_and_flags(u))
        labels.append(0)

    df = pd.DataFrame(rows)
    df["label"] = labels

    # Save features for inspection
    df.to_csv(FEATURES_OUT, index=False)
    print("Saved feature CSV:", FEATURES_OUT)

    # Features list: use numeric + flag columns (drop meta)
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    numeric_cols = ["sim_swaps", "total_reports", "unique_contacts", "avg_msg_len",
                    "median_msg_len", "short_msg_ratio", "long_msg_ratio",
                    "nocturnal_ratio", "active_days", "msg_count", "location_hops",
                    "image_gif_ratio", "group_high_rep_ratio", "group_name_change_count",
                    "early_msg_count"]
    feature_cols = [c for c in numeric_cols if c in df.columns] + flag_cols + ["flag_count"]

    X = df[feature_cols].fillna(0).astype(float)
    y = df["label"].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE)

    # Randomized search for RF hyperparams (fast)
    clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")
    param_dist = {
        "n_estimators": [100, 200, 400],
        "max_depth": [6, 10, 20, None],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.6, 0.8]
    }

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    rs = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=20, cv=cv, scoring="f1", n_jobs=-1, random_state=RANDOM_STATE, verbose=1)
    rs.fit(X_train, y_train)

    best = rs.best_estimator_
    print("Best params:", rs.best_params_)

    # Evaluate
    y_pred = best.predict(X_test)
    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred))

    # Save model + feature_cols
    joblib.dump({"model": best, "feature_cols": feature_cols}, out_model)
    print("Model bundle saved to", out_model)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suspicious", "-s", required=True, help="Path to suspicious JSON file or folder")
    ap.add_argument("--normal", "-n", required=True, help="Path to normal JSON file or folder")
    args = ap.parse_args()
    main(args.suspicious, args.normal)
