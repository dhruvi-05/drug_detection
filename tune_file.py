# tune_alpha.py
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score
import itertools
import argparse

# --- adjust to match your feature columns saved earlier
FEATURE_CSV = "features_labeled.csv"
MODEL_BUNDLE = "suspicion_model.pkl"

def precision_at_k(y_true_ids, ranked_ids, k):
    # y_true_ids: set of true suspicious ids (strings or ints)
    # ranked_ids: list of ids sorted by predicted score desc
    topk = ranked_ids[:k]
    hits = sum(1 for u in topk if u in y_true_ids)
    return hits / k

def load_features():
    df = pd.read_csv(FEATURE_CSV)
    # expects columns: user_id, label, plus feature cols
    return df

def compute_scores(bundle, X_df):
    # returns probabilities for the rows in X_df
    bundle_obj = joblib.load(bundle)
    if isinstance(bundle_obj, dict) and "model" in bundle_obj:
        model = bundle_obj["model"]
        feature_cols = bundle_obj["feature_cols"]
    else:
        model = bundle_obj
        feature_cols = list(getattr(model, "feature_names_in_", X_df.columns))
    X = X_df[feature_cols].fillna(0)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:,1]
    else:
        # fallback
        return model.predict(X).astype(float)

def run_grid(df, true_ids_set, feature_cols, alphas, rule_thresholds, k=5):
    results = []
    for alpha, rule_th in itertools.product(alphas, rule_thresholds):
        # create score = proba + alpha*(flag_count/max_flags)
        # assume df has columns user_id, flag_count, predicted_proba
        max_flags = max(1, int(df["flag_count"].max()))
        df["final_score"] = df["predicted_proba"] + alpha * (df["flag_count"] / float(max_flags))
        df["suspicious_priority"] = df["flag_count"] >= rule_th
        # rule-first ordering: suspicious_priority True first, then final_score
        df_sorted = df.sort_values(by=["suspicious_priority","final_score"], ascending=[False, False])
        ranked = list(df_sorted["user_id"].astype(str))
        p = precision_at_k(true_ids_set, ranked, k)
        results.append((alpha, rule_th, p))
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    df_all = load_features()
    # Use a hold-out validation split (simulate: use 20% as val)
    train_df, val_df = train_test_split(df_all, test_size=0.2, stratify=df_all["label"], random_state=42)

    # train a fresh model on train_df so predictions are made on val_df (better)
    feature_cols = [c for c in df_all.columns if c not in ("user_id", "label", "flag_count", "_flags_list")]
    X_train = train_df[feature_cols].fillna(0).astype(float)
    y_train = train_df["label"].astype(int)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y_train)

    # calibrate if you like
    calib = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
    calib.fit(X_train, y_train)  # quick calibration
    model_to_use = calib

    X_val_df = val_df.copy()
    # compute probabilities for val set
    val_features = X_val_df[feature_cols].fillna(0).astype(float)
    proba = model_to_use.predict_proba(val_features)[:,1]
    X_val_df["predicted_proba"] = proba
    # ensure flag_count exists
    if "flag_count" not in X_val_df.columns:
        X_val_df["flag_count"] = 0

    # set your true suspicious ids (if you have ground truth) — use label==1 here
    true_ids = set(val_df.loc[val_df["label"]==1, "user_id"].astype(str))

    alphas = np.linspace(0.0, 1.0, 11)  # 0..1 step 0.1
    rule_thresholds = [1,2,3,4,5,6]
    results = run_grid(X_val_df, true_ids, feature_cols, alphas, rule_thresholds, k=args.k)

    # print best top results by precision@k
    results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
    print("Top parameter combos by precision@%d:" % args.k)
    for a, r, p in results_sorted[:10]:
        print(f"alpha={a:.2f} rule_th={r} precision@{args.k}={p:.3f}")
