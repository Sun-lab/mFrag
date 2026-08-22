#!/usr/bin/env python3
"""Fit linear fusion of mFrag-Weight (S/L) + mFrag-Net (frag length) on training probs."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("/fh/working/sun_w/saishi_cui/mFrag_Project/mFrag")
os.environ.setdefault("MFRAG_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from mfrag import MFragNetFragLength
from mfrag.paths import LUNG_TRAIN_FRAG, LUNG_TRAIN_HREF, LUNG_TRAIN_LREF

OUT = Path(__file__).resolve().parent
PRED = ROOT / "data" / "predictions"
META = ROOT / "data" / "cohort" / "validation_testing_sample_list.tsv"
N_BOOT = 1000
RNG = np.random.default_rng(42)


def base_id(s: str) -> str:
    s = str(s)
    for suf in ("_lung_cancer", "_healthy"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s.split("_")[0]


def stage_grp(stage) -> str | None:
    if pd.isna(stage):
        return None
    s = str(stage).strip().upper()
    if s == "I":
        return "I"
    if s == "II":
        return "II"
    if s in {"III", "IV", "IIIA", "IIIB"}:
        return "III_IV"
    return None


def load_weight(split: str) -> pd.DataFrame:
    df = pd.read_csv(PRED / f"mfrag_weight_sl_ratio_{split}.csv")
    df["base"] = df["sample_id"].map(base_id)
    df = df.rename(columns={"predicted_proba_class1": "p_W", "y_true": "y"})
    return df[["sample_id", "base", "y", "p_W"]]


def ensure_net_train_preds() -> pd.DataFrame:
    out_csv = PRED / "mfrag_net_frag_length_training.csv"
    if out_csv.is_file():
        df = pd.read_csv(out_csv)
        print(f"Using existing {out_csv}")
        return df

    stems = sorted(p.stem for p in LUNG_TRAIN_FRAG.glob("*.csv"))
    print(f"Scoring Net on {len(stems)} training samples (CPU/GPU)...")
    net = MFragNetFragLength.from_pretrained(
        main_dir=LUNG_TRAIN_FRAG,
        healthy_ref_dir=LUNG_TRAIN_HREF,
        lung_ref_dir=LUNG_TRAIN_LREF,
    )
    rows = []
    for i, stem in enumerate(stems, 1):
        prob = net.predict_proba(stem)
        y = 1 if "lung_cancer" in stem else 0
        rows.append(
            {
                "sample_id": stem,
                "y_true": y,
                "predicted_proba_class1": prob,
            }
        )
        if i % 20 == 0 or i == len(stems):
            print(f"  {i}/{len(stems)}")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    return df


def load_net(split: str) -> pd.DataFrame:
    if split == "training":
        df = ensure_net_train_preds()
    else:
        df = pd.read_csv(PRED / "mfrag_net_frag_length_testing.csv")
    df["base"] = df["sample_id"].map(base_id)
    df = df.rename(columns={"predicted_proba_class1": "p_N", "y_true": "y"})
    return df[["sample_id", "base", "y", "p_N"]]


def merge_split(split: str) -> pd.DataFrame:
    w = load_weight(split)
    n = load_net(split)
    df = w.merge(n[["base", "p_N", "y"]], on="base", suffixes=("", "_n"))
    # prefer weight y; check consistency
    if not np.array_equal(df["y"].values, df["y_n"].values):
        raise RuntimeError(f"Label mismatch on {split}")
    df = df.drop(columns=["y_n"])
    return df


def metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return {"auc": np.nan, "prauc": np.nan}
    return {
        "auc": float(roc_auc_score(y, p)),
        "prauc": float(average_precision_score(y, p)),
    }


def bootstrap_metrics(y, p, n_boot=N_BOOT, seed=42):
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    aucs, prs = [], []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, pp))
        prs.append(average_precision_score(yy, pp))
    def summ(a):
        a = np.asarray(a, dtype=float)
        return {
            "median": float(np.median(a)),
            "q25": float(np.quantile(a, 0.25)),
            "q75": float(np.quantile(a, 0.75)),
            "n": int(len(a)),
        }
    return {"auc": summ(aucs), "prauc": summ(prs)}


def main() -> int:
    train = merge_split("training")
    test = merge_split("testing")
    print(f"train n={len(train)} cancer={int(train.y.sum())}")
    print(f"test  n={len(test)} cancer={int(test.y.sum())}")

    # Fit fusion on training
    Xtr = train[["p_W", "p_N"]].to_numpy()
    ytr = train["y"].to_numpy()
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    clf.fit(Xtr, ytr)
    beta0 = float(clf.intercept_[0])
    beta1, beta2 = map(float, clf.coef_[0])
    weights = {
        "model": "logistic_fusion_WeightSL_NetFragLength",
        "formula": "P(y=1|p_W,p_N)=sigmoid(beta0 + beta1*p_W + beta2*p_N)",
        "intercept_beta0": beta0,
        "beta_W": beta1,
        "beta_N": beta2,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "note": "Base-model train probs are in-sample (same protocol as prior Mathios SL+motif ensemble).",
    }
    (OUT / "fusion_weights.json").write_text(json.dumps(weights, indent=2))
    print("Fitted:", weights)

    def apply(df):
        out = df.copy()
        X = out[["p_W", "p_N"]].to_numpy()
        out["p_fuse"] = clf.predict_proba(X)[:, 1]
        out["p_mean"] = 0.5 * (out["p_W"] + out["p_N"])
        return out

    train = apply(train)
    test = apply(test)
    train.to_csv(OUT / "train_fused_predictions.csv", index=False)
    test.to_csv(OUT / "test_fused_predictions.csv", index=False)

    # Attach stage for test
    meta = pd.read_csv(META, sep="\t")
    meta["base"] = meta["sample_id"].map(base_id)
    meta["stage"] = meta["Stage"].map(stage_grp)
    test = test.merge(meta[["base", "stage"]].drop_duplicates("base"), on="base", how="left")

    rows = []
    for split_name, df in [("train", train), ("test", test)]:
        for name, col in [("WeightSL", "p_W"), ("NetFrag", "p_N"), ("Mean", "p_mean"), ("FuseLogit", "p_fuse")]:
            m = metrics(df["y"], df[col])
            rows.append({"split": split_name, "stage": "all", "model": name, **m, "n": len(df), "n_cancer": int(df.y.sum())})

    healthy = test["y"] == 0
    for st in ["I", "II", "III_IV"]:
        sub = test[healthy | ((test["y"] == 1) & (test["stage"] == st))].copy()
        for name, col in [("WeightSL", "p_W"), ("NetFrag", "p_N"), ("Mean", "p_mean"), ("FuseLogit", "p_fuse")]:
            m = metrics(sub["y"], sub[col])
            rows.append(
                {
                    "split": "test",
                    "stage": st,
                    "model": name,
                    **m,
                    "n": len(sub),
                    "n_cancer": int(sub.y.sum()),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "metrics_summary.csv", index=False)
    print(summary.to_string(index=False))

    # Bootstrap on test overall + stages for fuse vs singles
    boot_rows = []
    pools = [("all", test)]
    for st in ["I", "II", "III_IV"]:
        pools.append((st, test[healthy | ((test["y"] == 1) & (test["stage"] == st))].copy()))
    for st, df in pools:
        for name, col in [("WeightSL", "p_W"), ("NetFrag", "p_N"), ("Mean", "p_mean"), ("FuseLogit", "p_fuse")]:
            b = bootstrap_metrics(df["y"], df[col])
            boot_rows.append(
                {
                    "stage": st,
                    "model": name,
                    "n": len(df),
                    "n_cancer": int(df.y.sum()),
                    "auc_median": b["auc"]["median"],
                    "auc_q25": b["auc"]["q25"],
                    "auc_q75": b["auc"]["q75"],
                    "prauc_median": b["prauc"]["median"],
                    "prauc_q25": b["prauc"]["q25"],
                    "prauc_q75": b["prauc"]["q75"],
                    "n_boot": b["auc"]["n"],
                }
            )
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT / "bootstrap_summary.csv", index=False)
    print("\nBootstrap medians:")
    print(boot.to_string(index=False))
    print(f"\nOutputs in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
