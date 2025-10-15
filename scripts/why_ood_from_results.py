#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WHY-OOD analysis from existing OOD CSV
--------------------------------------
- DOES NOT re-judge OOD. Reads OOD labels from test_ood_knn.csv
- Rebuilds TRAIN-only preprocessing (optional prune + StandardScaler) to form
  a consistent reference for "why" metrics (Z-score / local prototypes / Mahalanobis).
- Saves per-sample top-K contributing features and aggregated feature stats.
- Plots simple bar charts for top-N features.

Usage (examples):
  python scripts/why_ood_from_results.py \
    --dataset jarvis22 --target e_form \
    --group_label elements --group_value F \
    --root_path /home/mao/Desktop/cm_repro_for_colab

  # No correlation pruning; show more features in bar chart
  python scripts/why_ood_from_results.py \
    --dataset jarvis22 --target e_form \
    --group_label elements --group_value F \
    --root_path /home/mao/Desktop/cm_repro_for_colab \
    --no_prune --topn_features_plot 40

  # Enable Mahalanobis (optional, might be slower on many features)
  python scripts/why_ood_from_results.py \
    --dataset jarvis22 --target e_form \
    --group_label elements --group_value F \
    --root_path /home/mao/Desktop/cm_repro_for_colab \
    --use_maha
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EmpiricalCovariance

# ---- your project helpers
from extra_funcs import load_data, get_split, iterative_corr_prune


# -------------------------------
# Utilities
# -------------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def save_barh_count(df_stats: pd.DataFrame, out_png: str, topN: int, title: str,
                    count_col="count", figsize_scale=0.28):
    if df_stats.empty:
        return
    topN = max(1, min(topN, len(df_stats)))
    head = df_stats.head(topN)
    fig, ax = plt.subplots(figsize=(6, max(2, figsize_scale * topN)))
    ax.barh(head.index[::-1], head[count_col][::-1])
    ax.set_xlabel("Appearance count among OOD top-K")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


# -------------------------------
# Analyses
# -------------------------------
def compute_train_reference(X_train: pd.DataFrame,
                            do_prune: bool,
                            prune_threshold: float,
                            prune_method: str) -> (pd.DataFrame, list):
    """Return (X_train_kept, kept_cols)."""
    if do_prune:
        print(f"[Prune] Iterative corr prune on TRAIN ONLY: "
              f"threshold={prune_threshold}, method={prune_method}")
        kept_cols, dropped_cols = iterative_corr_prune(
            X_train, None,  # y is not needed for pure unsupervised prune; pass None
            threshold=prune_threshold,
            method=prune_method,
            min_var=0.0,
            verbose=True,
        )
        print(f"Done. kept={len(kept_cols)}, dropped={len(dropped_cols)} "
              f"(threshold={prune_threshold}, method={prune_method})")
        return X_train.loc[:, kept_cols].copy(), kept_cols
    else:
        kept_cols = list(X_train.columns)
        print(f"[Prune] SKIPPED. kept={len(kept_cols)} (all features)")
        return X_train.copy(), kept_cols


def zscore_topk_for_ood(X_train_scaled: pd.DataFrame,
                        X_scaled: pd.DataFrame,
                        ood_ids: list,
                        topk: int,
                        out_dir: str):
    """
    Global Z-score against TRAIN mean/std.
    Saves:
      - ood_zscore_topk.csv (long table: id, feature, abs_z)
      - ood_zscore_feature_stats.csv (& plot)
    """
    print("[WHY-OOD-1] Global Z-score against TRAIN mean/std ...")
    mu = X_train_scaled.mean(axis=0)
    sd = X_train_scaled.std(axis=0).replace(0.0, np.nan)  # avoid div by 0
    X_ood = X_scaled.loc[ood_ids, X_train_scaled.columns]
    Z = (X_ood - mu) / sd
    Z = Z.fillna(0.0)
    absZ = np.abs(Z).to_numpy()  # (n_ood, n_feat)

    K = max(1, min(topk, absZ.shape[1]))
    row_arange = np.arange(absZ.shape[0])[:, None]
    topk_idx = np.argpartition(-absZ, kth=K-1, axis=1)[:, :K]
    order_in_topk = np.argsort(-absZ[row_arange, topk_idx])
    topk_sorted_idx = topk_idx[row_arange, order_in_topk]

    feat_names = X_train_scaled.columns.to_numpy()
    rows = []
    ood_ids_np = np.array(ood_ids, dtype=str)
    for i in range(len(ood_ids)):
        for j in range(K):
            rows.append((ood_ids_np[i], feat_names[topk_sorted_idx[i, j]],
                         float(absZ[i, topk_sorted_idx[i, j]])))
    df_topk = pd.DataFrame(rows, columns=["id", "feature", "abs_z"])
    fp = os.path.join(out_dir, "ood_zscore_topk.csv")
    df_topk.to_csv(fp, index=False)
    print(f"[Save] Global Z-score top-K -> {fp}")

    # aggregate
    stats = (df_topk.groupby("feature")["abs_z"]
             .agg(["count", "mean", "median", "max"])
             .sort_values(["count", "mean"], ascending=[False, False]))
    fp2 = os.path.join(out_dir, "ood_zscore_feature_stats.csv")
    stats.to_csv(fp2)
    print(f"[Save] Feature stats        -> {fp2}")

    # plot
    save_barh_count(stats,
                    out_png=os.path.join(out_dir, "features_frequency_zscore.png"),
                    topN=40,
                    title=f"Global Z-score contributions (top-K={K})")


def proto_diff_topk_for_ood(X_train_scaled: pd.DataFrame,
                            X_scaled: pd.DataFrame,
                            ood_ids: list,
                            topk_feat: int,
                            nn_for_proto: int,
                            out_dir: str,
                            topn_features_plot: int):
    """
    Local prototype (kNN on TRAIN) contributions.
    Saves:
      - ood_nn_contrib_topk.csv (id, feature, abs_proto_diff)
      - ood_nn_neighbors.csv
      - ood_nn_contrib_feature_stats.csv (& plot)
    """
    print(f"[WHY-OOD-2] Local prototype diff (k={nn_for_proto}) ...")

    if len(ood_ids) == 0:
        print("[WHY-OOD-2] No OOD samples in TEST; skip.")
        return

    # Align columns
    X_test_ood_scaled = X_scaled.loc[ood_ids, X_train_scaled.columns]
    X_train_vals = X_train_scaled.to_numpy()                 # (n_train, n_feat)
    X_test_ood_vals = X_test_ood_scaled.to_numpy()          # (n_ood, n_feat)

    # kNN on TRAIN
    nbrs = NearestNeighbors(n_neighbors=nn_for_proto, n_jobs=-1).fit(X_train_vals)
    nn_dist, nn_idx = nbrs.kneighbors(X_test_ood_vals, return_distance=True)  # idx: (n_ood, k)

    # training index labels for neighbors
    train_index_np = X_train_scaled.index.to_numpy()         # (n_train,)
    neighbor_ids = train_index_np[nn_idx]                    # (n_ood, k)

    # neighbor vectors: (n_ood, k, n_feat)
    neighbor_vals = np.take(X_train_vals, nn_idx, axis=0)
    # prototype: (n_ood, n_feat)
    proto_vals = neighbor_vals.mean(axis=1)
    # delta: x - proto
    delta = X_test_ood_vals - proto_vals                     # (n_ood, n_feat)
    abs_delta = np.abs(delta)

    # per-ood top-K features
    K = max(1, min(topk_feat, abs_delta.shape[1]))
    row_arange = np.arange(abs_delta.shape[0])[:, None]
    topk_idx = np.argpartition(-abs_delta, kth=K-1, axis=1)[:, :K]
    order_in_topk = np.argsort(-abs_delta[row_arange, topk_idx])
    topk_sorted_idx = topk_idx[row_arange, order_in_topk]

    feat_names_np = X_train_scaled.columns.to_numpy()
    rows = []
    for i, oid in enumerate(ood_ids):
        for j in range(K):
            rows.append((oid, feat_names_np[topk_sorted_idx[i, j]],
                         float(abs_delta[i, topk_sorted_idx[i, j]])))
    df_proto_topk = pd.DataFrame(rows, columns=["id", "feature", "abs_proto_diff"])
    fp = os.path.join(out_dir, "ood_nn_contrib_topk.csv")
    df_proto_topk.to_csv(fp, index=False)
    print(f"[Save] NN-prototype top-K   -> {fp}")

    # neighbor id dump
    neigh_cols = [f"nn{t+1}" for t in range(neighbor_ids.shape[1])]
    neigh_df = pd.DataFrame(neighbor_ids, index=ood_ids, columns=neigh_cols)
    fp2 = os.path.join(out_dir, "ood_nn_neighbors.csv")
    neigh_df.to_csv(fp2)
    print(f"[Save] NN neighbor ids      -> {fp2}")

    # aggregate stats
    stats = (df_proto_topk.groupby("feature")["abs_proto_diff"]
             .agg(["count", "mean", "median", "max"])
             .sort_values(["count", "mean"], ascending=[False, False]))
    fp3 = os.path.join(out_dir, "ood_nn_contrib_feature_stats.csv")
    stats.to_csv(fp3)
    print(f"[Save] Feature stats        -> {fp3}")

    # plot
    save_barh_count(stats,
                    out_png=os.path.join(out_dir, f"features_contrib_nn_top{topn_features_plot}.png"),
                    topN=topn_features_plot,
                    title=f"Local-prototype contributions (k={nn_for_proto}, top-K={K})")


def mahalanobis_for_ood(X_train_scaled: pd.DataFrame,
                        X_scaled: pd.DataFrame,
                        ood_ids: list,
                        topk_feat: int,
                        out_dir: str,
                        topn_features_plot: int):
    """
    Optional: Mahalanobis contributions via whitening.
    Saves:
      - ood_maha_topk.csv
      - ood_maha_feature_stats.csv (& plot)
    """
    print("[WHY-OOD-3] Mahalanobis (whitened Z) ...")
    if len(ood_ids) == 0:
        print("[WHY-OOD-3] No OOD samples in TEST; skip.")
        return

    X_train = X_train_scaled.to_numpy()
    X_ood = X_scaled.loc[ood_ids, X_train_scaled.columns].to_numpy()

    # Fit covariance
    cov = EmpiricalCovariance().fit(X_train)
    # whitened vectors: W = (X - mu) * Sigma^{-1/2}
    mu = X_train.mean(axis=0)
    # Use precision_cholesky (L) so that L.T @ L = precision
    L = cov.precision_cholesky_
    if L is None:
        # Fallback to precision_ if cholesky not present
        L = np.linalg.cholesky(cov.precision_)

    Xc = (X_ood - mu)  # center
    W = Xc @ L.T       # whitened
    absW = np.abs(W)

    K = max(1, min(topk_feat, absW.shape[1]))
    row_arange = np.arange(absW.shape[0])[:, None]
    topk_idx = np.argpartition(-absW, kth=K-1, axis=1)[:, :K]
    order_in_topk = np.argsort(-absW[row_arange, topk_idx])
    topk_sorted_idx = topk_idx[row_arange, order_in_topk]

    feat_names = X_train_scaled.columns.to_numpy()
    rows = []
    for i, oid in enumerate(ood_ids):
        for j in range(K):
            rows.append((oid, feat_names[topk_sorted_idx[i, j]],
                         float(absW[i, topk_sorted_idx[i, j]])))
    df_topk = pd.DataFrame(rows, columns=["id", "feature", "abs_maha"])
    fp = os.path.join(out_dir, "ood_mahalanobis_topk.csv")
    df_topk.to_csv(fp, index=False)
    print(f"[Save] Mahalanobis top-K    -> {fp}")

    stats = (df_topk.groupby("feature")["abs_maha"]
             .agg(["count", "mean", "median", "max"])
             .sort_values(["count", "mean"], ascending=[False, False]))
    fp2 = os.path.join(out_dir, "ood_mahalanobis_feature_stats.csv")
    stats.to_csv(fp2)
    print(f"[Save] Feature stats        -> {fp2}")

    save_barh_count(stats,
                    out_png=os.path.join(out_dir, f"features_contrib_maha_top{topn_features_plot}.png"),
                    topN=topn_features_plot,
                    title=f"Mahalanobis contributions (top-K={K})")


# -------------------------------
# Main
# -------------------------------
def run_analysis(dataset: str,
                 target: str,
                 group_label: str,
                 group_value: str,
                 root_path: str,
                 ood_csv: str | None,
                 prune: bool,
                 prune_threshold: float,
                 prune_method: str,
                 analysis_topk: int,
                 nn_for_proto: int,
                 topn_features_plot: int,
                 use_maha: bool):
    print(f"当前工作目录: {os.getcwd()}")
    # Load
    pkl_try = os.path.join(root_path, f"data/{dataset}/dat_featurized_matminer.pkl")
    print(f"尝试加载文件: {pkl_try}")
    df, X_all, y_all = load_data(dataset, target, root_path)
    print(f"加载数据完成: {len(df)} 条记录，{X_all.shape[1]} 个特征")

    # Split
    index_train, index_test = get_split(df, group_label, group_value)
    print(f"[Split] Train={len(index_train)}  Test={len(index_test)}  "
          f"(cond: {group_label}='{group_value}')")

    # Read OOD from CSV
    if ood_csv is None:
        ood_csv = os.path.join(root_path, f"data/{dataset}/umap_trainfit_ood/test_ood_knn.csv")
    ood_df = pd.read_csv(ood_csv, index_col=0)
    ood_df.index = ood_df.index.astype(str)
    # Only test subset
    ood_in_test = ood_df.reindex(index_test)["is_OOD"].fillna(False).astype(bool)
    ood_count = int(ood_in_test.sum())
    print(f"[Input-OOD] OOD in TEST (from CSV): {ood_count}/{len(index_test)} "
          f"({ood_count/len(index_test)*100:.2f}%)")

    # Train-only reference (optional prune)
    X_train = X_all.loc[index_train].copy()
    X_train_ref, kept_cols = compute_train_reference(
        X_train, do_prune=prune, prune_threshold=prune_threshold, prune_method=prune_method
    )
    print(f"[Features] Using: {len(kept_cols)} pruned features" if prune
          else f"[Features] Using: all {len(kept_cols)} features")

    # Scale: fit on TRAIN, apply to ALL
    scaler = StandardScaler().fit(X_train_ref)
    X_scaled = pd.DataFrame(scaler.transform(X_all.loc[:, kept_cols]),
                            columns=kept_cols, index=X_all.index)
    X_train_scaled = X_scaled.loc[index_train]
    print("[Scale] Standardized (fit on TRAIN, applied to ALL)")

    # Output dir
    out_dir = ensure_dir(os.path.join(
        root_path, f"data/{dataset}/umap_trainfit_ood/why_ood_from_results"
    ))
    # Save OOD list (for convenience)
    ood_list_path = os.path.join(out_dir, "ood_list.csv")
    pd.DataFrame({
        "id": index_test,
        "is_OOD": ood_in_test.astype(int)
    }).to_csv(ood_list_path, index=False)
    print(f"[Save] OOD list -> {ood_list_path}")

    # IDs of OOD in TEST
    ood_ids = ood_in_test[ood_in_test].index.tolist()

    # WHY-1: global zscore
    zscore_topk_for_ood(
        X_train_scaled=X_train_scaled,
        X_scaled=X_scaled,
        ood_ids=ood_ids,
        topk=analysis_topk,
        out_dir=out_dir,
    )

    # WHY-2: local prototype diff (fixed & vectorized)
    proto_diff_topk_for_ood(
        X_train_scaled=X_train_scaled,
        X_scaled=X_scaled,
        ood_ids=ood_ids,
        topk_feat=analysis_topk,
        nn_for_proto=nn_for_proto,
        out_dir=out_dir,
        topn_features_plot=topn_features_plot,
    )

    # WHY-3: optional mahalanobis
    if use_maha:
        mahalanobis_for_ood(
            X_train_scaled=X_train_scaled,
            X_scaled=X_scaled,
            ood_ids=ood_ids,
            topk_feat=analysis_topk,
            out_dir=out_dir,
            topn_features_plot=topn_features_plot,
        )

    print("[Done] WHY-OOD analysis completed.")


def parse_args():
    ap = argparse.ArgumentParser(description="WHY-OOD analysis from existing OOD CSV")
    ap.add_argument("--dataset", type=str, default="jarvis22")
    ap.add_argument("--target", type=str, default="e_form")
    ap.add_argument("--group_label", type=str, default="elements")
    ap.add_argument("--group_value", type=str, default="F")
    ap.add_argument("--root_path", type=str, default="/home/mao/Desktop/cm_repro_for_colab")

    ap.add_argument("--ood_csv", type=str, default=None, help="Path to test_ood_knn.csv")

    ap.add_argument("--no_prune", action="store_true", help="Disable corr-prune")
    ap.add_argument("--prune_threshold", type=float, default=0.8)
    ap.add_argument("--prune_method", type=str, default="spearman",
                    choices=["spearman", "pearson"])

    ap.add_argument("--analysis_topk", type=int, default=10, help="Top-K features per sample")
    ap.add_argument("--nn_for_proto", type=int, default=10, help="k for local prototype")
    ap.add_argument("--topn_features_plot", type=int, default=30)
    ap.add_argument("--use_maha", action="store_true", help="Enable Mahalanobis analysis")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        dataset=args.dataset,
        target=args.target,
        group_label=args.group_label,
        group_value=args.group_value,
        root_path=args.root_path,
        ood_csv=args.ood_csv,
        prune=(not args.no_prune),
        prune_threshold=args.prune_threshold,
        prune_method=args.prune_method,
        analysis_topk=args.analysis_topk,
        nn_for_proto=args.nn_for_proto,
        topn_features_plot=args.topn_features_plot,
        use_maha=args.use_maha,
    )
