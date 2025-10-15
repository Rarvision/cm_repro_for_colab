#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster OOD-driving features from existing result CSVs (CSV-only outputs, no plots)

Inputs (default paths can be changed by CLI args below):
- <root>/data/<dataset>/umap_trainfit_ood/why_ood_from_results/ood_list.csv
    columns: id, is_OOD (1/0), ... (other columns ignored)
- <root>/data/<dataset>/umap_trainfit_ood/why_ood_from_results/ood_zscore_topk.csv
    columns: id, feature, z (or abs_z), rank (optional)
- <root>/data/<dataset>/umap_trainfit_ood/why_ood_from_results/ood_nn_contrib_topk.csv
    columns: id, feature, abs_proto_diff, rank (optional)

Outputs (all CSV, no figures):
- clustered/clustered_features_zscore.csv
    columns: feature, cluster_id, freq, mean_score, median_score
- clustered/clusters_zscore_summary.csv
    columns: cluster_id, n_features, total_freq, mean_of_means, top_features

- clustered/clustered_features_nncontrib.csv
    columns: feature, cluster_id, freq, mean_score, median_score
- clustered/clusters_nncontrib_summary.csv
    columns: cluster_id, n_features, total_freq, mean_of_means, top_features

Notes
- We do hierarchical clustering on features using a correlation-distance
  computed from the feature-by-sample score matrix (only OOD samples).
- No figures are generated; everything is in CSV.
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

# --------------------------
# Utilities
# --------------------------
def ensure_exists(p):
    os.makedirs(p, exist_ok=True)

def read_with_fallback(path, required_cols=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
    return df

def detect_value_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of the candidate value columns exist: {candidates}")

def build_matrix_from_topk(df_topk, ood_ids, value_col):
    """
    Build feature x sample (OOD) matrix:
    - Rows: features
    - Cols: OOD sample ids
    - Values: score (z or abs_proto_diff), NaN if feature not selected for that sample
    Also return a binary matrix of 0/1 indicating presence.
    """
    df_topk = df_topk[df_topk["id"].isin(ood_ids)].copy()

    feats = sorted(df_topk["feature"].unique().tolist())
    samps = sorted(df_topk["id"].unique().tolist())

    f_idx = {f: i for i, f in enumerate(feats)}
    s_idx = {s: j for j, s in enumerate(samps)}

    M = np.full((len(feats), len(samps)), np.nan, dtype=float)
    Mb = np.zeros((len(feats), len(samps)), dtype=int)

    # If multiple (id, feature) rows exist, keep the max value
    df_topk = df_topk.sort_values(value_col, ascending=False).drop_duplicates(subset=["id", "feature"])

    for r in df_topk.itertuples(index=False):
        i = f_idx[getattr(r, "feature")]
        j = s_idx[getattr(r, "id")]
        val = float(getattr(r, value_col))
        M[i, j] = val
        Mb[i, j] = 1

    M_df = pd.DataFrame(M, index=feats, columns=samps)
    Mb_df = pd.DataFrame(Mb, index=feats, columns=samps)
    return feats, samps, M_df, Mb_df

def corr_distance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Correlation distance between rows (features). X: (n_features, n_samples)
    Distance d(i,j) = 1 - corr(row_i, row_j), with Nan-safety.
    If a row has zero variance (all NaN or constant), treat correlation with others as 0.
    """
    # Replace NaN with row-wise mean (fallback 0.0 if all-NaN)
    X_filled = X.copy()
    for i in range(X_filled.shape[0]):
        row = X_filled[i, :]
        mask = np.isnan(row)
        if mask.all():
            X_filled[i, :] = 0.0
        else:
            m = np.nanmean(row)
            row[mask] = m
            X_filled[i, :] = row

    # Normalize rows (z-norm) to mitigate scale
    row_mean = X_filled.mean(axis=1, keepdims=True)
    row_std = X_filled.std(axis=1, keepdims=True)
    row_std[row_std == 0] = 1.0
    Xn = (X_filled - row_mean) / row_std

    # Use correlation on rows -> distance = 1 - corr
    # pdist expects observations in rows; metric='correlation' returns 1 - corr
    D = pdist(Xn, metric='correlation')
    # NaNs can appear if remaining degeneracy; fill with 1.0 (max distance)
    D = np.nan_to_num(D, nan=1.0, posinf=1.0, neginf=1.0)
    return squareform(D)

def clustering_to_csv(M, Mb, out_feature_csv, out_cluster_csv, distance_cut=0.7):
    """
    Perform hierarchical clustering on features using correlation distance of M (scores),
    then cut the tree by distance threshold to get cluster labels.
    Save:
      - feature-level CSV: feature, cluster_id, freq, mean_score, median_score
      - cluster-level CSV: cluster_id, n_features, total_freq, mean_of_means, top_features
    """
    feats = M.index.tolist()

    # frequency of a feature being selected across OOD samples
    freq = Mb.sum(axis=1).astype(int)

    # summary per feature
    mean_score = M.mean(axis=1, skipna=True)
    median_score = M.median(axis=1, skipna=True)

    # correlation-distance between features
    D = corr_distance_matrix(M.values)
    # hierarchical clustering linkage
    Z = linkage(squareform(D, checks=False), method='average')
    # flat clusters
    clusters = fcluster(Z, t=distance_cut, criterion='distance')

    df_feat = pd.DataFrame({
        "feature": feats,
        "cluster_id": clusters,
        "freq": freq.values,
        "mean_score": mean_score.values,
        "median_score": median_score.values
    })
    # sort by (cluster_id, -freq, -mean_score)
    df_feat = df_feat.sort_values(by=["cluster_id", "freq", "mean_score"], ascending=[True, False, False]).reset_index(drop=True)
    df_feat.to_csv(out_feature_csv, index=False)

    # cluster-level aggregation
    grp = df_feat.groupby("cluster_id", as_index=False).agg(
        n_features=("feature", "count"),
        total_freq=("freq", "sum"),
        mean_of_means=("mean_score", "mean"),
    )
    # add a preview of top features per cluster
    top_names = (
        df_feat.groupby("cluster_id")
               .apply(lambda g: "; ".join(g.sort_values(["freq","mean_score"], ascending=[False,False])["feature"].head(5)))
               .rename("top_features")
               .reset_index()
    )
    df_cluster = grp.merge(top_names, on="cluster_id", how="left")
    df_cluster = df_cluster.sort_values(by=["total_freq", "mean_of_means"], ascending=[False, False]).reset_index(drop=True)
    df_cluster.to_csv(out_cluster_csv, index=False)

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Cluster OOD-driving features into CSVs (no plots).")
    parser.add_argument("--root", type=str, default="/home/mao/Desktop/cm_repro_for_colab",
                        help="Project root.")
    parser.add_argument("--dataset", type=str, default="jarvis22",
                        help="Dataset name (used for default paths).")
    parser.add_argument("--in_dir", type=str,
                        default=None,
                        help="Dir that contains ood_list.csv / ood_zscore_topk.csv / ood_nn_contrib_topk.csv. "
                             "Default: <root>/data/<dataset>/umap_trainfit_ood/why_ood_from_results")
    parser.add_argument("--out_dir", type=str,
                        default=None,
                        help="Output directory for CSVs. Default: <in_dir>/clustered")
    parser.add_argument("--min_freq_ratio", type=float, default=0.01,
                        help="Remove rare features whose frequency < ratio * #OOD_samples (e.g., 0.01=1%).")
    parser.add_argument("--distance_cut", type=float, default=0.7,
                        help="Flat-cluster distance threshold for hierarchical clustering (smaller -> more clusters).")
    args = parser.parse_args()

    in_dir = args.in_dir or os.path.join(args.root, "data", args.dataset, "umap_trainfit_ood", "why_ood_from_results")
    out_dir = args.out_dir or os.path.join(in_dir, "clustered")
    ensure_exists(out_dir)

    # ---- load inputs
    path_list = os.path.join(in_dir, "ood_list.csv")
    path_z    = os.path.join(in_dir, "ood_zscore_topk.csv")
    path_nn   = os.path.join(in_dir, "ood_nn_contrib_topk.csv")

    df_list = read_with_fallback(path_list, required_cols=["id", "is_OOD"])
    dfz     = read_with_fallback(path_z,    required_cols=["id", "feature"])
    dfn     = read_with_fallback(path_nn,   required_cols=["id", "feature"])

    # detect value columns
    z_col = detect_value_col(dfz, ["z", "abs_z", "z_score"])
    nn_col = detect_value_col(dfn, ["abs_proto_diff", "abs_diff", "contrib"])

    # keep OOD ids only
    ood_ids = df_list.loc[df_list["is_OOD"] == 1, "id"].astype(str).unique().tolist()
    if len(ood_ids) == 0:
        raise ValueError("No OOD ids found in ood_list.csv (is_OOD==1).")

    # --- Build Z-score matrix (feature x OOD samples)
    feats_z, samps_z, Mz, Mz_bin = build_matrix_from_topk(dfz, ood_ids, value_col=z_col)
    # --- Build NN-contrib matrix
    feats_n, samps_n, Mn, Mn_bin = build_matrix_from_topk(dfn, ood_ids, value_col=nn_col)

    # --- Frequency filter
    min_freq_z = max(1, int(np.ceil(args.min_freq_ratio * len(samps_z))))
    min_freq_n = max(1, int(np.ceil(args.min_freq_ratio * len(samps_n))))
    keep_z = Mz_bin.sum(axis=1) >= min_freq_z
    keep_n = Mn_bin.sum(axis=1) >= min_freq_n

    Mz_f = Mz.loc[keep_z]
    Mz_bin_f = Mz_bin.loc[keep_z]
    Mn_f = Mn.loc[keep_n]
    Mn_bin_f = Mn_bin.loc[keep_n]

    # --- Cluster & save CSVs (Z-score)
    clustering_to_csv(
        Mz_f,
        Mz_bin_f,
        out_feature_csv=os.path.join(out_dir, "clustered_features_zscore.csv"),
        out_cluster_csv=os.path.join(out_dir, "clusters_zscore_summary.csv"),
        distance_cut=args.distance_cut
    )

    # --- Cluster & save CSVs (NN-contrib)
    clustering_to_csv(
        Mn_f,
        Mn_bin_f,
        out_feature_csv=os.path.join(out_dir, "clustered_features_nncontrib.csv"),
        out_cluster_csv=os.path.join(out_dir, "clusters_nncontrib_summary.csv"),
        distance_cut=args.distance_cut
    )

    print("[Done] CSVs saved to:", out_dir)
    print(" - clustered_features_zscore.csv")
    print(" - clusters_zscore_summary.csv")
    print(" - clustered_features_nncontrib.csv")
    print(" - clusters_nncontrib_summary.csv")

if __name__ == "__main__":
    main()
