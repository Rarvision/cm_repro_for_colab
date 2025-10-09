# compute_similarity.py
# -*- coding: utf-8 -*-
"""
计算多种相似度/核，并输出：
1) 稀疏 KNN 图（.npz）
2) 邻居表（.csv.gz）
3) 汇总表（summary_k<k>.csv）
4) （可选）完整相似度/距离矩阵（.npy/.csv.gz）——通过 save_full 开关控制

建议：完整矩阵仅对采样集（<= 5000 样本）启用；全量 7.6 万请仅保存 KNN。
"""
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel, polynomial_kernel, rbf_kernel
from scipy.sparse import csr_matrix, save_npz
from numpy.linalg import LinAlgError


# ---------------------------
# 工具函数
# ---------------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def median_heuristic_gamma(X, subsample=2000, random_state=42):
    """对 RBF 的 gamma 使用中位数启发式：gamma = 1 / median(||xi-xj||^2)"""
    n = X.shape[0]
    if n > subsample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=subsample, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    D2 = pairwise_distances(Xs, Xs, metric="sqeuclidean")
    tri = np.triu_indices_from(D2, k=1)  # 非零上三角
    med = np.median(D2[tri])
    if med <= 0 or np.isnan(med):
        return 1e-3
    return 1.0 / med


def robust_cov_inv(X, ridge=1e-6):
    """计算逆协方差矩阵（马氏距离用），加入岭正则避免奇异。"""
    C = np.cov(X, rowvar=False)
    C_reg = C + ridge * np.eye(C.shape[0], dtype=C.dtype)
    try:
        C_inv = np.linalg.inv(C_reg)
    except LinAlgError:
        C_inv = np.linalg.pinv(C_reg)
    return C_inv


def knn_from_similarity(S, topk, keep_diag=False):
    """从相似度矩阵 S 构建 Top-k 稀疏邻接。"""
    n = S.shape[0]
    if not keep_diag:
        np.fill_diagonal(S, 0.0)
    idx_part = np.argpartition(-S, kth=topk-1, axis=1)[:, :topk]
    row = np.repeat(np.arange(n), topk)
    col = idx_part.flatten()
    data = S[np.arange(n)[:, None], idx_part].flatten()
    W = csr_matrix((data, (row, col)), shape=(n, n))
    return W


def knn_from_distance(D, topk):
    """从距离矩阵 D 构建 Top-k 稀疏邻接（相似度=1/(1+D)）。"""
    S = 1.0 / (1.0 + D)
    np.fill_diagonal(S, 0.0)
    return knn_from_similarity(S, topk)


def neighbor_table_from_sparse(W, index):
    """把稀疏 KNN 图转为易读 DataFrame（src, nbr, weight）。"""
    W = W.tocsr()
    rows, cols = W.nonzero()
    weights = W.data
    df_edges = pd.DataFrame({
        "src": index[rows],
        "nbr": index[cols],
        "weight": weights
    })
    return df_edges


def eval_neighbor_consistency(df_edges, target_series, agg="mean"):
    """
    邻居一致性评估：比较邻居 target 聚合值与自身 target 的差。返回 MAE/RMSE。
    """
    t = target_series.rename("target")
    df_edges = df_edges.join(t, on="src")
    df_edges = df_edges.join(t.rename("target_nbr"), on="nbr")
    if agg == "mean":
        nbr_pred = df_edges.groupby("src")["target_nbr"].mean()
    elif agg == "median":
        nbr_pred = df_edges.groupby("src")["target_nbr"].median()
    else:
        raise ValueError("agg must be 'mean' or 'median'.")
    y_true = t.loc[nbr_pred.index]
    y_pred = nbr_pred
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return mae, rmse


# ---------------------------
# 主流程
# ---------------------------
def main(args):
    ensure_dir(args.outdir)

    # 读取特征矩阵（DataFrame，索引为样本ID / jid）
    Xdf = pd.read_pickle(args.input)
    assert isinstance(Xdf, pd.DataFrame)
    index = Xdf.index
    X = Xdf.values.astype(np.float64)
    n = X.shape[0]
    print(f"[info] Loaded X: {X.shape} from {args.input}")

    # 读取 target（可选）
    target_series = None
    if args.target_csv and args.target_key and args.target_col:
        tgt = pd.read_csv(args.target_csv).set_index(args.target_key)
        if args.target_col not in tgt.columns:
            raise ValueError(f"target_col '{args.target_col}' not in {args.target_csv} columns.")
        target_series = tgt.loc[index, args.target_col]
        mask = target_series.notna().values
        if not mask.all():
            kept = mask.sum()
            print(f"[warn] Dropping {n - kept} rows due to missing target.")
            X = X[mask]
            index = index[mask]
            target_series = target_series[mask]
            n = X.shape[0]
        print(f"[info] Loaded target for {n} rows from {args.target_csv}")

    # 度量列表
    metrics = [m.lower() for m in args.metrics]
    metrics_allowed = {"euclidean", "cosine", "mahalanobis", "rbf", "linear", "poly"}
    unknown = set(metrics) - metrics_allowed
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}")

    # 自动判断是否允许保存完整矩阵
    will_save_full = bool(args.save_full) and (n <= args.full_save_max_n)
    if args.save_full and not will_save_full:
        print(f"[warn] save_full=True but n={n} > full_save_max_n={args.full_save_max_n}. "
              f"Full matrices will NOT be saved to avoid huge files.")

    print(f"[info] N={n}, topk={args.topk}. save_full={will_save_full}. "
          f"将输出每种度量的 KNN 稀疏图、邻居表，以及汇总。")

    C_inv = None
    gamma = None
    results_summary = []

    for m in metrics:
        print(f"\n===== Computing: {m} =====")
        S = None  # 相似度矩阵
        D = None  # 距离矩阵

        if m == "euclidean":
            D = pairwise_distances(X, metric="euclidean")
            W = knn_from_distance(D, topk=args.topk)

        elif m == "mahalanobis":
            if C_inv is None:
                print("[info] Computing inverse covariance (regularized)...")
                C_inv = robust_cov_inv(X, ridge=args.mahal_ridge)
            D = pairwise_distances(X, metric="mahalanobis", VI=C_inv)
            W = knn_from_distance(D, topk=args.topk)

        elif m == "cosine":
            S = cosine_similarity(X)
            W = knn_from_similarity(S, topk=args.topk)

        elif m == "linear":
            S = linear_kernel(X)
            W = knn_from_similarity(S, topk=args.topk)

        elif m == "poly":
            S = polynomial_kernel(X, degree=args.poly_degree, coef0=args.poly_coef0)
            W = knn_from_similarity(S, topk=args.topk)

        elif m == "rbf":
            if args.gamma is None:
                gamma = median_heuristic_gamma(X, subsample=min(3000, n))
                print(f"[info] RBF gamma (median heuristic): {gamma:.6g}")
            else:
                gamma = args.gamma
                print(f"[info] RBF gamma (user): {gamma:.6g}")
            S = rbf_kernel(X, gamma=gamma)
            W = knn_from_similarity(S, topk=args.topk)

        else:
            raise RuntimeError("unreachable")

        # ========= 新增：保存完整矩阵（统一处理） =========
        if will_save_full and (S is not None):
            sim_path_npy = os.path.join(args.outdir, f"similarity_{m}.npy")
            np.save(sim_path_npy, S)
            print(f"[save] full similarity matrix: {sim_path_npy}  shape={S.shape}")

            if args.save_full_csv:
                sim_path_csv = os.path.join(args.outdir, f"similarity_{m}.csv.gz")
                pd.DataFrame(S, index=index, columns=index).to_csv(sim_path_csv, compression="gzip")
                print(f"[save] full similarity CSV: {sim_path_csv}")

        if will_save_full and (D is not None):
            dist_path_npy = os.path.join(args.outdir, f"distance_{m}.npy")
            np.save(dist_path_npy, D)
            print(f"[save] full distance matrix: {dist_path_npy}  shape={D.shape}")

            if args.save_full_csv:
                dist_path_csv = os.path.join(args.outdir, f"distance_{m}.csv.gz")
                pd.DataFrame(D, index=index, columns=index).to_csv(dist_path_csv, compression="gzip")
                print(f"[save] full distance CSV: {dist_path_csv}")
        # ================================================

        # 保存稀疏图
        W_path = os.path.join(args.outdir, f"knn_{m}_k{args.topk}.npz")
        save_npz(W_path, W)
        print(f"[save] {W_path}  (nnz={W.nnz}, density={W.nnz/(n*n):.6e})")

        # 邻居表
        edges = neighbor_table_from_sparse(W, index=index)
        edges_path = os.path.join(args.outdir, f"neighbors_{m}_k{args.topk}.csv.gz")
        edges.to_csv(edges_path, index=False, compression="gzip")
        print(f"[save] {edges_path}  (rows={len(edges)})")

        # 邻居一致性评估（可选）
        mae = rmse = None
        if target_series is not None:
            mae, rmse = eval_neighbor_consistency(edges, target_series, agg="mean")
            print(f"[eval] neighbor consistency ({m}): MAE={mae:.6g}, RMSE={rmse:.6g}")

        results_summary.append({
            "metric": m,
            "nnz": int(W.nnz),
            "density": W.nnz/(n*n),
            "mae": mae,
            "rmse": rmse,
            "extra": {"gamma": gamma} if m == "rbf" else {}
        })

    # 汇总
    df_sum = pd.DataFrame(results_summary)
    sum_path = os.path.join(args.outdir, f"summary_k{args.topk}.csv")
    df_sum.to_csv(sum_path, index=False)
    print(f"\n[save] summary: {sum_path}")
    print(df_sum)


if __name__ == "__main__":
    # ============ 方式A：写死参数，PyCharm直接运行 ============
    class Args:
        # 输入/输出
        input = r"../data/jarvis22/cleaned/X_sample_scaled_n1000.pkl"
        outdir = r"../data/jarvis22/similarity"

        # KNN
        topk = 30

        # 计算哪些度量/核
        metrics = ["euclidean", "cosine", "mahalanobis", "rbf", "linear", "poly"]

        # RBF & Poly
        gamma = None            # None=用中位数启发式；或手动给数值如 0.001
        poly_degree = 3
        poly_coef0 = 1.0

        # 马氏距离
        mahal_ridge = 1e-6

        # 完整矩阵保存控制
        save_full = True        # 是否保存完整相似度/距离矩阵（小样本时建议 True）
        save_full_csv = True    # 同时保存为 .csv.gz（便于查看；体积更大）
        full_save_max_n = 5000  # 超过该 N 自动禁用完整矩阵保存

        # 可选：邻居一致性评估（需提供目标性质）
        target_csv = None
        target_key = None
        target_col = None

    args = Args()
    main(args)

    # ============ 方式B：命令行参数（需要时放开） ============
    # parser = argparse.ArgumentParser(description="Compute pairwise similarities/kernels and KNN graphs.")
    # parser.add_argument("--input", type=str, required=True,
    #                     help="Path to X_sample_scaled_*.pkl 或 X_full_scaled.pkl（DataFrame，索引为样本ID）")
    # parser.add_argument("--outdir", type=str, required=True, help="输出目录")
    # parser.add_argument("--topk", type=int, default=30, help="每个样本保留的邻居数")
    # parser.add_argument("--metrics", nargs="+",
    #                     default=["euclidean", "cosine", "mahalanobis", "rbf", "linear", "poly"],
    #                     help="要计算的相似度/核列表")
    # parser.add_argument("--gamma", type=float, default=None, help="RBF gamma；缺省用中位数启发式")
    # parser.add_argument("--poly_degree", type=int, default=3, help="多项式核的 degree")
    # parser.add_argument("--poly_coef0", type=float, default=1.0, help="多项式核的 coef0")
    # parser.add_argument("--mahal_ridge", type=float, default=1e-6, help="协方差矩阵的岭正则项")
    # parser.add_argument("--save_full", action="store_true", help="保存完整相似度/距离矩阵（仅小样本推荐）")
    # parser.add_argument("--save_full_csv", action="store_true", help="同时保存为CSV（体积更大）")
    # parser.add_argument("--full_save_max_n", type=int, default=5000, help="超过该N自动不保存完整矩阵")
    # parser.add_argument("--target_csv", type=str, default=None, help="包含目标性质的 CSV（可选）")
    # parser.add_argument("--target_key", type=str, default=None, help="target CSV 中与特征索引匹配的键列名（如 jid）")
    # parser.add_argument("--target_col", type=str, default=None, help="目标性质列名（如 formation_energy_per_atom）")
    # args = parser.parse_args()
    # main(args)
