# compute_similarity.py                        # 文件名：计算多种相似度/核并输出多种结果
# -*- coding: utf-8 -*-                        # 指定源文件编码为 UTF-8，确保中文注释/字符串不乱码
"""
计算多种相似度/核，并输出：
1) 稀疏 KNN 图（.npz）
2) 邻居表（.csv.gz）
3) 汇总表（summary_k<k>.csv）
4) 完整相似度/距离矩阵（.npy/.csv.gz）——通过 save_full 开关控制

建议：完整矩阵仅对采样集（<= 5000 样本）启用；大样本仅保存 KNN。
"""  # 顶部文档字符串：说明脚本功能、输出内容与使用建议

import os  # 操作路径/文件的标准库
import argparse  # 解析命令行参数
import numpy as np  # 数值计算库
import pandas as pd  # 数据表处理库
from pathlib import Path  # 更现代的路径对象

from sklearn.metrics import pairwise_distances  # sklearn 的通用成对距离计算函数
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel, polynomial_kernel, rbf_kernel  # 常见相似度/核函数
from scipy.sparse import csr_matrix, save_npz  # 稀疏矩阵类型与保存函数
from numpy.linalg import LinAlgError  # 线性代数错误类型（用于捕获逆矩阵失败）


# ---------------------------
# 工具函数
# ---------------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)  # 若目录不存在则递归创建；存在则忽略错误


def median_heuristic_gamma(X, subsample=1000, random_state=42):
    """对 RBF 的 gamma 使用中位数启发式：gamma = 1 / median(||xi-xj||^2)"""
    n = X.shape[0]  # 样本数
    if n > subsample:  # 若样本多于子采样上限
        rng = np.random.default_rng(random_state)  # 构造可复现实验的随机数生成器
        idx = rng.choice(n, size=subsample, replace=False)  # 无放回均匀采样 subsample 个索引
        Xs = X[idx]  # 取子集用于估计 gamma
    else:
        Xs = X  # 样本量不大则直接使用全部数据
    D2 = pairwise_distances(Xs, Xs, metric="sqeuclidean")  # 计算子集样本两两平方欧氏距离
    tri = np.triu_indices_from(D2, k=1)  # 取上三角（不含对角），避免重复和零距离
    med = np.median(D2[tri])  # 对上三角距离求中位数
    if med <= 0 or np.isnan(med):  # 保护性判断：异常值或数值问题
        return 1e-3  # 回退到一个较小的默认 gamma
    return 1.0 / med  # 返回启发式 gamma = 1 / 距离中位数


def robust_cov_inv(X, ridge=1e-6):
    """计算逆协方差矩阵（马氏距离用），加入岭正则避免奇异。"""
    C = np.cov(X, rowvar=False)  # 计算特征维度上的协方差矩阵（行是样本，列是特征）
    C_reg = C + ridge * np.eye(C.shape[0], dtype=C.dtype)  # 加入岭正则，缓解奇异或病态
    try:
        C_inv = np.linalg.inv(C_reg)  # 尝试求精确逆
    except LinAlgError:  # 如果失败（矩阵接近奇异）
        C_inv = np.linalg.pinv(C_reg)  # 使用伪逆作为替代，保证数值稳定
    return C_inv  # 返回（正则化的）逆协方差


def knn_from_similarity(S, topk, keep_diag=False):
    """从相似度矩阵 S 构建 Top-k 稀疏邻接。"""
    n = S.shape[0]  # 样本数
    if not keep_diag:
        np.fill_diagonal(S, 0.0)  # 通常不保留自环：对角置零，避免选到自己
    idx_part = np.argpartition(-S, kth=topk - 1, axis=1)[:, :topk]  # 对每行做近似 top-k（不完全排序，速度快）
    row = np.repeat(np.arange(n), topk)  # 行索引：每个样本重复 topk 次
    col = idx_part.flatten()  # 列索引：对应每行的 topk 列拼起来
    data = S[np.arange(n)[:, None], idx_part].flatten()  # 取出对应的相似度值并展平
    W = csr_matrix((data, (row, col)), shape=(n, n))  # 构造稀疏邻接矩阵（CSR 形式）
    return W  # 返回 KNN 稀疏图


def knn_from_distance(D, topk):
    """从距离矩阵 D 构建 Top-k 稀疏邻接（相似度=1/(1+D)）。"""
    S = 1.0 / (1.0 + D)  # 将距离映射为相似度（单调递减），范围 (0,1]
    np.fill_diagonal(S, 0.0)  # 不保留自环
    return knn_from_similarity(S, topk)  # 复用相似度版本的 KNN 构建逻辑


def neighbor_table_from_sparse(W, index):
    """把稀疏 KNN 图转为易读 DataFrame（src, nbr, weight）。"""
    W = W.tocsr()  # 确保是 CSR 类型，便于提取索引与数据
    rows, cols = W.nonzero()  # 取出非零条目的行列索引
    weights = W.data  # 非零条目的权重（相似度）
    df_edges = pd.DataFrame({  # 构建边表：源点、邻居点、权重
        "src": index[rows],  # 使用原始索引（如样本 ID）
        "nbr": index[cols],
        "weight": weights
    })
    return df_edges  # 返回邻居边表 DataFrame


def eval_neighbor_consistency(df_edges, target_series, agg="mean"):
    """
    邻居一致性评估：比较邻居 target 聚合值与自身 target 的差。返回 MAE/RMSE。
    """
    t = target_series.rename("target")  # 统一列名为 target
    df_edges = df_edges.join(t, on="src")  # 把源点的 target 拼到边表
    df_edges = df_edges.join(t.rename("target_nbr"), on="nbr")  # 把邻居点 target 拼到边表
    if agg == "mean":
        nbr_pred = df_edges.groupby("src")["target_nbr"].mean()  # 以邻居 target 的均值作为“预测”
    elif agg == "median":
        nbr_pred = df_edges.groupby("src")["target_nbr"].median()  # 或者使用中位数更稳健
    else:
        raise ValueError("agg must be 'mean' or 'median'.")  # 不支持的聚合方式直接报错
    y_true = t.loc[nbr_pred.index]  # 对齐真实值（源点自身 target）
    y_pred = nbr_pred  # 邻居聚合预测值
    mae = np.mean(np.abs(y_true - y_pred))  # 计算 MAE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))  # 计算 RMSE
    return mae, rmse  # 返回两个指标


# ---------------------------
# 主流程
# ---------------------------
def main(args):
    ensure_dir(args.outdir)  # 确保输出目录存在

    # 读取特征矩阵（DataFrame，索引为样本ID / jid）
    Xdf = pd.read_pickle(args.input)  # 从 pickle 读取特征 DataFrame
    assert isinstance(Xdf, pd.DataFrame)  # 断言类型正确
    index = Xdf.index  # 保存索引（样本 ID）
    X = Xdf.values.astype(np.float64)  # 取数值矩阵并转换为 float64 以便数值稳定
    n = X.shape[0]  # 样本数
    print(f"[info] Loaded X: {X.shape} from {args.input}")  # 打印加载信息

    # 读取 target
    target_series = None  # 初始化目标变量序列
    if args.target_csv and args.target_key and args.target_col:  # 若用户提供了目标 CSV 及关键字段
        tgt = pd.read_csv(args.target_csv).set_index(args.target_key)  # 读取并以 key 列为索引
        if args.target_col not in tgt.columns:
            raise ValueError(f"target_col '{args.target_col}' not in {args.target_csv} columns.")  # 校验列是否存在
        target_series = tgt.loc[index, args.target_col]  # 按特征索引顺序对齐提取目标
        mask = target_series.notna().values  # 生成非缺失掩码
        if not mask.all():  # 如果存在缺失目标
            kept = mask.sum()
            print(f"[warn] Dropping {n - kept} rows due to missing target.")  # 提示丢弃数量
            X = X[mask]  # 同步过滤特征矩阵
            index = index[mask]  # 同步过滤索引
            target_series = target_series[mask]  # 同步过滤目标
            n = X.shape[0]  # 更新样本数
        print(f"[info] Loaded target for {n} rows from {args.target_csv}")  # 打印目标加载信息

    # 度量列表
    metrics = [m.lower() for m in args.metrics]  # 统一小写，便于匹配
    metrics_allowed = {"euclidean", "cosine", "mahalanobis", "rbf", "linear", "poly"}  # 支持的度量/核
    unknown = set(metrics) - metrics_allowed  # 检查是否有未知选项
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}")  # 对未知度量直接报错

    # 自动判断是否允许保存完整矩阵
    will_save_full = bool(args.save_full) and (n <= args.full_save_max_n)  # 开关为真且样本量不超过阈值
    if args.save_full and not will_save_full:  # 用户要求保存但 N 过大
        print(f"[warn] save_full=True but n={n} > full_save_max_n={args.full_save_max_n}. "
              f"Full matrices will NOT be saved to avoid huge files.")  # 警告：为避免巨大文件，将不保存

    print(f"[info] N={n}, topk={args.topk}. save_full={will_save_full}. "
          f"将输出每种度量的 KNN 稀疏图、邻居表，以及汇总。")  # 运行参数总览

    C_inv = None  # 逆协方差缓存（用于马氏距离，避免重复求逆）
    gamma = None  # RBF 的 gamma 记录
    results_summary = []  # 汇总结果列表（最终写入 CSV）

    for m in metrics:  # 遍历每种度量/核
        print(f"\n===== Computing: {m} =====")  # 打印当前正在计算的度量
        S = None  # 相似度矩阵占位
        D = None  # 距离矩阵占位

        if m == "euclidean":  # 欧氏距离
            D = pairwise_distances(X, metric="euclidean")  # 计算全对距离
            W = knn_from_distance(D, topk=args.topk)  # 基于距离的 KNN 稀疏图

        elif m == "mahalanobis":  # 马氏距离
            if C_inv is None:
                print("[info] Computing inverse covariance (regularized)...")  # 提示即将计算逆协方差
                C_inv = robust_cov_inv(X, ridge=args.mahal_ridge)  # 带岭正则的逆协方差
            D = pairwise_distances(X, metric="mahalanobis", VI=C_inv)  # 使用预计算的 VI（inverse covariance）
            W = knn_from_distance(D, topk=args.topk)  # 转换相似度并取 KNN

        elif m == "cosine":  # 余弦相似度
            S = cosine_similarity(X)  # 直接得到相似度矩阵
            W = knn_from_similarity(S, topk=args.topk)  # 基于相似度的 KNN 稀疏图

        elif m == "linear":  # 线性核（X X^T），与余弦不同未归一化
            S = linear_kernel(X)  # 计算线性核
            W = knn_from_similarity(S, topk=args.topk)  # 取 KNN

        elif m == "poly":  # 多项式核
            S = polynomial_kernel(X, degree=args.poly_degree, coef0=args.poly_coef0)  # (γ 内部默认为 1/特征数)
            W = knn_from_similarity(S, topk=args.topk)  # 取 KNN

        elif m == "rbf":  # RBF/Gaussian 核
            if args.gamma is None:  # 若未用户指定 gamma
                gamma = median_heuristic_gamma(X, subsample=min(3000, n))  # 用中位数启发式估计
                print(f"[info] RBF gamma (median heuristic): {gamma:.6g}")  # 打印估计值
            else:
                gamma = args.gamma  # 使用用户给定值
                print(f"[info] RBF gamma (user): {gamma:.6g}")  # 打印用户值
            S = rbf_kernel(X, gamma=gamma)  # 计算 RBF 相似度矩阵
            W = knn_from_similarity(S, topk=args.topk)  # 取 KNN

        else:
            raise RuntimeError("unreachable")  # 理论上不会到达（前面已校验）

        # ========= 新增：保存完整矩阵（统一处理） =========
        if will_save_full and (S is not None):  # 若允许保存且有相似度矩阵
            sim_path_npy = os.path.join(args.outdir, f"similarity_{m}.npy")  # 构造 .npy 路径
            np.save(sim_path_npy, S)  # 保存相似度矩阵为二进制 .npy（高效）
            print(f"[save] full similarity matrix: {sim_path_npy}  shape={S.shape}")  # 打印保存信息

            if args.save_full_csv:  # 若还需要保存 CSV 版本（便于人工查看）
                sim_path_csv = os.path.join(args.outdir, f"similarity_{m}.csv.gz")  # 构造 .csv.gz 路径
                pd.DataFrame(S, index=index, columns=index).to_csv(sim_path_csv, compression="gzip")  # 附带索引保存
                print(f"[save] full similarity CSV: {sim_path_csv}")  # 打印保存信息

        if will_save_full and (D is not None):  # 若允许保存且有距离矩阵
            dist_path_npy = os.path.join(args.outdir, f"distance_{m}.npy")  # 构造 .npy 路径
            np.save(dist_path_npy, D)  # 保存距离矩阵
            print(f"[save] full distance matrix: {dist_path_npy}  shape={D.shape}")  # 打印保存信息

            if args.save_full_csv:  # 同步保存 CSV 压缩版
                dist_path_csv = os.path.join(args.outdir, f"distance_{m}.csv.gz")
                pd.DataFrame(D, index=index, columns=index).to_csv(dist_path_csv, compression="gzip")
                print(f"[save] full distance CSV: {dist_path_csv}")  # 打印保存信息
        # ================================================

        # 保存稀疏图
        W_path = os.path.join(args.outdir, f"knn_{m}_k{args.topk}.npz")  # KNN 稀疏图保存路径
        save_npz(W_path, W)  # 保存为 .npz（scipy 稀疏格式）
        print(f"[save] {W_path}  (nnz={W.nnz}, density={W.nnz / (n * n):.6e})")  # 打印非零元数与稠密度

        # 邻居表
        edges = neighbor_table_from_sparse(W, index=index)  # 稀疏图转为边表 DataFrame
        edges_path = os.path.join(args.outdir, f"neighbors_{m}_k{args.topk}.csv.gz")  # 边表保存路径
        edges.to_csv(edges_path, index=False, compression="gzip")  # 保存为压缩 CSV
        print(f"[save] {edges_path}  (rows={len(edges)})")  # 打印行数（边数）

        # 邻居一致性评估（可选）
        mae = rmse = None  # 初始化评估指标
        if target_series is not None:  # 当提供 target 时才评估
            mae, rmse = eval_neighbor_consistency(edges, target_series, agg="mean")  # 以均值聚合评估
            print(f"[eval] neighbor consistency ({m}): MAE={mae:.6g}, RMSE={rmse:.6g}")  # 打印评估结果

        results_summary.append({  # 将当前度量的摘要信息加入列表
            "metric": m,  # 度量名称
            "nnz": int(W.nnz),  # 稀疏图非零边数
            "density": W.nnz / (n * n),  # 稠密度（方便比较）
            "mae": mae,  # 邻居一致性 MAE（若无 target 则为 None）
            "rmse": rmse,  # 邻居一致性 RMSE（若无 target 则为 None）
            "extra": {"gamma": gamma} if m == "rbf" else {}  # 额外信息：RBF 的 gamma
        })

    # 汇总
    df_sum = pd.DataFrame(results_summary)  # 汇总结果转为 DataFrame
    sum_path = os.path.join(args.outdir, f"summary_k{args.topk}.csv")  # 汇总 CSV 路径
    df_sum.to_csv(sum_path, index=False)  # 保存汇总 CSV
    print(f"\n[save] summary: {sum_path}")  # 打印保存信息
    print(df_sum)  # 控制台再打印一份汇总便于查看


if __name__ == "__main__":  # 脚本作为主程序执行时的入口
    # ============ 方式A：写死参数，PyCharm直接运行 ============
    class Args:
        # 输入/输出
        input = r"../data/jarvis22/cleaned/X_sample_scaled_n1000.pkl"  # 特征 DataFrame 的 pickle 路径
        outdir = r"../data/jarvis22/similarity"  # 输出目录

        # KNN
        topk = 30  # 每个样本保留的邻居数

        # 计算哪些度量/核
        metrics = ["euclidean", "cosine", "mahalanobis", "rbf", "linear", "poly"]  # 默认全跑一遍

        # RBF & Poly
        gamma = None  # None=用中位数启发式；或手动给数值如 0.001
        poly_degree = 3  # 多项式核阶数
        poly_coef0 = 1.0  # 多项式核常数项（影响低阶贡献）

        # 马氏距离
        mahal_ridge = 1e-6  # 协方差正则强度（越大越稳健但偏差更大）

        # 完整矩阵保存控制
        save_full = True  # 是否保存完整相似度/距离矩阵（小样本时建议 True）
        save_full_csv = True  # 同时保存为 .csv.gz（直观但体积更大）
        full_save_max_n = 5000  # 超过该 N 自动禁用完整矩阵保存（避免爆内存/磁盘）

        # 邻居一致性评估（需提供目标性质）
        target_csv = None  # 包含目标的 CSV 路径；若不评估则保持 None
        target_key = None  # CSV 中作为行索引的键（需与特征索引对齐）
        target_col = None  # 需要评估的目标列名


    args = Args()  # 构造参数对象
    main(args)  # 直接调用主流程

    # ============ 方式B：命令行参数（需要时放开） ============
    # parser = argparse.ArgumentParser(description="Compute pairwise similarities/kernels and KNN graphs.")  # 构建命令行解析器
    # parser.add_argument("--input", type=str, required=True,
    #                     help="Path to X_sample_scaled_*.pkl 或 X_full_scaled.pkl（DataFrame，索引为样本ID）")  # 特征输入
    # parser.add_argument("--outdir", type=str, required=True, help="输出目录")                               # 输出目录
    # parser.add_argument("--topk", type=int, default=30, help="每个样本保留的邻居数")                        # K 值
    # parser.add_argument("--metrics", nargs="+",
    #                     default=["euclidean", "cosine", "mahalanobis", "rbf", "linear", "poly"],
    #                     help="要计算的相似度/核列表")                                                       # 多选度量
    # parser.add_argument("--gamma", type=float, default=None, help="RBF gamma；缺省用中位数启发式")         # RBF 参数
    # parser.add_argument("--poly_degree", type=int, default=3, help="多项式核的 degree")                    # 多项式阶数
    # parser.add_argument("--poly_coef0", type=float, default=1.0, help="多项式核的 coef0")                  # 多项式常数项
    # parser.add_argument("--mahal_ridge", type=float, default=1e-6, help="协方差矩阵的岭正则项")            # 正则强度
    # parser.add_argument("--save_full", action="store_true", help="保存完整相似度/距离矩阵（仅小样本推荐）")  # 是否保存完整矩阵
    # parser.add_argument("--save_full_csv", action="store_true", help="同时保存为CSV（体积更大）")           # 是否同时保存 CSV
    # parser.add_argument("--full_save_max_n", type=int, default=5000, help="超过该N自动不保存完整矩阵")      # 保存阈值
    # parser.add_argument("--target_csv", type=str, default=None, help="包含目标性质的 CSV（可选）")           # 目标文件路径
    # parser.add_argument("--target_key", type=str, default=None, help="target CSV 中与特征索引匹配的键列名（如 jid）")  # 对齐键
    # parser.add_argument("--target_col", type=str, default=None, help="目标性质列名（如 formation_energy_per_atom）")   # 目标列名
    # args = parser.parse_args()                                      # 解析命令行参数
    # main(args)                                                      # 以命令行参数方式运行
