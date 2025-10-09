# prepare_features.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==== 配置区 ====
INPUT_PKL = r"../data/jarvis22/dat_featurized_matminer.pkl"   # 原始 pkl 路径
OUTPUT_DIR = r"../data/jarvis22/cleaned"                      # 输出目录
MISSING_COL_THRESHOLD = 0.20   # 丢弃缺失率 > 20% 的列, 设为None禁用
SAMPLE_SIZE = 1000             # 采样行数, 设为None则不采样
RANDOM_STATE = 42              # 采样随机种子

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) 读取
    print(f"[1/6] Loading: {INPUT_PKL}")
    df = pd.read_pickle(INPUT_PKL)
    print(f"   - shape: {df.shape}")
    print(f"   - index name: {df.index.name}")
    print("   - first 5 columns & dtypes:\n", df.dtypes.head(12))

    # 2) 只保留数值列
    print("[2/6] Selecting numeric columns...")
    # df_num = df.select_dtypes(include=[np.number]).copy()
    # print(f"   - numeric shape: {df_num.shape}")
    df_num = df.iloc[:, 12:]
    print(f"   - after dropping first 12 columns: {df.shape}")
    print("   - first few columns now:\n", df.columns[:10])

    # 3) 按缺失率丢列 + 中位数填补
    if MISSING_COL_THRESHOLD is not None:
        miss_ratio = df_num.isna().mean()
        keep_cols = miss_ratio[miss_ratio <= MISSING_COL_THRESHOLD].index.tolist()
        dropped = [c for c in df_num.columns if c not in keep_cols]
        df_num = df_num.loc[:, keep_cols]
        print(f"[3/6] Drop columns with missing ratio > {MISSING_COL_THRESHOLD:.0%}")
        print(f"   - dropped {len(dropped)} columns")
    else:
        print("[3/6] Skip dropping by missing ratio")

    # 中位数填补
    medians = df_num.median(numeric_only=True)
    df_num = df_num.fillna(medians)
    still_nan = df_num.isna().sum().sum()
    print(f"   - after fillna(median), remaining NaNs: {still_nan}")

    # 4) 标准化（零均值单位方差）
    print("[4/6] Standardizing features (z-score)...")
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(df_num.values)
    df_scaled = pd.DataFrame(X_scaled, index=df_num.index, columns=df_num.columns)
    print(f"   - scaled shape: {df_scaled.shape}")

    # 5) 随机采样（用于相似度方法对比的“小集合”）
    if SAMPLE_SIZE is not None:
        sample_n = min(SAMPLE_SIZE, df_scaled.shape[0])
        df_sample = df_scaled.sample(n=sample_n, random_state=RANDOM_STATE)
        print(f"[5/6] Sampled {sample_n} rows for quick similarity experiments.")
    else:
        df_sample = None
        print("[5/6] Sampling disabled.")

    # 6) 保存输出
    print("[6/6] Saving outputs...")
    # 保存完整清洗+标准化矩阵
    full_pkl = os.path.join(OUTPUT_DIR, "X_full_scaled.pkl")
    full_csv = os.path.join(OUTPUT_DIR, "X_full_scaled.csv.gz")
    df_scaled.to_pickle(full_pkl)
    df_scaled.to_csv(full_csv, compression="gzip")
    # 保存采样矩阵
    if df_sample is not None:
        sample_pkl = os.path.join(OUTPUT_DIR, f"X_sample_scaled_n{len(df_sample)}.pkl")
        sample_csv = os.path.join(OUTPUT_DIR, f"X_sample_scaled_n{len(df_sample)}.csv.gz")
        df_sample.to_pickle(sample_pkl)
        df_sample.to_csv(sample_csv, compression="gzip")
    # 保存元信息：保留列名与简单统计
    meta_path = os.path.join(OUTPUT_DIR, "feature_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"Total rows (original): {df.shape[0]}\n")
        f.write(f"Original cols: {df.shape[1]}\n")
        f.write(f"Numeric cols kept: {df_scaled.shape[1]}\n")
        f.write(f"Dropped by missing ratio: {dropped if MISSING_COL_THRESHOLD is not None else 'N/A'}\n\n")
        f.write("Feature list:\n")
        for c in df_scaled.columns:
            f.write(f"{c}\n")

    print("Done.")
    print(f"Full scaled matrix: {full_pkl}")
    if df_sample is not None:
        print(f"Sample scaled matrix: {sample_pkl}")
    print(f"Meta: {meta_path}")

if __name__ == "__main__":
    main()
