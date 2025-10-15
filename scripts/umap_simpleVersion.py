#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UMAP visualization + OOD detection (kNN distance-based)
-------------------------------------------------------
- Fit StandardScaler and UMAP on TRAIN only
- Apply transform to all data
- Detect OOD test samples via kNN distance in scaled feature space
- Visualization:
    grey = training samples
    blue = in-distribution test samples
    red  = OOD test samples
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from umap import UMAP

from extra_funcs import load_data, get_split, iterative_corr_prune

# =========================
# Config
# =========================
dataset = 'jarvis22'
target = 'e_form'
group_label = 'elements'
group_value = 'F'   # test set contains this element
root_path = '/home/mao/Desktop/cm_repro_for_colab'
output_dir = f'{root_path}/data/{dataset}/umap_trainfit_ood/'
os.makedirs(output_dir, exist_ok=True)

# Visualization params
figsize = (4, 4)
train_color = 'grey'
test_in_color = 'blue'
test_ood_color = 'red'
train_alpha = 0.06
test_alpha = 0.25
train_size = 4
test_size = 8

# UMAP params
umap_n_neighbors = 50
umap_min_dist = 0.30
umap_random_state = 0

# kNN OOD params
knn_k = 10
ood_percentile = 95   # threshold based on training distance percentile

use_kept_cols_for_umap = True

# =========================
# Load data
# =========================
print(f"当前工作目录: {os.getcwd()}")
df, X_all, y_all = load_data(dataset, target, root_path)
print(f"加载数据完成: {len(df)} 条记录，{X_all.shape[1]} 个特征")

index_train, index_test = get_split(df, group_label, group_value)
print(f"划分完成: 训练集 {len(index_train)}，测试集 {len(index_test)} (条件: {group_label} contains '{group_value}')")

# =========================
# Iterative correlation pruning on TRAIN ONLY
# =========================
print("在训练集上进行迭代相关性裁剪")
kept_cols, dropped_cols = iterative_corr_prune(
    X_all.loc[index_train],
    y_all.loc[index_train],
    threshold=0.8,
    method="spearman",
    min_var=0.0,
    verbose=True,
)
print(f"裁剪完成: kept={len(kept_cols)}, dropped={len(dropped_cols)}")

# 使用裁剪后的特征
if use_kept_cols_for_umap:
    X = X_all.loc[:, kept_cols].copy()
    used_feat_desc = f"{len(kept_cols)} pruned features"
else:
    X = X_all.copy()
    used_feat_desc = f"all {X_all.shape[1]} features"

print(f"UMAP 将使用特征: {used_feat_desc}")

# =========================
# Scale: fit on TRAIN, apply to ALL
# =========================
scaler = StandardScaler().fit(X.loc[index_train])
X_scaled = pd.DataFrame(
    scaler.transform(X),
    columns=X.columns,
    index=X.index
)
print("标准化完成：在训练集上拟合并已应用到全体样本。")

# =========================
# kNN-based OOD detection (in high-dim scaled feature space)
# =========================
print(f"计算基于 {knn_k}-NN 的 OOD 距离阈值 ...")
nbrs = NearestNeighbors(n_neighbors=knn_k).fit(X_scaled.loc[index_train])
train_dists, _ = nbrs.kneighbors(X_scaled.loc[index_train])
train_mean_dists = train_dists.mean(axis=1)

threshold = np.percentile(train_mean_dists, ood_percentile)
print(f"训练集 {ood_percentile} 分位距离阈值: {threshold:.4f}")

test_dists, _ = nbrs.kneighbors(X_scaled.loc[index_test])
test_mean_dists = test_dists.mean(axis=1)
ood_mask = test_mean_dists > threshold

num_ood = np.sum(ood_mask)
num_test = len(index_test)
print(f"OOD 测试样本数: {num_ood}/{num_test} ({num_ood/num_test*100:.2f}%)")

# 保存 OOD 结果表
ood_df = pd.DataFrame({
    'formula': df.loc[index_test, 'formula'],
    'mean_knn_dist': test_mean_dists,
    'is_OOD': ood_mask
}, index=index_test)
ood_df.to_csv(os.path.join(output_dir, 'test_ood_knn.csv'))
print("已保存 OOD 判定结果 CSV。")

# =========================
# UMAP: fit on TRAIN, transform ALL
# =========================
umap = UMAP(
    n_components=2,
    n_neighbors=umap_n_neighbors,
    min_dist=umap_min_dist,
    random_state=umap_random_state,
)
umap.fit(X_scaled.loc[index_train])

X_umap_all = umap.transform(X_scaled)
X_umap = pd.DataFrame(X_umap_all, columns=['0', '1'], index=X.index)

X_umap_train = X_umap.loc[index_train]
X_umap_test = X_umap.loc[index_test]
X_umap_test_in = X_umap_test.loc[~ood_mask]
X_umap_test_ood = X_umap_test.loc[ood_mask]

# =========================
# Save CSV
# =========================
csv_path = os.path.join(output_dir, 'umap_result_trainfit_ood.csv')
X_umap.to_csv(csv_path)
print(f"已保存 UMAP 结果到: {csv_path}")

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=figsize)

# 绘制训练集 (灰)
ax.scatter(X_umap_train['0'], X_umap_train['1'],
           s=train_size, alpha=train_alpha, color=train_color, label='Training')

# 绘制测试集 (蓝)
ax.scatter(X_umap_test_in['0'], X_umap_test_in['1'],
           s=test_size, alpha=test_alpha, color=test_in_color, label='Test (in-distribution)')

# 绘制 OOD 测试点 (红)
ax.scatter(X_umap_test_ood['0'], X_umap_test_ood['1'],
           s=test_size*1.5, alpha=0.9, facecolors='none',
           edgecolors=test_ood_color, linewidths=0.8, label='Test (OOD)')

ax.set_xlabel('UMAP Component 1')
ax.set_ylabel('UMAP Component 2')

# Axis limits + reverse x-axis
rscale = 0.10
dx = (X_umap['0'].max() - X_umap['0'].min()) * rscale
dy = (X_umap['1'].max() - X_umap['1'].min()) * rscale
ax.set_xlim([X_umap['0'].max() + 0.5*dx, X_umap['0'].min() - 0.5*dx])
ax.set_ylim([X_umap['1'].min() - 0.5*dy, X_umap['1'].max() + 1.25*dy])
ax.set_xticks([])
ax.set_yticks([])

# Legend
train_patch = mlines.Line2D([], [], color=train_color, marker='o', linestyle='None', markersize=5, label='Training')
test_patch_in = mlines.Line2D([], [], color=test_in_color, marker='o', linestyle='None', markersize=5, label='Test (in)')
test_patch_ood = mlines.Line2D([], [], color=test_ood_color, marker='o', linestyle='None', markersize=6,
                               markerfacecolor='none', label='Test (OOD)')
ax.legend(handles=[train_patch, test_patch_in, test_patch_ood], loc='upper right', fontsize=9)

# Footnote
ax.text(0.02, 0.02,
        f'Features: {used_feat_desc}\nUMAP fit: TRAIN only\nOOD: kNN({knn_k})>{ood_percentile}%',
        transform=ax.transAxes, fontsize=7, ha='left', va='bottom')

# Save fig
png_path = os.path.join(output_dir, 'umap_trainfit_ood.png')
fig.savefig(png_path, transparent=False, facecolor='white', dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"已保存 UMAP+OOD 图像到: {png_path}")
print("Done.")
