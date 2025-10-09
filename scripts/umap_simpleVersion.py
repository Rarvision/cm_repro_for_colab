#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate UMAP visualization to show the distribution of training and test data.
No feature selection is performed, using all Matminer features directly.
"""

import pandas as pd
import os
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.preprocessing import StandardScaler
from extra_funcs import load_data, get_split, iterative_corr_prune

# Experiment configuration
dataset = 'jarvis22'
target = 'e_form'
group_label = 'elements'
group_value = 'F'
output_dir = f'/data/repro/cm_probing_ood/draft/results/{dataset}_{target}_{group_label}_{group_value}'
os.makedirs(output_dir, exist_ok=True)

# Load data
df, X, y = load_data(dataset, target)

index_train, index_test = get_split(df, group_label, group_value)

kept_cols, dropped_cols = iterative_corr_prune(
    X.loc[index_train],
    y.loc[index_train],      # 有监督保留策略；若做无监督就传 None
    threshold=0.7,           # 你的阈值
    method="spearman",       # 对材料数据更鲁棒；需要严格线性可用 "pearson"
    min_var=0.0,
    verbose=True,
)

X_train = X.loc[index_train, kept_cols].copy()
y_train = y.loc[index_train].copy()
X_test  = X.loc[index_test,  kept_cols].copy()
y_test  = y.loc[index_test].copy()

print(f"最终用于建模的特征数: {X_train.shape[1]}（训练/测试列一致）")

# Standardize features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# UMAP dimensionality reduction
umap = UMAP(n_components=2, n_neighbors=50, random_state=0)
X_umap = umap.fit_transform(X)
X_umap = pd.DataFrame(X_umap, columns=['0', '1'], index=X.index)
X_umap_train = X_umap.loc[index_train]
X_umap_test = X_umap.loc[index_test]

# Save UMAP results
X_umap.to_csv(f'{output_dir}/umap_result.csv')

# Create UMAP plot
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(X_umap_train['0'], X_umap_train['1'], color='grey', s=6, label='Training Set', alpha=0.1)
ax.scatter(X_umap_test['0'], X_umap_test['1'], color='red', s=3.6, label='Test Set (Fluorine)', alpha=0.02)

# Set axis labels
ax.set_xlabel('UMAP Component 1')
ax.set_ylabel('UMAP Component 2')

# Set axis limits
rscale = 0.1
d = (X_umap['0'].max() - X_umap['0'].min()) * rscale
xlims = [X_umap['0'].min() - d / 2, X_umap['0'].max() + d / 2]
d = (X_umap['1'].max() - X_umap['1'].min()) * rscale
ylims = [X_umap['1'].min() - 0.5 * d, X_umap['1'].max() + 1.25 * d]
ax.set_xlim(xlims[::-1])  # Reverse x-axis to match original
ax.set_ylim(ylims)
ax.set_xticks([])
ax.set_yticks([])

# Create legend
train_patch = mlines.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=5, label='Training Set')
test_patch = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=5, label='Test Set (Fluorine)')
ax.legend(handles=[train_patch, test_patch], loc='upper right', fontsize=10)

# Save figure
figname = f'{output_dir}/umap.png'
fig.savefig(figname, transparent=False, facecolor='white', dpi=300, bbox_inches='tight')
plt.close()