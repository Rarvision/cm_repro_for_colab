#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 matminer_feature_labels.txt，从 jarvis22_featurized_matminer.pkl 提取特征名称。
"""

import pandas as pd

# 加载数据
df = pd.read_pickle('/data/repro/cm_probing_ood/draft/data/jarvis22/dat_featurized_matminer.pkl')

# 排除非特征列
exclude_cols = [
    'spg_number', 'spg_symbol', 'formula', 'e_form', 'bandgap', 'atoms',
    'Tc_supercon', 'bulk_modulus', 'shear_modulus', 'hse_gap', 'reference', 'search'
]
matminer_features = [col for col in df.columns if col not in exclude_cols]

# 保存特征标签
with open('/data/repro/cm_probing_ood/draft/data/matminer_feature_labels.txt', 'w') as f:
    f.write('\n'.join(matminer_features))

print(f"生成 matminer_feature_labels.txt，包含 {len(matminer_features)} 个特征")
print(f"前5个特征: {matminer_features[:5]}")