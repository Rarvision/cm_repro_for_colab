#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
执行 leave-one-group-out 实验，训练 RandomForest 和 XGBoost 模型，
在含氟元素的数据上测试，并保存预测结果。
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from extra_funcs import load_data, get_split, get_scores_from_pred, iterative_corr_prune

# 实验配置
dataset = 'jarvis22'
target = 'e_form'
group_label = 'elements'
group_value = 'F'
modelnames = ['rf', 'xgb']
output_dir = f'/data/repro/cm_probing_ood/draft/results/{dataset}_{target}_{group_label}_{group_value}'
os.makedirs(output_dir, exist_ok=True)

# 加载数据
df, X, y = load_data(dataset, target)

# 数据划分
index_train, index_test = get_split(df, group_label, group_value)

kept_cols, dropped_cols = iterative_corr_prune(
    X.loc[index_train],
    y.loc[index_train],      # 有监督保留策略；若做无监督就传 None
    threshold=0.7,           # 你的阈值
    method="spearman",       # 对材料数据更鲁棒；需要严格线性可用 "pearson"
    min_var=0.0,
    verbose=True,
)


X_train, y_train = X.loc[index_train], y.loc[index_train]
X_test, y_test = X.loc[index_test], y.loc[index_test]

# 打印数据集大小
print(f'训练集大小: {len(y_train)}')
print(f'测试集大小: {len(y_test)}')

# 初始化结果字典
results = {}

# 遍历模型
for modelname in modelnames:
    print(f'\n训练模型: {modelname}')
    
    # 初始化模型
    if modelname == 'rf':
        model = RandomForestRegressor(
            n_estimators=100,
            max_features=1/3,
            n_jobs=-1,
            random_state=0
        )
    elif modelname == 'xgb':
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.25,
            reg_lambda=0.01,
            reg_alpha=0.1,
            subsample=0.85,
            colsample_bytree=0.3,
            colsample_bylevel=0.5,
            num_parallel_tree=4,
            tree_method='hist',
            device='cuda',
            random_state=0
        )
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred, index=y_test.index)
    
    # 计算性能指标
    mad, std, maes, rmse, r2, pearson_r, pearson_p_value, spearman_r, spearman_p_value, kendall_r, kendall_p_value = get_scores_from_pred(y_test, y_pred)
    
    # 保存结果
    results[modelname] = {
        'mad': mad,
        'std': std,
        'mae': maes,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
        'kendall_r': kendall_r
    }
    
    # 保存预测结果到 CSV
    df_pred = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    df_pred.to_csv(f'{output_dir}/{modelname}_pred.csv')
    
    # 打印性能
    print(f'{modelname} 测试集性能:')
    print(f'MAE: {maes:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}')
    print(f'Pearson r: {pearson_r:.3f}, Spearman r: {spearman_r:.3f}, Kendall r: {kendall_r:.3f}')

    # 生成并保存拟合结果图
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='y=x')
    plt.xlabel('True Formation Energy (eV/atom)')
    plt.ylabel('Predicted Formation Energy (eV/atom)')
    plt.title(f'{modelname.upper()} Parity Plot (R² = {r2:.3f})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{output_dir}/{modelname}_parity_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

# 保存性能指标到 CSV
results_df = pd.DataFrame(results).T
results_df.to_csv(f'{output_dir}/performance_summary.csv')