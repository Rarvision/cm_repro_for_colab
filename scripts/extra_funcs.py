#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
辅助函数模块，提供数据加载、分组、性能评估等功能。
用于支持LOGO实验和UMAP可视化。
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn import metrics
from pymatgen.core.periodic_table import Element
from pymatgen.core import Composition
import os


def load_data(dataset, target):
    """
    加载特征化数据，过滤无效值，添加元素属性。

    参数：
        dataset (str): 数据集名称，'jarvis22'。
        target (str): 目标变量， 'e_form'。

    返回：
        df (pd.DataFrame): 包含元信息的 DataFrame。
        X (pd.DataFrame): 特征矩阵（Matminer 特征）。
        y (pd.Series): 目标变量。
    """
    # 打印当前工作目录和文件路径
    print(f"当前工作目录: {os.getcwd()}")
    file_path = f'/data/repro/cm_probing_ood/draft/data/{dataset}/dat_featurized_matminer.pkl'
    print(f"尝试加载文件: {file_path}")

    # 读取特征化数据
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    df = pd.read_pickle(file_path)

    # 读取 Matminer 特征标签
    feature_file = '/data/repro/cm_probing_ood/draft/data/matminer_feature_labels.txt'
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"特征标签文件 {feature_file} 不存在")
    with open(feature_file, 'r') as f:
        matminer_features = f.read().splitlines()

    # 过滤 e_form > 5 的数据
    df = df[df['e_form'] < 5]

    # 删除目标变量或特征中的 NaN 值
    df = df.dropna(subset=[target] + matminer_features)

    # 添加元素相关属性
    df = add_elemental_attributes(df)

    # 提取特征和目标
    X = df[matminer_features].astype(float)
    y = df[target]

    # Remove highly correlated features
    corr_threshold = 0.7
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    X = X.drop(columns=to_drop)
    print(
        f"Removed {len(to_drop)} highly correlated features (threshold={corr_threshold}). Remaining features: {len(X.columns)}")

    # print(f"加载数据完成: {len(df)} 条记录，{len(matminer_features)} 个特征")
    print(f"加载数据完成: {len(df)} 条记录，{len(X.columns)} 个特征")
    return df, X, y


def add_elemental_attributes(df):
    """
    为数据集添加元素相关属性，如元素组、周期、元素数量。

    参数：
        df (pd.DataFrame): 输入 DataFrame，包含 'formula' 列。

    返回：
        df (pd.DataFrame): 添加了 'elements', 'group', 'period', 'nelements' 列的 DataFrame。
    """
    if 'formula' not in df.columns:
        raise KeyError("DataFrame 中缺少 'formula' 列")
    print("从 'formula' 列解析元素列表")
    df['elements'] = df['formula'].apply(lambda x: [str(e) for e in Composition(x).elements])
    elements = df['elements'].apply(lambda x: [Element(e) for e in x])
    df['group'] = elements.apply(lambda x: [e.group for e in x])
    df['period'] = elements.apply(lambda x: [e.row for e in x])
    df['nelements'] = elements.apply(lambda x: len(set(x)))
    return df


def get_split(df, group_label, group_value):
    """
    根据分组标签和值划分训练和测试集索引。

    参数：
        df (pd.DataFrame): 包含分组信息的 DataFrame。
        group_label (str): 分组标签，如 'elements'。
        group_value (str/int): 分组值，如 'F'。

    返回：
        index_train (pd.Index): 训练集索引。
        index_test (pd.Index): 测试集索引。
    """
    if group_label == 'elements':
        index_train = df[df[group_label].apply(lambda x: group_value not in x)].index
        index_test = df[df[group_label].apply(lambda x: group_value in x)].index
    else:
        raise NotImplementedError(f"分组标签 {group_label} 未实现")
    return index_train, index_test


def get_scores_from_pred(y_test, y_pred):
    """
    计算预测性能指标，包括 MAD、MAE、RMSE、R² 和相关系数。

    参数：
        y_test (pd.Series): 真实值。
        y_pred (pd.Series): 预测值。

    返回：
        tuple: 包含 mad, std, maes, rmse, r2 及其他相关系数。
    """
    mad = (y_test - y_test.mean()).abs().mean()
    std = y_test.std()
    maes = metrics.mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = metrics.r2_score(y_test, y_pred)
    pearson_r, pearson_p_value = pearsonr(y_test, y_pred)
    spearman_r, spearman_p_value = spearmanr(y_test, y_pred)
    kendall_r, kendall_p_value = kendalltau(y_test, y_pred)
    return mad, std, maes, rmse, r2, pearson_r, pearson_p_value, spearman_r, spearman_p_value, kendall_r, kendall_p_value