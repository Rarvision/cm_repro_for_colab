import pandas as pd

df = pd.read_pickle('/data/repro/cm_probing_ood/draft/data/jarvis22/dat_featurized_matminer.pkl')
print("列名:", df.columns.tolist())
print("是否有 'elements' 列:", 'elements' in df.columns)