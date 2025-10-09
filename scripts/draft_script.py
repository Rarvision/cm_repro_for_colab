import pandas as pd

df = pd.read_pickle("../data/jarvis22/dat_featurized_matminer.pkl")
print(df.shape)
print(df.head())