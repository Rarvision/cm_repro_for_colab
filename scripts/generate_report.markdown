Experiment Report: Leave-One-Group-Out (Fluorine)

Experiment Setup





Dataset: Jarvis22



Target Variable: Formation Energy (e_form)



Grouping: Elements, test set contains Fluorine (F)



Models:





RandomForest: n_estimators=100, max_features=1/3, n_jobs=-1, random_state=0



XGBoost: n_estimators=1000, learning_rate=0.25, reg_lambda=0.01, reg_alpha=0.1, subsample=0.85, colsample_bytree=0.3, colsample_bylevel=0.5, num_parallel_tree=4, tree_method='hist', device='cuda', random_state=0



Data Processing: Filter e_form > 5 and NaN values, no feature selection

Performance Metrics

The following are the performance metrics for RandomForest and XGBoost on the Fluorine-containing test set:







Model



MAE (eV/atom)



RMSE (eV/atom)



R²



Pearson r



Spearman r



Kendall r





{% for modelname in ['rf', 'xgb'] %}





























{% set results = pd.read_csv(f'/data/repro/cm_probing_ood/draft/results/jarvis22_e_form_elements_F/performance_summary.csv', index_col=0).loc[modelname] %}





























{{ modelname.upper() }}



{{ results['mae']:.3f }}



{{ results['rmse']:.3f }}



{{ results['r2']:.3f }}



{{ results['pearson_r']:.3f }}



{{ results['spearman_r']:.3f }}



{{ results['kendall_r']:.3f }}





{% endfor %}

























UMAP Visualization

The following plots show the UMAP dimensionality reduction results for the training set (non-Fluorine) and test set (Fluorine):

{% for modelname in ['rf', 'xgb'] %}

{{ modelname.upper() }} UMAP Plot

![{{ modelname }} UMAP](/data/repro/cm_probing_ood/draft/results/jarvis22_e_form_elements_F/{{ modelname }}_umap.png) R²: {{ pd.read_csv(f'/data/repro/cm_probing_ood/draft/results/jarvis22_e_form_elements_F/performance_summary.csv', index_col=0).loc[modelname]['r2']:.3f }} {% endfor %}

Conclusion





RandomForest and XGBoost achieved R² values of {{ pd.read_csv('/data/repro/cm_probing_ood/draft/results/jarvis22_e_form_elements_F/performance_summary.csv', index_col=0).loc['rf']['r2']:.3f }} and {{ pd.read_csv('/data/repro/cm_probing_ood/draft/results/jarvis22_e_form_elements_F/performance_summary.csv', index_col=0).loc['xgb']['r2']:.3f }} on the Fluorine test set, respectively.



The UMAP plots show the distribution of training and test data in the feature space, with the test set (Fluorine) points indicating the model's extrapolation performance.

Report generated on: September 24, 2025