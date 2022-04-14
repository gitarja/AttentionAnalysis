import pandas as pd
from statsmodels.stats.multitest import multipletests
import numpy as np

'''
To prevent false positive, we must perform p-value correction using Benjamini/Hochberg
'''


P_VALUE_FILE = "D:\\usr\\pras\\data\\googledrive\\Research-Doctoral\\Experiment-Result\\SpatioTemporal-2020-08-27\\p-values\\gaze-adjustment\\p-values-anova.csv"

p_data = pd.read_csv(P_VALUE_FILE)
alpha = 0.05
p_value = np.array(p_data.iloc[:, 1:].values).flatten()
correct = multipletests(p_value, alpha, "fdr_bh")
results = correct[0]
col_len = 1
for i in range(len(results)//col_len):
    print(results[i*col_len:(i+1)*col_len])
