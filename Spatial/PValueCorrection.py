import pandas as pd
from Conf.Settings import P_VALUE_FILE
from statsmodels.stats.multitest import multipletests
import numpy as np

p_data = pd.read_csv(P_VALUE_FILE)
alpha = 0.05
p_value = np.array(p_data.iloc[:, 1:].values).flatten()
print(p_data.iloc[:, 1:].values)
correct = multipletests(p_value, alpha, "fdr_bh")
results = correct[0]
col_len = 6
for i in range(len(results)//col_len):
    print(results[i*col_len:(i+1)*col_len])
# for index, row in p_data.iterrows():
#     # print(row[0])
#     correct = multipletests(row[1:], alpha, "fdr_bh")
#     # print(correct[0])
#     print(correct[1])