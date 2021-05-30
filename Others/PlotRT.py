import pandas as pd
import numpy as np
import matplotlib
from Conf.Settings import ASD_DW_RESULTS_PATH, TYPICAL_DW_RESULTS_PATH, PLOT_PATH
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
rt_length = 4

paths = [TYPICAL_DW_RESULTS_PATH , ASD_DW_RESULTS_PATH]
file_name = "RTSeries.csv"
titles = ["Typical", "High-risk"]
colors = ["#1b9e77", "#d95f02"]
i=0

typical_mean_data = []
typical_std_data = []
asd_mean_data = []
asd_std_data = []
for path in paths:

    for file in glob(path+"*"+file_name):
        if i == 0:
            typical = pd.read_csv(file, index_col=0)
            typical_mean = typical.groupby(typical.Time // 61).RT.mean().values
            if len(typical_mean) == 4:
                typical_mean_data.append(typical_mean)
                typical_std_data.append(typical.groupby(typical.Time // 61).RT.std().values )
        else:
            asd = pd.read_csv(file, index_col=0)
            asd_mean = asd.groupby(asd.Time // 61).RT.mean().values
            if len(asd_mean) == 4:
                asd_mean_data.append(asd_mean)
                asd_std_data.append(asd.groupby(asd.Time // 61).RT.std().values)

    i+=1

rt_avg = [np.average(typical_mean_data, 0) * 1000, np.average(asd_mean_data, 0) * 1000]
rt_std = [np.average(typical_std_data, 0) * 1000, np.average(asd_std_data, 0) * 1000]
plt.rcParams["font.family"] = "arial"
for j in range(len(rt_avg)):
    times = np.arange(1, rt_length + 1)
    ylim = [-100, 700]
    plt.figure(j)
    plt.plot(times, rt_avg[j], color=colors[j], label="Average RT(ms)")
    plt.fill_between(times, rt_avg[j] + rt_std[j], rt_avg[j] - rt_std[j], alpha=0.5, color=colors[j])
    plt.ylim(ylim)
    # plt.xlabel("Minutes")
    # plt.ylabel("ms")
    plt.legend(loc=4)
    # plt.show()
    plt.savefig(PLOT_PATH+str(j) +"_rt_summary.png")
    plt.close()




