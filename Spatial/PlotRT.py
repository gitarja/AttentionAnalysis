import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
rt_length = 4

paths = ["D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\Typical\\summary\\", "D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\High-risk\\summary\\"]
file = "summary_response_new.csv"
titles = ["Typical", "High-risk"]
colors = ["#1b9e77", "#d95f02"]
i=0
for path in paths:
    data = pd.read_csv(path + file)

    rt = np.average(data.iloc[:, 8:12].values, 0) * 1000
    rt_var = np.average(data.iloc[:, 12:].values, 0) * 1000

    times = np.arange(1, rt_length + 1)
    ylim = [-100, 900]
    plt.title(titles[i])
    #plt.subplot(len(subjects), 1, i)
    plt.plot(times, rt, color=colors[i], label="Average RT")
    plt.fill_between(times, rt + rt_var, rt - rt_var, alpha=0.5, color=colors[i])
    plt.ylim(ylim)
    plt.xlabel("Minutes")
    plt.ylabel("ms")
    i+=1
    # plt.show()
    plt.savefig(path+"rt_summary.png")
    plt.close()




