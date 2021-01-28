import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")


# path = "D:\\usr\\pras\\data\\AttentionTestData\\Hoikuen(2020-01-24)\\results\\analysis\\"
path = "D:\\usr\\pras\\data\\AttentionTestData\\Akza-Ardi\\result\\analysis\\"
# path = "D:\\usr\\pras\\data\\AttentionTestData\\Yokodai\\AttentionTest\\result\\analysis\\"

files = glob.glob(path + "*_avg_dist.csv")

avg = []
max = []
min = []
std = []
last = []
response = []
subjects = []
i = 0
for file in files:
    response_file = file.replace("_avg_dist.csv", "_responses.csv")
    avg_file = file
    std_file = file.replace("_avg_dist.csv", "_std_dist.csv")
    max_file = file.replace("_avg_dist.csv", "_max_dsit.csv")
    min_file = file.replace("_avg_dist.csv", "_min_dist.csv")
    last_file = file.replace("_avg_dist.csv", "_last_dist.csv")
    avg.append(np.loadtxt(file, delimiter=","))
    response.append(np.loadtxt(response_file, delimiter=","))
    std.append(np.loadtxt(std_file, delimiter=","))
    max.append(np.loadtxt(max_file, delimiter=","))
    min.append(np.loadtxt(min_file, delimiter= ","))
    last.append(np.loadtxt(last_file, delimiter= ","))
    subjects.append(np.ones(len(avg[i])) * i)
    i+=1

min_X_all = np.array(min)
avg_X = np.concatenate(avg, 0)
std_X = np.concatenate(std, 0)[avg_X!= 0]
max_X = np.concatenate(max, 0)[avg_X != 0]
min_X = np.concatenate(min, 0)[avg_X != 0]
last_X = np.concatenate(last, 0)[avg_X != 0]
response = np.concatenate(response, 0)[avg_X != 0]
subjects = np.concatenate(subjects, 0)[avg_X != 0]
avg_X = avg_X[avg_X != 0]



labels = ["GoPositive", "NoGoNegative", "GoNegative", "NoGoPositive"]
colors = []
data =[]

#by subjects
for s in np.unique(subjects):
    #by response
    for i in range(0, 4):

        print("Average of the minimum distance = %f, STD of the minimum distance = %f" % (np.average(last_X[(response == i) & (subjects==s)]), np.std(last_X[(response == i) & (subjects==s)])))
    print()

#     # by response
# for i in range(0, 4):
#     avg_response = np.average(avg_X[response == i])
#     std_response = np.average(std_X[response == i])
#     max_response = np.average(max_X[response == i])
#     min_response = np.average(min_X[response == i])
#     min_std_response = np.std(min_X[response == i])
#     data.append(min_X[response == i])
#     print("Average of the minimum distance = %f, STD of the minimum distance = %f" % (
#         np.average(last_X[response == i]), np.std(last_X[response == i])))
# sns.boxplot(data=data)
# plt.ylim([0, 1])
# plt.show()
