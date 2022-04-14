import numpy as np
import glob
import seaborn as sns
from Conf.Settings import MAX_LAG, MIN_D_N, TYPICAL_DW_RESULTS_PATH, ASD_DW_RESULTS_PATH, FREQ_GAZE, PLOT_RESULT_PATH, ASD_PATH
import pandas as pd
import scipy.stats as stats
from Utils.Lib import cohenD, arModel, autocorr
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
import warnings
warnings.filterwarnings("ignore")



sns.set_style("white")

paths = [TYPICAL_DW_RESULTS_PATH, ASD_DW_RESULTS_PATH]
asd_labels = pd.read_csv(ASD_PATH +  "labels.csv")
multi_class = False

'''
Label idx
0 = typical
1 = ASD
2 = ASD+AD
'''
# group_labels = ["Typical", "ASD without ADHD", "ASD with ADHD"]
# group_colors = ["#66c2a5", "#8da0cb", "#e78ac3"]
# group_colors = ["#252525", "#737373", "#bdbdbd"] #grey_scale

group_labels = ["Typical", "ASD"]
# group_colors = ["#66c2a5", "#fc8d62"] #colorful
group_colors = ["#1f97d5", "#25489d"] #colorful
# group_colors = ["#252525", "#bdbdbd"] #grey_scale
first_group_label = 0
second_group_label = 1
third_group_label = 2



ar_params = []
response = []
subjects= []
t_asd = []



i= 0
s_idx = 0

for path in paths:
    files = glob.glob(path + "*_ar_params.npy")

    for file in files:
        f_name = file.split(path)[-1].split("_ar_params.npy")[0]
        response_file = file.replace("_ar_params.npy", "_responses.npy")
        ar_params.append(np.load(file, allow_pickle=True))
        response.append(np.load(response_file, allow_pickle=True))
        subjects.append(np.ones(len(ar_params[s_idx])) * s_idx)
        if np.sum(asd_labels.id.values == f_name) > 0  and multi_class == True:
            t_asd.append(np.ones(len(ar_params[s_idx])) * asd_labels[asd_labels.id.values == f_name].Label.values)
        else:
            t_asd.append(np.ones(len(ar_params[s_idx])) * i)
        s_idx +=1
    i += 1

X = np.concatenate(ar_params, 0)
response = np.concatenate(response, 0)[np.sum(X, 1) != 0]
subjects = np.concatenate(subjects, 0)[np.sum(X, 1) != 0]
t_asd = np.concatenate(t_asd, 0)[np.sum(X, 1) != 0]
X_filter = X[np.sum(X, 1) != 0]


labels = ["GoPositive", "NoGoNegative", "GoNegative", "NoGoPositive"]
X_features = []
Y = []
responses = []
for s in np.unique(subjects):
    features = []
    for i in range(len(labels)):
        X_response = X_filter[(subjects == s) & (response == i)]
        features.append(np.mean(X_response, 0))
        responses.append(i)
        Y.append(np.mean(t_asd[(subjects == s) & (response == i)]))

    X_features.append(features)


# define data
X = np.concatenate(X_features, 0)
Y = np.array(Y)
responses = np.array(responses)

X_typical_all = X[Y == first_group_label]
X_asd = X[Y == second_group_label]
X_asd_ad = X[Y == third_group_label]

response_typical = responses[Y == first_group_label]
response_asd = responses[Y == second_group_label]
response_asd_ad = responses[Y == third_group_label]
ar_model = arModel(maxlag=MAX_LAG, min_len=MIN_D_N)



# plt.show()

# template = mix_template = ("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}")
template = (" {}, {}, {}, {}, {}, {}, {}")


print("-------------------------------------Average------------------------------------------------")
X_typical = X_typical_all
X_asd = X_asd
X_asd_ad = X_asd_ad

# mean and std
mean_typical = np.mean(X_typical, 0)
mean_asd = np.mean(X_asd, 0)
mean_asd_ad = np.mean(X_asd_ad, 0)

# # extrapolating data all
t_extra = ar_model.predict(np.mean(X_typical_all, 0), start=5, end=55) [5:] # extrapolation of typical
a_extra = ar_model.predict(np.mean(X_asd, 0), start=5, end=55)[5: ]  # extrapolation of ASD
a_ad_extra = ar_model.predict(np.mean(X_asd_ad, 0), start=5, end=55)[5: ]  # extrapolation of ASD

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title("Average")
# plot extrapolate results
idx_times = np.arange(0, len(t_extra), 1) / FREQ_GAZE
ax1.plot(idx_times, t_extra, label=group_labels[first_group_label], color=group_colors[first_group_label], linewidth=1.5)
ax1.plot(idx_times, a_extra, label=group_labels[second_group_label], color=group_colors[second_group_label], linewidth=1.5)
if multi_class:
    ax1.plot(idx_times, a_ad_extra, label=group_labels[third_group_label], color=group_colors[third_group_label], linewidth=1.5)
ax1.set_ylim([0.0, 0.9])
ax1.set_xlabel("Time(s)")
ax1.set_ylabel("Distance")
ax1.legend()
# plot autocorrelation function
t_extra_diff = np.abs(np.diff(t_extra))
a_extra_diff = np.abs(np.diff(a_extra))
a_ad_extra_diff = np.abs(np.diff(a_ad_extra))
print(str(np.mean(t_extra_diff)) + "," + str(np.mean(a_extra_diff)))
ax2.plot(idx_times[1:], t_extra_diff, ".", label=group_labels[first_group_label], color=group_colors[first_group_label], linewidth=1.5)
ax2.plot(idx_times[1:], a_extra_diff,".", label=group_labels[second_group_label], color=group_colors[second_group_label], linewidth=1.5)
if multi_class:
    ax2.plot(idx_times[1:], a_ad_extra_diff,".", label=group_labels[third_group_label], color=group_colors[third_group_label], linewidth=1.5)
ax2.set_ylim([0.0, 0.06])
ax2.set_xlabel("Lag")
ax2.set_ylabel("Gaze-adj velocity")
ax2.legend()
plt.tight_layout()
plt.savefig(PLOT_RESULT_PATH +  "average.eps")
#Analyze per label

for i in range(len(labels)):

    # mean and std
    mean_typical = np.mean(X_typical_all[response_typical==i], 0)
    mean_asd = np.mean( X_asd[response_asd==i], 0)
    mean_asd_ad = np.mean( X_asd_ad[response_asd_ad==i], 0)
    # #extrapolating data
    t_extra = ar_model.predict(mean_typical, start=5, end=55)[5:]  # extrapolation of typical
    a_extra = ar_model.predict(mean_asd, start=5, end=55)[5:]  # extrapolation of ASD
    a_ad_extra =  ar_model.predict(mean_asd_ad, start=5, end=55)[5:]  # extrapolation of ASD w/ AD
    # print(str(np.min(t_extra)) + ","+ str(np.min(a_extra)))
    plt.figure(i)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title(labels[i])
    #plot extrapolate results
    idx_times = np.arange(0, len(t_extra), 1) / FREQ_GAZE
    ax1.plot(idx_times, t_extra, label=group_labels[first_group_label], color=group_colors[first_group_label],linewidth=1.5)
    ax1.plot(idx_times, a_extra, label=group_labels[second_group_label], color=group_colors[second_group_label],linewidth=1.5)
    if multi_class:
        ax1.plot(idx_times, a_ad_extra, label=group_labels[third_group_label], color=group_colors[third_group_label],
             linewidth=1.5)
    ax1.set_ylim([0.0, 0.9])
    ax1.set_xlabel("Time(s)")
    ax1.set_ylabel("Distance")
    ax1.legend()
    #plot autocorrelation function
    t_extra_diff = np.abs(np.diff(t_extra))
    a_extra_diff = np.abs(np.diff(a_extra))
    a_ad_extra_diff = np.abs(np.diff(a_ad_extra))
    print(str(np.mean(t_extra_diff)) + "," + str(np.mean(a_extra_diff)))
    ax2.plot(idx_times[1:], t_extra_diff,".",  label=group_labels[first_group_label], color=group_colors[first_group_label], linewidth=1.5)
    ax2.plot(idx_times[1:], a_extra_diff,".", label=group_labels[second_group_label], color=group_colors[second_group_label], linewidth=1.5)
    if multi_class:
        ax2.plot(idx_times[1:], a_ad_extra_diff,".", label=group_labels[third_group_label], color=group_colors[third_group_label], linewidth=1.5)
    ax2.set_ylim([0.0, 0.06])
    ax2.set_xlabel("Time(s)")
    ax2.set_ylabel("Decreasing Rate")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(PLOT_RESULT_PATH + labels[i] + ".eps")
    # plt.show()







