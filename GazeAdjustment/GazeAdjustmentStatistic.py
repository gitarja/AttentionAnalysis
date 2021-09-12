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



sns.set_style("whitegrid")

paths = [TYPICAL_DW_RESULTS_PATH, ASD_DW_RESULTS_PATH]
asd_labels = pd.read_csv(ASD_PATH +  "labels.csv")
multi_class = True

'''
Label idx
0 = typical
1 = ASD
2 = ASD+AD
'''
group_labels = ["Typical", "ASD w/o AD", "ASD w/ AD"]
group_colors = ["#66c2a5", "#8da0cb", "#e78ac3"]
first_group_label = 1
second_group_label = 2



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
print("-------------------------------------ANOVA------------------------------------------------")
_, p_anova = f_classif(X, Y)
print(p_anova)

for i in range(len(labels)):
    X_label = X[responses==i]
    Y_label = Y[responses==i]
    _, p_anova = f_classif(X_label, Y_label)
    print(p_anova)
print("-------------------------------------End-ANOVA------------------------------------------------")

X_typical_all = X[Y == first_group_label]
X_asd_all = X[Y == second_group_label]

response_typical = responses[Y == first_group_label]
response_asd = responses[Y == second_group_label]
ar_model = arModel(maxlag=MAX_LAG, min_len=MIN_D_N)



# plt.show()

# template = mix_template = ("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}")
template = (" {}, {}, {}, {}, {}, {}, {}")


print("-------------------------------------Average------------------------------------------------")
X_typical = X_typical_all
X_asd = X_asd_all

# t-test between groups
F_t, F_p = stats.ttest_ind(X_typical, X_asd)

# mean and std
mean_typical = np.mean(X_typical, 0)
mean_asd = np.mean(X_asd, 0)
std_typical = np.std(X_typical, 0)
std_asd = np.std(X_asd, 0)

for j in range(3):
    # man-whitney between groups
    mw = stats.mannwhitneyu(X_typical[: j], X_asd[: j])

    # cohen
    cohen = cohenD(X_typical[:, j], X_asd[:, j])
    # print(template.format(F_p[j], p_typical[j], p_asd[j], mw[1], mw_typical[1], mw_asd[1], cohen, mean_typical[j], mean_asd[j], std_typical[j], std_asd[j]))
    print(template.format(F_p[j], mw[1], cohen, mean_typical[j],
                          mean_asd[j], std_typical[j], std_asd[j]))

# # extrapolating data all
t_extra = ar_model.predict(np.mean(X_typical_all, 0), start=5, end=55) [5:] # extrapolation of typical
a_extra = ar_model.predict(np.mean(X_asd_all, 0), start=5, end=55)[5: ]  # extrapolation of ASD

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title("Average")
# plot extrapolate results
idx_times = np.arange(0, len(t_extra), 1) / FREQ_GAZE
ax1.plot(idx_times, t_extra, label=group_labels[first_group_label], color=group_colors[first_group_label], linewidth=1.5)
ax1.plot(idx_times, a_extra, label=group_labels[second_group_label], color=group_colors[second_group_label], linewidth=1.5)
ax1.set_ylim([0.0, 0.9])
ax1.set_xlabel("Time(s)")
ax1.set_ylabel("Distance")
ax1.legend()
# plot autocorrelation function
t_extra_diff = np.abs(np.diff(t_extra))
a_extra_diff = np.abs(np.diff(a_extra))
print(str(np.mean(t_extra_diff)) + "," + str(np.mean(a_extra_diff)))
ax2.plot(idx_times[1:], t_extra_diff, ".", label=group_labels[first_group_label], color=group_colors[first_group_label], linewidth=.5)
ax2.plot(idx_times[1:], a_extra_diff, ".", label=group_labels[second_group_label], color=group_colors[second_group_label], linewidth=.5)
ax2.set_ylim([0.0, 0.06])
ax2.set_xlabel("Lag")
ax2.set_ylabel("Gaze-adj velocity")
ax2.legend()
plt.tight_layout()
plt.savefig(PLOT_RESULT_PATH +  "average.eps")
#Analyze per label

for i in range(len(labels)):
    print("-------------------------------------"+labels[i]+"------------------------------------------------")
    X_typical = X_typical_all[response_typical==i]
    X_asd = X_asd_all[response_asd==i]

    #t-test between groups
    F_t, F_p = stats.ttest_ind(X_typical, X_asd)


    # mean and std
    mean_typical = np.mean(X_typical, 0)
    mean_asd = np.mean(X_asd, 0)
    std_typical = np.std(X_typical, 0)
    std_asd = np.std(X_asd, 0)


    for j in range(3):
        # man-whitney between groups
        mw = stats.mannwhitneyu(X_typical[: j], X_asd[: j])

        #cohen
        cohen = cohenD(X_typical[:, j], X_asd[:, j])
        # print(template.format(F_p[j], p_typical[j], p_asd[j], mw[1], mw_typical[1], mw_asd[1], cohen, mean_typical[j], mean_asd[j], std_typical[j], std_asd[j]))
        print(template.format( F_p[j],  mw[1],  cohen, mean_typical[j],
                              mean_asd[j], std_typical[j], std_asd[j]))

    #
    # #extrapolating data
    t_extra = ar_model.predict(mean_typical, start=5, end=55)[5:]  # extrapolation of typical
    a_extra = ar_model.predict(mean_asd, start=5, end=55)[5:]  # extrapolation of ASD
    # print(str(np.min(t_extra)) + ","+ str(np.min(a_extra)))
    plt.figure(i)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title(labels[i])
    #plot extrapolate results
    idx_times = np.arange(0, len(t_extra), 1) / FREQ_GAZE
    ax1.plot(idx_times, t_extra, label=group_labels[first_group_label], color=group_colors[first_group_label],linewidth=1.5)
    ax1.plot(idx_times, a_extra, label=group_labels[second_group_label], color=group_colors[second_group_label],linewidth=1.5)
    ax1.set_ylim([0.0, 0.9])
    ax1.set_xlabel("Time(s)")
    ax1.set_ylabel("Distance")
    ax1.legend()
    #plot autocorrelation function
    t_extra_diff = np.abs(np.diff(t_extra))
    a_extra_diff = np.abs(np.diff(a_extra))
    print(str(np.mean(t_extra_diff)) + "," + str(np.mean(a_extra_diff)))
    ax2.plot(idx_times[1:], t_extra_diff, ".",  label=group_labels[first_group_label], color=group_colors[first_group_label], linewidth=.5)
    ax2.plot(idx_times[1:], a_extra_diff, ".", label=group_labels[second_group_label], color=group_colors[second_group_label], linewidth=.5)
    ax2.set_ylim([0.0, 0.06])
    ax2.set_xlabel("Time(s)")
    ax2.set_ylabel("Decreasing Rate")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(PLOT_RESULT_PATH + labels[i] + ".eps")
    # plt.show()

    #plotting the data
    # for j in range(3):
    #     plt.figure(j)
    #     plt.boxplot([X_typical[:, j], X_asd[:, j]], notch=True, widths=0.25, positions=[0.75, 1.25], \
    #             labels=['Typical', 'Diagnoised'])
    #     plt.title("Coefficient " + str(j+1))
    #     plt.savefig(PLOT_RESULT_PATH+"Coefficient " + str(j+1)+".png")
    #     plt.show()






