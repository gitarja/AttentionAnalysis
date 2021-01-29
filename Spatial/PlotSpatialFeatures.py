from Conf.Settings import  ASD_DW_RESULTS_PATH, TYPICAL_DW_RESULTS_PATH, PLOT_RESULT_PATH
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import  seaborn as sns


result_paths = [TYPICAL_DW_RESULTS_PATH, ASD_DW_RESULTS_PATH]

typical_results = result_paths[0] + "summary\\"
asd_results = result_paths[1] + "summary\\"

typical_results = pd.read_csv(os.path.join(typical_results, "summary_response_new.csv"))
asd_results = pd.read_csv(os.path.join(asd_results, "summary_response_new.csv"))

TYPICAL_LABELS = ["Typical" for x in range(len(typical_results))]
ASD_LABELS = ["ASD" for x in range(len(asd_results))]
labels = np.concatenate([TYPICAL_LABELS, ASD_LABELS])

all_results = pd.concat([typical_results, asd_results])
all_results["Label"] = labels.tolist()
all_results["Go"] = all_results["Go"] * 100
all_results["GoError"] = all_results["GoError"] * 100


plt.figure(1)
sns.boxplot(y="Go", x="Label",
                 data=all_results, palette="Set2").set_title('Go Positive (%)')
plt.savefig(PLOT_RESULT_PATH + "go.png")
plt.figure(2)
sns.boxplot(y="GoError", x="Label",
                 data=all_results, palette="Set2").set_title('Go Negative (%)')
plt.savefig(PLOT_RESULT_PATH + "go_error.png")
plt.figure(3)
sns.boxplot(y="Fixation_std", x="Label",
                 data=all_results, palette="Set2").set_title('Variance of Fixation Time (ms)')
plt.savefig(PLOT_RESULT_PATH + "fixation_std.png")
plt.figure(4)
sns.boxplot(y="Sampen_dist", x="Label",
                 data=all_results, palette="Set2").set_title('Sample Entropy of Gaze Distance')
plt.savefig(PLOT_RESULT_PATH + "sampen_dist.png")
plt.figure(5)
sns.boxplot(y="Sampen_angle", x="Label",
                 data=all_results, palette="Set2").set_title('Sample Entropy of Gaze Angle')
plt.savefig(PLOT_RESULT_PATH + "sampen_angle.png")
plt.figure(6)
sns.boxplot(y="GazeObj_entropy", x="Label",
                 data=all_results, palette="Set2").set_title('Gaze-to-obj Entropy')
plt.savefig(PLOT_RESULT_PATH + "gazeobj_entropy.png")
plt.figure(7)
sns.boxplot(y="Spectral_entropy", x="Label",
                 data=all_results, palette="Set2").set_title('Gaze-to-obj Spectral Entropy')
plt.savefig(PLOT_RESULT_PATH + "spectral_entropy.png")
plt.show()