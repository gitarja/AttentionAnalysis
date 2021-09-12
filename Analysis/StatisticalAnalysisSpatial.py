import pandas as pd
import scipy.stats as stats
from sklearn.feature_selection import f_classif
from Conf.Settings import ASD_PATH, ASD_DW_PATH, ASD_DW_RESULTS_PATH, MIN_D_N, MAX_LAG, CUT_OFF, TYPICAL_PATH, \
    TYPICAL_DW_PATH, TYPICAL_DW_RESULTS_PATH
from Utils.Lib import cohenD
import numpy as np

typical_result = pd.read_csv(TYPICAL_DW_RESULTS_PATH + "summary\\" + "summary_response_new.csv")
typical_len = int(len(typical_result.index) / 2)
asd_results = pd.read_csv(ASD_DW_RESULTS_PATH + "summary\\" + "summary_response_new.csv")
asd_labels = pd.read_csv(ASD_PATH +  "labels.csv")
asd_len = int(len(asd_results.index) / 2)
#
between_template = ("Between Groups F-val: {}, p-val: {}")
within_template = ("Typical Groups F-val: {}, p-val: {} | ASD Groups F-val: {}, p-val: {}")
mix_template = ("{}, {}, {}")

columns = ["Go", "GoError", "NoGo",
           "NoGoError", "RT", "RTVar", "Trajectory Area",
           "Velocity_avg",
           "Velocity_std",
           "Acceleration_avg",
           "Acceleration_std",
           "Fixation_avg",
           "Fixation_std",
           "Distance_avg",
           "Distance_std",
           "Angle_avg",
           "Angle_std",
           "Sampen_dist",
           "Sampen_angle",
           "Sampen_velocity",
           "Spatial_entropy",
           "GazeObj_entropy",
           "Sampen_gaze_obj",
           "Spectral_entropy",
           "Sampen_acceleration"]

# columns = [
#            "Distance_avg",
#            "Distance_std",
#            "Angle_avg",
#            "Angle_std",
# ]

#decide labels
asd_split = True

'''
Label desc:
1 = ASD
2 = ASD + AD
'''
if asd_split:
    typical_labels = np.zeros(len(typical_result))
    as_ad_labels = np.ones(len(asd_results))
    for idx, result in asd_results.iterrows():
        as_ad_labels[idx] = asd_labels[asd_labels.id.values == result.id].Label
    asd = asd_results[as_ad_labels == 1]
    asd_ad = asd_results[as_ad_labels == 2]

    #concatenate all labels
    labels = np.concatenate([typical_labels, as_ad_labels])

    print("-------------------Typical-Average--------------------------")
    print(typical_result.loc[:, columns].mean(axis=0))
    print(typical_result.loc[:, columns].std(axis=0))
    print("-------------------ASD-Average--------------------------")
    print(asd.loc[:, columns].mean(axis=0))
    print(asd.loc[:, columns].std(axis=0))
    print("-------------------ASD+AD-Average--------------------------")
    print(asd_ad.loc[:, columns].mean(axis=0))
    print(asd_ad.loc[:, columns].std(axis=0))

    print("-------------------TypicalxASD--------------------------")
    for col in columns:
        # between Typical and ASD
        _,t_p = stats.ttest_ind(typical_result[col].values, asd[col].values)
        _, w_p = stats.mannwhitneyu(typical_result[col].values, asd[col].values)
        # compute Cohen'sD
        d_val = cohenD(typical_result[col].values, asd[col].values)
        # print("-------------------"+col+"--------------------------")
        print(mix_template.format(t_p, w_p, d_val))

    print("-------------------ASDxASD+AD--------------------------")
    for col in columns:
        # between Typical and ASD
        _,t_p = stats.ttest_ind(asd[col].values, asd_ad[col].values)
        _, w_p = stats.mannwhitneyu(asd[col].values, asd_ad[col].values)
        # compute Cohen'sD
        d_val = cohenD(asd[col].values, asd_ad[col].values)
        # print("-------------------"+col+"--------------------------")
        print(mix_template.format(t_p, w_p, d_val))

    print("-------------------TypicalxASD+AD--------------------------")
    for col in columns:
        # between Typical and ASD
        _,t_p = stats.ttest_ind(typical_result[col].values, asd_ad[col].values)
        _, w_p = stats.mannwhitneyu(typical_result[col].values, asd_ad[col].values)
        # compute Cohen'sD
        d_val = cohenD(typical_result[col].values, asd_ad[col].values)
        # print("-------------------"+col+"--------------------------")
        print(mix_template.format(t_p, w_p, d_val))

    print("-------------------ANOVA--------------------------")
    X = np.concatenate([typical_result.loc[:,columns].values, asd.loc[:,columns].values, asd_ad.loc[:,columns].values])
    _, anova_p = f_classif(X, labels)
    for p in anova_p:
        print(p)



else:
    print("-------------------Typical-Average--------------------------")
    print(typical_result.mean(axis=0))
    print(typical_result.std(axis=0))
    print("-------------------ASD-Average--------------------------")
    print(asd_results.mean(axis=0))
    print(asd_results.std(axis=0))
    for col in columns:
        # between Typical and ASD
        F, p = stats.mannwhitneyu(typical_result[col].values, asd_results[col].values)
        # within group
        # F_typical, p_typical = stats.mannwhitneyu(typical_result.iloc[0:typical_len][col].values,
        #                                       typical_result.iloc[typical_len:][col].values)
        # F_asd, p_asd = stats.mannwhitneyu(asd_results.iloc[0:asd_len][col].values,
        #                               asd_results.iloc[asd_len:][col].values)

        # compute Cohen'sD
        d_val = cohenD(typical_result[col].values, asd_results[col].values)
        # print("-------------------"+col+"--------------------------")
        print(mix_template.format(F, p,  d_val))

