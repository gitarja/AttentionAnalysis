import pandas as pd
import scipy.stats as stats
from Conf.Settings import ASD_PATH, ASD_DW_PATH, ASD_DW_RESULTS_PATH, MIN_D_N, MAX_LAG, CUT_OFF, TYPICAL_PATH, TYPICAL_DW_PATH, TYPICAL_DW_RESULTS_PATH


typical_result = pd.read_csv(TYPICAL_DW_RESULTS_PATH + "summary\\" +  "summary_response_new.csv")
typical_len = int(len(typical_result.index) / 2)
asd_results = pd.read_csv(ASD_DW_RESULTS_PATH + "summary\\" +  "summary_response_new.csv")
asd_len = int(len(asd_results.index) / 2)
#
between_template = ("Between Groups F-val: {}, p-val: {}")
within_template = ("Typical Groups F-val: {}, p-val: {} | ASD Groups F-val: {}, p-val: {}")
mix_template = ("{}, {}, {}, {}, {}, {}")

columns = ["Go", "GoError" , "NoGo",
                 "NoGoError", "RT", "RTVar", "Trajectory Area",
                 "Velocity_avg",
                 "Velocity_std",
                 "Acceleration_avg",
                 "Acceleration_std",
                 "Fixation_avg",
                 "Fixation_std",
                 "Sampen_dist",
                 "Sampen_angle",
                 "Spatial_entropy",
                 "GazeObj_entropy",
                 "Sampen_gaze_obj",
                 "Spectral_entropy",
                 "Sampen_velocity",
                 "Sampen_acceleration"]
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
    F_typical, p_typical = stats.mannwhitneyu(typical_result.iloc[0:typical_len][col].values,
                                                 typical_result.iloc[typical_len:][col].values)
    F_asd, p_asd = stats.mannwhitneyu(asd_results.iloc[0:asd_len][col].values,
                                         asd_results.iloc[asd_len:][col].values)
    # print("-------------------"+col+"--------------------------")
    print(mix_template.format(F, p, F_typical, p_typical, F_asd, p_asd))
    # print(between_template.format(F, p))
    # print(within_template.format(F_typical, p_typical, F_asd, p_asd))



