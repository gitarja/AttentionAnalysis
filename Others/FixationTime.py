import pandas as pd
import glob
import numpy as np
from Utils.DataReader import DataReader
from Utils.Lib import filterFixation
from Utils.OutliersRemoval import OutliersRemoval
from Conf.Settings import ASD_PATH, ASD_DW_PATH, ASD_DW_RESULTS_PATH, AREA_FIX_TH, CUT_OFF, TYPICAL_PATH, TYPICAL_DW_PATH, TYPICAL_DW_RESULTS_PATH
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


#set the processing
#ASD or TYPICAL
game_paths = [TYPICAL_PATH, ASD_PATH]
gaze_paths = [TYPICAL_DW_PATH, ASD_DW_PATH]
result_paths = [TYPICAL_DW_RESULTS_PATH, ASD_DW_RESULTS_PATH]

# t_response: type of response
responses = ["GoPositive", "NoGoNegative", "GoNegative", "NoGoPositive"]
typical_asd = ["Typical", "ASD"]
t_response = []
fixation_times = []
labels = []
#Reader and removal
reader = DataReader()
removal = OutliersRemoval(cutoff=CUT_OFF)

class_idx = 0
for path in zip(game_paths, gaze_paths, result_paths):
    game_path = path[0]
    gaze_path = path[1]
    result_path = path[2]

    #read game data
    reader = DataReader()
    removal = OutliersRemoval(cutoff=CUT_OFF)
    data_game, game_f_names = reader.readGameResults(game_path)
    data_game = removal.transformGameResult(data_game)
    data_gaze, data_gaze_obj, gaze_f_names = reader.readGazeData(gaze_path, downsample=True)


    if not game_f_names.__eq__(gaze_f_names):
        warnings.simplefilter("Data do not match")


    for d in zip(data_game, data_gaze_obj, game_f_names):
        game = d[0]
        gaze = d[1]
        f_name = d[2]



        for _, data in game.iterrows():
            response_time = data["ResponseTime"]

            if response_time == -1:
                response_time = data["SpawnTime"] + 0.7

            gaze_t = gaze[(gaze["Time"] >= data["SpawnTime"]) & (gaze["Time"] <= response_time)]
            idx_filter = filterFixation(gaze_t["Distance"].values <= AREA_FIX_TH)
            res = np.argmax(data[["PosResponse", "NegResponse", "MissResponse"]].values)

            # response type
            # 0: GoPos
            # 1: NoGoNeg
            # 2: NoNeg
            # 3: NoGoPos
            if (res == 0) & (data["ResponseTime"] == -1):
                res = 3
            for idx in idx_filter:
                if len(idx) > 2:
                    distances_time = gaze_t["Time"].values

                    t_response.append(responses[res])

                    #compute fixation time
                    fix_time = distances_time[np.max(idx)] - distances_time[np.min(idx)]
                    fixation_times.append(fix_time)
                    labels.append(typical_asd[class_idx])

    class_idx += 1

#convert results
t_response = np.array(t_response)
fixation_times = np.array(fixation_times) * 1000
labels = np.array(labels)

#prepare dataframe
d = {"Fixation_Times": fixation_times, "Type_Response": t_response, "Label": labels}
data = pd.DataFrame(d)

print(data.groupby(['Type_Response', 'Label']).mean())
print(data.groupby(['Type_Response', 'Label']).std())



plt.figure(1)

sns.boxplot(y="Fixation_Times", x="Type_Response", hue="Label",
                 data=data, palette="Set2").set_title('Fixation Time (s)')
plt.show()














