import pandas as pd
from Utils.Lib import euclidianDist
import glob
import numpy as np
import matplotlib.pyplot as plt
import nolds


def relu(data, max=1.):
    data[data > max] = max
    return data


def anglesEstimation(data):
    angles = np.array(
        [np.dot(data[i, :], data[i + 1, :]) / (np.linalg.norm(data[i, :]) * np.linalg.norm(data[i + 1, :])) for i in
         range(len(data) - 1)])
    angles = np.rad2deg(np.arccos(relu(angles)))
    return angles

def sampleEntropy(data):
    return nolds.sampen(data)


paths = ["D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\Typical\\",
         "D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\High-risk\\"]

maxlag = 3
high_risk_en = []
typical_en = []
i = 0
for path in paths:

    files = glob.glob(path + "*_gazeHeadPose.csv")

    for file in files:

        filename = file.split(path)[1].split("_gazeHeadPose.csv")[0]
        filename_gaze = file

        filename_game = file.replace("_gazeHeadPose.csv", "_gameResults.csv")

        game_data = pd.read_csv(filename_game)

        gaze_data = pd.read_csv(filename_gaze)
        gaze_data = gaze_data.iloc[gaze_data.index % 2 == 1]

        gaze_obj = gaze_data[(gaze_data["ObjectX"] != -1) & (gaze_data["ObjectY"] != -1) & (gaze_data["GazeX"] >= 0) & (
                    gaze_data["GazeY"] >= 0)]
        avg_angles = []
        for _, data in game_data.iterrows():
            response_time = data["ResponseTime"]
            if (data["ResponseTime"] == -1) | (data["ResponseTime"] <= data["SpawnTime"]):
                response_time = data["SpawnTime"] + 0.7
            gaze_t = gaze_obj[(gaze_obj["Time"] >= data["SpawnTime"]) & (gaze_obj["Time"] <= response_time)]
            angles = anglesEstimation(gaze_t[["GazeX", "GazeY"]].values)
            if len(angles) > 1:
                avg_angles.append(angles)
        avg_angles = np.concatenate(avg_angles)
        if i == 0:
            typical_en.append(np.average(avg_angles))
        else:
            high_risk_en.append(np.average(avg_angles))

    i += 1

typical_en = np.array(typical_en)
high_risk_en = np.array(high_risk_en)

# plot the data
plt.figure(0)
fig1, ax1 = plt.subplots()
ax1.set_title('Angles')
ax1.boxplot(np.array([typical_en, high_risk_en]).transpose())

plt.xticks([1, 2], ["Typical", "Diagnosed"])
plt.figure(1)

plt.show()
# summarize the results
print("Typical-Average angles: %f, Typical-STD angles: %f, Diagnosed-Average angles: %f, Diagnosed-STD angles: %f " % (
    np.average(typical_en), np.std(typical_en), np.average(high_risk_en), np.std(high_risk_en)))
