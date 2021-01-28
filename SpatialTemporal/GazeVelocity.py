import pandas as pd
from Utils.Lib import movingAverage, computeVelocity
import glob
import numpy as np
import matplotlib.pyplot as plt
from Conf.Settings import TYPICAL_DW_PATH, ASD_DW_PATH, AVG_WIN_SIZE, FREQ_GAZE
import seaborn as sns
sns.set_style("whitegrid")



paths = [TYPICAL_DW_PATH, ASD_DW_PATH]

velocity_x_typical = []
velocity_y_typical = []

velocity_x_asd = []
velocity_y_asd = []

i = 0
for path in paths:
    files = glob.glob(path + "*_gazeHeadPose_downsample.csv")
    for file in files:
        gaze_data = pd.read_csv(file)
        gaze_obj = gaze_data[ (gaze_data["Time"] >= 1) & (gaze_data["GazeX"] > 0)  & (gaze_data["GazeX"] > 0)]

        time_avg = movingAverage(gaze_obj["Time"].values, AVG_WIN_SIZE)
        gazex_avg = movingAverage(gaze_obj["GazeX"].values, AVG_WIN_SIZE)
        gazey_avg = movingAverage(gaze_obj["GazeY"].values, AVG_WIN_SIZE)
        freq = int(FREQ_GAZE / 10)
        if i  == 0:
            velocity_x_typical.append(computeVelocity(time_avg, gazex_avg, freq))
            velocity_y_typical.append(computeVelocity(time_avg, gazey_avg, freq))
        else:
            velocity_x_asd.append(computeVelocity(time_avg, gazex_avg, freq))
            velocity_y_asd.append(computeVelocity(time_avg, gazey_avg, freq))

    i+=1

velocity_x_typical = np.concatenate(velocity_x_typical)
velocity_y_typical = np.concatenate(velocity_y_typical)


velocity_x_asd = np.concatenate(velocity_x_asd)
velocity_y_asd = np.concatenate(velocity_y_asd)


#prepare the data and convert to MS


velocity_x = np.concatenate([velocity_x_typical, velocity_x_asd])
velocity_y = np.concatenate([velocity_y_typical, velocity_y_asd])

#Prepare labels
#x-axis
TYPICAL_x_LABELS = ["Typical" for x in range(len(velocity_x_typical))]
ASD_x_LABELS = ["ASD" for x in range(len(velocity_x_asd))]
velocity_x_labels = np.concatenate([TYPICAL_x_LABELS, ASD_x_LABELS])
#y-axis
TYPICAL_y_LABELS = ["Typical" for x in range(len(velocity_y_typical))]
ASD_y_LABELS = ["ASD" for x in range(len(velocity_y_asd))]
velocity_y_labels = np.concatenate([TYPICAL_y_LABELS, ASD_y_LABELS])

#Prepare dataframe

velocity_x_data = {"Velocity": velocity_x, "Label": velocity_x_labels}
velocity_x_data = pd.DataFrame(velocity_x_data)

velocity_y_data = {"Velocity": velocity_y, "Label": velocity_y_labels}
velocity_y_data = pd.DataFrame(velocity_y_data)

#plot the data

plt.figure(1)
# plt.ylim([0, 0.01])
sns.boxplot(y="Velocity", x="Label",
                 data=velocity_x_data, palette="Set2").set_title('Velocity X (norm pixel / 100 ms)')

plt.figure(2)
# plt.ylim([0, 0.01])
sns.boxplot(y="Velocity", x="Label",
                 data=velocity_y_data, palette="Set2").set_title('Velocity Y (norm pixel / 100 ms)')
plt.show()

#summarize the results
print("Typical-Average velocity x: %f, STD velocity x: %f, Average velocity y: %f, STD velocity y: %f " % (
np.average(velocity_x_typical), np.std(velocity_x_typical), np.average(velocity_y_typical), np.std(velocity_y_typical)))

print("ASD-Average velocity x: %f, STD velocity x: %f, Average velocity y: %f, STD velocity y: %f " % (
np.average(velocity_x_asd), np.std(velocity_x_asd), np.average(velocity_y_asd), np.std(velocity_y_asd)))