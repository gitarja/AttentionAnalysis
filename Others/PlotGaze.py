import matplotlib.pyplot as plt
import pandas as pd
from Conf.Settings import ASD_DW_PATH, TYPICAL_DW_PATH
import matplotlib
import numpy as np

matplotlib.use('Qt5Agg')
th_min = 132.
th_max = 177.

GAZE_PATH = "D:\\usr\\pras\\data\\AttentionTestData\\PerceptualLearning\\AttentionTestResults\\csv\\downsample\\"
# open dataset
typical_data = pd.read_csv(GAZE_PATH + "PT_000006_gazeHeadPose_downsample_avg.csv")
typical_data_sample = typical_data[(typical_data.Time >= th_min) & (typical_data.Time <= th_max)]

asd_data = pd.read_csv(GAZE_PATH + "PT_000007_gazeHeadPose_downsample_avg.csv")
asd_data_sample = asd_data[(asd_data.Time >= th_min) & (asd_data.Time <= th_max)]

# generate axes object
_, ax = plt.subplots(1, 2)

# set limits
ax[0].set_xlim(0, 1)
ax[0].set_ylim(0, 1)

ax[1].set_xlim(0, 1)
ax[1].set_ylim(0, 1)

i = 0
plt.pause(5)
l = 0
for i in range(np.minimum(len(typical_data_sample.index), len(asd_data_sample.index))):
    asd = asd_data_sample.iloc[i]
    typical = typical_data_sample.iloc[i]


    # add something to typical axes
    if typical["GazeX"] > 0:
        ax[0].plot(typical["GazeX"], typical["GazeY"], 'bo', markersize=2.5, label="Gaze" if i == 0 else "")

    if typical["ObjectX"] > 0:
        ax[0].plot(typical["ObjectX"], typical["ObjectY"], 'rx', label="Object" if i == 0 else "")


        # add something to asd axes
    if asd["GazeX"] > 0:
        ax[1].plot(asd["GazeX"], asd["GazeY"], 'bo', markersize=2.5, label="Gaze" if i == 0 else "")

    if asd["ObjectX"] > 0:
        ax[1].plot(asd["ObjectX"], asd["ObjectY"], 'rx', label="Object" if i == 0 else "")



    plt.draw()
    plt.pause(0.0005)  # is necessary for the plot to update for some reason

    if i == 1:
        # draw the plot
        ax[0].legend()
        ax[1].legend()

    # start removing points if you don't want all shown

    if len(ax[0].lines) > 20:
        for i in range(len(ax[0].lines) - 20):
            ax[0].lines[0].remove()
    if len(ax[0].collections) > 0:
        ax[0].collections[0].remove()

    if len(ax[1].lines) > 20:
        for i in range(len(ax[1].lines) - 20):
            ax[1].lines[0].remove()
    if len(ax[1].collections) > 0:
        ax[1].collections[0].remove()

