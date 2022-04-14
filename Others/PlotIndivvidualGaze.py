import matplotlib.pyplot as plt
import pandas as pd
from Conf.Settings import ASD_DW_PATH, TYPICAL_DW_PATH
import matplotlib
import numpy as np

matplotlib.use('Qt5Agg')
th_min = 2.
th_max = 180.
# open dataset
typical_data = pd.read_csv("D:\\usr\\pras\\data\\AttentionTestData\\AttentionTest-2021-11-01\\csv_files\\ST_000002_gazeHeadPose.csv")
typical_data_sample = typical_data[(typical_data.Time >= th_min) & (typical_data.Time <= th_max)]

# generate axes object
_, ax = plt.subplots(1, 1)

# set limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)


i = 0
plt.pause(5)
l = 0
for i in range(len(typical_data_sample.index)):

    typical = typical_data_sample.iloc[i]


    # add something to typical axes
    if typical["GazeX"] > 0:
        ax.plot(typical["GazeX"], typical["GazeY"], 'bo', markersize=2.5, label="Gaze" if i == 0 else "")

    if typical["ObjectX"] > 0:
        ax.plot(typical["ObjectX"], typical["ObjectY"], 'rx', markersize=2.5, label="Object" if i == 0 else "")




    plt.draw()
    plt.pause(0.005)  # is necessary for the plot to update for some reason

    if i == 1:
        # draw the plot
        ax.legend()

    # start removing points if you don't want all shown
    if len(ax.lines) > 20:
        for i in range(len(ax.lines) - 20):
            ax.lines[0].remove()
    if len(ax.collections) > 0:
        ax.collections[0].remove()


