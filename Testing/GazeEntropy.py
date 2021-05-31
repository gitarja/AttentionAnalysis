import pandas as pd
import glob
import numpy as np
from Utils.DataReader import DataReader
from Conf.Settings import ASD_PATH, ASD_DW_PATH, ASD_DW_RESULTS_PATH, AREA_FIX_TH, CUT_OFF, TYPICAL_PATH, TYPICAL_DW_PATH, TYPICAL_DW_RESULTS_PATH, AVG_WIN_SIZE
from Utils.Lib import movingAverage
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.neighbors import KernelDensity
from matplotlib import cm
# instantiate and fit the KDE model
from nolds import sampen



gaze_paths = [TYPICAL_DW_PATH, ASD_DW_PATH]
#Reader and removal
reader = DataReader()

#edges
xedges, yedges = np.arange(0, 101, 1), np.arange(0, 101, 1)
for gaze_path in gaze_paths:
    data_gaze, data_gaze_obj, gaze_f_names = reader.readGazeData(gaze_path, downsample=True)
    print("------------------------------------------------------------------------")
    for gaze in data_gaze:
        gazex_avg = movingAverage(gaze["GazeX"].values, AVG_WIN_SIZE)
        gazey_avg = movingAverage(gaze["GazeY"].values, AVG_WIN_SIZE)
        # sampen_x = sampen(gazex_avg, 7)
        # sampen_y = sampen(gazey_avg, 7)


        H, _, _ = np.histogram2d(gazex_avg * 100, gazey_avg * 100, bins=(xedges, yedges), density=True)

        p = H.flatten()
        p = p[p>0]
        entropy = np.sum(p * np.log(1/p))
        # print("sampen_x: %f, sampen_y: %f, entropy: %f" % (sampen_x, sampen_y, entropy))


        xedges, yedges = np.meshgrid(np.arange(1, 101, 1), np.arange(1, 101, 1))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(xedges / 100, yedges / 100, H,cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        plt.show()