import glob
import pandas as pd
import yaml
from Utils.Lib import computeVelocity, computeAcceleration, computeVelocityAccel
from Utils.Lib import movingAverage
from Conf.Settings import AVG_WIN_SIZE
from scipy.signal._savitzky_golay import savgol_filter
import matplotlib.pyplot as plt
import numpy as np



def powerPlot(x_ori, x_filtered, freq, title):

    freq = 1 / freq
    # x_ori
    ps_ori = np.abs(np.fft.fft(x_ori ** 2))
    freqs_ori = np.fft.fftfreq(x_ori.size, freq)
    idx_ori = np.argsort(freqs_ori)

    # x_filtered
    ps_filt = np.abs(np.fft.fft(x_filtered ** 2))
    freqs_filt = np.fft.fftfreq(x_filtered.size, freq)
    idx_filt = np.argsort(freqs_filt)


    plt.plot(freqs_ori[idx_ori], ps_ori[idx_ori], label="original")
    plt.plot(freqs_filt[idx_filt], ps_filt[idx_filt], label="filtered")

    plt.legend(loc="upper right")
    plt.title(title)


def plotVelocityAcc(time, x_ori, title):
    skip =1
    # x_ori
    velocity_ori, time_velocity_ori = computeVelocity(time, x_ori,
                                              n=skip, time_constant=True)  # velocity axis for every 55 ms (Ref: eye movement Frederick Bartlet)
    acceleration_ori = computeAcceleration(time_velocity_ori, velocity_ori, n=skip, time_constant=True)  # compute acceleration


    # x_filtered
    # first derivative
    gaze_x_d1 = savgol_filter(x_ori[:, 0], AVG_WIN_SIZE, polyorder=3, deriv=1)
    gaze_y_d1 = savgol_filter(x_ori[:, 1], AVG_WIN_SIZE, polyorder=3, deriv=1)

    gaze_d1 =np.array([gaze_x_d1, gaze_y_d1]).transpose()

    #second derivative
    gaze_x_d2 = savgol_filter(x_ori[:, 0], AVG_WIN_SIZE, polyorder=3, deriv=2)
    gaze_y_d2 = savgol_filter(x_ori[:, 1], AVG_WIN_SIZE, polyorder=3, deriv=2)

    gaze_d2 = np.array([gaze_x_d2, gaze_y_d2]).transpose()

    velocity_filtered = np.sqrt(np.sum(np.power(gaze_d1, 2), -1))

    acceleration_filtered = np.sqrt(np.sum(np.power(gaze_d2, 2), -1))

    plt.figure(1)
    plt.plot(velocity_ori, label="Velocity original")
    plt.plot(velocity_filtered, label="Velocity filtered")
    plt.legend(loc="upper right")
    plt.title(title)

    plt.figure(2)
    plt.plot(acceleration_ori, label="Acceleration original")
    plt.plot(acceleration_filtered, label="Acceleration filtered")

    plt.legend(loc="upper right")
    plt.title(title)


    #power
    plt.figure(3)
    powerPlot(np.array(velocity_ori), np.array(velocity_filtered), 72, title)
    plt.figure(4)
    powerPlot(np.array(acceleration_ori), np.array(acceleration_filtered), 72, title)




def gazePlot(x_ori, x_iltered, title):


    plt.plot(x_ori, label="original")
    plt.plot(x_iltered, label="filtered")
    plt.legend(loc="upper right")
    plt.title(title)
    plt.close()




paths = ["D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\Typical-Hoikuen\\", "D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\High-risk\\"]
# paths = ["D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\Typical_Hoikuen\\"]

normalize_col = ["ObjectX","ObjectY"]

for path in paths:
    files = glob.glob(path + "*_gazeHeadPose.csv")

    for file in files:
        print(file.split("\\")[-1])

        data = pd.read_csv(file)
        data = data[data["Time"].values > 1]
        down_sample = data.loc[data.index % 2 == 1]

        down_avg_data = pd.DataFrame(columns=down_sample.columns.values)
        down_sample = down_sample[(down_sample.GazeX >= 0) & (down_sample.GazeY >= 0)]
        for col in down_sample:

            #applying savgol filter
            #window= 3 samples (3/72 = 41.6 ms) with poly order = 2
            #ref:https://doi.org/10.3758/BRM.42.1.188

            if col in normalize_col:
                down_avg_data[col] = down_sample[col].values
            else:
                down_avg_data[col] = savgol_filter(down_sample[col].values, AVG_WIN_SIZE, polyorder=3)


        # powerPlot(down_sample["GazeX"].values[1000:1200], down_avg_data["GazeX"].values[1000:1200], 72, "n=3, poly=2") #power plot
        # gaze_avg = np.array([down_sample["GazeX"].values[1000:1200], down_sample["GazeY"].values[1000:1200]]).transpose()
        # gaze_avg_f = np.array(
        #     [down_avg_data["GazeX"].values[1000:1200], down_avg_data["GazeY"].values[1000:1200]]).transpose()
        # plotVelocityAcc(down_sample["Time"].values[1000:1200], gaze_avg, "n=3, poly=2") #velocity and acc plot

        #remove object < 0
        removed_idx = (down_avg_data["ObjectX"].values < 0.1) | (down_avg_data["ObjectY"].values < 0.1)
        down_avg_data["ObjectX"].loc[removed_idx] = -1
        down_avg_data["ObjectY"].loc[removed_idx] = -1

        #compute velocity and acceleration using Savgol
        gaze_avg = np.array([down_sample["GazeX"].values, down_sample["GazeY"].values]).transpose()
        velocity, acceleration = computeVelocityAccel(down_sample["Time"].values, gaze_avg, AVG_WIN_SIZE, 3)
        down_avg_data["Velocity"] = velocity
        down_avg_data["Acceleration"] = acceleration

        down_sample_file = file.replace("_gazeHeadPose.csv", "_gazeHeadPose_downsample_avg.csv")
        down_avg_data.to_csv(down_sample_file, index=False, float_format='%.7f') #save data


# paths = ["D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\Typical_Hoikuen\\GameResults\\"]
#
# for path in paths:
#     files = glob.glob(path + "*.yml")
#
#     for file in files:
#         with codecs.open(file, 'r', 'utf-8') as f:
#             parsed_yaml_file = yaml.load(f)
#             print(parsed_yaml_file["id"] +","+ parsed_yaml_file["sex"])
