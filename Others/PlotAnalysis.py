import pandas as pd
import os
from Utils.Lib import createDir, computeVelocity, euclidianDistT, anglesEstimation, removeNoObjectData, gazeEntropy, \
    spectralEntropy, computeAcceleration
from Spatial.Libs.DataProcessor import DataProcessor
from Utils.DataReader import DataReader
import numpy as np
from Utils.OutliersRemoval import OutliersRemoval
from Conf.Settings import ASD_PATH, ASD_DW_PATH, ASD_DW_RESULTS_PATH, MIN_D_N, MAX_LAG, CUT_OFF, TYPICAL_PATH, \
    TYPICAL_DW_PATH, TYPICAL_DW_RESULTS_PATH, AVG_WIN_SIZE, FREQ_GAZE, AREA_FIX_TH, PLOT_PATH
from Utils.Lib import filterFixation
import matplotlib.pyplot as plt
from nolds import sampen
from sklearn.neighbors import KernelDensity
import warnings
from scipy.stats import gaussian_kde as kde

# set the processing
# ASD or TYPICAL
game_paths = [TYPICAL_PATH, ASD_PATH]
gaze_paths = [TYPICAL_DW_PATH, ASD_DW_PATH]
result_paths = [TYPICAL_DW_RESULTS_PATH, ASD_DW_RESULTS_PATH]

# Reader and removal
reader = DataReader()
removal = OutliersRemoval(cutoff=CUT_OFF)

# edges
xedges, yedges = np.arange(0, 101, 1), np.arange(0, 101, 1)
# x_grid, y_grid = np.mgrid[0:1:100j, 0:1:100j]
# positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
for path in zip(game_paths, gaze_paths, result_paths):
    game_path = path[0]
    gaze_path = path[1]
    result_path = path[2]

    # read game data

    data_game, game_f_names = reader.readGameResults(game_path)
    data_game = removal.transformGameResult(data_game)  # remove response with RT less that TH
    data_gaze, data_gaze_obj, gaze_f_names = reader.readGazeData(gaze_path, downsample=True)

    # data process
    process = DataProcessor()

    # dataframe for the results

    columns = ["id", "Go", "GoError", "NoGo",
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
               "Spatial_entropy",
               "GazeObj_entropy",
               "Sampen_gaze_obj",
               "Spectral_entropy",
               "Sampen_velocity",
               "Sampen_acceleration"]

    data = pd.DataFrame(columns=columns)
    i = 0
    for d in zip(data_game, data_gaze, data_gaze_obj, game_f_names):
        game = d[0]  # game data
        gaze_data = d[1]  # all gaze data
        gaze_obj_data = d[2]  # gaze data when stimulus appear
        f_name = d[3]  # file name

        area = process.convexHullArea(gaze_data)  # compute gaze area

        # compute RT, RTVar, Correct and Incorrect Percentage
        response = game[game["ResponseTime"] != -1]
        Go_response = len(response[response["PosResponse"] == 1]) / len(game)
        NoGo_E_reponse = len(game[game["NegResponse"] == 1]) / len(game)
        Go_E_reponse = len(game[game["MissResponse"] == 1]) / len(game)
        NoGo_reponse = len(game[(game["PosResponse"] == 1) & (game["ResponseTime"] == -1)]) / len(game)

        RT = process.computeResponseTimeGroup(response).transpose().flatten()
        RT_AVG, RT_Var_AVG = np.mean(response.RT[response.RT >= 0]), np.std(response.RT[response.RT >= 0])

        # Save response time
        RT_series = pd.DataFrame({"Time": response["SpawnTime"].values, "RT": response["RT"].values})
        RT_series.to_csv(result_path + f_name + "_RTSeries.csv")

        # compute gaze velocity
        skip = 1
        time = gaze_data["Time"].values
        gazex = gaze_data["GazeX"].values
        gazey = gaze_data["GazeY"].values
        gaze_avg = np.array([gazex, gazey]).transpose()
        velocity = gaze_data["Velocity"].values
        acceleration = gaze_data["Acceleration"].values

        #plot gaze
        # plt.plot(gazex, gazey, linewidth=0.75, c="k")
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # plt.savefig(PLOT_PATH + f_name + ".eps")

        # compute sample entropy and angle (1e-25 to avoid NAN)

        # dist_avg = euclidianDistT(gaze_avg, skip=2)  # compute euclidian distance for consecutive gaze
        # angle_avg = anglesEstimation(gaze_avg, skip=2)  # compute angle distance for consecutive gaze

        # plot angle_avg
        # plt.plot(acceleration[0:720], linewidth=0.75, c="k")
        # plt.ylim([0, 0.12])
        # plt.savefig(PLOT_PATH + f_name + ".eps")

        #
        # # compute gaze-object-entropy
        gaze_obj = removeNoObjectData(
            gaze_obj_data[['Time', 'GazeX', 'GazeY', 'ObjectX', 'ObjectY']].dropna().to_numpy())
        gaze_point = gaze_obj[:, 1:3]
        obj_point = gaze_obj[:, 3:]
        # gaze_obj_en = gazeEntropy(gaze_point - obj_point)
        # spectral_entropy = spectralEntropy(gaze_point - obj_point)
        xy = gaze_point - obj_point
        #plot gaze-to-obj

        # plt.plot(xy, linewidth=0.75, c="k")
        # plt.ylim([-1, 1])
        # plt.savefig(PLOT_PATH + f_name + ".eps")



        #density
        from mpl_toolkits.mplot3d import axes3d
        from matplotlib import cm
        import matplotlib

        matplotlib.use('Qt5Agg')
        est = kde(xy.transpose())
        xgrid, ygrid = np.mgrid[-1:1:51j, -1:1:51j]
        H = np.array([est.pdf([x, y]) for (x, y) in zip(xgrid, ygrid)]).ravel()
        H = H.reshape(51, 51)
        fig = plt.figure()
        ax = fig.add_subplot()
        # c = ax.contourf(xgrid, ygrid, H / 35.,  vmin=0., vmax=1.) #the max val is about 35
        c = ax.pcolormesh(xgrid, ygrid, H / 35, cmap=cm.coolwarm,
                         vmin=0., vmax=1.)


        fig.colorbar(c, ax=ax)
        plt.savefig(PLOT_PATH + f_name + ".png")



        plt.close()



        #
        # #  start compute fixation time avg sample dist and sample angle
        # fixation_times = []
        # sampen_gaze_objs = []
        # for _, g in game.iterrows():
        #     response_time = g["ResponseTime"]
        #     if (g["ResponseTime"] == -1) | (g["ResponseTime"] <= g["SpawnTime"]):
        #         response_time = g["SpawnTime"] + 0.7
        #     # take gaze based on response time
        #     gaze_t = gaze_obj_data[(gaze_obj_data["Time"] >= g["SpawnTime"]) & (gaze_obj_data["Time"] <= response_time)]
        #     idx_filter = filterFixation(gaze_t["Distance"].values <= AREA_FIX_TH)
        #
        #     # compute sample entropy
        #     gaze_samp_avg = np.array([gaze_t["GazeX"].values, gaze_t["GazeY"].values]).transpose()
        #     obj_samp_avg = np.array([gaze_t["ObjectX"].values, gaze_t["ObjectY"].values]).transpose()
        #     gaze_to_obj = np.linalg.norm(gaze_samp_avg - obj_samp_avg, axis=-1)
        #
        #     # the embeding is 6 so it requires minimum lenth of 8
        #     if len(gaze_to_obj) > 10:
        #         sampen_gaze_obj = sampen(gaze_to_obj, 2)  # sample entropy of gaze-to-obj
        #         if np.isinf(sampen_gaze_obj) == False:
        #             sampen_gaze_objs.append(sampen_gaze_obj)
        #     # end compute the avg sample entropy
        #
        #     for idx in idx_filter:
        #         if len(idx) > 2:
        #             distances_time = gaze_t["Time"].values
        #             # compute fixation time
        #             fix_time = distances_time[np.max(idx)] - distances_time[np.min(idx)]
        #             fixation_times.append(fix_time)
        # fixation_times = np.array(fixation_times) * 1000
        # #  end compute fixation time
        # if (len(sampen_gaze_objs) == 0):
        #     print(f_name)
        #
