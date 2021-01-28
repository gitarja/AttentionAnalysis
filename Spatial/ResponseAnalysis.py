import pandas as pd
import os
from Utils.Lib import createDir, computeVelocity, movingAverage, euclidianDistT, anglesEstimation, removeNoObjectData, gazeEntropy, spectralEntropy
from Spatial.Libs.DataProcessor import DataProcessor
from Utils.DataReader import DataReader
import numpy as np
from Utils.OutliersRemoval import OutliersRemoval
from Conf.Settings import ASD_PATH, ASD_DW_PATH, ASD_DW_RESULTS_PATH, MIN_D_N, MAX_LAG, CUT_OFF, TYPICAL_PATH, \
    TYPICAL_DW_PATH, TYPICAL_DW_RESULTS_PATH, AVG_WIN_SIZE, FREQ_GAZE, AREA_FIX_TH
from Utils.Lib import filterFixation
import matplotlib.pyplot as plt
from nolds import sampen
from sklearn.neighbors import KernelDensity
import warnings

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
x_grid, y_grid = np.mgrid[0:1:100j, 0:1:100j]
positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
for path in zip(game_paths, gaze_paths, result_paths):
    game_path = path[0]
    gaze_path = path[1]
    result_path = path[2]

    # read game data

    data_game, game_f_names = reader.readGameResults(game_path)
    data_game = removal.transformGameResult(data_game)
    data_gaze, data_gaze_obj, gaze_f_names = reader.readGazeData(gaze_path, downsample=True)

    # data process
    process = DataProcessor()

    # dataframe for the results

    columns = ["id", "Go", "GoError", "NoGo", "NoGoError", "RT", "RTVar", "Trajectory Area", "VelocityX_avg",
               "VelocityY_avg", "VelocityX_std", "VelocityY_std", "Fixation_avg", "Fixation_std", "Sampen_dist",
               "Sampen_angle", "Spatial_entropy", "GazeObj_entropy",  "Sampen_gaze_obj", "Spectral_entropy"]

    data = pd.DataFrame(columns=columns)
    i = 0
    for d in zip(data_game, data_gaze, data_gaze_obj, game_f_names):
        game = d[0]
        gaze_data = d[1]
        gaze_obj_data = d[2]
        f_name = d[3]

        area = process.convexHullArea(gaze_data)

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
        freq = int(FREQ_GAZE / 10)
        time = gaze_data["Time"].values
        gazex = gaze_data["GazeX"].values
        gazey = gaze_data["GazeY"].values
        velocity_x = computeVelocity(time, gazex, freq)
        velocity_y = computeVelocity(time, gazey, freq)

        # compute sample entropy
        gaze_avg = np.array([gazex, gazey]).transpose()
        dist_avg = euclidianDistT(gaze_avg, skip=2)
        angle_avg = anglesEstimation(gaze_avg, skip=2)

        sampen_dist = sampen(dist_avg, freq) + 1e-25
        sampen_angle = sampen(angle_avg, freq) + 1e-25

        # compute entropy
        H, _, _ = np.histogram2d(gazex * 100, gazey * 100, bins=(xedges, yedges), density=True)

        p = H.flatten()
        p = p[p > 0]
        spatial_entropy = np.sum(p * np.log(1 / p)) / np.log(len(p))

        #compute gaze-object-entropy
        gaze_obj = removeNoObjectData(gaze_obj_data[['Time', 'GazeX', 'GazeY', 'ObjectX', 'ObjectY']].dropna().to_numpy())
        gaze_point = gaze_obj[:, 1:3]
        obj_point = gaze_obj[:, 3:]
        gaze_obj_en = gazeEntropy(gaze_point - obj_point)


        #  start compute fixation time avg sample dist and sample angle
        fixation_times = []
        sampen_gaze_objs = []
        spectral_entropys = []
        for _, g in game.iterrows():
            response_time = g["ResponseTime"]
            if (g["ResponseTime"] == -1) | (g["ResponseTime"] <= g["SpawnTime"]):
                response_time = g["SpawnTime"] + 0.7
            # take gaze based on response time
            gaze_t = gaze_obj_data[(gaze_obj_data["Time"] >= g["SpawnTime"]) & (gaze_obj_data["Time"] <= response_time)]
            idx_filter = filterFixation(gaze_t["Distance"].values <= AREA_FIX_TH)

            #compute sample entropy
            gaze_samp_avg = np.array([gaze_t["GazeX"].values, gaze_t["GazeY"].values]).transpose()
            obj_samp_avg = np.array([gaze_t["ObjectX"].values, gaze_t["ObjectY"].values]).transpose()
            gaze_to_obj = np.linalg.norm(gaze_samp_avg - obj_samp_avg, axis=-1)

            #the embeding is 6 so it requires minimum lenth of 8
            if len(gaze_to_obj) > 10:
                sampen_gaze_obj = sampen(gaze_to_obj, freq) + 1e-25
                spectral_entropy  = spectralEntropy(gaze_samp_avg - obj_samp_avg)
                if np.isinf(sampen_gaze_obj) == False:
                    sampen_gaze_objs.append(sampen_gaze_obj)
                    spectral_entropys.append(spectral_entropy)




            #end compute the avg sample entropy

            for idx in idx_filter:
                if len(idx) > 2:
                    distances_time = gaze_t["Time"].values
                    # compute fixation time
                    fix_time = distances_time[np.max(idx)] - distances_time[np.min(idx)]
                    fixation_times.append(fix_time)
        fixation_times = np.array(fixation_times) * 1000
        #  end compute fixation time
        if len (sampen_gaze_objs) == 0:
            print(f_name)
        if len(RT) > 2:
            data = data.append(
                {"id": f_name, "Go": Go_response, "GoError": Go_E_reponse, "NoGo": NoGo_reponse,
                 "NoGoError": NoGo_E_reponse, "RT": RT_AVG, "RTVar": RT_Var_AVG, "Trajectory Area": area,
                 "VelocityX_avg": np.average(velocity_x),
                 "VelocityY_avg": np.average(velocity_y),
                 "VelocityX_std": np.std(velocity_x),
                 "VelocityY_std": np.std(velocity_y),
                 "Fixation_avg": np.average(fixation_times),
                 "Fixation_std": np.std(fixation_times),
                 "Sampen_dist": sampen_dist,
                 "Sampen_angle": sampen_angle,
                 "Spatial_entropy": spatial_entropy,
                 "GazeObj_entropy": gaze_obj_en,
                 "Sampen_gaze_obj": np.average(sampen_gaze_objs),
                 "Spectral_entropy" : np.average(spectral_entropys),
                 }, ignore_index=True)

            i += 1

    result_summary_path = result_path + "summary\\"
    createDir(result_summary_path)
    data.to_csv(os.path.join(result_summary_path, "summary_response_new.csv"), columns=columns, index=False)
