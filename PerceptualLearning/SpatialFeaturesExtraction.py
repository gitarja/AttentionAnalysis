import pandas as pd
import os
from Utils.Lib import createDir, computeVelocity, euclidianDistT, anglesEstimation, removeNoObjectData, gazeEntropy, \
    spectralEntropy, computeAcceleration
from Spatial.Libs.DataProcessor import DataProcessor
from Utils.DataReader import DataReader
import numpy as np
from Utils.OutliersRemoval import OutliersRemoval
from Utils.Lib import filterFixation
from Conf.Settings import CUT_OFF, AREA_FIX_TH
from nolds import sampen


#New data
game_paths = ["D:\\usr\\pras\\data\\AttentionTestData\\PerceptualLearning\\AttentionTestResults\\csv\\"]
gaze_paths = ["D:\\usr\\pras\\data\\AttentionTestData\\PerceptualLearning\\AttentionTestResults\\csv\\downsample\\"]
result_paths = ["D:\\usr\\pras\\data\\AttentionTestData\\PerceptualLearning\\AttentionTestResults\\csv\\summary\\"]

#session
num_stimulus = 9 #number of stimulus in half of trial


# Reader and removal
reader = DataReader()
removal = OutliersRemoval(cutoff=CUT_OFF)


# edges
xedges, yedges = np.arange(0, 101, 1), np.arange(0, 101, 1)
# x_grid, y_grid = np.mgrid[0:1:100j, 0:1:100j]
# positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
all_features = []
for path in zip(game_paths, gaze_paths, result_paths):
    game_path = path[0]
    gaze_path = path[1]
    result_path = path[2]
    print(game_path)


    # read game data

    data_game, game_f_names = reader.readGameResults(game_path)
    data_game = removal.transformGameResult(data_game)  # remove response with RT less that TH
    data_gaze, data_gaze_obj, gaze_f_names = reader.readGazeData(gaze_path, downsample=True)

    # data process
    process = DataProcessor()

    for d in zip(data_game, data_gaze, data_gaze_obj, game_f_names):
        game = d[0]  # game data
        gaze_data = d[1]  # all gaze data
        gaze_obj_data = d[2]  # gaze data when stimulus appear
        f_name = d[3]  # file name

        area = process.convexHullArea(gaze_data)  # compute gaze area

        # compute RT, RTVar, Correct and Incorrect Percentage
        response = game[game["ResponseTime"] != -1]

        RT = process.computeResponseTimeSession(response, num_stimulus=num_stimulus).transpose().flatten() * 1000
        gaze_related_features = []
        for i in range(len(game.index)//num_stimulus):
            time_start = game.iloc[i]["SpawnTime"]
            time_end = game.iloc[((i+1) * num_stimulus) - 1]["SpawnTime"]

            gaze_data_session = gaze_data[(gaze_data["Time"] >= time_start) & (gaze_data["Time"] < time_end)]
            gaze_obj_data_session = gaze_obj_data[(gaze_obj_data["Time"] >= time_start) & (gaze_obj_data["Time"] < time_end)]

            # compute gaze velocity
            skip = 1
            time = gaze_data_session["Time"].values
            gazex = gaze_data_session["GazeX"].values
            gazey = gaze_data_session["GazeY"].values
            gaze_avg = np.array([gazex, gazey]).transpose()
            velocity = gaze_data_session["Velocity"].values
            acceleration = gaze_data_session["Acceleration"].values
            sampen_velocity = sampen(velocity, 2)  # computed sample entropy of gaze velocity
            sampen_acceleration = sampen(acceleration, 2)  # compute sample entropy of gaze acceleration

            # compute sample entropy and angle (1e-25 to avoid NAN)

            dist_avg = euclidianDistT(gaze_avg, skip=2)  # compute euclidian distance for consecutive gaze
            angle_avg = anglesEstimation(gaze_avg, skip=2)  # compute angle distance for consecutive gaze

            # compute sample entropy of gaze distance
            sampen_dist = sampen(dist_avg, 2)
            sampen_angle = sampen(angle_avg, 2)

            # compute spatial entropy
            H, _, _ = np.histogram2d(gazex * 100, gazey * 100, bins=(xedges, yedges), density=True)

            p = H.flatten()
            p = p[p > 0]
            spatial_entropy = np.sum(p * np.log(1 / p)) / np.log(len(p))

            # compute gaze-object-entropy
            gaze_obj = removeNoObjectData(
                gaze_obj_data_session[['Time', 'GazeX', 'GazeY', 'ObjectX', 'ObjectY']].dropna().to_numpy())
            gaze_point = gaze_obj[:, 1:3]
            obj_point = gaze_obj[:, 3:]
            gaze_obj_en = gazeEntropy(gaze_point - obj_point)
            spectral_entropy = spectralEntropy(gaze_point - obj_point)

            #  start compute fixation time avg sample dist and sample angle
            fixation_times = []
            sampen_gaze_objs = []
            for _, g in game.iterrows():
                response_time = g["ResponseTime"]
                if (g["ResponseTime"] == -1) | (g["ResponseTime"] <= g["SpawnTime"]):
                    response_time = g["SpawnTime"] + 0.7
                # take gaze based on response time
                gaze_t = gaze_obj_data_session[
                    (gaze_obj_data_session["Time"] >= g["SpawnTime"]) & (gaze_obj_data_session["Time"] <= response_time)]
                idx_filter = filterFixation(gaze_t["Distance"].values <= AREA_FIX_TH)

                # compute sample entropy
                gaze_samp_avg = np.array([gaze_t["GazeX"].values, gaze_t["GazeY"].values]).transpose()
                obj_samp_avg = np.array([gaze_t["ObjectX"].values, gaze_t["ObjectY"].values]).transpose()
                gaze_to_obj = np.linalg.norm(gaze_samp_avg - obj_samp_avg, axis=-1)

                # the embeding is 6 so it requires minimum lenth of 8
                if len(gaze_to_obj) > 10:
                    sampen_gaze_obj = sampen(gaze_to_obj, 2)  # sample entropy of gaze-to-obj
                    if np.isinf(sampen_gaze_obj) == False:
                        sampen_gaze_objs.append(sampen_gaze_obj)
                # end compute the avg sample entropy

                for idx in idx_filter:
                    if len(idx) > 2:
                        distances_time = gaze_t["Time"].values
                        # compute fixation time
                        fix_time = distances_time[np.max(idx)] - distances_time[np.min(idx)]
                        fixation_times.append(fix_time)
            fixation_times = np.array(fixation_times) * 1000

            # gaze_related_features.append([np.average(acceleration), np.std(acceleration), np.average(fixation_times), np.std(fixation_times), np.average(dist_avg), np.std(dist_avg), np.average(angle_avg), np.std(angle_avg), sampen_dist, sampen_angle, spatial_entropy, gaze_obj_en, np.average(sampen_gaze_objs), spectral_entropy, sampen_velocity, sampen_acceleration])
            gaze_related_features.append([np.std(fixation_times), np.average(sampen_gaze_objs), np.average(acceleration)])
        gaze_related_features = np.array(gaze_related_features).transpose().flatten()
        all_features.append([np.concatenate([RT, gaze_related_features])])


    all_features = np.concatenate(all_features)
    # save features
    result_summary_path = result_path
    createDir(result_summary_path)
    np.savetxt(result_path+"summary.csv", all_features, delimiter=",")


