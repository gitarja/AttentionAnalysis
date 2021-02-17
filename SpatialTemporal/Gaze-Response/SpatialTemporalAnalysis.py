import pandas as pd
import glob
import numpy as np
from Utils.DataReader import DataReader, euclidianDist
from Utils.Lib import arParams, createDir, computeCutoff
from Utils.OutliersRemoval import OutliersRemoval
from Conf.Settings import ASD_PATH, ASD_DW_PATH, ASD_DW_RESULTS_PATH, MIN_D_N, MAX_LAG, CUT_OFF, TYPICAL_PATH, TYPICAL_DW_PATH, TYPICAL_DW_RESULTS_PATH, MIN_D_N
import warnings


#set the processing
#ASD or TYPICAL
game_paths = [TYPICAL_PATH, ASD_PATH]
gaze_paths = [TYPICAL_DW_PATH, ASD_DW_PATH]
result_paths = [TYPICAL_DW_RESULTS_PATH, ASD_DW_RESULTS_PATH]

RT = []

#Reader and removal
reader = DataReader()
removal = OutliersRemoval(cutoff=CUT_OFF)
for path in zip(game_paths, gaze_paths, result_paths):
    game_path = path[0]
    gaze_path = path[1]
    result_path = path[2]

    #read game data
    reader = DataReader()
    data_game, game_f_names = reader.readGameResults(game_path) #read game results
    data_game = removal.transformGameResult(data_game) #remove outlier
    data_gaze, data_gaze_obj, gaze_f_names = reader.readGazeData(gaze_path, downsample=True) #read gaze data


    if not game_f_names.__eq__(gaze_f_names):
        warnings.simplefilter("Data do not match")


    for d in zip(data_game, data_gaze_obj, game_f_names):
        game = d[0]
        gaze = d[1]
        f_name = d[2]

        # samp_e: sample entropy
        # max_dist: maximum distance within that period
        # mid_dist: minimum distance within that period
        # avg_dist: avg distance within that period
        # std_dist: std distance within that period
        # ar_params: autoregressive params representing the distances
        # t_response: type of response
        samp_e_list = []
        max_dist = []
        min_dist = []
        avg_dist = []
        std_dist = []
        last_dist = []
        ar_params = []
        t_response = []

        for _, data in game.iterrows():

            response_time = data["ResponseTime"]
            if (data["ResponseTime"] == -1) | (data["ResponseTime"] <= data["SpawnTime"]):
                # when the response is NoGo Positive or Go Negative assume that the subject look at the
                # stimulus for 700 ms

                response_time = data["SpawnTime"] + 0.7
            gaze_t = gaze[
                (gaze["Time"] >= data["SpawnTime"]) & (gaze["Time"] <= response_time) & (gaze["ObjectX"] != -1) & (
                            gaze["ObjectY"] != -1)]
            # distances = gaze_t["Distance"].values #distance between gaze and object
            distances = euclidianDist(gaze_t[["GazeX", "GazeY"]].values, gaze_t[["ObjectX", "ObjectY"]].values) #distance between gaze and object
            times = gaze_t["Time"].values
            # only proceed the distance longer than threshold
            # shorter distance may cause error
            if len(distances) >= MIN_D_N:

                #response type
                # 0: GoPos
                # 1: NoGoNeg
                # 2: NoNeg
                # 3: NoGoPos
                res = np.argmax(data[["PosResponse", "NegResponse", "MissResponse"]].values)
                if (res == 0) & (data["ResponseTime"] == -1):
                    res = 3
                t_response.append(res)

                max_dist.append(np.max(distances))
                min_dist.append(np.min(distances))
                avg_dist.append(np.average(distances))
                std_dist.append(np.std(distances))
                last_dist.append(distances[-1])

                try:
                    #fit the data into the model
                    params, loglike_score = arParams(distances, times=times, min_len=MIN_D_N, maxlag=MAX_LAG)
                    mean_coeff = np.average(params)
                    ar_params.append(params)
                except:
                    #when fail set to zeros and will not be proceed further
                    mean_coeff = 0
                    ar_params.append([0, 0, 0])
                    samp_e_list.append(0)

        # save results
        createDir(result_path)
        np.save(result_path + f_name + "_max_dsit.npy", np.array(max_dist))
        np.save(result_path + f_name + "_min_dist.npy", np.array(min_dist))
        np.save(result_path + f_name + "_avg_dist.npy", np.array(avg_dist))
        np.save(result_path + f_name + "_std_dist.npy", np.array(std_dist))
        np.save(result_path + f_name + "_last_dist.npy", np.array(last_dist))
        np.save(result_path + f_name + "_ar_params.npy", np.array(ar_params))
        np.save(result_path + f_name + "_responses.npy", np.array(t_response))
















