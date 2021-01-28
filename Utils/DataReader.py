import glob
import pandas as pd
import warnings
from Utils.Lib import timeDiff, euclidianDist
from joblib import Parallel, delayed

class DataReader:

    def readGameResults(self, data_path):
        '''
        :param data_path: path of game results
        :return: all game results file
        '''
        game_results = []
        file_names = []
        file_prefix = "_gameResults.csv"
        files = glob.glob(data_path + "*"+file_prefix)
        try:
            for file in files:
                game_result = pd.read_csv(file)
                game_result["RT"] = [timeDiff(game_result.loc[i]["ResponseTime"], game_result.loc[i]["SpawnTime"]) for i
                                     in range(len(game_result.index))]
                file_name = file.split(data_path)[-1]
                file_names.append(file_name.split(file_prefix)[0])
                game_results.append(game_result)

            return game_results, file_names
        except:
            warnings.simplefilter("cannot read files")
            return None

    def readGazeData(self, data_path, downsample=False):
        gaze_results = []
        gaze_obj_results = []
        file_names = []
        file_prefix = "_gazeHeadPose.csv"
        if downsample:
            file_prefix = "_gazeHeadPose_downsample_avg.csv"
        files = glob.glob(data_path + "*"+file_prefix)
        try:
            for file in files:
                gaze = pd.read_csv(file)
                gaze = gaze[(gaze["Time"].values > 0.1)]
                gaze_t = gaze[(gaze["ObjectX"].values != -1)& (gaze["ObjectY"].values != -1)].copy()
                gaze_t.loc[:, "Distance"] = euclidianDist(gaze_t[["GazeX", "GazeY"]].values, (gaze_t[["ObjectX", "ObjectY"]].values))
                file_name = file.split(data_path)[-1]
                file_names.append(file_name.split(file_prefix)[0])
                #add gaze and gaze-object to list
                gaze_obj_results.append(gaze_t)
                gaze_results.append(gaze)

            return gaze_results, gaze_obj_results, file_names
        except:
            warnings.simplefilter("cannot read files")
            return None
