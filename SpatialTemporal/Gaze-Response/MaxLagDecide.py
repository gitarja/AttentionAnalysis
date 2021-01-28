import pandas as pd
from Utils.Lib import euclidianDist
import glob
import numpy as np
from Utils.DataReader import DataReader
from Utils.Lib import arParams, createDir
from Utils.OutliersRemoval import OutliersRemoval
from Conf.Settings import ASD_PATH, ASD_DW_PATH, MIN_D_N, CUT_OFF, TYPICAL_PATH, TYPICAL_DW_PATH
from scipy import interpolate
from statsmodels.tsa.ar_model import AR


max_lag = np.zeros(20)
max_llf = np.zeros(20)

# read game data
reader = DataReader()
removal = OutliersRemoval(cutoff=CUT_OFF)
# read typical data
typical_game, _ = reader.readGameResults(TYPICAL_PATH)
typical_game = removal.transformGameResult(typical_game)
_, typical_gaze_obj, _ = reader.readGazeData(TYPICAL_DW_PATH, downsample=True)

# read asd data
asd_game, _ = reader.readGameResults(ASD_PATH)
asd_game = removal.transformGameResult(asd_game)
_, asd_gaze_obj, _ = reader.readGazeData(ASD_DW_PATH, downsample=True)

data_game = typical_game + asd_game
data_gaze = typical_gaze_obj + asd_gaze_obj

for d in zip(data_game, data_gaze):
    game = d[0]
    gaze = d[1]

    for _, data in game.iterrows():
        response_time = data["ResponseTime"]
        if (data["ResponseTime"] == -1) | (data["ResponseTime"] <= data["SpawnTime"]):
            response_time = data["SpawnTime"] + 0.7
        gaze_t = gaze[(gaze["Time"] >= data["SpawnTime"]) & (gaze["Time"] <= response_time)]
        distances = gaze_t["Distance"].values

        # response type
        res = np.argmax(data[["PosResponse", "NegResponse", "MissResponse"]].values)
        if (res == 0) & (data["ResponseTime"] == -1):
            res = 3

        if len(distances) >= MIN_D_N:
                model = AR(distances)
                model_fitted = model.fit( ic="aic", maxiter=200)
                max_lag[model_fitted.k_ar] = max_lag[model_fitted.k_ar] + 1
                max_llf[model_fitted.k_ar] += model_fitted.llf



optimal_lag = np.argmax(max_lag)
print(optimal_lag)
print(max_llf[optimal_lag] / np.max(max_lag))

