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
aic_lag = np.zeros(20)
bic_lag = np.zeros(20)

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
        gaze_t = gaze[(gaze["Time"] >= data["SpawnTime"]) & (gaze["Time"] <= response_time) & (gaze["ObjectX"] != -1) & (gaze["ObjectY"] != -1)]
        # distances = gaze_t["Distance"].values
        distances = euclidianDist(gaze_t[["GazeX", "GazeY"]].values, gaze_t[["ObjectX", "ObjectY"]].values)

        # response type
        res = np.argmax(data[["PosResponse", "NegResponse", "MissResponse"]].values)
        if (res == 0) & (data["ResponseTime"] == -1):
            res = 3

        if (len(distances) >= MIN_D_N) and (np.std(distances) != 0) :
            try:
                model = AR(distances)
                model_fitted = model.fit(ic="bic", maxiter=100)
                max_lag[model_fitted.k_ar] = max_lag[model_fitted.k_ar] + 1
                max_llf[model_fitted.k_ar] += model_fitted.llf
                aic_lag[model_fitted.k_ar] += model_fitted.aic
                bic_lag[model_fitted.k_ar] += model_fitted.bic
            except:
                print(distances)



optimal_lag = np.argmax(max_lag)
print(optimal_lag)
print(max_llf[optimal_lag] / np.max(max_lag))


print(max_lag)
print(max_llf / np.max(max_lag))
print(aic_lag / np.max(max_lag))
print(bic_lag / np.max(max_lag))

