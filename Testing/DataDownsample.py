import glob
import pandas as pd
import yaml
import codecs
from Utils.Lib import movingAverage
from Conf.Settings import AVG_WIN_SIZE

paths = ["D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\Typical-Hoikuen\\", "D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\High-risk\\"]
# paths = ["D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\Typical_Hoikuen\\"]

normalize_col = ["Time","GazeX","GazeY","ObjectX","ObjectY"]
for path in paths:
    files = glob.glob(path + "*_gazeHeadPose.csv")

    for file in files:
        print(file.split("\\")[-1])

        data = pd.read_csv(file)
        data = data[data["Time"].values > 1]
        down_sample = data.loc[data.index % 2 == 1]

        down_avg_data = pd.DataFrame(columns=down_sample.columns.values)
        for col in down_sample:
            down_sample = down_sample[(down_sample.GazeX >= 0) & (down_sample.GazeY >= 0)]
            if col in normalize_col:
                down_avg_data[col] = movingAverage(down_sample[col].values, AVG_WIN_SIZE, remove_neg=True)
            else:
                down_avg_data[col] = movingAverage(down_sample[col].values, AVG_WIN_SIZE, remove_neg=False)
        down_sample_file = file.replace("_gazeHeadPose.csv", "_gazeHeadPose_downsample_avg.csv")
        down_avg_data.to_csv(down_sample_file, index=False, float_format='%.7f')


# paths = ["D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\Typical_Hoikuen\\GameResults\\"]
#
# for path in paths:
#     files = glob.glob(path + "*.yml")
#
#     for file in files:
#         with codecs.open(file, 'r', 'utf-8') as f:
#             parsed_yaml_file = yaml.load(f)
#             print(parsed_yaml_file["id"] +","+ parsed_yaml_file["sex"])
