import cv2
import matplotlib.pyplot as plt
import glob
import pandas as pd
def plotGazeObject(gaze, file_name=None):
    '''
    :param figure: the plotting figure
    :param canvas: the canvas where the figure is attached into
    The player's gaze position and object position
    '''

    # Open game background
    img = cv2.imread("..\\Conf\\game_bg.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, 1, 0, 1])
    # ax.plot(self.gazeObject[0], self.gazeObject[1], '-o', label="Object", alpha=0.5, color="green")
    ax.plot(gaze[0], gaze[1], 'o', label="Gaze", alpha=0.7, color="yellow")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    plt.axis('off')
    plt.savefig(file_name)



# gaze_paths = "D:\\usr\\pras\\data\\AttentionTestData\\PerceptualLearning\\AttentionTestResults\\csv\\downsample\\"
# result_path = "D:\\usr\\pras\\data\\AttentionTestData\\PerceptualLearning\\AttentionTestResults\\csv\\summary\\"

gaze_paths = "D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\High-risk\\downsample_avg_5\\"
result_path = "D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\High-risk\\downsample_avg_5\\summary\\"
prefix = "_gazeHeadPose_downsample_avg.csv"
files = glob.glob(gaze_paths + "*" + prefix)

for file in files:
    data = pd.read_csv(file)[["GazeX","GazeY"]].values.transpose()
    file_name = file.split("\\")[-1].split(prefix)[0]
    plotGazeObject(data, result_path + file_name + ".png")
