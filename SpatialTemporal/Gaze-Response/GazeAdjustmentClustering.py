import numpy as np
import glob
from Utils.Lib import arModel, createDir
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from Conf.Settings import MAX_LAG, MIN_D_N, TYPICAL_DW_RESULTS_PATH, ASD_DW_RESULTS_PATH, ANALYSIS_RESULT_PATH, MAX_CLUSTER, FREQ_GAZE
from sklearn.ensemble import IsolationForest

import seaborn as sns
from joblib import dump, load

sns.set_style("whitegrid")


paths = [TYPICAL_DW_RESULTS_PATH, ASD_DW_RESULTS_PATH]
paths_label = ["Typical", "ASD"]
# sex (0: man, 1:woman)
# sex = [0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,0]

#labels
response_labels = ["GoPositive", "NoGoNegative", "GoNegative", "NoGoPositive"]
# response_labels = ["Go", "NoGo"]
# response_labels = ["Positive", "Negative"]


colors = []

# mean and std of features
X_mean = np.array([ 0.00614283,  0.8159654 ,  0.61965095, -0.29654615, -0.25634407,
        0.07089135])

X_std = np.array([0.18147366, 0.60033625, 0.57644222, 0.6631802 , 0.7000391 ,
       0.79502488])

# X_mean = np.array([ 0.04364135,  0.81553428,  0.42240785, -0.36420452, -0.04945678])
#
# X_std = np.array([0.29233338, 0.29716318, 0.48513484, 0.42207222, 0.13045445])

#iterations

p_idx = 0



for path in paths:
    files = glob.glob(path + "*_ar_params.npy")
    result_path = ANALYSIS_RESULT_PATH + paths_label[p_idx] + "\\"
    createDir(result_path)
    ar_params_list = []
    response_list = []
    subjects_list = []
    #subject index
    s_idx = 0

    for file in files:
        # if sex[j] == 1:
        response_file = file.replace("_ar_params.npy", "_responses.npy")
        ar_params_list.append(np.load(file, allow_pickle=True))
        response_list.append(np.load(response_file, allow_pickle=True))
        subjects_list.append(np.ones(len(ar_params_list[s_idx])) * s_idx)
        s_idx += 1

    X = np.concatenate(ar_params_list, 0)

    response = np.concatenate(response_list, axis=0)[np.sum(X, axis=-1) != 0]
    subjects = np.concatenate(subjects_list, axis=0)[np.sum(X, axis=-1) != 0]
    X_filter = X[np.sum(X, axis=-1) != 0]



    #pre-processing cluster

    X_norm = (X_filter - X_mean) / X_std
    ar_model = arModel(maxlag=MAX_LAG, min_len=MIN_D_N)
    # iterations index
    # r: number of response
    # c: number of centroids

    # remove outlier
    clf = IsolationForest(random_state=0, contamination=0.1)
    outlier_score = clf.fit_predict(X_norm)
    X_norm = X_norm[outlier_score != -1]
    subjects = subjects[outlier_score != -1]
    response = response[outlier_score != -1]

    for r in range(len(response_labels)):
        #four responses
        X_response = X_norm[response == r]
        subjects_response = subjects[response == r]

        #two stimulus: Go and NoGo
        # X_response = X_norm[(response == r) | (response == r+2)]
        # subjects_response = subjects[(response == r) | (response == r+2)]

        #two response: positif and negatif
        # X_response = X_norm[(response == r) | (response == r + (3 - r))]
        # subjects_response = subjects[(response == r) | (response == r + (3-r))]





        # clustering
        scores = np.ones(MAX_CLUSTER + 1) * 10000
        for k in range(2, MAX_CLUSTER + 1):
            gmm = GaussianMixture(n_components=k, max_iter=200, init_params="kmeans", covariance_type="full",
                                  random_state=0)
            gmm.fit(X_response)
            scores[k] = gmm.aic(X_response)
            # print("K= %f, Score= %f" % (k, score))

        # c = 4
        cluster_n = np.argmin(scores)
        gmm = GaussianMixture(n_components=cluster_n, max_iter=200, init_params="kmeans", covariance_type="full", random_state=0)
        gmm.fit(X_response)
        dump(gmm, result_path + response_labels[r] + ".joblib")
        centroids = gmm.means_

        #extrapolate the average of centroids
        X_avg = np.mean(centroids, 0)
        # ar_model = ARModel()


        a = ar_model.predict((X_avg * X_std)  +  X_mean, start=5, end=55)
        plt.figure(1)
        plt.plot(np.arange(0, len(a), 1) / FREQ_GAZE, a, label=response_labels[r])
        print(response_labels[r] + ":" + str(np.min(a)))
        plt.xlabel("Time(s)")
        plt.ylabel("Distance")
        # plt.ylim([0, 1])
        #extrapolate each centroid
        predicts = gmm.predict(X_response)
        plt.figure(r + 2)
        for c in range(len(centroids)):
            # ar_model = ARModel()
            if len(subjects_response[predicts == c]) > 1:
                # print(centroids[j])
                # print("Response = %s, cluster = %d, num_member = %f" % (
                #     response_labels[r], c, len(subjects_response[predicts == c])))
                # print(np.unique(subjects_response[predicts == j]))
                a_c = ar_model.predict((centroids[c] * X_std) + X_mean, start=5, end=55)
                plt.xlabel("Time(s)")
                plt.ylabel("Distance")
                plt.plot(np.arange(0, len(a_c), 1)/ FREQ_GAZE, a_c, label="Centroid" + str(c))
                plt.title(response_labels[r])
                # plt.ylim([0, 1])

        plt.figure(r + 2)
        plt.legend()
        plt.savefig(result_path + response_labels[r] + ".eps")
        plt.savefig(result_path + response_labels[r] + ".png")

    plt.figure(1)
    plt.legend()
    plt.savefig(result_path + "summary.eps")
    plt.savefig(result_path + "summary.png")
    plt.show()
    #move to the next path
    p_idx +=1
