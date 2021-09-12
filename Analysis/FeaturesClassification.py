import numpy as np
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from Conf.Settings import  TYPICAL_DW_RESULTS_PATH, ASD_DW_RESULTS_PATH, ANALYSIS_RESULT_PATH, ADULT_DW_RESULTS_PATH, ASD_PATH
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, roc_curve, auc, roc_auc_score
import pandas as pd
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from scipy.stats import kurtosis, skew
import pickle

import matplotlib.pyplot as plt

sns.set_style("whitegrid")
adult = False
multi_class = True

ar_params = []
spatial_features = []
response = []
subjects = []
Y = []
s_idx = 0
class_idx = 0
f = []

# spatial features
typical_spatial_features = pd.read_csv(TYPICAL_DW_RESULTS_PATH + "summary\\summary_response_new.csv")
asd_spatial_features = pd.read_csv(ASD_DW_RESULTS_PATH + "summary\\summary_response_new.csv")
adult_spatial_features = pd.read_csv(ADULT_DW_RESULTS_PATH + "summary\\summary_response_new.csv")

#asd labels
asd_labels = pd.read_csv(ASD_PATH + "labels.csv")
if adult:
    paths = [ADULT_DW_RESULTS_PATH]
    spatial_features_concat = pd.concat([adult_spatial_features], sort=True)
else:
    paths = [TYPICAL_DW_RESULTS_PATH, ASD_DW_RESULTS_PATH]
    spatial_features_concat = pd.concat([typical_spatial_features, asd_spatial_features], sort=True)

#all spatial features
# features_cols = ["Go", "GoError", "NoGo",
#            "NoGoError", "RT", "RTVar", "Trajectory Area",
#            "Velocity_avg",
#            "Velocity_std",
#            "Acceleration_avg",
#            "Acceleration_std",
#            "Fixation_avg",
#            "Fixation_std",
#            "Distance_avg",
#            "Distance_std",
#            "Angle_avg",
#            "Angle_std",
#            "Sampen_dist",
#            "Sampen_angle",
#            "Spatial_entropy",
#            "GazeObj_entropy",
#            "Sampen_gaze_obj",
#            "Spectral_entropy",
#            "Sampen_velocity",
#            "Sampen_acceleration"
#                  ]



# significant features
features_cols = [
    "Sampen_velocity",
    "Acceleration_avg",
    "Fixation_std",
    "Sampen_dist",
    "Sampen_angle",
    "GazeObj_entropy",
    "Sampen_gaze_obj",
    "Spectral_entropy",

]

# significant features ANOVA
# features_cols = [
#     "Sampen_velocity",
#     "Acceleration_avg",
#     "Fixation_std",
#     "Sampen_dist",
#     "Sampen_angle",
#     "GazeObj_entropy",
#     "Sampen_gaze_obj",
#     "Spectral_entropy",
#     "Distance_avg",
#     "Distance_std",
#      "Velocity_avg",
#      "Velocity_std",
#      "Acceleration_avg",
# ]


# significant features + performance
# features_cols = [
#     "Sampen_velocity",
#     "Acceleration_avg",
#     "Fixation_std",
#     "Sampen_dist",
#     "Sampen_angle",
#     "GazeObj_entropy",
#     "Sampen_gaze_obj",
#     "Spectral_entropy",
#     "Go",
#     "GoError",
#     # "NoGo",
#     # "NoGoError",
#     # "RT",
#     "RTVar",
#     "Acceleration_std",
# ]

for path in paths:
    files = glob.glob(path + "*_ar_params.npy")
    candidate_labels = np.zeros(len(files)) + class_idx
    f.append(files)
    idx = 0
    for file in files:
        # if sex[j] == 1:
        f_name = file.split(path)[-1].split("_ar_params.npy")[0]
        response_file = file.replace("_ar_params.npy", "_responses.npy")
        ar_params.append(np.load(file, allow_pickle=True))
        response.append(np.load(response_file, allow_pickle=True))
        s_features = spatial_features_concat.loc[spatial_features_concat['id'] == f_name][features_cols].values
        spatial_features.append(s_features)
        subjects.append(np.ones(len(ar_params[s_idx])) * s_idx)
        if np.sum(asd_labels.id.values == f_name) > 0  and multi_class == True:
            candidate_labels[idx] = asd_labels[asd_labels.id.values == f_name].Label

        s_idx += 1
        idx +=1
    Y.append(candidate_labels)
    class_idx += 1

# print(f)
X = np.concatenate(ar_params, 0)
response = np.concatenate(response, 0)[np.sum(X, 1) != 0]
subjects = np.concatenate(subjects, 0)[np.sum(X, 1) != 0]
X_filter = X[np.sum(X, 1) != 0]
labels = ["GoPositive", "NoGoNegative", "GoNegative", "NoGoPositive"]
colors = []

X_features = []
for s in np.unique(subjects):
    features = []
    # for i in range(1):
    #     X_response = X_filter[(subjects == s), 0:3]
    for i in range(len(labels)):
    # for i in [0, 3]:
        X_response = X_filter[((subjects == s) & (response == i))]

        features.append(np.concatenate(
                [skew(X_response, axis=0), kurtosis(X_response, axis=0), np.std(X_response, axis=0),
                 np.mean(X_response, axis=0)]))
    X_features.append(np.concatenate(features))

# define data
X_gaze = np.array(X_features)
X_spatial = np.concatenate(spatial_features)
Y = np.concatenate(Y)
f = np.concatenate(f)

if adult:
    norm_scaller = pickle.load(open("norm_scaller", "rb"))
    scaler = pickle.load(open("scaler", "rb"))

else:
    # preprocessing
    norm_scaller = StandardScaler()
    # reduce the dimension of gaze-adjustmen features
    scaler = make_pipeline(StandardScaler(),
                           NeighborhoodComponentsAnalysis(n_components=5,
                                                          random_state=0))
    scaler.fit(X_gaze, Y)
    norm_scaller.fit(X_spatial, Y)
    pickle.dump(norm_scaller, open("norm_scaller", "wb"))
    pickle.dump(scaler, open("scaler", "wb"))

X_gaze = scaler.transform(X_gaze)
X_spatial = norm_scaller.transform(X_spatial)




# combine features
# X_final = X_spatial
X_final = np.concatenate([X_gaze, X_spatial], -1)

# save features to npy and csv
# if adult:
#     # npy
#     np.save("data_adults.npy", X_final)
#     np.save("labels_adults.npy", Y)
#     # pandas CSV
#     df = pd.DataFrame({"subject_id": spatial_features_concat['id'].values, "labels": Y})
#     df.to_csv("adults_id.csv")
#     np.savetxt("features_gazeperformance_adults.csv", X_final, delimiter=",")
# else:
#     # npy
#     np.save("features_gazeperformance.npy", X_final)
#     np.save("labels_gazeperformance.npy", Y)
#     # pandas CSV
#     df = pd.DataFrame({"subject_id" :spatial_features_concat['id'].values, "labels": Y})
#     df.to_csv("subjects_id.csv")
#     np.savetxt("features_gazeperformance.csv", X_final, delimiter=",")
#     np.savetxt("labels_gazeperformance.csv", Y, delimiter=",")


# find the optimal value for the Adaboost and Decision Tree
parameters = {'n_estimators': [15, 25, 50, 75], "learning_rate": [0.5, 0.75], "base_estimator__max_depth":[1, 3, 5, 7], "base_estimator__max_features":[.5], "base_estimator__max_leaf_nodes":[3, 5, 7]}
# base estimator for AdaBoost
base_estimator = DecisionTreeClassifier(criterion="entropy",
                                        class_weight="balanced",
                                        random_state=0)
base_model = AdaBoostClassifier(base_estimator=base_estimator,
                                random_state=0)
clf = GridSearchCV(base_model, parameters)
clf.fit(X_final, Y)
best_params = clf.best_params_


acc = []
roc_auc_values = []
mcc = []
# perform k-fold cross validation
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
fold = 0

for train_index, test_index in kf.split(X_final, Y):
    X_train = X_final[train_index]
    X_test = X_final[test_index]

    Y_train = Y[train_index]
    Y_test = Y[test_index]

    # np.save("data_train"+str(fold)+".npy", X_train)
    # np.save("data_test" + str(fold) + ".npy", X_test)
    #
    # np.save("label_train" + str(fold) + ".npy", Y_train)
    # np.save("label_test" + str(fold) + ".npy", Y_test)

    fold+=1

    best_model = AdaBoostClassifier(learning_rate=best_params["learning_rate"],
                                    base_estimator=DecisionTreeClassifier(criterion="entropy", max_depth=best_params["base_estimator__max_depth"], max_features=best_params["base_estimator__max_features"], max_leaf_nodes=best_params["base_estimator__max_leaf_nodes"],
                                        class_weight="balanced",
                                        random_state=0),
                                    n_estimators=best_params["n_estimators"], random_state=0) #create adaboost classifier with the optimized parameter values



    # print(f[test_index])
    best_model.fit(X_train, Y_train)  # fit the data into the model
    score = best_model.score(X_test, Y_test)  # test the model with test data
    acc.append(score)

    if multi_class:
        auc = roc_auc_score(Y_test, best_model.predict_proba(X_test), multi_class="ovo")  # compute fpr and tpr
    else:
        auc = roc_auc_score(Y_test, best_model.predict_proba(X_test)[:, 1]) # compute fpr and tpr
    roc_auc_values.append(auc)

    mcc.append(matthews_corrcoef(Y_test, best_model.predict(X_test)))
    # print(score)
    # print(matthews_corrcoef(Y_test, best_model.predict(X_test)))  # matthews ccc
    # print(roc_auc) #auc
    print(classification_report(Y_test, best_model.predict(X_test)))  # compute precision recall and F1-score
    print(confusion_matrix(Y_test, best_model.predict(X_test)))  # compute confusion matrix


    # print(f[test_index][Y_test !=best_model.predict(X_test)])

for i in range(len(acc)):
    print("%f, %f, %f" %  (acc[i], mcc[i], roc_auc_values[i]))  # average ACC
print(np.average(acc))
