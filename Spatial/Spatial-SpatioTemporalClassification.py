import numpy as np
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from Conf.Settings import MAX_LAG, MIN_D_N, TYPICAL_DW_RESULTS_PATH, ASD_DW_RESULTS_PATH, ANALYSIS_RESULT_PATH
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, roc_curve, auc
import pandas as pd
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from scipy.stats import kurtosis, skew
from sklearn.tree import export_graphviz
import pydotplus
from dtreeviz.trees import dtreeviz
from io import StringIO
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

paths = [TYPICAL_DW_RESULTS_PATH, ASD_DW_RESULTS_PATH]

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
spatial_features_concat = pd.concat([typical_spatial_features, asd_spatial_features], sort=True)

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




# features_cols = [
#     "Sampen_velocity",
#     "Acceleration_avg",
#     "Fixation_std",
#     "Sampen_dist",
#     "Sampen_angle",
#     "GazeObj_entropy",
#     "Sampen_gaze_obj",
#     "Spectral_entropy",
#
# ]


features_cols = [
    "Sampen_velocity",
    "Acceleration_avg",
    "Fixation_std",
    "Sampen_dist",
    "Sampen_angle",
    "GazeObj_entropy",
    "Sampen_gaze_obj",
    "Spectral_entropy",
    "Go",
    "GoError",
    "NoGo",
    "NoGoError",
    "RT",
    "RTVar",
]

for path in paths:
    files = glob.glob(path + "*_ar_params.npy")
    Y.append(np.zeros(len(files)) + class_idx)
    f.append(files)
    for file in files:
        # if sex[j] == 1:
        f_name = file.split(path)[-1].split("_ar_params.npy")[0]
        response_file = file.replace("_ar_params.npy", "_responses.npy")
        ar_params.append(np.load(file, allow_pickle=True))
        response.append(np.load(response_file, allow_pickle=True))
        s_features = spatial_features_concat.loc[spatial_features_concat['id'] == f_name][features_cols].values
        spatial_features.append(s_features)
        subjects.append(np.ones(len(ar_params[s_idx])) * s_idx)
        s_idx += 1

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
        X_response = X_filter[((subjects == s) & (response == i))]
        features.append(np.concatenate(
            [skew(X_response, axis=0), kurtosis(X_response, axis=0), np.std(X_response, axis=0),
             np.mean(X_response, axis=0)]))
    X_features.append(np.concatenate(features))

# define data
X = np.array(X_features)
X_spatial = np.concatenate(spatial_features)
Y = np.concatenate(Y)
f = np.concatenate(f)

# preprocessing
norm_scaller = StandardScaler()
# reduce the dimension of gaze-adjustmen features
scaler = make_pipeline(norm_scaller,
                       NeighborhoodComponentsAnalysis(n_components=5,
                                                      random_state=0))

# transform features
X_norm = scaler.fit_transform(X, Y)
# X_norm = scaler.fit_transform(np.concatenate([X, X_spatial], -1), Y)
X_spatial_norm = norm_scaller.fit_transform(X_spatial, Y)

# combine features
X_norm = np.concatenate([X_norm, X_spatial_norm], -1)
# save features to csv
np.savetxt("features.csv", X_norm, delimiter=",")
np.savetxt("labels.csv", Y, delimiter=",")


# find the optimal value for the Adaboost and Decision Tree
parameters = {'n_estimators': [15, 25, 50, 75], "learning_rate": [0.5, 0.75], "base_estimator__max_depth":[1, 3, 5, 7], "base_estimator__max_features":[.5], "base_estimator__max_leaf_nodes":[3, 5, 7]}
# base estimator for AdaBoost
base_estimator = DecisionTreeClassifier(criterion="entropy",
                                        class_weight="balanced",
                                        random_state=0)
base_model = AdaBoostClassifier(base_estimator=base_estimator,
                                random_state=0)
clf = GridSearchCV(base_model, parameters)
clf.fit(X_norm, Y)
best_params = clf.best_params_

# print(clf.best_estimator_.feature_importances_)



acc = []
roc_auc_values = []
mcc = []
# perform k-fold cross validation
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
fold = 0

for train_index, test_index in kf.split(X_norm, Y):
    X_train = X_norm[train_index]
    X_test = X_norm[test_index]

    Y_train = Y[train_index]
    Y_test = Y[test_index]

    np.save("data_train"+str(fold)+".npy", X_train)
    np.save("data_test" + str(fold) + ".npy", X_test)

    np.save("label_train" + str(fold) + ".npy", Y_train)
    np.save("label_test" + str(fold) + ".npy", Y_test)

    fold+=1

    # best_model = AdaBoostClassifier(learning_rate=best_params["learning_rate"],
    #                                 base_estimator=DecisionTreeClassifier(criterion="entropy", max_depth=best_params["base_estimator__max_depth"], max_features=best_params["base_estimator__max_features"], max_leaf_nodes=best_params["base_estimator__max_leaf_nodes"],
    #                                     class_weight="balanced",
    #                                     random_state=0),
    #                                 n_estimators=best_params["n_estimators"], random_state=0)
    #
    # # print(f[test_index])
    # best_model.fit(X_train, Y_train)  # fit the data into the model
    # score = best_model.score(X_test, Y_test)  # test the model with test data
    # acc.append(score)
    #
    # fpr, tpr, _ = roc_curve(Y_test, best_model.predict_proba(X_test)[:, 1])
    # roc_auc = auc(fpr, tpr)
    # roc_auc_values.append(roc_auc)
    #
    # mcc.append(matthews_corrcoef(Y_test, best_model.predict(X_test)))
    # # print(score)
    # # print(matthews_corrcoef(Y_test, best_model.predict(X_test)))  # matthews ccc
    # # print(roc_auc) #auc
    # print(classification_report(Y_test, best_model.predict(X_test)))  # compute precision recall and F1-score
    # print(confusion_matrix(Y_test, best_model.predict(X_test)))  # compute confusion matrix


    # plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc="lower right")
    # plt.show()
    # print(f[test_index][Y_test !=best_model.predict(X_test)])

# for i in range(len(acc)):
#     print("%f, %f, %f" %  (acc[i], mcc[i], roc_auc_values[i]))  # average ACC
# print(np.average(acc))
# visualize tree
# best_model = DecisionTreeClassifier(criterion="entropy", max_depth=7, max_leaf_nodes=5, random_state=0, class_weight="balanced")
# best_model.fit(X_norm, Y)
# print(best_model.feature_importances_)
# dot_data = StringIO()
# features_cols = ["C1", "C2", "C3", "C4", "C5"] + features_cols
# viz = dtreeviz(best_model, X_norm, Y, target_name="class", feature_names=features_cols, class_names=['Typical', 'ASD'])
# viz.save('tree_all_spatial.svg')
