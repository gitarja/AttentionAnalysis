import numpy as np
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from Conf.Settings import MAX_LAG, MIN_D_N, TYPICAL_DW_RESULTS_PATH, ASD_DW_RESULTS_PATH, ANALYSIS_RESULT_PATH
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from scipy.stats import kurtosis, skew
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
        # s_features = spatial_features_concat.loc[spatial_features_concat['id'] == f_name][
        #     ["Go", "GoError", "NoGo", "NoGoError", "RT", "RTVar", "Trajectory Area", "VelocityX_avg",
        #        "VelocityY_avg", "VelocityX_std", "VelocityY_std", "Fixation_avg", "Fixation_std", "Sampen_dist",
        #        "Sampen_angle", "Spatial_entropy", "GazeObj_entropy"]].values

        s_features = spatial_features_concat.loc[spatial_features_concat['id'] == f_name][
            ["Go", "GoError", "Fixation_std", "Sampen_dist",
               "Sampen_angle", "GazeObj_entropy"]].values
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
    for i in range(len(labels)):
        X_response = X_filter[(subjects == s) & (response == i)]
        features.append(np.concatenate([skew(X_response, axis=0), kurtosis(X_response, axis=0), np.std(X_response, axis=0), np.mean(X_response, axis=0)]) )
    X_features.append(np.concatenate(features))

# define data
X = np.array(X_features)
X_spatial = np.concatenate(spatial_features)
Y = np.concatenate(Y)
f = np.concatenate(f)

# preprocessing
norm_scaller = StandardScaler()
scaler = make_pipeline(norm_scaller,
                       NeighborhoodComponentsAnalysis(n_components=5,
                                                      random_state=0))

# transform features
X_norm = scaler.fit_transform(X, Y)
X_spatial_norm = norm_scaller.fit_transform(X_spatial, Y)

# combine features
X_norm = np.concatenate([X_norm, X_spatial_norm], -1)
#save features to csv
np.savetxt("features.csv", X_norm, delimiter=",")
np.savetxt("labels.csv", Y, delimiter=",")
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

# compute Mutual Information
mi = mutual_info_classif(X_norm, Y)
print(mi)
# for i in range(len(labels)):
#     print(mi[(i*7):(i+1)*7])
# print("MI of %s = %f" %(labels[i], np.sum(mi[(i*7):(i+1)*7])))
# parameters = {'n_estimators': [25, 50, 75], 'max_samples': [0.3, 0.5, 0.7], 'max_features': [0.3, 0.5, 0.7]}
# rf = ExtraTreesClassifier(random_state=0, class_weight='balanced', bootstrap=True)

parameters = {'C': [0.25]}
svc = SVC(random_state=0,  kernel="linear", class_weight="balanced")

clf = GridSearchCV(svc, parameters)
clf.fit(X_norm, Y)
best_params = clf.best_params_

# compute features importance
print(best_params)
acc = []
for train_index, test_index in kf.split(X_norm, Y):
    X_train = X_norm[train_index]
    X_test = X_norm[test_index]

    Y_train = Y[train_index]
    Y_test = Y[test_index]

    # clf = AdaBoostClassifier(n_estimators=100, algorithm='SAMME', base_estimator=estimator,random_state=0)
    # best_model = RandomForestClassifier(n_estimators=best_params["n_estimators"],
    #                                   max_features=best_params["max_features"], max_samples=best_params["max_samples"],
    #                                   random_state=0, class_weight='balanced', bootstrap=True)

    best_model = SVC(C= best_params["C"],random_state=0, kernel="linear", class_weight="balanced")
    best_model.fit(X_train, Y_train)
    score = best_model.score(X_test, Y_test)
    acc.append(score)
    print(score)
    print(classification_report(Y_test, best_model.predict(X_test)))
    print(confusion_matrix(Y_test, best_model.predict(X_test)))
    # print(clf.predict(X_test))
    # print(clf.predict_proba(X_test))
    # print(Y_test)
    print(f[test_index])
    # print(clf.feature_importances_)

print("AVG-Classification %f" %(np.mean(acc)) )