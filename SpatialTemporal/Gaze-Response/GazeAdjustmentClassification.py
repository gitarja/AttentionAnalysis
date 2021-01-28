import numpy as np
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from Conf.Settings import MAX_LAG, MIN_D_N, TYPICAL_DW_RESULTS_PATH, ASD_DW_RESULTS_PATH, ANALYSIS_RESULT_PATH
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
sns.set_style("whitegrid")

paths = [TYPICAL_DW_RESULTS_PATH, ASD_DW_RESULTS_PATH]

ar_params = []
response = []
subjects = []
Y = []
s_idx = 0
class_idx = 0
f = []
for path in paths:
    files = glob.glob(path + "*_ar_params.npy")
    Y.append(np.zeros(len(files)) + class_idx)
    f.append(files)
    for file in files:
        # if sex[j] == 1:
        response_file = file.replace("_ar_params.npy", "_responses.npy")
        ar_params.append(np.load(file, allow_pickle=True))
        response.append(np.load(response_file, allow_pickle=True))
        subjects.append(np.ones(len(ar_params[s_idx])) * s_idx)
        s_idx += 1

    class_idx += 1

X = np.concatenate(ar_params, 0)
response = np.concatenate(response, 0)[np.sum(X, 1) != 0]
subjects = np.concatenate(subjects, 0)[np.sum(X, 1) != 0]
X_filter = X[np.sum(X, 1) != 0]
labels = ["GoPositive", "NoGoNegative", "GoNegative", "NoGoPositive"]
# labels = ["Go", "NoGo"]
colors = []


X_features = []
for s in np.unique(subjects):
    features =[]
    for i in range(len(labels)):
        # two stimulus: Go and NoGo
        # X_response = X_filter[(subjects == s) & ((response == i) | (response == i+2))]

        #two response: positif and negatif
        # X_response = X_filter[(subjects == s) &((response == i) | (response == i + (3 - i)))]

        # four responses
        X_response = X_filter[(subjects == s) & (response==i)]
        features.append(np.concatenate([np.average(X_response, axis=0), np.std(X_response, axis=0)]))
    X_features.append(np.concatenate(features))


#define data
X = np.array(X_features)
Y = np.concatenate(Y)
f = np.concatenate(f)

#preprocessing
scaler = make_pipeline(StandardScaler(),
                    NeighborhoodComponentsAnalysis(n_components=5,
                                                   random_state=0))
X_norm = scaler.fit_transform(X, Y)



kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


#compute Mutual Information

# for i in range(len(labels)):
#     print(mi[(i*7):(i+1)*7])
    #print("MI of %s = %f" %(labels[i], np.sum(mi[(i*7):(i+1)*7])))
# estimator = LinearSVC()

#features selector


# parameters = {'n_estimators':[25, 50], 'max_samples':[0.3, 0.5, 0.7], 'max_features':[0.3, 0.5, 0.7]}
# rf = ExtraTreesClassifier(random_state=0, class_weight='balanced', bootstrap=True)

parameters = {'C': [1, 1.5, 2.5]}
svc = SVC(random_state=0,  kernel="linear")
clf = GridSearchCV(svc, parameters)
clf.fit(X_norm, Y)
best_params = clf.best_params_
print(best_params)
for train_index, test_index in kf.split(X_norm, Y):
    X_train = X_norm[train_index]
    X_test = X_norm[test_index]

    Y_train = Y[train_index]
    Y_test = Y[test_index]

    # clf = AdaBoostClassifier(n_estimators=100, algorithm='SAMME', base_estimator=estimator,random_state=0)
    # best_model = ExtraTreesClassifier(n_estimators=best_params["n_estimators"], max_features=best_params["max_features"], max_samples=best_params["max_samples"], random_state=0, class_weight='balanced', bootstrap=True)
    best_model = SVC(C=best_params["C"], random_state=0, kernel="linear")
    best_model.fit(X_train, Y_train)
    score = best_model.score(X_test, Y_test)

    print(score)
    print(classification_report(Y_test, best_model.predict(X_test)))
    print(confusion_matrix(Y_test, best_model.predict(X_test)))
    # print(clf.predict(X_test))
    # print(clf.predict_proba(X_test))
    # print(Y_test)
    print(f[test_index])
    # print(clf.feature_importances_)

