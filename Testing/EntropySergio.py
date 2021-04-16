# sampleMeasuresCode.py
# Statistical computations for selected measures (entropy, velocity, acceleration).
# Sergio A. Alvarez, updated Feb. 14, 2021

from glob import glob
from pandas import read_csv
from scipy.stats import gaussian_kde as kde, entropy, ttest_ind, mannwhitneyu, median_test
from scipy.signal import welch, savgol_filter
from numpy import min, max, mean, median, std, mgrid, array, zeros
from math import log2
from matplotlib.pyplot import boxplot, title, show, xticks, yticks
from Conf.Settings import TYPICAL_DW_PATH, ASD_DW_PATH

def removeNoObjectData(numpyData):
    return numpyData[numpyData[:, 1] != -1, :]


def objectPresent(datum):
    return datum[1] != -1


def objectAppearanceIndices(D):
    return [i for i in range(1, D.shape[0]) if not objectPresent(D[i - 1, :]) and objectPresent(D[i, :])]


def timeGazeTrajectory(filename):
    df = read_csv(filename)
    print(filename, ': read', str(df.shape[0]), 'data points')
    return df[['Time', 'ObjectX', 'ObjectY', 'GazeX', 'GazeY']].dropna().to_numpy()


def gazeEntropy(xy, relative=True):
    est = kde(xy.transpose())
    if relative:
        xgrid, ygrid = mgrid[-1:1:51j, -1:1:51j]
    else:
        xgrid, ygrid = mgrid[0:1:51j, 0:1:51j]
    return entropy(array([est.pdf([x, y]) for (x, y) in zip(xgrid, ygrid)]).ravel()) \
           / log2(len(xgrid.ravel()))


def spectralEntropy(xy, fs=72):  # fs defaults to downsampled frequency
    _, spx = welch(xy[:, 0], fs, nperseg=fs / 2)  # scipy.signal.welch, sensitive to nperseg
    _, spy = welch(xy[:, 1], fs, nperseg=fs / 2)  # equal spectrum discretization for x, y
    return entropy(spx + spy) / log2(len(_))  # scipy.stats.entropy


def velocity(xy, n=5, p=2):
    vx = savgol_filter(xy[:, 0], window_length=n, polyorder=p, deriv=1)
    vy = savgol_filter(xy[:, 1], window_length=n, polyorder=p, deriv=1)
    return mean(vx ** 2 + vy ** 2) ** 0.5


def acceleration(xy, n=5, p=2):
    accx = savgol_filter(xy[:, 0], window_length=n, polyorder=p, deriv=2)
    accy = savgol_filter(xy[:, 1], window_length=n, polyorder=p, deriv=2)
    return mean(accx ** 2 + accy ** 2) ** 0.5


def measureFromName(measureType):
    if 'entropy' in measureType:
        if 'spectral' in measureType:
            return spectralEntropy
        else:
            return gazeEntropy
    elif 'velocity' in measureType:
        return velocity
    elif 'acceleration' in measureType:
        return acceleration


def measureBySegments(f, D, ind):
    vals = zeros((len(ind),))
    for seg in range(len(ind) - 1):
        vals[seg] = f(D[ind[seg]:ind[seg + 1], :])
    return mean(vals), std(vals)


def testMeasure(filePrefix, measureType):
    f = measureFromName(measureType)
    H, S = [], []
    for file in glob(filePrefix + '*gazeHeadPose*'):
        D = timeGazeTrajectory(file)
        if 'by segments' in measureType:
            ind = objectAppearanceIndices(D)
        else:
            D = removeNoObjectData(D)
        ab = D[:, 1:3]
        xy = D[:, 3:]
        if 'gaze-to-object' in measureType:
            D = xy - ab  # relative gaze-to-object measure
        else:
            D = xy  # "grounded" gaze-only measure
        if 'by segments' in measureType:
            m, s = measureBySegments(f, D, ind)
            H.append(m)
            S.append(s)
        else:
            H.append(f(D))
            S.append(f(D))
    return H, S


def plotResults(distH, distA, titleString, digits=5):
    boxplot([distH, distA], notch=True, widths=0.25, positions=[0.75, 1.25], \
            labels=['Diagnosed', 'Typical'])
    title(titleString, fontsize=18)
    xticks(fontsize=14)
    yticks(fontsize=10)
    show()
    stat, pValue = ttest_ind(distH, distA)
    print(titleString)
    print('mean / med:\t, diagnosed:', round(mean(distH), digits), '/', round(median(distH), digits), \
          '\ttypical:', round(mean(distA), digits), '/', round(median(distA), digits))
    print('t significance level p =', round(pValue, digits))
    stat, pValue = mannwhitneyu(distH, distA)
    print('U significance level p =', round(pValue, digits))
    stat, pValue, m, table = median_test(distH, distA, correction=False)
    print('median significance level p =', round(pValue, digits))
    print()


# Appropriate directory names and figure title must be inserted below
# measureType is a string that describes the measure to be computed;
# options include entropy, velocity, acceleration;
# if measureType includes the term 'gaze-to-object', then relative measure is used;
# if it includes the term 'by segments', then the mean over obj-appearance segments is used;
# these descriptors can be combined, as in 'gaze-to-object velocity by segments'

measureType = 'spectral entropy'
ParentDirectory = './PrasSavgolDataFeb2021/'
Hdiag, Sdiag = testMeasure(TYPICAL_DW_PATH, measureType)
Htyp, Styp = testMeasure(ASD_DW_PATH, measureType)
plotResults(Hdiag, Htyp, measureType)
