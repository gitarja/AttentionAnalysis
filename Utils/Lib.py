import numpy as np
import nolds
from scipy import interpolate
from statsmodels.tsa.ar_model import AR, AutoReg
from os import path, mkdir
from scipy.stats import gaussian_kde as kde, entropy
from scipy.signal import welch

def createDir(dir):
    if not path.exists(dir):
        mkdir(dir)


def timeDiff(x1, x2):
    '''
    :param x1: end
    :param x2: start
    :return: time difference (s)
    '''
    if x1 < 0:
        return -1
    timediff = (x1 - x2)
    if timediff >= 0:
        return timediff
    else:
        return -1


def euclidianDist(x1, x2):
    dist = np.sqrt(np.sum(np.power(x1 - x2, 2), -1))
    return dist

def euclidianDistT(x, skip=3):
    dist = np.array([euclidianDist(x[i], x[i-skip]) for i in range(skip, len(x), 1)])
    return dist



def movingAverage(a, n=3, remove_neg=False):
    a = np.nan_to_num(a)
    # a[a < 0] = 0
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    avg_result = ret[n - 1:] / n
    if remove_neg:
        avg_result[avg_result < 0] = -1
    return avg_result


def computeVelocity(time, gaze, n):
    acc = []
    for i in range(n, len(time), n):
        dt = (time[i] - time[i - n])
        gt = (gaze[i] - gaze[i - n])

        if (dt < 0.1):
            acc.append(np.abs(gt) / dt)
    # print(np.max([np.abs((gaze[i] - gaze[i - n])) for i in range(n, len(time), n)]))
    return acc


def spectralEntropy(xy, fs=72):     # defaults to downsampled frequency
    _, spx = welch(xy[:,0], fs, nperseg=10)     # scipy.signal.welch
    _, spy = welch(xy[:,1], fs, nperseg=10)     # equal spectrum discretization for x, y
    return entropy(spx + spy)/np.log(len(_))  # scipy.stats.entropy

def relu(data, max=1.):
    data[data > max] = max
    return data


def anglesEstimation(data, skip=3):
    angles = np.array(
        [np.dot(data[i, :], data[i - skip, :]) / (np.linalg.norm(data[i, :]) * np.linalg.norm(data[i - skip, :])) for i in range(skip, len(data), 1)])
    angles = np.arccos(relu(angles))
    return angles





def arParams(x, times=None, min_len=20, maxlag=3):
    # delta_times = times[1:] -times[0:-1]
    # velocity = x[1:] / delta_times
    model = AR(endog=x)
    model_fitted = model.fit(maxlag=maxlag, maxiter=200)
    # model = AutoReg(endog=x, lags=maxlag)
    # model_fitted = model.fit()
    loglike_score = model.loglike(model_fitted.params)

    return model_fitted.params, loglike_score


def arModel(min_len=20, maxlag=4):
    start = 1.1
    end = 1.05
    prior = np.arange(start, end, (end - start) / min_len)
    model = AR(prior)
    model.fit(maxlag=maxlag)
    # model = AutoReg(prior, lags=maxlag)
    # model.fit()
    return model

def filterFixation(fixation):
    fixation_list = []
    fixation_g = []
    for i in range(1, len(fixation)):
        if (fixation[i-1] == True) & (fixation[i] == False):
            fixation_list.append(fixation_g)
            fixation_g = []
        if fixation[i]:
            fixation_g.append(i)

    if len(fixation_g) > 0:
        fixation_list.append(fixation_g)


    return fixation_list

#compute cutoff to remove outlier
def computeCutoff(x, c=2):
    mad = 1.4826 * np.median(np.absolute(x - np.median(x)))
    return np.median(x)  - (c * mad)

#only take gaze when the stimulus appears
def removeNoObjectData(numpyData):
    return numpyData[numpyData[:,1]!= -1, :]

def gazeEntropy(xy, relative=True):
    est = kde(xy.transpose())
    if relative:
        xgrid, ygrid = np.mgrid[-1:1:51j, -1:1:51j]
    else:
        xgrid, ygrid = np.mgrid[0:1:51j, 0:1:51j]
    return entropy(np.array([est.pdf([x,y]) for (x,y) in zip(xgrid, ygrid)]).ravel())\
           /np.log2(len(xgrid.ravel()))



