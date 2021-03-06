import numpy as np
import nolds
from scipy import interpolate
from statsmodels.tsa.ar_model import AR, AutoReg
from os import path, mkdir
from scipy.stats import gaussian_kde as kde, entropy
from scipy.signal import welch
import warnings
from scipy.signal._savitzky_golay import savgol_filter
from statsmodels.tsa.stattools import acf
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
    ''' compute euclidian distance of two points
    :param x1:
    :param x2:
    :return:
    '''
    dist =np.sqrt(np.sum(np.power(x1-x2, 2), -1))
    return dist

def euclidianDistT(x, time=None, skip=3):
    ''' compute the euclidian distance for the series
    :param x: must be series > skip + 3
    :param skip:
    :return:
    '''
    if len(x) < (skip + 3):
        raise ValueError("length of x must be greater than skip + 3")
    dist = np.array([euclidianDist(x[i], x[i-skip]) for i in range(skip, len(x), 1)])
    return dist



def movingAverage(a, n=3, remove_neg=False):
    ''' moving average with window equals n
    :param a: a series
    :param n: the window length
    :param remove_neg:
    :return:
    '''
    a = np.nan_to_num(a)
    # a[a < 0] = 0
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    avg_result = ret[n - 1:] / n
    if remove_neg:
        avg_result[avg_result < 0] = -1
    return avg_result


def computeVelocity(time, gaze, n, time_constant=False):
    '''
    :param time: a series contains time
    :param gaze: a series contains gaze values
    :param n: the window length
    :return: series of velocity, time
    '''
    velc = []
    times = []
    for i in range(n, len(time), n):
        dt = (time[i] - time[i - n])
        gt = np.linalg.norm(gaze[i] - gaze[i - n])

        # if (dt <= 0.5):
        if time_constant == False:
            velc.append(gt / dt)
        else:
            velc.append(gt)
        times.append(time[i])
    # print(np.max([np.abs((gaze[i] - gaze[i - n])) for i in range(n, len(time), n)]))
    return velc, times

def computeAcceleration(time, velocity, n, time_constant=False):
    '''
    :param dt: time difference
    :param velocity: a series of velocity
    :param n: the window lenght
    :return: series of acceleration
    '''
    accs = []
    for i in range(n, len(velocity), n):
        dt = (time[i] - time[i - n])
        vt = (velocity[i] - velocity[i - n])
        # if (dt <= 0.2):
        if time_constant == False:
            accs.append((vt/dt) + 1e-25) #add 1-e25 to prevent zero
        else:
            accs.append(vt)  # add 1-e25 to prevent zero

    return accs

def computeVelocityAccel(time, gaze, n, poly):
    # x_filtered
    # first derivative
    gaze_x_d1 = savgol_filter(gaze[:, 0], n, polyorder=poly, deriv=1)
    gaze_y_d1 = savgol_filter(gaze[:, 1], n, polyorder=poly, deriv=1)
    # time_d1 =  savgol_filter(time, n, polyorder=poly, deriv=1)

    gaze_d1 =np.array([gaze_x_d1, gaze_y_d1]).transpose()

    #second derivative
    gaze_x_d2 = savgol_filter(gaze[:, 0], n, polyorder=poly, deriv=2)
    gaze_y_d2 = savgol_filter(gaze[:, 1], n, polyorder=poly, deriv=2)
    # time_d2 =  savgol_filter(savgol_filter(time, n, polyorder=poly, deriv=0), n, polyorder=poly, deriv=1)

    gaze_d2 = np.array([gaze_x_d2, gaze_y_d2]).transpose()

    velocity_filtered = np.sqrt(np.sum(np.power(gaze_d1, 2), -1))

    acceleration_filtered = np.sqrt(np.sum(np.power(gaze_d2, 2), -1))


    return velocity_filtered, acceleration_filtered





def relu(data, max=1.):
    ''' relu activation function
    :param data: must be a series
    :param max:
    :return:
    '''
    data[data > max] = max
    return data


def anglesEstimation(data, skip=3):
    ''' compute angles of consecutive gazes
    :param data: a series
    :param skip: stride
    :return: a series of angles
    '''
    angles = np.array(
        [np.dot(data[i, :], data[i - skip, :]) / (np.linalg.norm(data[i, :]) * np.linalg.norm(data[i - skip, :])) for i in range(skip, len(data), 1)])
    angles = np.arccos(relu(angles))
    return angles





def arParams(x, times=None, min_len=20, maxlag=3):
    '''
    :param x: input of the AR model
    :param times:
    :param min_len:
    :param maxlag: maximum lag
    :return: params, and loglike score
    '''
    model = AR(endog=x)
    model_fitted = model.fit(maxlag=maxlag, maxiter=200)
    loglike_score = model.loglike(model_fitted.params)
    return model_fitted.params, loglike_score


def arModel(min_len=20, maxlag=4):
    ''' create an AR model with prior inputs
    :param min_len:
    :param maxlag: maximum leg
    :return: AR model
    '''
    start = 1.0
    end = 0.95
    prior = np.arange(start, end, (end - start) / min_len)
    model = AR(prior)
    model.fit(maxlag=maxlag)
    # model = AutoReg(prior, lags=maxlag)
    # model.fit()
    return model

def filterFixation(fixation):
    ''' decide when the gaze enter and leave the fixation area
    :param fixation: a series of distance between gaze and stimulus
    :return: the series of fixation
    '''
    fixation_list = []
    fixation_g = []
    for i in range(1, len(fixation)):
        if (fixation[i-1] == True) and (fixation[i] == False):
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
    ''' proposed by Sergio A. Alvarez
    :param xy: distance between gaze and obj
    :param relative:
    :return: the entropy of heatmap
    '''
    est = kde(xy.transpose())
    if relative:
        xgrid, ygrid = np.mgrid[-1:1:51j, -1:1:51j]
    else:
        xgrid, ygrid = np.mgrid[0:1:51j, 0:1:51j]
    return entropy(np.array([est.pdf([x,y]) for (x,y) in zip(xgrid, ygrid)]).ravel())\
           /np.log2(len(xgrid.ravel()))

def spectralEntropy(xy, fs=72):     # defaults to downsampled frequency
    ''' proposed by Sergio A. Alvarez
    :param xy: gaze - object series
    :param fs:
    :return:
    '''
    _, spx = welch(xy[:,0], fs, nperseg=fs/2)     # scipy.signal.welch
    _, spy = welch(xy[:,1], fs, nperseg=fs/2)     # equal spectrum discretization for x, y
    return entropy(spx + spy)/np.log2(len(_))  # scipy.stats.entropy


def cohenD(a, b):
    n_a = len(a)
    n_b = len(b)
    return (np.mean(a)-np.mean(b))/\
           np.sqrt(((n_a-1)*np.var(a, ddof=1) + (n_b-1)*np.var(b, ddof=1)) / (n_a + n_b -2))

def autocorr(x, max_lag=20, normalize=True):
    if normalize:
        return acf(x, nlags=max_lag, fft=False)
    else:
        result = np.correlate(x, x, mode='full')
        return result[result.size // 2:]

