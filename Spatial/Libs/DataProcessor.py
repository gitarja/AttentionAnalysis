import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
class DataProcessor:
    def __init__(self):
        # group response spawntime for every minute
        self.groupConsRes = None
        # group data spawntime for every minute
        self.groupCons = None

    def ProceedGameResult(self, gameResultPath=None, gazeHeadPath=None, gameResults=None, gazeHeadResults=None):
        '''
        :param gameResultPath: the path of game results
        :param gazeHeadPath: the path of gaze and head tracking results
        :return:
        '''
        startTime = 0
        if (gameResultPath is not None):
            self.dataGame = pd.read_csv(gameResultPath)
            self.dataGame = self.dataGame.fillna(0)
            self.dataGame  = self.dataGame .assign(DiffTime=pd.Series(self.dataGame .ResponseTime - self.dataGame .SpawnTime).values)

            # proceed data, remove the first one minute data
            self.processedData = self.dataGame.loc[(self.dataGame.SpawnTime >= 0)].reset_index(drop=True)
            # self.processedData.SpawnTime = self.processedData.SpawnTime - startTime
            # self.processedData.ResponseTime = self.processedData.ResponseTime - startTime
            #normalize the spawn time so it starts from zero
            startTime = self.dataGame.SpawnTime[0]
            self.dataGame.SpawnTime = self.dataGame.SpawnTime - startTime
            self.dataGame.loc[self.dataGame.ResponseTime != -1, "ResponseTime"] = self.dataGame.ResponseTime.loc[self.dataGame.ResponseTime!=-1] - startTime
            #good response: data with response time and pos response = 1
            self.goodResponse = self.dataGame.loc[(self.dataGame.PosResponse == 1) & (self.dataGame.ResponseTime != -1.0)]
            #neg response: data with response time and neg response = 1
            self.negResponse = self.dataGame.loc[(self.dataGame.NegResponse == 1) & (self.dataGame.ResponseTime != -1.0)]
            #miss response: data with response time and miss response = 1
            self.missResponse = self.dataGame.loc[(self.dataGame.MissResponse == 1) & (self.dataGame.ResponseTime != -1.0)]
            #reponse data with response = 1
            self.response = self.dataGame.loc[self.dataGame.ResponseTime != -1.0]
            #group response spawntime for every minute
            self.groupConsRes = np.floor(self.response.SpawnTime // 61).astype(int)
            #group data spawntime for every minute
            self.groupCons = np.floor(self.dataGame.SpawnTime // 61).astype(int)


            #proceed data
            self.processedGoodResponse = self.processedData.loc[(self.processedData.PosResponse == 1) & (self.processedData.ResponseTime != -1.0)]
            self.processedNegResponse = self.processedData.loc[(self.processedData.NegResponse == 1) & (self.processedData.ResponseTime != -1.0)]
            self.processedMissResponse = self.processedData.loc[(self.processedData.MissResponse == 1) & (self.processedData.ResponseTime != -1.0)]
            self.processedResponse = self.processedData.loc[self.processedData.ResponseTime != -1.0]





        if (gazeHeadPath is not None):
            self.gazeData = pd.read_csv(gazeHeadPath)
            self.gazeData = self.gazeData.fillna(0)
            self.gazeData = self.gazeData[self.gazeData.Time >= startTime]
            self.gazeData.Time = self.gazeData.Time - startTime

    def bernouliSTD(self, data):
        p = data.mean()
        std = p * (1-p)
        return std

    def computeResponseTime(self, spawnTime, responseTime):
        '''
        :param spawnTime: time when the object appears
        :param responseTime: time when the player responses
        :returns: average and std response time
        '''
        timeDiff = (responseTime - spawnTime)
        timeDiff = timeDiff[timeDiff >= 0]

        #compute both avg and std if there are response times
        if len(timeDiff.values) > 0:
            avgTime = np.average(timeDiff.values)
            stdTime = np.std(timeDiff.values)
            #stdTime = self.bernouliSTD(responseTime.values)
        else:
            avgTime = 0
            stdTime = 0

        return avgTime, stdTime

    def averageResponseTimes(self, mode=1):
        '''
        :param mode: 1 for using unprocessed data, 2 otherwise
        :return: [avg response time, std response time, compute overall avg and std of response time]
        '''
        if mode == 1:
            response = self.response
        elif mode == 2:
            response = self.processedResponse
        avgTime, stdTime = self.computeResponseTime(spawnTime=response.SpawnTime, responseTime=response.ResponseTime)
        return [avgTime, stdTime], self.computeResponseTimeNS(self.response)

    def averageGoodResponseTime(self, mode=1):
        """
        :param mode: 1 for using unprocessed data, 2 otherwise
        :return:
        """
        if mode==1:
            dataGame = self.dataGame
            goodResponse = self.goodResponse
        elif mode == 2:
            dataGame = self.processedData
            goodResponse = self.processedGoodResponse
        #overall avg and std positif response
        avg = dataGame.PosResponse.mean()
        #std = dataGame.PosResponse.std()
        std = self.bernouliSTD(dataGame.PosResponse)

        #compute avg and std response time of positif response
        avgTime, stdTime = self.computeResponseTime(spawnTime=goodResponse.SpawnTime,
                                                    responseTime=goodResponse.ResponseTime)
        #compute avg and std positif response for each minute
        avgTimes = self.dataGame.PosResponse.groupby(self.groupCons).mean().values.reshape(-1, 1)
        #stdTimes = self.dataGame.PosResponse.groupby(self.groupCons).std().values.reshape(-1, 1)
        stdTimes = self.dataGame.PosResponse.groupby(self.groupCons).apply(self.bernouliSTD).values.reshape(-1, 1)

        return [avg, std, avgTime, stdTime], [avgTimes, stdTimes, self.computeResponseTimeGroup(self.goodResponse)]

    def averageNegResponseTime(self, mode=1):
        """
        :param mode: 1 for using unprocessed data, 2 otherwise
        :return:
        """
        if mode == 1:
            dataGame = self.dataGame
            negResponse = self.negResponse
        elif mode==2:
            dataGame = self.processedData
            negResponse = self.processedNegResponse

        # overall avg and std negative response
        avg = dataGame.NegResponse.mean()
        #std = dataGame.NegResponse.std()
        std = self.bernouliSTD(dataGame.NegResponse)

        # compute avg and std response time of negative response
        avgTime, stdTime = self.computeResponseTime(spawnTime=negResponse.SpawnTime,
                                                    responseTime=negResponse.ResponseTime)

        # compute avg and std negative response for each minute
        avgTimes = self.dataGame.NegResponse.groupby(self.groupCons).mean().values.reshape(-1, 1)
        #stdTimes = self.dataGame.NegResponse.groupby(self.groupCons).std().values.reshape(-1, 1)
        stdTimes = self.dataGame.NegResponse.groupby(self.groupCons).apply(self.bernouliSTD).values.reshape(-1, 1)


        return [avg, std, avgTime, stdTime], [avgTimes, stdTimes, self.computeResponseTimeGroup(self.negResponse)]


    def averageMissResponseTime(self, mode=1):
        """
        :param mode: 1 for using unprocessed data, 2 otherwise
        :return:
        """
        if mode == 1:
            dataGame = self.dataGame
        elif mode == 2:
            dataGame = self.processedData

        # overall avg and std miss response
        avg = dataGame.MissResponse.mean()
        #std = dataGame.MissResponse.std()
        std = self.bernouliSTD(dataGame.MissResponse)
        # compute avg and std response time of miss response
        avgTimes = self.dataGame.MissResponse.groupby(self.groupCons).mean().values.reshape(-1, 1)
        # stdTimes = self.dataGame.MissResponse.groupby(self.groupCons).std().values.reshape(-1, 1)
        stdTimes = self.dataGame.MissResponse.groupby(self.groupCons).apply(self.bernouliSTD).values.reshape(-1, 1)


        return [avg, std], [avgTimes, stdTimes]

    def computeResponseTimeGroup(self, response):
        '''
        :param response: response (responseTime, spawnTime)
        :return:
        '''
        if (self.groupConsRes is None):
            self.groupConsRes = np.floor(response.SpawnTime // 60).astype(int)
        responseTimes = (response.ResponseTime - response.SpawnTime)
        responseTimes = responseTimes[responseTimes>=0]
        #group by response times for each one minute and compute their avg and std
        reponseTimesMean = responseTimes.groupby(self.groupConsRes).mean().values.reshape(-1, 1)
        reponseTimesStd = responseTimes.groupby(self.groupConsRes).std().values.reshape(-1, 1)
        #reponseTimesStd = responseTimes.groupby(self.groupConsRes).apply(self.bernouliSTD).values.reshape(-1, 1)


        return np.concatenate([reponseTimesMean, reponseTimesStd], axis=-1)



    def computeResponseTimeSession(self, response, num_stimulus=18):
        #num_stimulus the number of stimulus in each session
        groupConsRes = np.floor(response.index // num_stimulus).astype(int)
        responseTimes = (response.ResponseTime - response.SpawnTime)
        responseTimes = responseTimes[responseTimes >= 0]
        # group by response times for each one minute and compute their avg and std
        reponseTimesMean = responseTimes.groupby(groupConsRes).mean().values.reshape(-1, 1)
        reponseTimesStd = responseTimes.groupby(groupConsRes).std().values.reshape(-1, 1)
        # reponseTimesStd = responseTimes.groupby(self.groupConsRes).apply(self.bernouliSTD).values.reshape(-1, 1)

        return np.concatenate([reponseTimesMean, reponseTimesStd], axis=-1)

    def computeResponseTimeNS(self, response):
        responseTimes = (response.ResponseTime - response.SpawnTime)
        groupedResponseTimes = responseTimes.groupby(self.groupCons)

        return groupedResponseTimes



    def computeGazeObject(self):
        #filter out the object and the player's gaze position when he gives correct answer
        data = self.dataGame[(self.dataGame.ResponseTime >=0) & (self.dataGame.GazeX >= 0) & (self.dataGame.GazeY >= 0) & (self.dataGame.PosResponse == 1)]

        return [data.ObjectX.values.flatten(), data.ObjectY.values.flatten(), data.GazeX.values.flatten(), data.GazeY.values.flatten()]
    def computeGaze(self):
        #filterout the gaze, only the validate ones,
        data = self.gazeData[(self.gazeData.GazeX >= 0) & (self.gazeData.GazeY >= 0)]

        return [data.GazeX.values.flatten(), data.GazeY.values.flatten()]

    def computeGazeArea(self):
        area = self.convexHullArea(self.gazeData[["Time", "GazeX", "GazeY"]])

        return area

    def computeGazeVTime(self):
        #compute the rotations and movements of head by computing distance between the following data with the previous data

        dataGaze = self.gazeData[["Time", "GazeX", "GazeY"]]

        groupCons = np.floor(dataGaze.Time // 61).astype(int)
        gazeTrajectoryArea = dataGaze.groupby(groupCons).apply(self.convexHullArea)

        return gazeTrajectoryArea

    def computeHead(self):
        #compute the rotations and movements of head by computing distance between the following data with the previous data

        dataHeadPos = self.gazeData[["Time", "HeadPosX", "HeadPosY", "HeadPosZ"]]
        dataHeadRot = self.gazeData[["Time", "HeadRotX", "HeadRotY", "HeadRotZ"]]
        groupCons = np.floor(dataHeadPos.Time // 1).astype(int)
        # Average the values
        dataHeadPos = dataHeadPos.groupby(groupCons).mean()
        dataHeadRot = dataHeadRot.groupby(groupCons).mean()
        # sampleAfter = (dataHeadPos.index) % 5 == 0
        # sampleBefore = (dataHeadPos.index + 2) % 5 == 0
        dataBeforePos = dataHeadPos.iloc[:-1]
        dataAfterPos = dataHeadPos.iloc[1:]

        dataBeforeRot = dataHeadRot.iloc[:-1]
        dataAfterRot = dataHeadRot.iloc[1:]

        headPosChanges = np.sum(np.sqrt(np.square(dataAfterPos.values - dataBeforePos.values)), axis=-1)

        headRotChanges = np.sum(np.sqrt(np.square(dataAfterRot.values - dataBeforeRot.values)), axis=-1)

        dataHead = pd.DataFrame(
            {"Time": dataHeadPos.Time[1:].values, "HeadPosChange": headPosChanges, "HeadRotChange": headRotChanges})

        #group  by times for each one minute
        groupCons = np.floor(dataHead.Time // 61).astype(int)
        dataHead = dataHead.groupby(groupCons)
        return dataHead

    #----------------------Gaze and Object Processing--------------------------------------#

    def responseGaze(self, reponse=1):
        """
        :param reponse: 1 for positive, 2 for negative, 3 for miss, 4 for all data
        :return: distance between gaze and object and gaze positions
        """
        if reponse==1:
            data = self.processedGoodResponse
        elif reponse==2:
            data = self.processedNegResponse
        elif reponse==3:
            data = self.processedMissResponse
        else:
            data = self.processedData

        if len(data) > 0:
            #compute response
            data = data.assign(DiffTime=pd.Series(data.ResponseTime - data.SpawnTime).values)
            #compute the distance between object and gaze
            objectPosition = data[['ObjectX','ObjectY']].values
            gazePosition = data[['GazeX','GazeY']].values

            distances = np.sqrt(np.sum(np.power(objectPosition - gazePosition, 2), axis=1))
            data = data.assign(Distance=pd.Series(distances).values)

            #compute distance between object
            objDistance = np.sqrt(np.sum(np.power(objectPosition[1:] - objectPosition[:-1], 2), axis=1))
            objDistance = np.insert(objDistance, 0, [0], axis=0)

            data = data.assign(ObjDistance = pd.Series(objDistance).values)
            data.loc[data.ResponseTime == -1, "ObjDistance"] = -1
            if len(data[:-1].ResponseTime == -1) > 1:
                data.loc[data[:-1].loc[data[:-1].ResponseTime == -1].index + 1, "ObjDistance"] = -1

        return data
    #compute convex hull of given points
    def convexHullArea(self, data):
        hull = ConvexHull(points=data[["GazeX", "GazeY"]].values)
        return hull.volume

