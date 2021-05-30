import glob
import pandas as pd
import numpy as np
import random

class DataGenerator:

    def __init__(self, features_file, labels_file, offline=False, triplet=False):
        self.i = 0

        X = np.load(features_file)
        y = np.load(labels_file)

        if offline:
            if triplet:
                self.data = self.readTripletOffline(X, y)
            else:
                self.data = self.readOffline(X, y)
        else:
            self.data = self.readData(X, y)

        self.train_n = len( self.data)



    def fetch(self):
        i = 0
        while i < len(self.data):
            data_i = self.data[i]
            yield data_i[0], data_i[1] #return data and the label
            i+=1

    def fetch_offline(self):
        i = 0

        while i < len(self.data):
            data_i = self.data[i]
            yield data_i[0], data_i[1], data_i[2]  # return anchor, positif, negative
            i += 1


    def fetch_triplet_offline(self):
        i = 0

        while i < len(self.data):
            data_i = self.data[i]
            yield data_i[0], data_i[1], data_i[2], data_i[3]  # return anchor, positif, negative achor and the negative
            i += 1

    def readData(self, X, y):
        data = []

        for i in range(len(X)):
            data.append([X[i], y[i]])
        return data


    def readOffline(self, X, y):
        data = []
        X_positif = X[y==1]
        X_negatif = X[y==0]


        for i in range(len(X_positif) - 1):
                for j in range(len(X_negatif)):
                    data.append([X_positif[i], X_positif[i+1], X_negatif[j]])

        for i in range(len(X_negatif)//2):
            for j in range(len(X_positif)):
                data.append([X_positif[i], X_positif[i + 1], X_negatif[j]])

        return data

    def readTripletOffline(self, X, y):
        data = []
        X_positif = X[y==1]
        X_negatif = X[y==0]


        for i in range(len(X_positif) - 1):
            for k in range(i+1, len(X_positif)):
                for j in range(len(X_negatif)//2):
                    for l in range(j+1, len(X_negatif)//3):
                        data.append([X_positif[i], X_positif[k], X_negatif[j], X_negatif[l]])



        return data





