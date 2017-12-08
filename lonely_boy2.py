import pandas as pd
import matplotlib.pyplot as plt
import numpy.matlib
import numpy as np
from tqdm import tqdm


def addOneHot(df):
    return df.join(pd.get_dummies(df))

class BSAS:
    def __init__(self, theta=None, q=None):
        # theta: Dissimarity Threshold
        # q: Max #Clusters
        self.theta = theta; self.q = q
        self.clusters = {}; self.centroids = {}
        
    def __getCentroid(self, X, Y):
        try:
            probe = Y[1]
            return np.divide(X, Y[0])
        except:
            return X
    
    def __findClosestCluster(self, clusters, centroids, sample):
        centID = 0
        cluster_population = clusters[centID].shape
        centroid = self.__getCentroid(centroids[centID], cluster_population)

        minDist = euclidean(centroid, sample)
        try:
            for cntID in centroids:
                if (cntID == 0):
                    continue
                cluster_population = clusters[cntID].shape
                centroid = self.__getCentroid(centroids[cntID], cluster_population)

                if (tmp < minDist):
                    minDist = tmp
                    centID = cntID
        except:
            pass
        return minDist, centID
    
    def fit(self, data, order):
        m = 1 #Clusters/Centroids
        clusters = {}; centroids = {}
        
        first_sample = data[:,order[0]]
        clusters[m-1] = first_sample; centroids[m-1] = np.add(np.zeros_like(first_sample), first_sample)
        
        N, l = data.shape
        for i in range(1,l):
            sample = data[:,order[i]]
            dist, k = self.__findClosestCluster(clusters, centroids, sample)
            if ((dist > self.theta) and (m < self.q)):
                m += 1
                clusters[m-1] = sample; centroids[m-1] = np.add(np.zeros_like(sample), sample)
            else:
                clusters[k] = np.vstack((clusters[k], sample))
                centroids[k] = np.add(centroids[k], sample)
            
        self.clusters = clusters
        self.centroids = centroids
        
    def predict(self):
        real_centroids = {}
        for key in self.clusters:
            real_centroids[key] = self.__getCentroid(self.centroids[key], self.clusters[key].shape)
        return self.clusters, real_centroids