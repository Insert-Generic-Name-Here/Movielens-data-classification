# MatPlotLib Configuration
import PyQt5
get_ipython().magic('matplotlib qt')
from matplotlib import style;  style.use('ggplot')
# Import Crucial Libraries
import numpy as np
import numpy.matlib
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


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
                tmp = euclidean(centroid, sample)
                if (tmp < minDist):
                    minDist = tmp
                    centID = cntID
        except:
            pass
        return minDist, centID
    
    
    def __getEuclideanDistances(self, data, size):
        minED = np.inf; maxED = -np.inf
        
        for column_i in tqdm(range(size), desc='Computing (Min/Max) Euclidean Distances...'):                
            for column_j in range(size):
                if (column_i == column_j):
                    continue
                dist = euclidean(data[:,column_i], data[:,column_j])
                if (dist < minED):
                    minED = dist
                if (dist > maxED):
                    maxED = dist
                    
        return minED, maxED
    
    
    def __findIndexofMax(self, dct):
        minVal = np.inf       
        minKey = None
        for key in dct:
            tmp = dct[key]
            if (tmp < minVal):
                minVal = tmp
                minKey = key
        return minKey
    
    
    def __findOptimalCluster(self, clusters):
        clusters_frq = {}
        min_cluster = np.min(clusters)
        for cluster in tqdm(clusters, desc='Finding Optimal Cluster...'):
            if (cluster == min_cluster):
                continue
            try:
                clusters_frq[cluster] += 1
            except:
                clusters_frq[cluster] = 1
        opt_cluster = None; frq_opt_cluster = -np.inf
        
        for key in clusters_frq:
            tmp = clusters_frq[key]
            if (tmp > frq_opt_cluster):
                frq_opt_cluster = tmp
                opt_cluster = key
        return opt_cluster
    
    
    def __findOptimalTheta(self, opt_cluster, clusters, theta):
        cl_start = 0; cl_fin = 0; cl_key = None
        found = False
        cl_ranges = {}
        
        for i in range(len(clusters)):
            if (clusters[i] == opt_cluster):
                if (not found):
                    cl_start = i
                    cl_fin = i
                    found = True
                else:
                    cl_fin += 1
            else:
                if (found):
                    tmp = [cl_start, cl_fin, (cl_fin-cl_start)]
                    cl_ranges[i] = tmp
        
        for key in cl_ranges:
            max_range = -np.inf
            val = cl_ranges[key][2]
            if (val > max_range):
                max_range = val
                cl_key = key
        
        opt_theta_range = cl_ranges[cl_key]
        theta_avg = 0
        for i in range(opt_theta_range[0], opt_theta_range[1]+1):
            theta_avg += theta[i]
        
        theta_avg = theta_avg / (opt_theta_range[1] - opt_theta_range[0] + 1)
        return theta_avg   
    
    
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
    
    
    def fit_best(self, data, n_times=20, first_time=True, plot_graph=True):
        N, l = data.shape
        if (first_time):
            minDist, maxDist = self.__getEuclideanDistances(data, l)
            dists = np.save('comp-data/2-bsas-comp-data/min-max-euclidean-distances.npy', np.array([minDist, maxDist], dtype=np.float))
        else:
            minDist, maxDist = np.load('comp-data/2-bsas-comp-data/min-max-euclidean-distances.npy')

        meanDist = (minDist + maxDist)/2
        theta_min = 0.25 * meanDist; theta_max = 1.75 * meanDist
        n_theta = 50
        s = (theta_max - theta_min)/(n_theta - 1)
        
        if (first_time):
            total_clusters = []
            total_theta = np.arange(theta_min, theta_max+s, s)
            for theta in tqdm(total_theta, desc=('Running BSAS...')):
                max_clusters = -np.inf
                for i in np.arange(n_times):
                    clf = BSAS(theta=theta,q=l)
                    order = np.random.permutation(range(l))
                    clf.fit(data, order)
                    clusters, centroids = clf.predict()
                    clustersN = len(clusters)
                    if (clustersN > max_clusters):
                        max_clusters = clustersN
                total_clusters = total_clusters + [max_clusters]
        
            np.save('comp-data/2-bsas-comp-data/total_clusters.npy', np.array(total_clusters, dtype=np.int))
            np.save('comp-data/2-bsas-comp-data/total_theta.npy', np.array(total_theta, dtype=np.float))
        else:
            total_clusters = np.load('comp-data/2-bsas-comp-data/total_clusters.npy')
            total_theta = np.load('comp-data/2-bsas-comp-data/total_theta.npy')
        
        
        if (plot_graph==True):
            plt.plot(total_theta, total_clusters, 'b-') 
            plt.xlabel('theta')
            plt.ylabel('#clusters')
            plt.title('#clusters versus theta')
            plt.grid()
            plt.show()
        
        opt_cluster = self.__findOptimalCluster(total_clusters) #print (opt_cluster)
        opt_theta = self.__findOptimalTheta(opt_cluster, total_clusters, total_theta) #print (opt_theta)
        
        self.theta = opt_theta
        self.q = opt_cluster
        
    
    def predict(self):
        real_centroids = {}
        for key in self.clusters:
            real_centroids[key] = self.__getCentroid(self.centroids[key], self.clusters[key].shape)
        return self.clusters, real_centroids
    
    
    def specs(self):
        return self.theta, self.q