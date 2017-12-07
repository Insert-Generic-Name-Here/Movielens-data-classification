import matplotlib.pyplot as plt
import lonely_boy2 as lb2
import pandas as pd
import numpy.matlib
import numpy as np


X = np.load('C:\\Users\\This PC\\Documents\\Coding\\Python\\pattern-recognition-project-2017\\finaldf.npy'); X = np.asmatrix(X); X= X.T
q = 15; theta = 2.5

[bel_new, repre_new] = lb2.screw_bsas(X[:,:1000].copy())
print("Clustering Labels: ", bel_new)
print("Centroids: ", repre_new)

[bel_opt, repre_opt] = lb2.clust_reassign(X.copy(), repre_new.copy(), bel_new.copy())
print("[Optimal] Clustering Labels: ", bel_new)
print("[Optimal] Centroids: ", repre_new)