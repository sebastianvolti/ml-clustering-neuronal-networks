import copy
import numpy as np
import pandas as pd
import utils
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

#centroide_init
SMART = 'k-means++' 
RANDOM ='random' 

#alg_type
AUTO="auto"
FULL="full"
ELKAN="elkan"

def preprocess(dataset):
    X = np.array(dataset[['income', 'education', 'age-group','race', 'religion']])
    y = np.array(dataset['ideology'])
    X.shape
    return X, y

def means(dataset, runs_number, limit_iter, plot_grafics):

    #parametros del algoritmo kmeans, no vale la pena dejar configurables
    clusters_number = 3
    k_means_type = 0
    tol_param =  0.0001
    alg_type = AUTO
    centroide_init = SMART
    kmeans_data, kmeans_y = preprocess(dataset)
    kmeans_result = KMeans(n_clusters=clusters_number, init=centroide_init, n_init=runs_number, max_iter=limit_iter, tol=tol_param, algorithm=alg_type).fit(kmeans_data)

    if (plot_grafics > 0):
        labels = kmeans_result.predict(kmeans_data)
        # Getting the cluster centers
        C = kmeans_result.cluster_centers_
        colores=['red','green','blue']
        asignar=[]
        for row in labels:
            asignar.append(colores[int(row)])
            
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(kmeans_data[:, 0], kmeans_data[:, 1], kmeans_data[:, 2], c=asignar,s=60)
        ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)
        plt.show()