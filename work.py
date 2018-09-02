#!/usr/bin/env python
# -*- coding: utf-8 -*-

import kmeans_runner, dbscan_runner, gmem_runner
import matplotlib.pyplot as plt
import numpy as np
import os
from cleaning_dataset import cleaning
from globals import DEFAULT_CLEAN_DATA_FNAME, INDEXES_FNAME, RAW_INPUT_FNAME
import sys
from cycler import cycler
import itertools

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


create_folder('result_images')
create_folder('results_data')

if not os.path.isfile(DEFAULT_CLEAN_DATA_FNAME) or not os.path.isfile(INDEXES_FNAME):
    if not os.path.isfile(RAW_INPUT_FNAME):
        print('No se encontro el archivo `tweets.txt`')
        sys.exit(0)
    cleaning()


kmeans_k_little = np.arange(2, 21, 2, dtype=int)
kmeans_k_big = np.arange(100, 501, 100, dtype=int)
kmeans_k = np.concatenate((kmeans_k_little, kmeans_k_big))

gmmem_k = np.arange(2, 11, 2, dtype=int)



print('running dbscan')
eps_vals = np.arange(0.01, 1, 0.05)
msamples = np.arange(4, 30, 3)


def get_color_cycler():
    return itertools.cycle(cycler(color=[
        '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0',
        '#f032e6', '#d2f53c', '#fabebe', '#008080', '#e6beff', '#aa6e28', '#fffac8',
        '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000080', '#808080'
    ]))


def plot_silhouette(X, Y, title, labels, xlabel, ylabel, xticks=None, yticks=None):
    fname = title.replace(' ', '_')
    color_cycler = get_color_cycler()
    if os.path.isfile(fname):
        return
    plt.figure(figsize=(10, 7))
    for i in range(len(X)):
        XX = X[i]
        YY = Y[i]
        none_p = list(filter(lambda x: YY[x] is not None, [i for i in range(len(XX))]))
        XX = [XX[u] for u in none_p]
        YY = [YY[u] for u in none_p]
        plt.plot(XX, YY, linestyle='-.', marker='D', label=labels[i], **next(color_cycler))


    plt.title(title)
    if len(labels) > 1 and labels[0] != '':
        plt.legend()
    plt.grid(alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    plt.savefig('result_images/%s.png' % fname)
    plt.close()



print('running kmeans')
silhouettes = []
for k in kmeans_k:
    extra = kmeans_runner.runner_with_visualization(num_clusters=k, pca_num_components=1000, draw=False)
    silhouette_avg = extra['silhouette_avg']
    silhouettes.append(silhouette_avg)
plot_silhouette(X=[kmeans_k_little],
                Y=[silhouettes[:len(kmeans_k_little)]],
                title=u'KMeans Silhouette, Pequeños k',
                labels=[''],
                xlabel=u'# de Centroides (k)',
                ylabel=u'Score promedio',
                xticks=np.arange(kmeans_k_little[0], kmeans_k_little[-1]+0.5, 2))
plot_silhouette(X=[kmeans_k_big],
                Y=[silhouettes[len(kmeans_k_little):]],
                title=u'KMeans Silhouette, Grandes k',
                labels=[''],
                xlabel=u'# de Centroides (k)',
                ylabel=u'Score promedio',
                xticks=np.arange(kmeans_k_big[0],
                                 kmeans_k_big[-1]+1,
                                 int((kmeans_k_big[-1]-kmeans_k_big[0])/len(kmeans_k_big))))

all_sh = []
for msample in msamples:
    silhouettes = []
    for eps in eps_vals:
        extra = dbscan_runner.runner_with_visualization(eps=eps, min_samples=int(msample), pca_num_components=1000, draw=False)
        silhouette_avg = extra['silhouette_avg']
        silhouettes.append(silhouette_avg)
    all_sh.append(silhouettes)
plot_silhouette(X=[eps_vals for i in range(len(all_sh))],
                Y=all_sh,
                title='DBSCAN Silhouette plot',
                labels=[u'Muestras mínimas/core = %d' % msample for msample in msamples],
                xlabel=u'Distancia Mínima (eps)',
                ylabel=u'Score promedio',
                xticks=np.arange(0, 1.05, 0.1))



print('running GMM EM')
silhouettes = []
for k in gmmem_k:
    extra = gmem_runner.runner_with_visualization(num_clusters=k, pca_num_components=1000, draw=False)
    silhouette_avg = extra['silhouette_avg']
    silhouettes.append(silhouette_avg)
plot_silhouette(X=[gmmem_k],
                Y=[silhouettes],
                title=u'GMM-EM Silhouette',
                labels=[''],
                xlabel=u'# de Centroides (k)',
                ylabel=u'Score promedio',
                xticks=np.arange(gmmem_k[0], gmmem_k[-1]+1,int((gmmem_k[-1]-gmmem_k[0])/len(gmmem_k))))
