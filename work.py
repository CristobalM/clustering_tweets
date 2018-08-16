import kmeans_runner, dbscan_runner
#import matplotlib.pyplot as plt
import numpy as np
import os
from cleaning_dataset import cleaning
from globals import DEFAULT_CLEAN_DATA_FNAME, INDEXES_FNAME, RAW_INPUT_FNAME
import sys


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


kmeans_k = np.concatenate((np.arange(2, 21, 2, dtype=int), np.arange(100, 501, 100)))
print('running kmeans')
for k in kmeans_k:
    kmeans_runner.runner_with_visualization(num_clusters=k, pca_num_components=1000, draw=False)


print('running dbscan')
eps_vals = np.arange(0.01, 1, 0.05)
msamples = np.arange(4, 30, 3)

for msample in msamples:    
    for eps in eps_vals:
        dbscan_runner.runner_with_visualization(eps=eps, min_samples=int(msample), pca_num_components=1000, draw=False)

