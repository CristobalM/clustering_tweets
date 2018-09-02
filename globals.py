import os.path
import pickle


def cluster_result_fname(algo_name, **params):
    fname_base = 'results_data/%s_CLUSTERING_RESULT_' % algo_name
    params_list = []
    for key, value in params.items():
        params_list.append('%s=%s' % (str(key), str(value)))

    params_string = '_'.join(params_list)

    return fname_base + params_string


def load_if_exists(fname):
    fname = '%s' % fname
    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            result = pickle.load(f)
            return True, result

    return False, None


def save_result(fname, result):
    with open(fname, 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)



KMEANS_NAME = 'KMeans'
DBSCAN_NAME = 'DBScan'
GMMEM_NAME = 'GMMem'

DEFAULT_CLEAN_DATA_FNAME = 'clean_data.txt'
INDEXES_FNAME = 'indexes.txt'
RAW_INPUT_FNAME = 'tweets.txt'