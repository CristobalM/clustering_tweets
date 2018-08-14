"""
import descriptors_production
import pprint as pp
import k_means
import pickle
import re
import clean_utils

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('spanish')
the_stop_words = stopwords.words('spanish')


def dist_fun_nv(nv_a, nv_b):
    '''
    :param nv_a:
    :type nv_a descriptors_production.NiceVector
    :param nv_b:
    :type nv_b descriptors_production.NiceVector
    :return: float
    '''
    return nv_a.diff_vec_metric(nv_b)


NUM_CLUSTERS = 7

with open('clean_data.txt', 'r') as f:
    dhandler = descriptors_production.DocumentsHandler()
    words_in_sentences_raw = []
    for line in f:
        line = re.sub('\n', '', line).strip().lower()
        words_in_line = line.split(' ')
        words_in_sentences_raw.append(words_in_line)
        words_in_line, _ = clean_utils.without_stop_words(words_in_line)
        line = ' '.join(words_in_line)
        dhandler.enter_document(line)

    dhandler.convert_to_tfidf()

    clusters, result, centroids = k_means.k_means(dhandler.documents_nice_vectors, NUM_CLUSTERS, dist_fun_nv)
    with open('cluster_' + str(NUM_CLUSTERS) + '.pickle', 'wb') as f:
        pickle.dump([clusters, result, centroids, words_in_sentences_raw], f, pickle.HIGHEST_PROTOCOL)
    print(clusters)
    print(result)
    print(centroids)
# dhandler.documents_nice_vectors
"""

import clustering_runner
import k_means

def dist_fun_nv(nv_a, nv_b):
    '''
    :param nv_a:
    :type nv_a descriptors_production.NiceVector
    :param nv_b:
    :type nv_b descriptors_production.NiceVector
    :return: float
    '''
    return nv_a.diff_vec_metric(nv_b)


def modif_k_means(data, **parameters):
    assert(all(['k' in parameters, 'dist_fun' in parameters]))
    data = data.documents_nice_vectors

    k = parameters['k']
    dist_fun = parameters['dist_fun']

    return k_means.k_means(data, k, dist_fun) + ('cluster_' + str(NUM_CLUSTERS),)


clean_data_fname = 'clean_data.txt'
NUM_CLUSTERS = 7

kmeans_runner = clustering_runner.ClusteringRunner(clean_data_fname, modif_k_means, k=NUM_CLUSTERS, dist_fun=dist_fun_nv)
kmeans_runner.run_algorithm()
