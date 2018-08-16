from sklearn.cluster import KMeans

import clustering_runner

from globals import cluster_result_fname, KMEANS_NAME, DEFAULT_CLEAN_DATA_FNAME, load_if_exists, save_result
from cluster_visualization import KMeansClusterVisualization

default_clusters_num = 10


def k_means_with_sklearn(data, **parameters):
    assert('k' in parameters)

    k = parameters['k']
    fname = cluster_result_fname(KMEANS_NAME, k=k)
    fname_check = '%s_pre_result.pickle' % fname
    already_exists, kmeans = load_if_exists(fname_check)
    if not already_exists:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        save_result(fname_check, kmeans)

    return kmeans.labels_, k, fname


def runner(input_filename, num_clusters):
    kmeans_runner = clustering_runner.ClusteringRunner(input_filename, k_means_with_sklearn, k=num_clusters)
    kmeans_runner.run_algorithm()


def runner_with_visualization(input_filename=DEFAULT_CLEAN_DATA_FNAME,
                              num_clusters=default_clusters_num, pca_num_components=100, draw=True):
    print('with runner')
    runner(input_filename, num_clusters)
    print('runner OK')
    print('with visualization')
    KMeansClusterVisualization(input_filename, num_clusters, load_now=True, visualize_now=True,
                               pca_num_components=pca_num_components, draw=draw)
    print('visualization OK')