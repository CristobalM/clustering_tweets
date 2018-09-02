from sklearn.mixture import GaussianMixture

import clustering_runner

from globals import cluster_result_fname, GMMEM_NAME, DEFAULT_CLEAN_DATA_FNAME, load_if_exists, save_result
from cluster_visualization import GMMemClusterVisualization

default_clusters_num = 10


def gmmem_sklearn(data, **parameters):
    assert('k' in parameters)

    k = parameters['k']
    fname = cluster_result_fname(GMMEM_NAME, k=k)
    fname_check = '%s_pre_result.pickle' % fname
    already_exists, gmmem = load_if_exists(fname_check)
    if not already_exists:
        gmmem = GaussianMixture(n_components=k, random_state=0).fit(data).predict(data)
        save_result(fname_check, gmmem)

    return gmmem, k, fname


def runner(input_filename, num_clusters):
    gmmem_runner = clustering_runner.ClusteringRunner(input_filename, gmmem_sklearn, k=num_clusters)
    gmmem_runner.run_algorithm(with_svd=True)


def runner_with_visualization(input_filename=DEFAULT_CLEAN_DATA_FNAME,
                              num_clusters=default_clusters_num, pca_num_components=100, draw=True):
    print('with runner')
    runner(input_filename, num_clusters)
    print('runner OK')
    print('with visualization')
    viz = GMMemClusterVisualization(input_filename, num_clusters, load_now=True, visualize_now=True,
                               pca_num_components=pca_num_components, draw=draw)
    print('visualization OK')
    return viz.extra_data()