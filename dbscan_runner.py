import clustering_runner
import sklearn.cluster as skcluster
from globals import cluster_result_fname, DBSCAN_NAME, DEFAULT_CLEAN_DATA_FNAME, load_if_exists, save_result
from cluster_visualization import DBScanClusterVisualization
default_eps = 0.7
default_min_samples = 5


def modif_dbscan(data, **parameters):
    assert(all(['eps' in parameters, 'min_samples' in parameters, 'metric' in parameters, 'algo' in parameters]))

    eps = parameters['eps']
    min_samples = parameters['min_samples']
    metric = parameters['metric']
    algo = parameters['algo']

    fname = cluster_result_fname(DBSCAN_NAME, eps='%.3f'%eps, minsamples=min_samples)
    fname_check = '%s_pre_result.pickle' % fname
    already_exists, dbscan = load_if_exists(fname_check)

    if not already_exists:
        dbscan = skcluster.DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algo).fit(data)
        save_result(fname_check, dbscan)

    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return labels, n_clusters, fname


def runner(input_filename, eps, min_samples, metric='cosine', algo='auto'):
    dbscann_runner = clustering_runner.ClusteringRunner(input_filename, modif_dbscan,
                                                        eps=eps, min_samples=min_samples,
                                                        metric=metric, algo=algo)
    dbscann_runner.run_algorithm()
    """
    dbscann_runner = clustering_runner.ClusteringRunner(clean_data_fname, modif_dbscan, eps=0.7, min_samples=5, metric='euclidean', algo='auto')
    dbscann_runner.run_algorithm()
    """


def runner_with_visualization(input_filename=DEFAULT_CLEAN_DATA_FNAME,
                              eps=0.7, min_samples=default_min_samples, pca_num_components=100, draw=True):
    runner(input_filename, eps, min_samples)
    viz = DBScanClusterVisualization(input_filename, eps, min_samples,
                               load_now=True, visualize_now=True, pca_num_components=pca_num_components, draw=draw)
    return viz.extra_data()
