import clustering_runner
import sklearn.cluster as skcluster


clean_data_fname = 'clean_data.txt'


def modif_dbscan(data, **parameters):
    assert(all(['eps' in parameters, 'min_samples' in parameters, 'metric' in parameters, 'algo' in parameters]))
    if 'mytfidf' in parameters and parameters['mytfidf']:
        data, sparsity = data.to_csr_matrix()
        print("sparsity before dbscan = %f" % sparsity)
    eps = parameters['eps']
    min_samples = parameters['min_samples']
    metric = parameters['metric']
    algo = parameters['algo']
    dbscan = skcluster.DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algo).fit(data)
    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters, 'clustering_DBSCAN_eps.%.10f_minsamples.%d_metric.%s' % (eps, min_samples, metric)


dbscann_runner = clustering_runner.ClusteringRunner(clean_data_fname, modif_dbscan, eps=0.7, min_samples=5, metric='euclidean', algo='auto')
dbscann_runner.run_algorithm()
