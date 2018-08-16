import pickle
import matplotlib.pyplot as plt
from globals import cluster_result_fname, KMEANS_NAME, DBSCAN_NAME

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from cycler import cycler
import itertools

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import numpy as np

import os.path

stemmer = SnowballStemmer('spanish')
the_stop_words = stopwords.words('spanish')

DEFAULT_PCA_NUM_COMPONENTS = 100


class ClusterVisualization:
    """Base Class that deals with plotting of data and writing result files

    Attributes:
        filename (str): Name of the file containing the data that will be visualized, the string before '.pickle'
            extension is expected.
        clean_input_filename (str): Full filename of the clean data

        **kwargs: Optional parameters such as 'pca_num_components'(int), 'load_now'(bool), 'visualize_now'(bool)
    """
    def __init__(self, filename, clean_input_filename, **kwargs):
        self.filename = filename
        self.clean_input_filename = clean_input_filename
        self.data, self.labels, self.cluster_num, self.words_in_sentences_raw, self.tdidf, self.vocab_size = (None,)*6
        self.outliers = []
        self.labels_indexes = []
        self.clusters = {}
        self.extra_title = '' if not hasattr(self, 'extra_title') else self.extra_title
        self.draw = True if 'draw' not in kwargs else kwargs['draw']

        if 'pca_num_components' in kwargs:
            self.pca_num_components = kwargs['pca_num_components']
        else:
            self.pca_num_components = DEFAULT_PCA_NUM_COMPONENTS

        load_now = 'load_now' in kwargs and kwargs['load_now']
        visualize_now = 'visualize_now' in kwargs and kwargs['visualize_now']

        if load_now:
            self.load_file()

        if visualize_now:
            self.visualize()

    def load_file(self):
        """Loads data stored in self.filename which was passed to the constructor"""
        with open('%s.pickle' % self.filename, 'rb') as f:
            self.data = pickle.load(f)
            [self.labels, self.cluster_num, self.words_in_sentences_raw, self.tdidf, self.vocab_size] = self.data
            self.obtain_clusters()

    def obtain_clusters(self):
        """This obtains a better representation for the clusters which is used by the visualize() step"""
        for i, label in enumerate(self.labels):
            if label == -1:
                self.outliers.append(i)
                continue

            if label not in self.clusters:
                self.labels_indexes.append(label)
                self.clusters[label] = []

            self.clusters[label].append(i)


    @staticmethod
    def get_style_it():
        """Gives style to different points in a cluster

        Returns: iterator
        """
        return itertools.cycle(cycler(marker=[
            's', '8', 'X', 'P', 'D', 'p', '*', 'H',
            9, 1, 2, 3, 4, '_', 'x', '|', 10, 11, 4]) * cycler(color=[
            '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0',
            '#f032e6', '#d2f53c', '#fabebe', '#008080', '#e6beff', '#aa6e28', '#fffac8',
            '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000080', '#808080'
        ]))

    def calc_vis_data_if_not_saved(self, X, pca_num_components=100, tsne_num_components=2):
        """Retrieves the positions of the points in a 2-dimensional space, which are calculated and the saved to disk
        or only loaded from disk if they already exist

        Args:
            X (numpy.matrix): A (sparse) 2-dimensional numpy matrix storing high dimensional points in its rows
            pca_num_components (int): PCA target dimension before applying t-SNE
            tsne_num_components (int): t-SNE target dimension, works only with value 2
        """
        fname = '%s.reduced_dim_PCA%d_SNE%d.pickle' % (self.clean_input_filename, pca_num_components, tsne_num_components)
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                Y = pickle.load(f)
        else:
            if pca_num_components == -1:
                pre_red_data = X
            else:
                print("Before PCA")
                pre_red_data = PCA(n_components=pca_num_components).fit_transform(X)
                print("After PCA")

            print("BEFORE TSNE")
            embeddings = TSNE(n_components=tsne_num_components)
            Y = embeddings.fit_transform(pre_red_data)
            with open(fname, 'wb') as f:
                pickle.dump(Y, f, pickle.HIGHEST_PROTOCOL)
            print("AFTER TSNE")

        return Y

    def output_clustered_tweets_to_file(self):
        """Writes the tweets to a file organized by cluster"""
        original_tweets = []

        with open('tweets.txt', 'r') as orig_tw_f:
            for line in orig_tw_f:
                original_tweets.append(line.replace('\n', ''))

        indexes_orig = []
        with open('indexes.txt', 'r') as indexes_f:
            for line in indexes_f:
                indexes_orig.append(int(line.replace('\n', '')))

        with open('%s_result_clustered_tweets_.txt' % self.filename ,
                  'w') as f:
            c_id = 0
            for label in self.labels_indexes:
                cluster = self.clusters[label]
                f.write('cluster #' + str(c_id + 1) + ' tweets\n')
                for idx in cluster:
                    clean_idx = idx
                    real_idx = indexes_orig[clean_idx]
                    real_sentence = original_tweets[real_idx]
                    output = '<<<' + real_sentence + '>>> :: at line: ' + str(real_idx) + '\n'
                    f.write(output)

                f.write('\n\n\n\n')

                c_id += 1

    @staticmethod
    def algo_name():
        return 'Base'

    def set_extra_title(self, extra):
        self.extra_title = extra

    def visualize(self):
        """Uses matplotlib to plot the data in a 2-D Graph"""
        if self.data is None:
            print("ERROR: Data has not been loaded")
            return

        tsne_num_components = 2
        X = self.tdidf.todense()
        Y = self.calc_vis_data_if_not_saved(X, pca_num_components=self.pca_num_components,
                                            tsne_num_components=tsne_num_components)

        style_c_m = self.get_style_it()

        fig, ax = plt.subplots()

        fig.set_figheight(15)
        fig.set_figwidth(15)
        max_labels = 60
        for idx, label in enumerate(self.labels_indexes):
            cluster = self.clusters[label]
            reduced_data_cluster = Y[cluster]

            A, B = zip(*reduced_data_cluster)
            if idx < max_labels:
                ax.scatter(A, B, **next(style_c_m), alpha=0.8, label='C#%d' % (int(label) + 1))
            else:
                ax.scatter(A, B, **next(style_c_m), alpha=0.8)

        outliers_reduced = Y[self.outliers]
        if outliers_reduced is not None and len(outliers_reduced) > 0:
            A, B = zip(*outliers_reduced)
            ax.scatter(A, B, c='black', s=0.3, alpha=0.3, label='Ruido')

        ax.grid(True, alpha=0.1)

        rows_num = self.cluster_num / 20
        cols_num = int(self.cluster_num / rows_num)

        ax.legend(bbox_to_anchor=(-0.15, 1.05, 1.25, 0.202), loc=3, ncol=cols_num, mode="expand", borderaxespad=0.)

        XX, YY = zip(*Y)

        min_x, max_x = int(np.min(XX)/10)*10, int(np.max(XX)/10)*10
        min_y, max_y = int(np.min(YY)/10)*10, int(np.max(YY)/10)*10
        extra_title = ', %s' % self.extra_title if len(self.extra_title) > 0 else ''
        title = 'Clustering tweets, algoritmo: %s%s' % (self.algo_name(), extra_title)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xticks(np.arange(min_x-10,max_x+30, 10))
        ax.set_yticks(np.arange(min_y-10,max_y+30, 10))
        fig.subplots_adjust(top=0.85)
        plt.ylabel('Y', rotation=0)
        if self.draw:
            plt.draw()
        plt.savefig('result_images/%s.png' % title.replace(' ', '_'))
        if not self.draw:
            plt.close()
        self.output_clustered_tweets_to_file()
        #print('Tamano vocabulario %d' %self.vocab_size)


class KMeansClusterVisualization(ClusterVisualization):
    """Configures ClusterVisualization to receive KMeans parameters

    Attributes:
        clean_input_filename (str): Full filename of the clean data
        centroid_num (int): Number of centroids for kmeans.
        **kwargs: Optional parameters such as 'pca_num_components'(int), 'load_now'(bool), 'visualize_now'(bool)
    """
    @staticmethod
    def algo_name():
        return KMEANS_NAME

    def __init__(self, clean_input_filename, centroid_num, **kwargs):
        self.set_extra_title('k = %d' % centroid_num)
        filename = cluster_result_fname(self.algo_name(), k=centroid_num)
        super().__init__(filename, clean_input_filename, **kwargs)


class DBScanClusterVisualization(ClusterVisualization):
    """Configures ClusterVisualization to receive DBSCAN parameters

     Attributes:
        clean_input_filename (str): Full filename of the clean data
        eps (float): Neighbours distance
        min_samples (int): Minimum number of neighbours to be a core point
        **kwargs: Optional parameters such as 'pca_num_components'(int), 'load_now'(bool), 'visualize_now'(bool)
    """
    @staticmethod
    def algo_name():
        return DBSCAN_NAME

    def __init__(self, clean_input_filename, eps, min_samples, **kwargs):
        self.set_extra_title('eps=%.3f, min samples = %d' % (eps, min_samples))
        filename = cluster_result_fname(self.algo_name(), eps='%.3f'%eps, minsamples=min_samples)
        super().__init__(filename, clean_input_filename, **kwargs)


