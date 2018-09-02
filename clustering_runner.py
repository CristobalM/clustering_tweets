import pickle
import re
import clean_utils

from sklearn.feature_extraction.text import TfidfVectorizer as TfidfVectorizer

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.decomposition import TruncatedSVD
from globals import save_result, load_if_exists

stemmer = SnowballStemmer('spanish')
the_stop_words = stopwords.words('spanish')


class ClusteringRunner:
    """Has common functionality for running and algorithm and saving its data

        Arguments:
            clean_input_filename (str): Full filename of the clean data
            algorithm (function): Algorithm to run
    """
    def __init__(self, clean_input_filename, algorithm, **algorithm_parameters):
        self.algorithm = algorithm
        self.clean_input_filename = clean_input_filename
        self.algorithm_parameters = algorithm_parameters

    def run_algorithm(self, with_svd=False):
        with open(self.clean_input_filename, 'r') as f:
            words_in_sentences_raw = []
            docs = []
            for line in f:
                line = re.sub('\n', '', line).strip().lower()
                words_in_line = line.split(' ')
                words_in_sentences_raw.append(words_in_line)
                #words_in_line, _ = clean_utils.without_stop_words(words_in_line)
                line = ' '.join(words_in_line)
                docs.append(line)

            vectorizer = TfidfVectorizer(min_df=1, ngram_range=(3, 3))
            #vectorizer = TfidfVectorizer(min_df=1)
            tdidf = vectorizer.fit_transform(docs)
            vocab_size = len(vectorizer.vocabulary_)

            data_for_alg = tdidf

            if with_svd:
                svd_components = 2500
                svd_file = 'results_data/SVD%d_%s'% (svd_components, self.clean_input_filename)

                already_exists, svd_result = load_if_exists(svd_file)
                if not already_exists:
                    print('in SVD STEP')
                    svd_result = TruncatedSVD(n_components=svd_components, n_iter=7, random_state=0).fit_transform(tdidf)
                    print('SVD STEP OK')
                    save_result(svd_file, svd_result)
                else:
                    print('SVD ALREADY SAVED')
                data_for_alg = svd_result
            parameters_str = ','.join(['%s=%s' % (str(key), str(value)) for key, value in self.algorithm_parameters.items()])
            print('RUNNING ALGORITHM with parameters: %s' % parameters_str)
            result, extra, output_fname = self.algorithm(data_for_alg, **self.algorithm_parameters)
            #result, extra, output_fname = self.algorithm(tdidf, **self.algorithm_parameters)
            print('ALGORITHM OK')

            with open('%s.pickle' % output_fname, 'wb') as f:
                pickle.dump([result, extra, words_in_sentences_raw, tdidf, vocab_size], f, pickle.HIGHEST_PROTOCOL)

            #print(list(result))
            #print('Extra: %d' % extra)