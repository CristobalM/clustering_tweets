import descriptors_production
import pprint as pp
import k_means
import pickle
import re
import clean_utils
from sklearn.feature_extraction.text import TfidfVectorizer as TfidfVectorizer

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('spanish')
the_stop_words = stopwords.words('spanish')


class ClusteringRunner:
    def __init__(self, clean_data_filename, algorithm, **algorithm_parameters):
        self.algorithm = algorithm
        self.clean_data_filename = clean_data_filename
        self.algorithm_parameters = algorithm_parameters

    def run_algorithm(self):
        with open(self.clean_data_filename, 'r') as f:
            dhandler = descriptors_production.DocumentsHandler()
            words_in_sentences_raw = []
            docs = []
            for line in f:
                line = re.sub('\n', '', line).strip().lower()
                words_in_line = line.split(' ')
                words_in_sentences_raw.append(words_in_line)
                words_in_line, _ = clean_utils.without_stop_words(words_in_line)
                line = ' '.join(words_in_line)
                docs.append(line)
                dhandler.enter_document(line)

            vectorizer = TfidfVectorizer(min_df=1)
            tdidf = vectorizer.fit_transform(docs)
            #dhandler.convert_to_tfidf()
            #tdidf, sparsity = dhandler.to_csr_matrix()

            result, extra, output_fname = self.algorithm(tdidf, mytfidf=False, **self.algorithm_parameters)
            with open('%s.pickle' % output_fname, 'wb') as f:
                pickle.dump([result, extra, words_in_sentences_raw, tdidf], f, pickle.HIGHEST_PROTOCOL)
            print(list(result))
            print(extra)
            """
            result, extra, output_fname = self.algorithm(tdidf, **self.algorithm_parameters)
            with open('%s.pickle' % output_fname, 'wb') as f:
                pickle.dump([result, extra, words_in_sentences_raw], f, pickle.HIGHEST_PROTOCOL)
            print(list(result))
            print(extra)
            """

