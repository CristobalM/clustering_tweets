from sklearn.cluster import KMeans
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import mmh3
import pickle

from k_means import jaccard_dist, k_means

#import nltk
#nltk.download('stopwords')

stemmer = SnowballStemmer('spanish')
the_stop_words = stopwords.words('spanish')

NUM_CLUSTERS = 4



def strcmp(a, b):
    len_a = len(a)
    len_b = len(b)
    min_len = min(len_a, len_b)
    i = 0
    while i < min_len:
        if a[i] < b[i]:
            return -1
        elif a[i] > b[i]:
            return 1
        else:
            i += 1

    if len_a != len_b:
        if min_len == len_a:
            return -1
        else:
            return 1

    return 0

def without_stop_words(a_list):
    out = []

    for word in a_list:
        if word not in the_stop_words:
            out.append(word)

    return out

def remove_consecutive_repeated(a_list):
    out = []
    prev = ''
    for word in a_list:
        if word != prev:
            out.append(word)
        prev = word

    return out

def normalize_data():
    words_in_sentences_raw = []
    counter = 0
    mapping_lines = []
    normalized = []
    with open('clean_data.txt', 'r') as f:
        for line in f:
            no_endline_line = re.sub('\n', '', line).strip()
            #no_endline_line = parsetree(no_endline_line, lemmatta=True)
            words_in_line = no_endline_line.split(' ')
            words_in_line = list(map(lambda x: x.lower(), words_in_line))
            words_in_line = without_stop_words(words_in_line)
            words_in_line.sort()
            words_in_line = remove_consecutive_repeated(words_in_line)
            #words_in_line = list(filter(lambda x: x != 'bachelet', words_in_line))
            words_in_sentences_raw.append(words_in_line)
            words_in_line = list(map(stemmer.stem, words_in_line))
            words_in_line = list(map(lambda x: mmh3.hash(x), words_in_line))
            words_in_line.sort()
            if len(words_in_line) > 5:
                normalized.append(words_in_line)
                mapping_lines.append(counter)

            counter += 1


    return normalized, mapping_lines, words_in_sentences_raw



normalized, mapping_lines, words_in_sentences_raw = normalize_data()

#for sentence in normalized:
#    print(sentence)

clusters, result, centroids = k_means(normalized, NUM_CLUSTERS, jaccard_dist)


#print(clusters)
with open('cluster_'+ str(NUM_CLUSTERS)  + '.pickle', 'wb') as f:
    pickle.dump([clusters, result, centroids, mapping_lines, words_in_sentences_raw], f, pickle.HIGHEST_PROTOCOL)