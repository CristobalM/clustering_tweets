from sklearn.cluster import KMeans
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import mmh3
import pickle

import matplotlib.pyplot as plt
import numpy as np


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
            #words_in_line = remove_consecutive_repeated(words_in_line)
            #words_in_line = list(filter(lambda x: x != 'bachelet', words_in_line))
            words_in_sentences_raw.append(words_in_line)
            #words_in_line = list(map(stemmer.stem, words_in_line))
            #words_in_line = list(map(lambda x: mmh3.hash(x), words_in_line))
            words_in_line.sort()
            if len(words_in_line) > 5:
                normalized.append(words_in_line)
                mapping_lines.append(counter)
            counter += 1
    return normalized, mapping_lines, words_in_sentences_raw



normalized, mapping_lines, words_in_sentences_raw = normalize_data()

#for sentence in normalized:
#    print(sentence)
indexed_words = []
mapped_hash = {}
reverse_mapped_hash = {}
print(normalized)

for tweet in normalized:
    for word in tweet:
        hashed = mmh3.hash(word)
        if hashed in mapped_hash:
            indexed_words[mapped_hash[hashed]][0] += 1
        else:
            new_index = len(indexed_words)
            mapped_hash[hashed] = len(indexed_words)
            reverse_mapped_hash[hashed] = word
            indexed_words.append([1, hashed])

sorted_indexed_words = sorted(indexed_words, key=lambda a: a[0], reverse=True)
descendant_frequency = list(map(lambda a: a[0], sorted_indexed_words))
cut_freq = list(filter(lambda f: f >= 100, descendant_frequency))
print(len(sorted_indexed_words))
print(sorted_indexed_words)
#ten most frequent words
tmfw = [reverse_mapped_hash[sorted_indexed_words[i][1]] for i in range(0, min(10, len(sorted_indexed_words)))]
tmfw_freq = descendant_frequency[0:10]
print(tmfw)
print(tmfw_freq)

x_dom = np.arange(0, len(cut_freq), 1)
plt.plot(x_dom, cut_freq, marker='o')
#plt.xticks(x_dom)
plt.xlabel('Palabras')
plt.ylabel('Frecuencia')
plt.grid()
plt.figure()

plt.bar(range(len(tmfw_freq)), tmfw_freq, align='center', edgecolor='black', color='lightblue')
plt.xticks(range(len(tmfw)), tmfw, size='small')
plt.xlabel('Palabras')
plt.ylabel('Frecuencia')
plt.grid()

plt.show()
