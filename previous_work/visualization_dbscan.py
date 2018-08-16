import pickle
import matplotlib.pyplot as plt
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('spanish')
the_stop_words = stopwords.words('spanish')

BINS_NUM = 7

eps=0.7
min_samples = 5
metric = 'euclidean'

with open('clustering_DBSCAN_eps.%.10f_minsamples.%d_metric.%s.pickle' % (eps, min_samples, metric), 'rb') as f:
    data = pickle.load(f)


labels = data[0]
cluster_num = data[1]
words_in_sentences_raw = data[2]
tdidf = data[3]
histograms = []
outliers = []

clusters = {}
labels_indexes = []
for i, label in enumerate(labels):
    if label == -1:
        outliers.append(i)
        continue

    if label not in clusters:
        labels_indexes.append(label)
        clusters[label] = []

    clusters[label].append(i)


for label in labels_indexes:
    cluster = clusters[label]
    histo_words = []
    histo_occurrences = []
    dict_words_to_list = dict()
    for idx in cluster:
        real_idx = idx
        sentence = words_in_sentences_raw[real_idx]
        for word in sentence:
            if word.lower() in the_stop_words:
                continue
            if word in dict_words_to_list.keys():
                histo_idx = dict_words_to_list[word]
                histo_occurrences[histo_idx] += 1
            else:
                histo_words.append(word)
                histo_occurrences.append(1)
                dict_words_to_list[word] = len(histo_words) - 1

    to_sort = []
    for i in range(0, len(histo_words)):
        to_sort.append((histo_occurrences[i], histo_words[i]))

    to_sort.sort(reverse=True)
    histo_words = []
    histo_occurrences = []
    count = 0
    for (occurrences, word) in to_sort:
        histo_words.append(word)
        histo_occurrences.append(occurrences)
        count += 1
        if count >= BINS_NUM:
            break

    histograms.append((histo_words, histo_occurrences))

original_tweets = []

with open('tweets.txt', 'r') as orig_tw_f:
    for line in orig_tw_f:
        original_tweets.append(line.replace('\n', ''))

indexes_orig = []
with open('indexes.txt', 'r') as indexes_f:
    for line in indexes_f:
        indexes_orig.append(int(line.replace('\n', '')))

with open('RESULT_clustering_DBSCAN_eps.%.10f_minsamples.%d_metric.%s.txt' % (eps, min_samples, metric), 'w') as f:
    c_id = 0
    for label in labels_indexes:
        cluster = clusters[label]
        f.write('cluster #' + str(c_id + 1) + ' tweets\n')
        for idx in cluster:
            clean_idx = idx
            real_idx = indexes_orig[clean_idx]
            real_sentence = original_tweets[real_idx]
            output = '<<<' + real_sentence + '>>> :: at line: ' + str(real_idx) + '\n'
            f.write(output)

        f.write('\n\n\n\n')

        c_id += 1

from cycler import cycler
import itertools

jet = plt.get_cmap('jet')
colors = iter(jet(np.linspace(0, 1, 10)))
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
pca_num_components = 2
tsne_num_components = 2
X = tdidf.todense()
print("Before PCA")
reduced_data = PCA(n_components=pca_num_components).fit_transform(X)

pre_red_data = PCA(n_components=50).fit_transform(X)

print("After PCA")

fig, ax = plt.subplots()

fig.set_figheight(15)
fig.set_figwidth(15)


def get_style_it():
    return itertools.cycle(cycler(marker=[
        's', '8', 'X', 'P', 'D', 'p', '*', 'H',
        9, 1, 2, 3, 4, '_', 'x', '|', 10, 11, 4]) * cycler(color=[
        '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0',
        '#f032e6', '#d2f53c', '#fabebe', '#008080', '#e6beff', '#aa6e28', '#fffac8',
        '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000080', '#808080'
    ]))

style_c_m = get_style_it()
print("len of reduced data: %d" %len(reduced_data))
print(reduced_data)
print("cluster num: %d" % cluster_num)


for idx, label in enumerate(labels_indexes):
    cluster = clusters[label]
    reduced_data_cluster = reduced_data[cluster]

    A, B = zip(*reduced_data_cluster)
    ax.scatter(A, B, **next(style_c_m), alpha=0.8, label='C#%d' % (int(label)+1))

outliers_reduced = reduced_data[outliers]
A, B = zip(*outliers_reduced)
ax.scatter(A, B, c='black', s=0.3, alpha=0.3)

title1 = 'Tweets. Visualización con PCA a dimension 2'
ax.set_title(title1)
ax.legend(bbox_to_anchor=(0., 1.08, 1., .102), loc=3, ncol=int(cluster_num/(cluster_num/20)), mode="expand", borderaxespad=0.)
fig.subplots_adjust(top=0.85)

ax.grid(True)

fig2, ax2 = plt.subplots()

fig2.set_figheight(15)
fig2.set_figwidth(15)

print("BEFORE TSNE")
embeddings = TSNE(n_components=tsne_num_components)
Y = embeddings.fit_transform(pre_red_data)
print("AFTER TSNE")
style_c_m = get_style_it()

for idx, label in enumerate(labels_indexes):
    cluster = clusters[label]
    reduced_data_cluster = Y[cluster]

    A, B = zip(*reduced_data_cluster)
    ax2.scatter(A, B, **next(style_c_m), alpha=0.8, label='C#%d' % (int(label)+1))

outliers_reduced = Y[outliers]
A, B = zip(*outliers_reduced)
ax2.scatter(A, B, c='black', s=0.3, alpha=0.3)
ax2.grid(True)
ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=int(cluster_num/(cluster_num/20)), mode="expand", borderaxespad=0.)

title2= 'Tweets. PCA Hasta dim=50 + SNE hasta dim=2'
ax2.set_title(title2)

fig2.subplots_adjust(top=0.85)

plt.show()
#plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
#plt.show()