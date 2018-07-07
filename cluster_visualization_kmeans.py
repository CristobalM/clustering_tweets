import pickle
import matplotlib.pyplot as plt
import numpy as np
import heapq

NUM_CLUSTERS = 4
BINS_NUM = 7

with open('cluster_' + str(NUM_CLUSTERS) + '.pickle', 'rb') as f:
    data = pickle.load(f)


clusters = data[0]
result = data[1]
centroids = data[2]
mapping_lines = data[3]
words_in_sentences_raw = data[4]
histograms = []
for cluster in clusters:
    histo_words = []
    histo_occurrences = []
    dict_words_to_list = dict()
    for idx in cluster:
        real_idx = mapping_lines[idx]
        sentence = words_in_sentences_raw[real_idx]
        for word in sentence:
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

with open('results_clusters_'+str(NUM_CLUSTERS)+'.txt', 'w') as f:
    c_id = 0
    for cluster in clusters:
        f.write('cluster #' + str(c_id +1) + ' tweets\n')
        the_centroid_clean_idx = mapping_lines[centroids[c_id]]
        the_centroid_real_idx = indexes_orig[the_centroid_clean_idx]
        real_centroid = original_tweets[the_centroid_real_idx]
        f.write('centroid is: \n' + real_centroid + '\n\n\n')
        for idx in cluster:
            clean_idx = mapping_lines[idx]
            real_idx = indexes_orig[clean_idx]
            real_sentence = original_tweets[real_idx]
            output = '<<<' + real_sentence + '>>> :: at line: ' + str(real_idx) + '\n'
            f.write(output)

        f.write('\n\n\n\n')

        c_id += 1


for i in range(0, NUM_CLUSTERS):
    plt.figure()
    plt.gca().yaxis.grid(True)

    X_words = histograms[i][0]
    Y_occurrences = histograms[i][1]

    print(X_words)
    print(Y_occurrences)

    plt.bar(range(len(Y_occurrences)), Y_occurrences, align='center', edgecolor='black', color='lightblue')
    plt.xticks(range(len(Y_occurrences)), X_words, size='small')
    plt.title("Cluster #"+ str(i+1) + "/" + str(NUM_CLUSTERS))

plt.show()

