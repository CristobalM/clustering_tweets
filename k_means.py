import random
import numpy as np

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

def intersect_sentences_num(s_a, s_b):
    i = 0
    j = 0
    count = 0
    while i < len(s_a) and j < len(s_b):
        word_a = s_a[i]
        word_b = s_b[j]

        cmp = word_b - word_a
        #cmp = strcmp(word_a, word_b)
        if cmp == 0:
            count += 1
            i += 1
            j += 1

        elif cmp > 0:
            i += 1

        else:
            j += 1

    return count


def union_sentences_num(s_a, s_b):
    intersection_num = intersect_sentences_num(s_a, s_b)
    return len(s_a) + len(s_b) - intersection_num


def jaccard_dist(set_a, set_b):
    if len(set_a) == 0 or len(set_b) == 0: # distance with empty set
        return 1.0
    intersect_num = intersect_sentences_num(set_a, set_b)
    union_num = len(set_a) + len(set_b) - intersect_num

    return 1.0 - float(intersect_num) / float(union_num)


def init_clusters(k):
    clusters = []
    for i in range(0, k):
        clusters.append([])

    return clusters

"""
@:param data: List<List>
@:param k: Int
@:param dist_fun: Fun
@:returns (List<List>, List, List)

"""
def k_means(data, k, dist_fun):
    centroids = []
    clusters = init_clusters(k)
    random_permutation = np.random.permutation(len(data))
    shift = 0
    for i in range(0, k):
        r_idx = random_permutation[i]
        while len(data[r_idx]) <= 5:
            shift += 1
            r_idx = random_permutation[(i + shift) % len(data)]
        centroids.append(r_idx)

    result = np.zeros(len(data), dtype=int)

    previous_centroids = centroids[:]

    count = 0

    to_stop = 0

    while True:
        print("in iteration number %d of k means" % count)
        count += 1
        clusters = init_clusters(k)
        for i in range(0, len(data)):
            current_element = data[i]
            min_dist = np.inf
            selected_centroid = -1
            for j in range(0, k):
                current_centroid = data[centroids[j]]
                dist = dist_fun(current_element, current_centroid)
                if dist < min_dist:
                    selected_centroid = j
                    min_dist = dist

            result[i] = selected_centroid
            clusters[selected_centroid].append(i)

        print("initialized centroids")
        if to_stop >= 2:
            break

        for cluster_idx in range(0, k):
            cluster = clusters[cluster_idx]
            cluster_length = len(cluster)
            dist_matrix = np.zeros((cluster_length, cluster_length))
            for i in range(0, cluster_length):
                for l in range(i+1, cluster_length):
                    dist_matrix[i, l] = dist_fun(data[cluster[i]], data[cluster[l]])
                    dist_matrix[l, i] = dist_matrix[i, l]

            min_sum = np.inf
            new_centroid = centroids[cluster_idx]
            for i in range(0, cluster_length):
                if len(data[cluster[i]]) == 0:
                    row_sum = np.inf
                else:
                    row_sum = np.sqrt(np.sum(dist_matrix[i, :]**2))/len(data[cluster[i]])
                if row_sum < min_sum:
                    new_centroid = cluster[i]
                    min_sum = row_sum

            centroids[cluster_idx] = new_centroid

        print("updated centroids")

        if centroids == previous_centroids:
            to_stop += 1
        else:
            previous_centroids = centroids[:]
            to_stop = 0

    return clusters, result, centroids






