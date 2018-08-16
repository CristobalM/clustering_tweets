import numpy as np
import scipy.sparse as sp

class IndexAndValue:
    def __init__(self, index, value):
        self.index = index
        self.value = value

    def __repr__(self):
        return str((self.index, self.value))


class NiceVector:
    def __init__(self, vector):
        self.nv = vector

    def __getitem__(self, key):
        return self.nv[key]

    def __str__(self):
        return str(self.nv)

    def __repr__(self):
        return repr(self.nv)

    def __len__(self):
        return len(self.nv)

    def dot(self, to):
        i = 0
        j = 0
        to_vec = to.nv
        result = 0
        while i < len(self.nv) and j < len(to_vec):
            my_index = self.nv[i].index
            my_value = self.nv[i].value

            other_index = to_vec[j].index
            other_value = to_vec[j].value

            if my_index == other_index:
                result += my_value * other_value
                i += 1
                j += 1
            elif my_index < other_index:
                i += 1
            else:
                j += 1

        return result

    def distance(self, to):
        dot_result = self.dot(to)
        similarity = dot_result
        return 1 - similarity

    def diff_vec_metric(self, to):
        dot_result = self.dot(to)
        return np.sqrt(2*(1 - dot_result))

    def normalize(self):
        norm = np.sqrt(self.dot(self))
        if norm != 0:
            for iac in self.nv:
                iac.value /= norm


class DocumentsHandler:
    def __init__(self):
        self.vocabulary = Vocabulary()
        self.document_frequency = {}
        self.number_of_documents = 0
        self.documents_nice_vectors = []
        self.documents = []

    def get_vocabulary(self):
        return self.vocabulary

    def enter_document(self, document_string):
        self.documents.append(document_string)
        list_of_words = document_string.split()
        freq_in_doc = {}
        unique_words_index = []
        document_index = len(self.documents_nice_vectors)
        for word in list_of_words:
            self.vocabulary.add_word(word)
            word_index = self.vocabulary.get_index(word)
            if word_index not in freq_in_doc:
                freq_in_doc[word_index] = 0
                unique_words_index.append(word_index)
            freq_in_doc[word_index] += 1

        nice_list = []
        # Update document frequency
        for word_index in unique_words_index:
            if word_index not in self.document_frequency:
                self.document_frequency[word_index] = []
            self.document_frequency[word_index].append(IndexAndValue(document_index, freq_in_doc[word_index]))

        # Build Nice Vector
        for word_index in unique_words_index:
            iav = IndexAndValue(word_index, freq_in_doc[word_index])
            nice_list.append(iav)
            nice_list.sort(key=lambda x: x.index)

        self.number_of_documents += 1
        self.documents_nice_vectors.append(NiceVector(nice_list))

    def get_idf_by_word_name(self, word):
        word_index = self.vocabulary.get_index(word)
        different_documents_count = len(self.document_frequency[word_index])
        return np.log(self.number_of_documents / different_documents_count) if different_documents_count > 0 else 0

    def convert_to_tfidf(self):
        for nv in self.documents_nice_vectors:
            doc_words = 0
            for iav in nv:
                doc_words += iav.value
            if doc_words <= 0:
                continue
            for iav in nv:
                freq = float(iav.value)  # /float(doc_words)
                occurrences = float(len(self.document_frequency.get(iav.index, 0))) + 1
                iav.value = (1 + np.log(freq)) * np.log(
                    float(self.number_of_documents) / occurrences) if freq > 0 else 0

            nv.normalize()

    def to_csr_matrix(self):
        row = []
        col = []
        data =[]
        for i, dnv in enumerate(self.documents_nice_vectors):
            for iac in dnv:
                j = iac.index
                v = iac.value
                row.append(i)
                col.append(j)
                data.append(v)

        m = len(self.vocabulary)
        n = len(self.documents_nice_vectors)
        non_zero_elements = len(data)
        total_elements = n*m
        zero_elements = total_elements - non_zero_elements
        sparsity = float(zero_elements) / float(total_elements)
        result = sp.csr_matrix((data, (row, col)), shape=(n, m))

        return result, sparsity


class Vocabulary:
    def __init__(self):
        self.vocabulary = []
        self.word_index = {}

    def add_word(self, word):
        if self.word_index.get(word, -1) == -1:
            self.word_index[word] = len(self.vocabulary)
            self.vocabulary.append(word)

    def has(self, word):
        return self.word_index.get(word, -1) != -1

    def get_by_index(self, index):
        try:
            return self.vocabulary[index]
        except Exception as e:
            print("Index out of range")
            raise e

    def get_index(self, word):
        return self.word_index.get(word, -1)

    def __len__(self):
        return len(self.vocabulary)


