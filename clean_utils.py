import re
import unidecode
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('spanish')
the_stop_words = stopwords.words('spanish')


def remove_url(regex_url, text):
    m = re.search(regex_url, text)
    if m is not None:
        text = text.replace(m.group(0), '')

    return text


def is_bad_string(s):
    m = re.search(r'^(\s+|\s+\n|\n)$', s)
    return m is not None


def normalize_special_characters(text):
    return unidecode.unidecode(text)


def reduce_some_characters(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text


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
    found_sw = False
    for word in a_list:
        if word not in the_stop_words:
            out.append(word)
            found_sw = True

    return out, found_sw


def remove_consecutive_repeated(a_list):
    out = []
    prev = ''
    for word in a_list:
        if word != prev:
            out.append(word)
        prev = word

    return out
