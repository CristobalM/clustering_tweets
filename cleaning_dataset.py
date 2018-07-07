import re
import unidecode

data_extracted = []


regex_url = r'https://t\.co(\w|/)+'


def remove_url(text):
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

counter = 0
mapping_lines = []
with open('tweets.txt', 'r') as file:
    for line in file:
        line = re.sub(r'\s+', ' ', line).strip()
        line = re.sub(r'\s+', ' ', line).strip()
        line = re.sub('\r\n', '\n', line)

        if len(line) == 0 or is_bad_string(line):
            continue

        if len(line) >= 2 and line[0] == line[-1] == '"':
            line = line[1:-1]
        elif len(line) >= 1 and line[0] == '"':
            line = line[1:]
        elif len(line) >= 1 and line[-1] == '"':
            line = line[:-1]


        line = remove_url(line)
        line = remove_url(line)
        line = remove_url(line)
        line = remove_url(line)
        line = normalize_special_characters(line)
        line = reduce_some_characters(line)

        line = re.sub(r'\s+', ' ', line).strip()


        #print(line)
        if len(line) > 0:
            data_extracted.append(line)
            mapping_lines.append(counter)

        counter += 1


print(data_extracted)

with open('clean_data.txt', 'w') as f:
    for line in data_extracted:
        f.write(line + '\n')

with open('indexes.txt', 'w') as f:
    for index in mapping_lines:
        f.write(str(index) + '\n')

