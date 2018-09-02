from clean_utils import *


def cleaning():
    data_extracted = []

    regex_url = r'https://t\.co(\w|/)+'

    counter = 0
    mapping_lines = []

    tweets_to_save = {}

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

            line = remove_url(regex_url, line)
            line = remove_url(regex_url, line)
            line = remove_url(regex_url, line)
            line = remove_url(regex_url, line)
            line = normalize_special_characters(line)
            line = reduce_some_characters(line)

            line = re.sub(r'\s+', ' ', line).strip()
            line = re.sub('\n', '', line).strip().lower()

            if len(line) > 0:
                if line not in tweets_to_save:
                    data_extracted.append(line)
                    mapping_lines.append(counter)
                    tweets_to_save[line] = True

            counter += 1

    with open('clean_data.txt', 'w') as f:
        for line in data_extracted:
            f.write(line + '\n')

    with open('indexes.txt', 'w') as f:
        for index in mapping_lines:
            f.write(str(index) + '\n')
