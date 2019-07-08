import os
import json


def covert_pos(sentences, old_pos):
    sentence_number = int(old_pos.split(':')[0]) - 1
    word_number = int(old_pos.split(':')[1])
    previous_word_num = 0
    for i, s in enumerate(sentences):
        if i < sentence_number:
            previous_word_num += len(s)
    return previous_word_num + word_number


def get_coreference_data_from_medical_folder(tmp_folder_name):
    file_names = os.listdir(tmp_folder_name + '/docs')
    tmp_data = list()
    for file_name in file_names:
        tmp_sentences = list()
        all_w = list()
        with open(tmp_folder_name + '/docs/' + file_name, 'r', encoding='utf-8') as f:
            for line in f:
                words = line[:-1].split(' ')
                tmp_sentences.append(words)
                all_w += words
        with open(tmp_folder_name + '/chains/' + file_name + '.chains', 'r', encoding='utf-8') as f:
            all_clusters = list()
            for line in f:
                tmp_cluster = list()
                tmp_words = line[:-1].split('||')
                for w in tmp_words:
                    if w[0] == 'c':
                        raw_start_pos = w.split(' ')[-2]
                        raw_end_pos = w.split(' ')[-1]
                        tmp_mention = [covert_pos(tmp_sentences, raw_start_pos), covert_pos(tmp_sentences, raw_end_pos)]
                        tmp_cluster.append(tmp_mention)
                        # tmp_mention_words = all_w[tmp_mention[0]:tmp_mention[1]+1]
                        # print('end')
                all_clusters.append(tmp_cluster)
        tmp_data.append({'sentences': tmp_sentences, 'clusters': all_clusters, 'doc_key':'medical/'+file_name.split('.')[0]})
    return tmp_data


train_data = get_coreference_data_from_medical_folder(
    'medical_data/Train_Beth') + get_coreference_data_from_medical_folder('medical_data/Train_Partners')
test_data = get_coreference_data_from_medical_folder(
    'medical_data/Test_Beth') + get_coreference_data_from_medical_folder('medical_data/Test_Partners')

with open('medical_data/train.jsonlines', 'w') as f:
    for e in train_data:
        f.write(json.dumps(e))
        f.write('\n')
with open('medical_data/test.jsonlines', 'w') as f:
    for e in test_data:
        f.write(json.dumps(e))
        f.write('\n')

print('end')
