import ujson as json

import os
#
import tensorflow as tf
# import coref_model as cm
import util
from util import *
from tqdm import tqdm
import pickle



def find_sentence_index(example, span):
    separate_sentence_range = list()
    all_sentence = list()
    for s in example['sentences']:
        separate_sentence_range.append((len(all_sentence), len(all_sentence) + len(s) - 1))
        all_sentence += s
    for j, sentence_s_e in enumerate(separate_sentence_range):
        if sentence_s_e[0] <= span[0] <= sentence_s_e[1]:
            return j


def convert_conll_data_to_pronoun_format(tmp_name):
    number_of_candidates = list()
    number_of_correct_candidates = list()
    output_name = tmp_name.split('.')[0]+'.pronoun.'+tmp_name.split('.')[1]
    test_data = list()
    print('Start to process data...')
    pronoun_count = dict()
    for pronoun_type in interested_pronouns:
        pronoun_count[pronoun_type] = 0

    new_data = list()
    all_examples = list()
    with open(tmp_name, 'r') as f:
        counter = 0
        for line in f:
            counter += 1
            tmp_example = json.loads(line)
            all_examples.append(tmp_example)
    for tmp_example in tqdm(all_examples):
        example_with_pronoun_info = tmp_example
        example_with_pronoun_info['pronoun_info'] = list()
        all_sentence = list()
        for s in tmp_example['sentences']:
            all_sentence += s
        all_clusters = list()
        for c in tmp_example['clusters']:
            tmp_c = list()
            for w in c:
                tmp_w = list()
                for token in all_sentence[w[0]:w[1] + 1]:
                    tmp_w.append(token)
                tmp_c.append((w, tmp_w))
            all_clusters.append(tmp_c)
        tmp_all_NPs = list()
        Pronoun_dict = dict()
        for pronoun_type in interested_pronouns:
            Pronoun_dict[pronoun_type] = list()
        for c in all_clusters:
            for i in range(len(c)):
                if len(c[i][1]) != 1 or c[i][1][0] not in all_pronouns:
                    tmp_all_NPs.append(c[i][0])
        for c in all_clusters:
            for i in range(len(c)):
                if len(c[i][1]) == 1 and c[i][1][0] in all_pronouns:
                    for pronoun_type in interested_pronouns:
                        if c[i][1][0] in all_pronouns_by_type[pronoun_type]:
                            pronoun_index = find_sentence_index(tmp_example, c[i][0])
                            correct_NPs = list()
                            for j in range(len(c)):
                                if len(c[j][1]) != 1 or c[j][1][0] not in all_pronouns:
                                    if -2 <= find_sentence_index(tmp_example, c[j][0]) - pronoun_index <= 0:
                                        correct_NPs.append(tuple(c[j][0]))
                            if len(correct_NPs) > 0:
                                pronoun_count[pronoun_type] += 1
                                candidate_NPs = list()
                                for NP in tmp_all_NPs:
                                    if -2 <= find_sentence_index(tmp_example, NP) - pronoun_index <= 0:
                                        candidate_NPs.append(tuple(NP))
                                tmp_pronoun_info = dict()
                                tmp_pronoun_info['current_pronoun'] = tuple(c[i][0])
                                tmp_pronoun_info['candidate_NPs'] = candidate_NPs
                                number_of_candidates.append(len(candidate_NPs))
                                number_of_correct_candidates.append(len(correct_NPs))
                                tmp_pronoun_info['correct_NPs'] = correct_NPs
                                example_with_pronoun_info['pronoun_info'].append(tmp_pronoun_info)
        if len(example_with_pronoun_info['pronoun_info']) > 0:
            new_data.append(example_with_pronoun_info)
    with open(output_name, 'w') as f:
        for e in new_data:
            f.write(json.dumps(e))
            f.write('\n')


if __name__ == "__main__":
    convert_conll_data_to_pronoun_format('conll_data/train.jsonlines')
    convert_conll_data_to_pronoun_format('conll_data/dev.jsonlines')
    convert_conll_data_to_pronoun_format('conll_data/test.jsonlines')
    convert_conll_data_to_pronoun_format('medical_data/train.jsonlines')
    convert_conll_data_to_pronoun_format('medical_data/test.jsonlines')

    print('end')
