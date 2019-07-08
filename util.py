from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
import codecs
import collections
import json
import math
import shutil
import sys

import numpy as np
import tensorflow as tf
import pyhocon
from pycorenlp import StanfordCoreNLP


def initialize_from_env():
    if "GPU" in os.environ:
        set_gpus(int(os.environ["GPU"]))
    else:
        set_gpus()

    name = sys.argv[1]
    print("Running experiment: {}".format(name))

    config = pyhocon.ConfigFactory.parse_file("experiments.conf")[name]
    config["log_dir"] = mkdirs(os.path.join(config["log_root"], name))

    print(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def copy_checkpoint(source, target):
    for ext in (".index", ".data-00000-of-00001"):
        shutil.copyfile(source + ext, target + ext)


def make_summary(value_dict):
    return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()])


def flatten(l):
    return [item for sublist in l for item in sublist]


def set_gpus(*gpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    print("Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def load_char_dict(char_vocab_path):
    vocab = [u"<unk>"]
    with codecs.open(char_vocab_path, encoding="utf-8") as f:
        vocab.extend(l.strip() for l in f.readlines())
    char_dict = collections.defaultdict(int)
    char_dict.update({c: i for i, c in enumerate(vocab)})
    return char_dict


def maybe_divide(x, y):
    return 0 if y == 0 else x / float(y)


def projection(inputs, output_size, initializer=None):
    return ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)


def highway(inputs, num_layers, dropout):
    for i in range(num_layers):
        with tf.variable_scope("highway_{}".format(i)):
            j, f = tf.split(projection(inputs, 2 * shape(inputs, -1)), 2, -1)
            f = tf.sigmoid(f)
            j = tf.nn.relu(j)
            if dropout is not None:
                j = tf.nn.dropout(j, dropout)
            inputs = f * j + (1 - f) * inputs
    return inputs


def shape(x, dim):
    # print(x.get_shape())
    return x.get_shape()[dim].value or tf.shape(x)[dim]


def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
    if len(inputs.get_shape()) > 3:
        raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))

    if len(inputs.get_shape()) == 3:
        batch_size = shape(inputs, 0)
        seqlen = shape(inputs, 1)
        emb_size = shape(inputs, 2)
        current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
    else:
        current_inputs = inputs

    for i in range(num_hidden_layers):
        try:
            hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size])
        except:
            print(shape(current_inputs, 1), hidden_size)
            hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size])
        hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size])
        current_outputs = tf.nn.relu(tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))

        if dropout is not None:
            current_outputs = tf.nn.dropout(current_outputs, dropout)
        current_inputs = current_outputs

    output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size],
                                     initializer=output_weights_initializer)
    output_bias = tf.get_variable("output_bias", [output_size])
    outputs = tf.nn.xw_plus_b(current_inputs, output_weights, output_bias)

    if len(inputs.get_shape()) == 3:
        outputs = tf.reshape(outputs, [batch_size, seqlen, output_size])
    return outputs


def cnn(inputs, filter_sizes, num_filters):
    num_words = shape(inputs, 0)
    num_chars = shape(inputs, 1)
    input_size = shape(inputs, 2)
    outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.variable_scope("conv_{}".format(i)):
            w = tf.get_variable("w", [filter_size, input_size, num_filters])
            b = tf.get_variable("b", [num_filters])
        conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID")  # [num_words, num_chars - filter_size, num_filters]
        h = tf.nn.relu(tf.nn.bias_add(conv, b))  # [num_words, num_chars - filter_size, num_filters]
        pooled = tf.reduce_max(h, 1)  # [num_words, num_filters]
        outputs.append(pooled)
    return tf.concat(outputs, 1)  # [num_words, num_filters * len(filter_sizes)]


def batch_gather(emb, indices):
    batch_size = shape(emb, 0)
    seqlen = shape(emb, 1)
    if len(emb.get_shape()) > 2:
        emb_size = shape(emb, 2)
    else:
        emb_size = 1
    flattened_emb = tf.reshape(emb, [batch_size * seqlen, emb_size])  # [batch_size * seqlen, emb]
    offset = tf.expand_dims(tf.range(batch_size) * seqlen, 1)  # [batch_size, 1]
    gathered = tf.gather(flattened_emb, indices + offset)  # [batch_size, num_indices, emb]
    if len(emb.get_shape()) == 2:
        gathered = tf.squeeze(gathered, 2)  # [batch_size, num_indices]
    return gathered


class RetrievalEvaluator(object):
    def __init__(self):
        self._num_correct = 0
        self._num_gold = 0
        self._num_predicted = 0

    def update(self, gold_set, predicted_set):
        self._num_correct += len(gold_set & predicted_set)
        self._num_gold += len(gold_set)
        self._num_predicted += len(predicted_set)

    def recall(self):
        return maybe_divide(self._num_correct, self._num_gold)

    def precision(self):
        return maybe_divide(self._num_correct, self._num_predicted)

    def metrics(self):
        recall = self.recall()
        precision = self.precision()
        f1 = maybe_divide(2 * recall * precision, precision + recall)
        return recall, precision, f1


class EmbeddingDictionary(object):
    def __init__(self, info, normalize=True, maybe_cache=None):
        self._size = info["size"]
        self._normalize = normalize
        self._path = info["path"]
        if maybe_cache is not None and maybe_cache._path == self._path:
            assert self._size == maybe_cache._size
            self._embeddings = maybe_cache._embeddings
        else:
            self._embeddings = self.load_embedding_dict(self._path)

    @property
    def size(self):
        return self._size

    def load_embedding_dict(self, path):
        print("Loading word embeddings from {}...".format(path))
        default_embedding = np.zeros(self.size)
        embedding_dict = collections.defaultdict(lambda: default_embedding)
        if len(path) > 0:
            vocab_size = None
            with open(path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    word_end = line.find(" ")
                    word = line[:word_end]
                    embedding = np.fromstring(line[word_end + 1:], np.float32, sep=" ")
                    assert len(embedding) == self.size
                    embedding_dict[word] = embedding
            if vocab_size is not None:
                assert vocab_size == len(embedding_dict)
            print("Done loading word embeddings.")
        return embedding_dict

    def __getitem__(self, key):
        embedding = self._embeddings[key]
        if self._normalize:
            embedding = self.normalize(embedding)
        return embedding

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        else:
            return v


class CustomLSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, batch_size, dropout):
        self._num_units = num_units
        self._dropout = dropout
        self._dropout_mask = tf.nn.dropout(tf.ones([batch_size, self.output_size]), dropout)
        self._initializer = self._block_orthonormal_initializer([self.output_size] * 3)
        initial_cell_state = tf.get_variable("lstm_initial_cell_state", [1, self.output_size])
        initial_hidden_state = tf.get_variable("lstm_initial_hidden_state", [1, self.output_size])
        self._initial_state = tf.contrib.rnn.LSTMStateTuple(initial_cell_state, initial_hidden_state)

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self.output_size, self.output_size)

    @property
    def output_size(self):
        return self._num_units

    @property
    def initial_state(self):
        return self._initial_state

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "CustomLSTMCell"
            c, h = state
            h *= self._dropout_mask
            concat = projection(tf.concat([inputs, h], 1), 3 * self.output_size, initializer=self._initializer)
            i, j, o = tf.split(concat, num_or_size_splits=3, axis=1)
            i = tf.sigmoid(i)
            new_c = (1 - i) * c + i * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            return new_h, new_state

    def _orthonormal_initializer(self, scale=1.0):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            M1 = np.random.randn(shape[0], shape[0]).astype(np.float32)
            M2 = np.random.randn(shape[1], shape[1]).astype(np.float32)
            Q1, R1 = np.linalg.qr(M1)
            Q2, R2 = np.linalg.qr(M2)
            Q1 = Q1 * np.sign(np.diag(R1))
            Q2 = Q2 * np.sign(np.diag(R2))
            n_min = min(shape[0], shape[1])
            params = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
            return params

        return _initializer

    def _block_orthonormal_initializer(self, output_sizes):
        def _initializer(shape, dtype=np.float32, partition_info=None):
            assert len(shape) == 2
            assert sum(output_sizes) == shape[1]
            initializer = self._orthonormal_initializer()
            params = np.concatenate([initializer([shape[0], o], dtype, partition_info) for o in output_sizes], 1)
            return params

        return _initializer


def verify_correct_NP_match(predicted_NP, gold_NPs, model):
    if model == 'exact':
        for tmp_gold_NP in gold_NPs:
            if tmp_gold_NP[0] == predicted_NP[0] and tmp_gold_NP[1] == predicted_NP[1]:
                return True
    elif model == 'cover':
        for tmp_gold_NP in gold_NPs:
            if tmp_gold_NP[0] <= predicted_NP[0] and tmp_gold_NP[1] >= predicted_NP[1]:
                return True
            if tmp_gold_NP[0] >= predicted_NP[0] and tmp_gold_NP[1] <= predicted_NP[1]:
                return True
    return False

def filter_stop_words(input_sentence, stop_words):
    result = list()
    for w in input_sentence:
        if w in stop_words:
            continue
        result.append(w)
    return result


def get_coverage(w_list1, w_list2):
    if len(w_list1) == 0:
        return 0
    tmp_count = 0
    for w in w_list1:
        if w in w_list2:
            tmp_count += 1
    # if tmp_count > 0:
    #     print('')
    return tmp_count / len(w_list1)


def verify_match(coreference_pair, OMCS_pair, limitation=0.9):
    if get_coverage(coreference_pair[0], OMCS_pair[0]) >= limitation and get_coverage(coreference_pair[1],
                                                                                      OMCS_pair[1]) >= limitation:
        return True
    if get_coverage(coreference_pair[0], OMCS_pair[1]) >= limitation and get_coverage(coreference_pair[1],
                                                                                      OMCS_pair[0]) >= limitation:
        return True
    return False


def get_pronoun_related_words(example, pronoun_position):
    related_words = list()
    separate_sentence_range = list()
    all_sentence = list()
    for s in example['sentences']:
        separate_sentence_range.append((len(all_sentence), len(all_sentence) + len(s) - 1))
        all_sentence += s
    target_sentence = ''
    sentence_position = 0
    for j, sentence_s_e in enumerate(separate_sentence_range):
        if sentence_s_e[0] <= pronoun_position[0] <= sentence_s_e[1]:
            for w in example['sentences'][j]:
                target_sentence += ' '
                target_sentence += w
            sentence_position = pronoun_position[0] - sentence_s_e[0]
            break
    if len(target_sentence) > 0:
        target_sentence = target_sentence[1:]
    tmp_output = nlp_list[0].annotate(target_sentence,
                                      properties={'annotators': 'tokenize,depparse,lemma', 'outputFormat': 'json'})
    for s in tmp_output['sentences']:
        enhanced_dependency_list = s['enhancedPlusPlusDependencies']
        for relation in enhanced_dependency_list:
            if relation['dep'] == 'ROOT':
                continue
            governor_position = relation['governor']
            dependent_position = relation['dependent']
            if relation[
                'governorGloss'] in all_pronouns and sentence_position <= governor_position <= sentence_position + 2:
                if relation['dep'] in ['dobj', 'nsubj']:
                    related_words.append(relation['dependentGloss'])

            if relation[
                'dependentGloss'] in all_pronouns and sentence_position <= dependent_position <= sentence_position + 2:
                if relation['dep'] in ['dobj', 'nsubj']:
                    related_words.append(relation['governorGloss'])

        # Before_length += len(s['tokens'])
    return related_words


# def filter_NP_based_on_ner()


# def post_ranking(example, pronoun_position, top_NPs):
#     if len(top_NPs) == 0:
#         return []
#     pronoun_related_words = get_pronoun_related_words(example, pronoun_position)
#     # top_NP_words = list()
#     NP_match_scores = list()
#     all_sentence = list()
#     for s in example['sentences']:
#         all_sentence += s
#     print('pronoun related words:', pronoun_related_words)
#     for NP_position in top_NPs:
#         # top_NP_words.append(all_sentence[NP_position])
#         current_NP = all_sentence[NP_position[0]:NP_position[1]+1]
#         print(current_NP, NP_position)
#         tmp_score = 0
#         for related_word in pronoun_related_words:
#             for OMCS_pair in OMCS_data:
#                 if related_word not in stop_words and verify_match((filter_stop_words(current_NP, stop_words), [related_word]), OMCS_pair[1:]):
#                     tmp_score += 1
#         NP_match_scores.append(tmp_score)
#     print(NP_match_scores)
#     return top_NPs, NP_match_scores
    # return top_NPs

def get_pronoun_type(input_pronoun):
    for tmp_type in interested_pronouns:
        if input_pronoun in all_pronouns_by_type[tmp_type]:
            return tmp_type



stop_words = list()
with open('nltk_english.txt', 'r') as f:
    for line in f:
        stop_words.append(line[:-1])
stop_words = set(stop_words)

# with open('test_data_for_analyzing.json', 'r') as f:
#     all_test_data = json.load(f)

# OMCS_data = list()
# with open('OMCS/new_omcs600.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         words = line.split('\t')
#         OMCS_data.append((words[0], words[1].split(' '), words[2].split(' ')))


third_personal_pronouns = ['she', 'her', 'he', 'him', 'them', 'they', 'She', 'Her', 'He', 'Him', 'Them',
                           'They', 'it', 'It']

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

neutral_pronoun = ['it', 'It']

first_and_second_personal_pronouns = ['I', 'me', 'we', 'us', 'you', 'Me', 'We', 'Us', 'You']
relative_pronouns = ['that', 'which', 'who', 'whom', 'whose', 'whichever', 'whoever', 'whomever',
                     'That', 'Which', 'Who', 'Whom', 'Whose', 'Whichever', 'Whoever', 'Whomever']
demonstrative_pronouns = ['this', 'these', 'that', 'those', 'This', 'These', 'That', 'Those']
indefinite_pronouns = ['anybody', 'anyone', 'anything', 'each', 'either', 'everybody', 'everyone', 'everything',
                       'neither', 'nobody', 'none', 'nothing', 'one', 'somebody', 'someone', 'something', 'both',
                       'few', 'many', 'several', 'all', 'any', 'most', 'some',
                       'Anybody', 'Anyone', 'Anything', 'Each', 'Either', 'Everybody', 'Everyone', 'Everything',
                       'Neither', 'Nobody', 'None', 'Nothing', 'One', 'Somebody', 'Someone', 'Something', 'Both',
                       'Few', 'Many', 'Several', 'All', 'Any', 'Most', 'Some']
reflexive_pronouns = ['myself', 'ourselves', 'yourself', 'yourselves', 'himself', 'herself', 'itself', 'themselves',
                      'Myself', 'Ourselves', 'Yourself', 'Yourselves', 'Himself', 'Herself', 'Itself', 'Themselves']
interrogative_pronouns = ['what', 'who', 'which', 'whom', 'whose', 'What', 'Who', 'Which', 'Whom', 'Whose']
all_possessive_pronoun = ['my', 'your', 'his', 'her', 'its', 'our', 'your', 'their', 'mine', 'yours', 'his', 'hers', 'ours',
                      'yours', 'theirs', 'My', 'Your', 'His', 'Her', 'Its', 'Our', 'Your', 'Their', 'Mine', 'Yours',
                      'His', 'Hers', 'Ours', 'Yours', 'Theirs']
possessive_pronoun = ['his', 'hers', 'its', 'their', 'theirs', 'His', 'Hers', 'Its', 'Their', 'Theirs']

all_pronouns_by_type = dict()
all_pronouns_by_type['first_and_second_personal'] = first_and_second_personal_pronouns
all_pronouns_by_type['third_personal'] = third_personal_pronouns
all_pronouns_by_type['neutral'] = neutral_pronoun
all_pronouns_by_type['relative'] = relative_pronouns
all_pronouns_by_type['demonstrative'] = demonstrative_pronouns
all_pronouns_by_type['indefinite'] = indefinite_pronouns
all_pronouns_by_type['reflexive'] = reflexive_pronouns
all_pronouns_by_type['interrogative'] = interrogative_pronouns
all_pronouns_by_type['possessive'] = possessive_pronoun

all_pronouns = list()
for pronoun_type in all_pronouns_by_type:
    all_pronouns += all_pronouns_by_type[pronoun_type]

all_pronouns = set(all_pronouns)

interested_pronouns = ['third_personal', 'possessive', 'demonstrative']

interested_entity_types = ['NATIONALITY', 'ORGANIZATION', 'PERSON', 'DATE', 'CAUSE_OF_DEATH', 'CITY', 'LOCATION',
                           'NUMBER', 'TITLE', 'TIME', 'ORDINAL', 'DURATION', 'MISC', 'COUNTRY', 'SET', 'PERCENT',
                           'STATE_OR_PROVINCE', 'MONEY', 'CRIMINAL_CHARGE', 'IDEOLOGY', 'RELIGION', 'URL', 'EMAIL']

plural_pronouns = ['them', 'they', 'Them', 'They', 'their', 'theirs', 'Their', 'Theirs']
single_pronouns = ['it', 'It', 'she', 'her', 'he', 'him', 'She', 'Her', 'He', 'Him', 'his', 'hers', 'its', 'His',
                   'Hers', 'Its']
# number_both_pronoun = []

# person_pronouns = ['she', 'her', 'he', 'him', 'She', 'Her', 'He', 'Him']
male_pronouns = ['he', 'him', 'his', 'He', 'Him', 'His']
female_pronouns = ['she', 'her', 'hers', 'She', 'Her', 'Hers']
object_pronouns = ['it', 'It', 'its', 'Its']
both_pronouns = ['them', 'they', 'their', 'theirs', 'Them', 'They', 'Their', 'Theirs']


no_nlp_server = 15
nlp_list = [StanfordCoreNLP('http://localhost:900%d' % (i)) for i in range(no_nlp_server)]
special_words = ["'", '.', '@', '?', '!', '#', '(', ')', '-', '/', ':', ';', '+', '=', ',']
tmp_nlp = nlp_list[0]
