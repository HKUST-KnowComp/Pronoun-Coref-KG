from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import operator
import random
import math
import json
import threading
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import h5py

import util
from util import *
import time
from tqdm import tqdm


class PronounCorefKGModel(object):
    def __init__(self, config, model='Train'):
        self.config = config
        self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
        self.head_embeddings = util.EmbeddingDictionary(config["head_embeddings"], maybe_cache=self.context_embeddings)
        self.char_embedding_size = config["char_embedding_size"]
        self.char_dict = util.load_char_dict(config["char_vocab_path"])
        self.max_span_width = config["max_span_width"]
        self.genres = {g: i for i, g in enumerate(config["genres"])}
        self.softmax_threshold = config['softmax_threshold']
        if config["lm_path"]:
            self.lm_file = h5py.File(self.config["lm_path"], "r")
        else:
            self.lm_file = None
        self.kg_lm_file = h5py.File(self.config["kg_lm_path"], "r")
        self.lm_layers = self.config["lm_layers"]
        self.lm_size = self.config["lm_size"]
        self.eval_data = None  # Load eval data lazily.
        print('Start to load the eval data')
        st = time.time()
        # self.load_kg('final_kg.json')
        self.kg_embedding_size = 300
        self.load_simple_kg('final_kg.json')
        self.load_eval_data()
        self.load_test_data()
        print("Finished in {:.2f}".format(time.time() - st))
        input_props = []
        input_props.append((tf.string, [None, None]))  # Tokens.
        input_props.append((tf.float32, [None, None, self.context_embeddings.size]))  # Context embeddings.
        input_props.append((tf.float32, [None, None, self.head_embeddings.size]))  # Head embeddings.
        input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers]))  # LM embeddings.
        input_props.append((tf.int32, [None, None, None]))  # Character indices.
        input_props.append((tf.int32, [None]))  # Text lengths.
        input_props.append((tf.int32, [None]))  # Speaker IDs.
        input_props.append((tf.int32, []))  # Genre.
        input_props.append((tf.bool, []))  # Is training.
        input_props.append((tf.int32, [None]))  # gold_starts.
        input_props.append((tf.int32, [None]))  # gold_ends.
        input_props.append((tf.float32, [None, None, None]))  # related kg embeddings.
        input_props.append((tf.int32, [None, None]))  # candidate_positions.
        input_props.append((tf.int32, [None, None]))  # pronoun_positions.
        input_props.append((tf.bool, [None, None]))  # labels
        input_props.append((tf.float32, [None, None]))  # candidate_masks

        self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
        dtypes, shapes = zip(*input_props)
        queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
        self.enqueue_op = queue.enqueue(self.queue_input_tensors)
        self.input_tensors = queue.dequeue()

        self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.reset_global_step = tf.assign(self.global_step, 0)
        learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                                   self.config["decay_frequency"], self.config["decay_rate"],
                                                   staircase=True)
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
        optimizers = {
            "adam": tf.train.AdamOptimizer,
            "sgd": tf.train.GradientDescentOptimizer
        }
        optimizer = optimizers[self.config["optimizer"]](learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)

    def start_enqueue_thread(self, session):
        with open(self.config["train_path"]) as f:
            train_examples = [json.loads(jsonline) for jsonline in f.readlines()]

        def _enqueue_loop():
            while True:
                random.shuffle(train_examples)
                for example in train_examples:
                    tensorized_example = self.tensorize_pronoun_example(example, is_training=True)
                    feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
                    session.run(self.enqueue_op, feed_dict=feed_dict)

        enqueue_thread = threading.Thread(target=_enqueue_loop)
        enqueue_thread.daemon = True
        enqueue_thread.start()

    def restore(self, session, log_path=None):
        # Don't try to restore unused variables from the TF-Hub ELMo module.
        vars_to_restore = [v for v in tf.global_variables() if "module/" not in v.name]
        saver = tf.train.Saver(vars_to_restore)
        if log_path:
            checkpoint_path = os.path.join(log_path, "model.max.ckpt")
        else:
            checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
        print("Restoring from {}".format(checkpoint_path))
        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint_path)

    def load_lm_embeddings(self, doc_key):
        if self.lm_file is None:
            return np.zeros([0, 0, self.lm_size, self.lm_layers])
        file_key = doc_key.replace("/", ":")
        group = self.lm_file[file_key]
        num_sentences = len(list(group.keys()))
        sentences = [group[str(i)][...] for i in range(num_sentences)]
        lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.lm_size, self.lm_layers])
        for i, s in enumerate(sentences):
            lm_emb[i, :s.shape[0], :, :] = s
        return lm_emb

    def load_kg_lm_embeddings(self, raw_data):
        print('number of knowledge:', len(raw_data))
        print('max mention length:', max(len(m) for m in raw_data))
        print('lm_size:', self.lm_size)
        print('lm_layers:', self.lm_layers)
        overall_kg_lm = np.zeros([len(raw_data), max(len(m) for m in raw_data), self.lm_size, self.lm_layers])
        for i, tmp_mention in enumerate(raw_data):
            tmp_key = '-'.join(tmp_mention)
            tmp_group = self.kg_lm_file[tmp_key]
            tmp_emb = tmp_group[0][...]
            overall_kg_lm[i, :len(tmp_mention), :, :] = tmp_emb
        return overall_kg_lm

    def tensorize_mentions(self, mentions):
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def tensorize_span_labels(self, tuples, label_dict):
        if len(tuples) > 0:
            starts, ends, labels = zip(*tuples)
        else:
            starts, ends, labels = [], [], []
        return np.array(starts), np.array(ends), np.array([label_dict[c] for c in labels])

    def tensorize_pronoun_example(self, example, is_training):
        all_tokens = list()
        for s in example['sentences']:
            all_tokens += s
        num_words = len(all_tokens)
        word_index_to_sentence_index = list()
        for i, s in enumerate(example['sentences']):
            for w in s:
                word_index_to_sentence_index.append(i)

        gold_mentions = list()
        for pronoun_example in example['pronoun_info']:
            for tmp_np in pronoun_example['candidate_NPs']:
                if tmp_np not in gold_mentions and tmp_np[1] - tmp_np[0] < self.config["max_span_width"]:
                    gold_mentions.append(tmp_np)
            gold_mentions.append(pronoun_example['current_pronoun'])

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)
        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)
        speakers = util.flatten(example["speakers"])

        assert num_words == len(speakers)

        max_sentence_length = max(len(s) for s in sentences)
        max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))
        text_len = np.array([len(s) for s in sentences])
        tokens = [[""] * max_sentence_length for _ in sentences]
        context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings.size])
        head_word_emb = np.zeros([len(sentences), max_sentence_length, self.head_embeddings.size])
        char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                tokens[i][j] = word
                context_word_emb[i, j] = self.context_embeddings[word]
                head_word_emb[i, j] = self.head_embeddings[word]
                char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]
        tokens = np.array(tokens)

        speaker_dict = {s: i for i, s in enumerate(set(speakers))}
        speaker_ids = np.array([speaker_dict[s] for s in speakers])

        doc_key = example["doc_key"]
        genre = self.genres[doc_key[:2]]

        lm_emb = self.load_lm_embeddings(doc_key)

        if self.config['gold_candidate']:
            gold_mentions = sorted(gold_mentions)
            gold_mention_kgs = list()
            for m in gold_mentions:
                tmp_key = '-'.join(all_tokens[m[0]:m[1] + 1]).lower()
                # print(tmp_key)
                if tmp_key in self.kg_key_to_positions:
                    gold_mention_kgs.append(self.kg_key_to_positions[tmp_key])
                    tmp_raw_tails = list()
                    for tmp_id in self.kg_key_to_positions[tmp_key]:
                        tmp_raw_tails.append(self.raw_tails[tmp_id - 1])
                    # print(tmp_key, tmp_raw_tails)
                else:
                    gold_mention_kgs.append([])
                    # print(tmp_key, [])

            max_kg_number = self.config['max_knowledge']
            kg_knowledge_embeddings = np.zeros([len(gold_mentions), max_kg_number, self.kg_embedding_size])
            for i, related_kg_positions in enumerate(gold_mention_kgs):
                # tmp_raw_data = list()
                for j, tmp_position in enumerate(related_kg_positions):
                    if j >= max_kg_number:
                        continue
                    kg_knowledge_embeddings[i][j] = self.kg_tail_embeddings[tmp_position]
                    # print(self.raw_tails[tmp_position-1], self.kg_tail_embeddings[tmp_position][:5])
                    # tmp_raw_data.append(self.raw_tails[tmp_position-1])
            gold_candidates = list()
            for i, pronoun_example in enumerate(example['pronoun_info']):
                gold_candidates.append(pronoun_example['candidate_NPs'])
            max_candidate_NP_length = max(len(candidates) for candidates in gold_candidates)
            candidate_NP_positions = np.zeros([len(example['pronoun_info']), max_candidate_NP_length])
            pronoun_positions = np.zeros([len(example['pronoun_info']), 1])
            labels = np.zeros([len(example['pronoun_info']), max_candidate_NP_length], dtype=bool)
            candidate_mask = np.zeros([len(example['pronoun_info']), max_candidate_NP_length])
            for i, pronoun_example in enumerate(example['pronoun_info']):
                for j, tmp_np in enumerate(pronoun_example['candidate_NPs']):
                    candidate_mask[i, j] = 1
                    for k, tmp_tuple in enumerate(gold_mentions):
                        if tmp_tuple == tmp_np:
                            candidate_NP_positions[i, j] = k
                            break

                    if tmp_np in pronoun_example['correct_NPs']:
                        labels[i, j] = 1
                for k, tmp_tuple in enumerate(gold_mentions):
                    if tmp_tuple == pronoun_example['current_pronoun']:
                        pronoun_positions[i, 0] = k
                        break

            example_tensors = (
                tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training,
                gold_starts, gold_ends, kg_knowledge_embeddings, candidate_NP_positions, pronoun_positions, labels,
                candidate_mask)
        else:
            all_candidate_mentions = list()
            for i in range(num_words):
                for j in range(self.max_span_width):
                    if i + j < num_words and word_index_to_sentence_index[i] == word_index_to_sentence_index[i + j]:
                        if 'conll' in self.config['train_path']:
                            all_candidate_mentions.append((i, i + j))
                        elif 'medical' in self.config['train_path']:
                            tmp_words = all_tokens[i:i + j + 1]
                            no_special_words = True
                            for special_word in special_words:
                                if special_word in tmp_words:
                                    no_special_words = False
                                    break
                            if no_special_words:
                                all_candidate_mentions.append((i, i + j))
            all_candidate_mentions = sorted(all_candidate_mentions)
            candidate_starts, candidate_ends = self.tensorize_mentions(all_candidate_mentions)
            candidate_mention_kgs = list()
            for m in all_candidate_mentions:
                tmp_key = '-'.join(all_tokens[m[0]:m[1] + 1]).lower()
                # print(tmp_key)
                if tmp_key in self.kg_key_to_positions:
                    tmp_positions = self.kg_key_to_positions[tmp_key]
                    random.shuffle(tmp_positions)
                    candidate_mention_kgs.append(tmp_positions)
                    tmp_raw_tails = list()
                    for tmp_id in self.kg_key_to_positions[tmp_key]:
                        tmp_raw_tails.append(self.raw_tails[tmp_id - 1])
                else:
                    candidate_mention_kgs.append([])

            max_kg_number = self.config['max_knowledge']
            kg_knowledge_embeddings = np.zeros([len(candidate_mention_kgs), max_kg_number, self.kg_embedding_size])
            for i, related_kg_positions in enumerate(candidate_mention_kgs):
                for j, tmp_position in enumerate(related_kg_positions):
                    if j >= max_kg_number:
                        continue
                    kg_knowledge_embeddings[i][j] = self.kg_tail_embeddings[tmp_position]
            candidate_NPs = list()
            for i, pronoun_example in enumerate(example['pronoun_info']):
                tmp_pronoun = pronoun_example['current_pronoun']
                tmp_sentence = word_index_to_sentence_index[tmp_pronoun[0]]
                tmp_candidates = list()
                for j, tmp_candidate in enumerate(all_candidate_mentions):
                    # print(tmp_candidate)
                    if -2 < word_index_to_sentence_index[tmp_candidate[0]] - tmp_sentence <= 0:
                        tmp_candidates.append(tmp_candidate)

            max_candidate_NP_length = max(len(candidates) for candidates in candidate_NPs)
            candidate_NP_positions = np.zeros([len(example['pronoun_info']), max_candidate_NP_length])
            pronoun_positions = np.zeros([len(example['pronoun_info']), 1])
            labels = np.zeros([len(example['pronoun_info']), max_candidate_NP_length], dtype=bool)
            candidate_mask = np.zeros([len(example['pronoun_info']), max_candidate_NP_length])
            for i, pronoun_example in enumerate(example['pronoun_info']):
                for j, tmp_np in enumerate(candidate_NPs[i]):
                    candidate_mask[i, j] = 1
                    if list(tmp_np) in pronoun_example['correct_NPs']:
                        labels[i, j] = 1
                    for k, tmp_tuple in enumerate(all_candidate_mentions):
                        if tmp_tuple == tmp_np:
                            candidate_NP_positions[i, j] = k
                            break
                for k, tmp_tuple in enumerate(all_candidate_mentions):
                    if list(tmp_tuple) == pronoun_example['current_pronoun']:
                        pronoun_positions[i, 0] = k
                        break

            example_tensors = (
                tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training,
                candidate_starts, candidate_ends, kg_knowledge_embeddings, candidate_NP_positions, pronoun_positions,
                labels,
                candidate_mask)

        return example_tensors

    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        same_start = tf.equal(tf.expand_dims(labeled_starts, 1),
                              tf.expand_dims(candidate_starts, 0))  # [num_labeled, num_candidates]
        same_end = tf.equal(tf.expand_dims(labeled_ends, 1),
                            tf.expand_dims(candidate_ends, 0))  # [num_labeled, num_candidates]
        same_span = tf.logical_and(same_start, same_end)  # [num_labeled, num_candidates]
        candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span))  # [1, num_candidates]
        candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_candidates]
        return candidate_labels

    def get_dropout(self, dropout_rate, is_training):
        return 1 - (tf.to_float(is_training) * dropout_rate)

    def load_simple_kg(self, kg_path):
        print('Start to load KG')
        with open(kg_path, 'r') as kg_f:
            raw_kg = json.load(kg_f)
        raw_heads = list()
        raw_tails = list()
        interested_relations = ['medical', 'ner', 'plurity', 'gender', 'nsubj', 'dobj', 'OMCS']
        for relation in interested_relations:
            for edge in raw_kg[relation]:
                if edge[1][0] not in ['na']:
                    raw_heads.append(edge[0])
                    raw_tails.append(edge[1])
        self.kg_key_to_positions = dict()
        for i, tmp_head in enumerate(raw_heads):
            tmp_key = '-'.join(tmp_head).lower()
            if tmp_key not in self.kg_key_to_positions:
                self.kg_key_to_positions[tmp_key] = list()
            self.kg_key_to_positions[tmp_key].append(i + 1)
        head_embeddings = list()
        tail_embeddings = list()
        head_embeddings.append(np.zeros(300))
        tail_embeddings.append(np.zeros(300))
        for mention in tqdm(raw_heads):
            tmp_embeddings = list()
            for w in mention:
                tmp_embeddings.append(self.head_embeddings[w])
            head_embeddings.append(np.average(np.asarray(tmp_embeddings), axis=0))
        for mention in tqdm(raw_tails):
            tmp_embeddings = list()
            for w in mention:
                tmp_embeddings.append(self.head_embeddings[w])
            tail_embeddings.append(np.average(np.asarray(tmp_embeddings), axis=0))
        self.kg_head_embeddings = np.asarray(head_embeddings)
        self.kg_tail_embeddings = np.asarray(tail_embeddings)
        self.raw_tails = raw_tails

    def get_predictions_and_loss(self, tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len,
                                 speaker_ids, genre, is_training, gold_starts, gold_ends, kg_knowledge_embeddings,
                                 candidate_positions,
                                 pronoun_positions,
                                 labels, candidate_mask):
        all_k = util.shape(candidate_positions, 0)
        all_c = util.shape(candidate_positions, 1)
        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
        self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
        self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

        num_sentences = tf.shape(context_word_emb)[0]
        max_sentence_length = tf.shape(context_word_emb)[1]

        context_emb_list = [context_word_emb]
        head_emb_list = [head_word_emb]

        if self.config["char_embedding_size"] > 0:
            char_emb = tf.gather(
                tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]),
                char_index)  # [num_sentences, max_sentence_length, max_word_length, emb]
            flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2),
                                                       util.shape(char_emb,
                                                                  3)])  # [num_sentences * max_sentence_length, max_word_length, emb]
            flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config[
                "filter_size"])  # [num_sentences * max_sentence_length, emb]
            aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length,
                                                                             util.shape(flattened_aggregated_char_emb,
                                                                                        1)])  # [num_sentences, max_sentence_length, emb]
            context_emb_list.append(aggregated_char_emb)
            head_emb_list.append(aggregated_char_emb)

        if not self.lm_file:
            elmo_module = hub.Module("https://tfhub.dev/google/elmo/2")
            lm_embeddings = elmo_module(
                inputs={"tokens": tokens, "sequence_len": text_len},
                signature="tokens", as_dict=True)
            word_emb = lm_embeddings["word_emb"]  # [num_sentences, max_sentence_length, 512]
            lm_emb = tf.stack([tf.concat([word_emb, word_emb], -1),
                               lm_embeddings["lstm_outputs1"],
                               lm_embeddings["lstm_outputs2"]], -1)  # [num_sentences, max_sentence_length, 1024, 3]
        lm_emb_size = util.shape(lm_emb, 2)
        lm_num_layers = util.shape(lm_emb, 3)
        with tf.variable_scope("lm_aggregation"):
            self.lm_weights = tf.nn.softmax(
                tf.get_variable("lm_scores", [lm_num_layers], initializer=tf.constant_initializer(0.0)))
            self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))
        flattened_lm_emb = tf.reshape(lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])
        flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(self.lm_weights,
                                                                                 1))  # [num_sentences * max_sentence_length * emb, 1]
        aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
        aggregated_lm_emb *= self.lm_scaling
        if self.config['use_elmo']:
            context_emb_list.append(aggregated_lm_emb)

        context_emb = tf.concat(context_emb_list, 2)  # [num_sentences, max_sentence_length, emb]
        head_emb = tf.concat(head_emb_list, 2)  # [num_sentences, max_sentence_length, emb]
        context_emb = tf.nn.dropout(context_emb, self.lexical_dropout)  # [num_sentences, max_sentence_length, emb]
        head_emb = tf.nn.dropout(head_emb, self.lexical_dropout)  # [num_sentences, max_sentence_length, emb]

        text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)  # [num_sentence, max_sentence_length]

        context_outputs = self.lstm_contextualize(context_emb, text_len, text_len_mask)  # [num_words, emb]
        num_words = util.shape(context_outputs, 0)

        genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len(self.genres), self.config["feature_size"]]),
                              genre)  # [emb]

        flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask)  # [num_words]

        top_span_starts = gold_starts
        top_span_ends = gold_ends
        top_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, top_span_starts, top_span_ends)
        candidate_NP_embeddings = tf.gather(top_span_emb, candidate_positions)  # [k, max_candidate, embedding]
        candidate_starts = tf.gather(top_span_starts, candidate_positions)  # [k, max_candidate]
        pronoun_starts = tf.gather(top_span_starts, pronoun_positions)  # [k, 1]
        top_span_speaker_ids = tf.gather(speaker_ids, candidate_starts)  # [k]
        candidate_kg_embeddings = tf.gather(kg_knowledge_embeddings, candidate_positions)

        pronoun_embedding = tf.gather(top_span_emb, pronoun_positions)  # [k, embedding]
        pronoun_speaker_id = tf.gather(speaker_ids, pronoun_starts)  # [k, 1]
        pronoun_kg_embeddings = tf.gather(kg_knowledge_embeddings, pronoun_positions)

        mention_offsets = tf.range(util.shape(top_span_emb, 0)) + 1
        candidate_NP_offsets = tf.gather(mention_offsets, candidate_positions)
        pronoun_offsets = tf.gather(mention_offsets, pronoun_positions)
        k = util.shape(pronoun_positions, 0)
        dummy_scores = tf.zeros([k, 1])  # [k, 1]
        if self.config['kg_attention']:
            candidate_kg_embeddings, pronoun_kg_embeddings = self.kg_attention(candidate_NP_embeddings,
                                                                               pronoun_embedding,
                                                                               candidate_kg_embeddings,
                                                                               pronoun_kg_embeddings)
        else:
            candidate_kg_embeddings = tf.reshape(candidate_kg_embeddings,
                                                 [k, all_c,
                                                  self.config[
                                                      'max_knowledge'] * self.kg_embedding_size])  # [k, c, max_number_of_knowledge*kg_embedding_length]
            pronoun_kg_embeddings = tf.tile(
                tf.reshape(pronoun_kg_embeddings, [k, 1, self.config['max_knowledge'] * self.kg_embedding_size]),
                [1, all_c, 1])  # [k, c, max_number_of_knowledge*kg_embedding_length]
        for i in range(self.config["coref_depth"]):
            with tf.variable_scope("coref_layer", reuse=(i > 0)):
                if self.config['use_kg']:
                    knowledge_scores = self.get_overall_coreference_score_with_kg(candidate_NP_embeddings,
                                                                                  pronoun_embedding,
                                                                                  candidate_kg_embeddings,
                                                                                  pronoun_kg_embeddings,
                                                                                  top_span_speaker_ids,
                                                                                  pronoun_speaker_id, genre_emb,
                                                                                  candidate_NP_offsets,
                                                                                  pronoun_offsets)  # [k, c]
                    mention_score = self.get_mention_scores_kg(candidate_NP_embeddings, candidate_kg_embeddings)
                    coreference_scores = knowledge_scores + mention_score
                else:
                    coreference_scores = self.get_coreference_score(candidate_NP_embeddings, pronoun_embedding,
                                                                    top_span_speaker_ids,
                                                                    pronoun_speaker_id, genre_emb, candidate_NP_offsets,
                                                                    pronoun_offsets)  # [k, c]
                    knowledge_scores = tf.zeros([all_k, all_c])
                    mention_score = tf.zeros([all_k, all_c])
        score_after_softmax = tf.nn.softmax(coreference_scores, 1)  # [k, c]
        threshold = tf.zeros([all_k, all_c]) - tf.ones([all_k, all_c])
        ranking_mask = tf.to_float(tf.greater(score_after_softmax, threshold))  # [k, c]

        top_antecedent_scores = tf.concat([dummy_scores, coreference_scores], 1)  # [k, c + 1]
        labels = tf.logical_and(labels, tf.greater(score_after_softmax, threshold))

        dummy_mask_1 = tf.ones([k, 1])
        dummy_mask_0 = tf.zeros([k, 1])
        mask_for_prediction = tf.concat([dummy_mask_0, candidate_mask], 1)
        ranking_mask_for_prediction = tf.concat([dummy_mask_0, ranking_mask], 1)
        if self.config['random_sample_training']:
            random_mask = tf.greater(tf.random_uniform([all_k, all_c]), tf.ones([all_k, all_c]) * 0.5)
            labels = tf.logical_and(labels, random_mask)
            ranking_mask = ranking_mask * tf.to_float(random_mask)
        dummy_labels = tf.logical_not(tf.reduce_any(labels, 1, keepdims=True))  # [k, 1]
        top_antecedent_labels = tf.concat([dummy_labels, labels], 1)  # [k, c + 1]
        mask_for_training = tf.concat([dummy_mask_1, candidate_mask], 1)
        ranking_mask_for_training = tf.concat([dummy_mask_1, ranking_mask], 1)
        loss = self.softmax_loss(top_antecedent_scores * mask_for_training * ranking_mask_for_training,
                                 top_antecedent_labels)
        loss = tf.reduce_sum(loss)  # []

        return [top_antecedent_scores * mask_for_prediction * ranking_mask_for_prediction,
                score_after_softmax * candidate_mask, coreference_scores, knowledge_scores, mention_score], loss

    def get_mention_scores_kg(self, candidate_NP_embeddings, candidate_kg_embeddings):
        mention_representation = tf.concat([candidate_NP_embeddings, candidate_kg_embeddings], 2)
        with tf.variable_scope("mention_scores"):
            mention_score = util.ffnn(mention_representation, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                      self.dropout)  # [k, c, 1]
        mention_score = tf.squeeze(mention_score, 2)
        return mention_score

    def kg_attention(self, candidate_NPs_emb, pronoun_emb, candidate_kg_embeddings, pronoun_kg_embeddings):
        k = util.shape(candidate_NPs_emb, 0)
        c = util.shape(candidate_NPs_emb, 1)
        original_pair_emb = tf.concat([candidate_NPs_emb, tf.tile(pronoun_emb, [1, c, 1])], 2)
        candidate_kg_attention_score = self.get_kg_attention(original_pair_emb,
                                                             candidate_kg_embeddings)  # [k, c, max_knowledge]
        candidate_kg_embeddings = candidate_kg_embeddings * tf.tile(tf.expand_dims(candidate_kg_attention_score, 3),
                                                                    [1, 1, 1, self.kg_embedding_size])
        candidate_kg_embeddings = tf.reduce_sum(candidate_kg_embeddings, 2)  # [k, c, self.kg_embedding_size]

        pronoun_kg_embeddings = tf.tile(pronoun_kg_embeddings, [1, c, 1, 1])
        pronoun_kg_attention_score = self.get_kg_attention(original_pair_emb, pronoun_kg_embeddings)
        pronoun_kg_embeddings = pronoun_kg_embeddings * tf.tile(tf.expand_dims(pronoun_kg_attention_score, 3),
                                                                [1, 1, 1, self.kg_embedding_size])
        pronoun_kg_embeddings = tf.reduce_sum(pronoun_kg_embeddings, 2)  # [k, c, self.kg_embedding_size]
        return candidate_kg_embeddings, pronoun_kg_embeddings

    def get_overall_coreference_score_with_kg(self, candidate_NPs_emb, pronoun_emb, candidate_kg_embeddings,
                                              pronoun_kg_embeddings, candidate_NPs_speaker_ids, pronoun_speaker_id,
                                              genre_emb, candidate_NP_offsets, pronoun_offsets):
        k = util.shape(candidate_NPs_emb, 0)
        c = util.shape(candidate_NPs_emb, 1)
        feature_emb_list = []
        if self.config["use_metadata"]:
            same_speaker = tf.equal(candidate_NPs_speaker_ids, tf.tile(pronoun_speaker_id, [1, c]))  # [k, c]
            speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]),
                                         tf.to_int32(same_speaker))  # [k, c, emb]
            feature_emb_list.append(speaker_pair_emb)

            tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, c, 1])  # [k, c, emb]
            feature_emb_list.append(tiled_genre_emb)

        if self.config["use_features"]:
            antecedent_distance_buckets = self.bucket_distance(
                tf.nn.relu(tf.tile(pronoun_speaker_id, [1, c]) - candidate_NP_offsets))  # [k, c]
            antecedent_distance_emb = tf.gather(
                tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]]),
                antecedent_distance_buckets)  # [c, emb]
            feature_emb_list.append(antecedent_distance_emb)

        feature_emb = tf.concat(feature_emb_list, 2)  # [k, c, emb]
        feature_emb = tf.nn.dropout(feature_emb, self.dropout)  # [k, c, emb]

        target_emb = tf.tile(pronoun_emb, [1, c, 1])  # [k, c, emb]
        candidate_NPs_emb = tf.concat([candidate_NPs_emb, candidate_kg_embeddings], 2)
        target_emb = tf.concat([target_emb, pronoun_kg_embeddings], 2)
        similarity_emb = candidate_NPs_emb * target_emb  # [k, c, emb]

        pair_emb = tf.concat([target_emb, candidate_NPs_emb, similarity_emb, feature_emb], 2)  # [k, c, emb]

        with tf.variable_scope("slow_antecedent_scores"):
            slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                               self.dropout)  # [k, c, 1]
        slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2)  # [k, c]
        return slow_antecedent_scores  # [c]

    def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
        span_emb_list = []

        span_start_emb = tf.gather(context_outputs, span_starts)  # [k, emb]
        span_emb_list.append(span_start_emb)

        span_end_emb = tf.gather(context_outputs, span_ends)  # [k, emb]
        span_emb_list.append(span_end_emb)

        span_width = 1 + span_ends - span_starts  # [k]

        if self.config["use_features"]:
            span_width_index = span_width - 1  # [k]
            span_width_emb = tf.gather(
                tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]),
                span_width_index)  # [k, emb]
            span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
            span_emb_list.append(span_width_emb)

        if self.config["model_heads"]:
            span_indices = tf.expand_dims(tf.range(self.config["max_span_width"]), 0) + tf.expand_dims(span_starts,
                                                                                                       1)  # [k, max_span_width]
            span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices)  # [k, max_span_width]
            span_text_emb = tf.gather(head_emb, span_indices)  # [k, max_span_width, emb]
            with tf.variable_scope("head_scores"):
                self.head_scores = util.projection(context_outputs, 1)  # [num_words, 1]
            span_head_scores = tf.gather(self.head_scores, span_indices)  # [k, max_span_width, 1]
            span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.config["max_span_width"], dtype=tf.float32),
                                       2)  # [k, max_span_width, 1]
            span_head_scores += tf.log(span_mask)  # [k, max_span_width, 1]
            span_attention = tf.nn.softmax(span_head_scores, 1)  # [k, max_span_width, 1]
            span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1)  # [k, emb]
            span_emb_list.append(span_head_emb)

        span_emb = tf.concat(span_emb_list, 1)  # [k, emb]
        return span_emb  # [k, emb]

    def get_mention_scores(self, span_emb):
        with tf.variable_scope("mention_scores"):
            return util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout)  # [k, 1]

    def softmax_loss(self, antecedent_scores, antecedent_labels):
        gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))  # [k, max_ant + 1]
        marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1])  # [k]
        log_norm = tf.reduce_logsumexp(antecedent_scores, [1])  # [k]
        return log_norm - marginalized_gold_scores  # [k]

    def bucket_SP_score(self, sp_scores):
        logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(sp_scores)) / math.log(2))) + 3
        use_identity = tf.to_int32(sp_scores <= 4)
        combined_idx = use_identity * sp_scores + (1 - use_identity) * logspace_idx
        return tf.clip_by_value(combined_idx, 0, 9)

    def bucket_distance(self, distances):
        logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances)) / math.log(2))) + 3
        use_identity = tf.to_int32(distances <= 4)
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return tf.clip_by_value(combined_idx, 0, 9)

    def get_kg_attention(self, original_emb, kg_emb):
        k = util.shape(original_emb, 0)
        c = util.shape(original_emb, 1)
        mention_emb_size = util.shape(original_emb, 2)
        original_emb = tf.tile(tf.expand_dims(original_emb, 2),
                               [1, 1, self.config['max_knowledge'], 1])  # [k, c, max_knowedge, knowedge_emb]
        emb_for_attention_score = tf.reshape(tf.concat([original_emb, kg_emb], 3), [k * c, self.config['max_knowledge'],
                                                                                    mention_emb_size + self.kg_embedding_size])  # [k*c, max_knowledge, knowledge_emb+mention_emb]
        with tf.variable_scope('kg_attention', reuse=tf.AUTO_REUSE):
            kg_attention_score = util.ffnn(emb_for_attention_score, self.config["ffnn_depth"], self.config["ffnn_size"],
                                           1, self.dropout)  # [k*c, max_knowledge, 1]
        kg_attention_score = tf.nn.softmax(tf.squeeze(kg_attention_score, 2), 1)  # [k*c, max_knowledge]
        return tf.reshape(kg_attention_score, [k, c, self.config['max_knowledge']])  # [k, c, max_knowledge]

    def get_coreference_score(self, candidate_NPs_emb, pronoun_emb, candidate_NPs_speaker_ids, pronoun_speaker_id,
                              genre_emb, candidate_NP_offsets, pronoun_offsets):
        k = util.shape(candidate_NPs_emb, 0)
        c = util.shape(candidate_NPs_emb, 1)

        feature_emb_list = []

        if self.config["use_metadata"]:
            same_speaker = tf.equal(candidate_NPs_speaker_ids, tf.tile(pronoun_speaker_id, [1, c]))  # [k, c]
            speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]),
                                         tf.to_int32(same_speaker))  # [k, c, emb]
            feature_emb_list.append(speaker_pair_emb)

            tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, c, 1])  # [k, c, emb]
            feature_emb_list.append(tiled_genre_emb)

        if self.config["use_features"]:
            antecedent_distance_buckets = self.bucket_distance(
                tf.nn.relu(tf.tile(pronoun_speaker_id, [1, c]) - candidate_NP_offsets))  # [k, c]
            antecedent_distance_emb = tf.gather(
                tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]]),
                antecedent_distance_buckets)  # [c, emb]
            feature_emb_list.append(antecedent_distance_emb)

        feature_emb = tf.concat(feature_emb_list, 2)  # [k, c, emb]
        feature_emb = tf.nn.dropout(feature_emb, self.dropout)  # [k, c, emb]

        target_emb = tf.tile(pronoun_emb, [1, c, 1])  # [k, c, emb]
        similarity_emb = candidate_NPs_emb * target_emb  # [k, c, emb]

        pair_emb = tf.concat([target_emb, candidate_NPs_emb, similarity_emb, feature_emb], 2)  # [k, c, emb]

        with tf.variable_scope("slow_antecedent_scores"):
            slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                               self.dropout)  # [k, c, 1]
        slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2)  # [k, c]
        return slow_antecedent_scores  # [c]

    def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = tf.shape(emb)[0]
        max_sentence_length = tf.shape(emb)[1]

        emb_rank = len(emb.get_shape())
        if emb_rank == 2:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
        elif emb_rank == 3:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))

    def lstm_contextualize(self, text_emb, text_len, text_len_mask):
        num_sentences = tf.shape(text_emb)[0]

        current_inputs = text_emb  # [num_sentences, max_sentence_length, emb]

        for layer in range(self.config["contextualization_layers"]):
            with tf.variable_scope("layer_{}".format(layer)):
                with tf.variable_scope("fw_cell"):
                    cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences,
                                                  self.lstm_dropout)
                with tf.variable_scope("bw_cell"):
                    cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences,
                                                  self.lstm_dropout)
                state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]),
                                                         tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
                state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]),
                                                         tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))

                (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=current_inputs,
                    sequence_length=text_len,
                    initial_state_fw=state_fw,
                    initial_state_bw=state_bw)

                text_outputs = tf.concat([fw_outputs, bw_outputs], 2)  # [num_sentences, max_sentence_length, emb]
                text_outputs = tf.nn.dropout(text_outputs, self.lstm_dropout)
                if layer > 0:
                    highway_gates = tf.sigmoid(util.projection(text_outputs, util.shape(text_outputs,
                                                                                        2)))  # [num_sentences, max_sentence_length, emb]
                    text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
                current_inputs = text_outputs

        return self.flatten_emb_by_sentence(text_outputs, text_len_mask)

    def get_predicted_antecedents(self, antecedents, antecedent_scores):
        predicted_antecedents = []
        for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if index < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedents[i, index])
        return predicted_antecedents

    def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
        mention_to_predicted = {}
        predicted_clusters = []
        for i, predicted_index in enumerate(predicted_antecedents):
            if predicted_index < 0:
                continue
            assert i > predicted_index
            predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
            if predicted_antecedent in mention_to_predicted:
                predicted_cluster = mention_to_predicted[predicted_antecedent]
            else:
                predicted_cluster = len(predicted_clusters)
                predicted_clusters.append([predicted_antecedent])
                mention_to_predicted[predicted_antecedent] = predicted_cluster

            mention = (int(top_span_starts[i]), int(top_span_ends[i]))
            predicted_clusters[predicted_cluster].append(mention)
            mention_to_predicted[mention] = predicted_cluster

        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

        return predicted_clusters, mention_to_predicted

    def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gc in gold_clusters:
            for mention in gc:
                mention_to_gold[mention] = gc

        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends,
                                                                               predicted_antecedents)
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters

    def load_eval_data(self):
        print('path name:', self.config["eval_path"])
        if self.eval_data is None:
            raw_eval_data = list()
            with open(self.config["eval_path"]) as f:
                for line in f:
                    raw_eval_data.append(json.loads(line))
            self.eval_data = list()
            for tmp_raw_data in tqdm(raw_eval_data):
                self.eval_data.append((self.tensorize_pronoun_example(tmp_raw_data, is_training=False), tmp_raw_data))
            print("Loaded {} eval examples.".format(len(self.eval_data)))

    def load_test_data(self):
        print('path name:', self.config["test_path"])
        if self.test_data is None:
            raw_test_data = list()
            with open(self.config["test_path"]) as f:
                for line in f:
                    raw_test_data.append(json.loads(line))
            self.test_data = list()
            for tmp_raw_data in tqdm(raw_test_data):
                self.test_data.append((self.tensorize_pronoun_example(tmp_raw_data, is_training=False), tmp_raw_data))
            print("Loaded {} test examples.".format(len(self.test_data)))

    def evaluate(self, session, official_stdout=False):
        all_coreference = 0
        predict_coreference = 0
        corrct_predict_coreference = 0
        result_by_pronoun_type = dict()
        for tmp_pronoun_type in interested_pronouns:
            result_by_pronoun_type[tmp_pronoun_type] = {'all_coreference': 0, 'predict_coreference': 0,
                                                        'correct_predict_coreference': 0}

        for example_num, (tensorized_example, example) in enumerate(self.eval_data):

            tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, kg_knowledge_embeddings, candidate_NP_positions, pronoun_positions, labels, candidate_mask = tensorized_example
            feed_dict = {i: t for i, t in zip(self.input_tensors, tensorized_example)}
            predictions = session.run(
                self.predictions, feed_dict=feed_dict)
            pronoun_coref_scores = predictions[0]
            gold_starts = gold_starts.tolist()
            gold_ends = gold_ends.tolist()
            new_candidate_NP_positions = list()
            for tmp_positions in candidate_NP_positions:
                new_candidate_NP_positions.append(tmp_positions.tolist())
            candidate_NP_positions = new_candidate_NP_positions

            all_sentence = list()
            for s in example['sentences']:
                all_sentence += s

            for i, pronoun_coref_scores_by_example in enumerate(pronoun_coref_scores):
                tmp_pronoun_example = example['pronoun_info'][i]
                tmp_pronoun = all_sentence[tmp_pronoun_example['current_pronoun'][0]]
                current_pronoun_type = get_pronoun_type(tmp_pronoun)

                correct_cluster = None
                for c in example['clusters']:
                    if verify_correct_NP_match(tmp_pronoun_example['current_pronoun'], c, 'exact'):
                        correct_cluster = c
                        break

                pronoun_coref_scores_by_example = pronoun_coref_scores_by_example[1:]
                valid_candidate_positions = list()
                valid_candidate_scores = list()
                for j, tmp_score in enumerate(pronoun_coref_scores_by_example.tolist()):
                    if tmp_score > 0:
                        tmp_NP = all_sentence[
                                 gold_starts[int(candidate_NP_positions[i][j])]:gold_ends[int(
                                     candidate_NP_positions[i][j])] + 1]
                        tmp_NP_position = [gold_starts[int(candidate_NP_positions[i][j])], gold_ends[int(
                            candidate_NP_positions[i][j])]]
                        if len(tmp_NP) == 1 and tmp_NP[0] in all_pronouns and verify_correct_NP_match(tmp_NP_position,
                                                                                                      correct_cluster,
                                                                                                      'exact'):
                            continue
                        valid_candidate_positions.append(j)
                        valid_candidate_scores.append(tmp_score)
                if len(valid_candidate_positions) > 0:
                    candidate_scores_after_softmax = softmax(np.asarray(valid_candidate_scores)).tolist()
                    for k, score_after_softmax in enumerate(candidate_scores_after_softmax):
                        tmp_NP = (gold_starts[int(candidate_NP_positions[i][valid_candidate_positions[k]])],
                                  gold_ends[int(candidate_NP_positions[i][valid_candidate_positions[k]])])
                        threshold = 0
                        if 'conll' in self.config['eval_path']:
                            threshold = 0.01
                        elif 'medical' in self.config['eval_path']:
                            threshold = 0.00000001
                        if score_after_softmax > threshold:
                            result_by_pronoun_type[current_pronoun_type]['predict_coreference'] += 1
                            predict_coreference += 1
                            if labels[i][valid_candidate_positions[k]]:
                                result_by_pronoun_type[current_pronoun_type]['correct_predict_coreference'] += 1
                                corrct_predict_coreference += 1
                for l in labels[i]:
                    if l:
                        result_by_pronoun_type[current_pronoun_type]['all_coreference'] += 1
                        all_coreference += 1
            if example_num % 50 == 0:
                if predict_coreference > 0 and all_coreference > 0:
                    p = corrct_predict_coreference / predict_coreference
                    r = corrct_predict_coreference / all_coreference
                    if p == 0 and r == 0:
                        f1 = 0
                    else:
                        f1 = 2 * p * r / (p + r)
                    print("Average F1 (py): {:.2f}%".format(f1 * 100))
                    print("Average precision (py): {:.2f}%".format(p * 100))
                    print("Average recall (py): {:.2f}%".format(r * 100))
                    print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))
                else:
                    print('there is no positive prediction')
        summary_dict = {}
        if predict_coreference > 0:
            for tmp_pronoun_type in interested_pronouns:
                try:
                    print('Pronoun type:', tmp_pronoun_type)
                    tmp_p = result_by_pronoun_type[tmp_pronoun_type]['correct_predict_coreference'] / \
                            result_by_pronoun_type[tmp_pronoun_type]['predict_coreference']
                    tmp_r = result_by_pronoun_type[tmp_pronoun_type]['correct_predict_coreference'] / \
                            result_by_pronoun_type[tmp_pronoun_type]['all_coreference']
                    tmp_f1 = 2 * tmp_p * tmp_r / (tmp_p + tmp_r)
                    print('p:', tmp_p)
                    print('r:', tmp_r)
                    print('f1:', tmp_f1)
                except:
                    pass
            p = corrct_predict_coreference / predict_coreference
            r = corrct_predict_coreference / all_coreference
            f1 = 2 * p * r / (p + r)
            summary_dict["Average F1 (py)"] = f1
            print("Average F1 (py): {:.2f}%".format(f1 * 100))
            summary_dict["Average precision (py)"] = p
            print("Average precision (py): {:.2f}%".format(p * 100))
            summary_dict["Average recall (py)"] = r
            print("Average recall (py): {:.2f}%".format(r * 100))
        else:
            summary_dict["Average F1 (py)"] = 0
            summary_dict["Average precision (py)"] = 0
            summary_dict["Average recall (py)"] = 0
            print('there is no positive prediction')
            f1 = 0

        return util.make_summary(summary_dict), f1

    def test(self, session, official_stdout=False):
        all_coreference = 0
        predict_coreference = 0
        corrct_predict_coreference = 0
        result_by_pronoun_type = dict()
        for tmp_pronoun_type in interested_pronouns:
            result_by_pronoun_type[tmp_pronoun_type] = {'all_coreference': 0, 'predict_coreference': 0,
                                                        'correct_predict_coreference': 0}

        for example_num, (tensorized_example, example) in enumerate(self.test_data):

            tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, kg_knowledge_embeddings, candidate_NP_positions, pronoun_positions, labels, candidate_mask = tensorized_example
            feed_dict = {i: t for i, t in zip(self.input_tensors, tensorized_example)}
            predictions = session.run(
                self.predictions, feed_dict=feed_dict)
            pronoun_coref_scores = predictions[0]
            gold_starts = gold_starts.tolist()
            gold_ends = gold_ends.tolist()
            new_candidate_NP_positions = list()
            for tmp_positions in candidate_NP_positions:
                new_candidate_NP_positions.append(tmp_positions.tolist())
            candidate_NP_positions = new_candidate_NP_positions

            all_sentence = list()
            for s in example['sentences']:
                all_sentence += s

            for i, pronoun_coref_scores_by_example in enumerate(pronoun_coref_scores):
                tmp_pronoun_example = example['pronoun_info'][i]
                tmp_pronoun = all_sentence[tmp_pronoun_example['current_pronoun'][0]]
                current_pronoun_type = get_pronoun_type(tmp_pronoun)

                correct_cluster = None
                for c in example['clusters']:
                    if verify_correct_NP_match(tmp_pronoun_example['current_pronoun'], c, 'exact'):
                        correct_cluster = c
                        break

                pronoun_coref_scores_by_example = pronoun_coref_scores_by_example[1:]
                valid_candidate_positions = list()
                valid_candidate_scores = list()
                for j, tmp_score in enumerate(pronoun_coref_scores_by_example.tolist()):
                    if tmp_score > 0:
                        tmp_NP = all_sentence[
                                 gold_starts[int(candidate_NP_positions[i][j])]:gold_ends[int(
                                     candidate_NP_positions[i][j])] + 1]
                        tmp_NP_position = [gold_starts[int(candidate_NP_positions[i][j])], gold_ends[int(
                            candidate_NP_positions[i][j])]]
                        if len(tmp_NP) == 1 and tmp_NP[0] in all_pronouns and verify_correct_NP_match(tmp_NP_position,
                                                                                                      correct_cluster,
                                                                                                      'exact'):
                            continue
                        valid_candidate_positions.append(j)
                        valid_candidate_scores.append(tmp_score)
                if len(valid_candidate_positions) > 0:
                    candidate_scores_after_softmax = softmax(np.asarray(valid_candidate_scores)).tolist()
                    for k, score_after_softmax in enumerate(candidate_scores_after_softmax):
                        tmp_NP = (gold_starts[int(candidate_NP_positions[i][valid_candidate_positions[k]])],
                                  gold_ends[int(candidate_NP_positions[i][valid_candidate_positions[k]])])
                        threshold = 0
                        if 'conll' in self.config['eval_path']:
                            threshold = 0.01
                        elif 'medical' in self.config['eval_path']:
                            threshold = 0.00000001
                        if score_after_softmax > threshold:
                            result_by_pronoun_type[current_pronoun_type]['predict_coreference'] += 1
                            predict_coreference += 1
                            if labels[i][valid_candidate_positions[k]]:
                                result_by_pronoun_type[current_pronoun_type]['correct_predict_coreference'] += 1
                                corrct_predict_coreference += 1
                for l in labels[i]:
                    if l:
                        result_by_pronoun_type[current_pronoun_type]['all_coreference'] += 1
                        all_coreference += 1
            if example_num % 50 == 0:
                if predict_coreference > 0 and all_coreference > 0:
                    p = corrct_predict_coreference / predict_coreference
                    r = corrct_predict_coreference / all_coreference
                    if p == 0 and r == 0:
                        f1 = 0
                    else:
                        f1 = 2 * p * r / (p + r)
                    print("Average F1 (py): {:.2f}%".format(f1 * 100))
                    print("Average precision (py): {:.2f}%".format(p * 100))
                    print("Average recall (py): {:.2f}%".format(r * 100))
                    print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))
                else:
                    print('there is no positive prediction')
        summary_dict = {}
        if predict_coreference > 0:
            for tmp_pronoun_type in interested_pronouns:
                try:
                    print('Pronoun type:', tmp_pronoun_type)
                    tmp_p = result_by_pronoun_type[tmp_pronoun_type]['correct_predict_coreference'] / \
                            result_by_pronoun_type[tmp_pronoun_type]['predict_coreference']
                    tmp_r = result_by_pronoun_type[tmp_pronoun_type]['correct_predict_coreference'] / \
                            result_by_pronoun_type[tmp_pronoun_type]['all_coreference']
                    tmp_f1 = 2 * tmp_p * tmp_r / (tmp_p + tmp_r)
                    print('p:', tmp_p)
                    print('r:', tmp_r)
                    print('f1:', tmp_f1)
                except:
                    pass
            p = corrct_predict_coreference / predict_coreference
            r = corrct_predict_coreference / all_coreference
            f1 = 2 * p * r / (p + r)
            summary_dict["Average F1 (py)"] = f1
            print("Average F1 (py): {:.2f}%".format(f1 * 100))
            summary_dict["Average precision (py)"] = p
            print("Average precision (py): {:.2f}%".format(p * 100))
            summary_dict["Average recall (py)"] = r
            print("Average recall (py): {:.2f}%".format(r * 100))
        else:
            summary_dict["Average F1 (py)"] = 0
            summary_dict["Average precision (py)"] = 0
            summary_dict["Average recall (py)"] = 0
            print('there is no positive prediction')
            f1 = 0

        return util.make_summary(summary_dict), f1
