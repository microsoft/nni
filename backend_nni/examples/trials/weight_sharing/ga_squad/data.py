# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''
Data processing script for the QA model.
'''

import csv
import json
from random import shuffle

import numpy as np


class WhitespaceTokenizer:
    '''
    Tokenizer for whitespace
    '''

    def tokenize(self, text):
        '''
        tokenize function in Tokenizer.
        '''
        start = -1
        tokens = []
        for i, character in enumerate(text):
            if character == ' ' or character == '\t':
                if start >= 0:
                    word = text[start:i]
                    tokens.append({
                        'word': word,
                        'original_text': word,
                        'char_begin': start,
                        'char_end': i})
                    start = -1
            else:
                if start < 0:
                    start = i
        if start >= 0:
            tokens.append({
                'word': text[start:len(text)],
                'original_text': text[start:len(text)],
                'char_begin': start,
                'char_end': len(text)
            })
        return tokens


def load_from_file(path, fmt=None, is_training=True):
    '''
    load data from file
    '''
    if fmt is None:
        fmt = 'squad'
    assert fmt in ['squad', 'csv'], 'input format must be squad or csv'
    qp_pairs = []
    if fmt == 'squad':
        with open(path) as data_file:
            data = json.load(data_file)['data']
            for doc in data:
                for paragraph in doc['paragraphs']:
                    passage = paragraph['context']
                    for qa_pair in paragraph['qas']:
                        question = qa_pair['question']
                        qa_id = qa_pair['id']
                        if not is_training:
                            qp_pairs.append(
                                {'passage': passage, 'question': question, 'id': qa_id})
                        else:
                            for answer in qa_pair['answers']:
                                answer_begin = int(answer['answer_start'])
                                answer_end = answer_begin + len(answer['text'])
                                qp_pairs.append({'passage': passage,
                                                 'question': question,
                                                 'id': qa_id,
                                                 'answer_begin': answer_begin,
                                                 'answer_end': answer_end})
    else:
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            line_num = 0
            for row in reader:
                qp_pairs.append(
                    {'passage': row[1], 'question': row[0], 'id': line_num})
                line_num += 1
    return qp_pairs


def tokenize(qp_pair, tokenizer=None, is_training=False):
    '''
    tokenize function.
    '''
    question_tokens = tokenizer.tokenize(qp_pair['question'])
    passage_tokens = tokenizer.tokenize(qp_pair['passage'])
    if is_training:
        question_tokens = question_tokens[:300]
        passage_tokens = passage_tokens[:300]
    passage_tokens.insert(
        0, {'word': '<BOS>', 'original_text': '<BOS>', 'char_begin': 0, 'char_end': 0})
    passage_tokens.append(
        {'word': '<EOS>', 'original_text': '<EOS>', 'char_begin': 0, 'char_end': 0})
    qp_pair['question_tokens'] = question_tokens
    qp_pair['passage_tokens'] = passage_tokens


def collect_vocab(qp_pairs):
    '''
    Build the vocab from corpus.
    '''
    vocab = set()
    for qp_pair in qp_pairs:
        for word in qp_pair['question_tokens']:
            vocab.add(word['word'])
        for word in qp_pair['passage_tokens']:
            vocab.add(word['word'])
    return vocab


def shuffle_step(entries, step):
    '''
    Shuffle the step
    '''
    answer = []
    for i in range(0, len(entries), step):
        sub = entries[i:i+step]
        shuffle(sub)
        answer += sub
    return answer


def get_batches(qp_pairs, batch_size, need_sort=True):
    '''
    Get batches data and shuffle.
    '''
    if need_sort:
        qp_pairs = sorted(qp_pairs, key=lambda qp: (
            len(qp['passage_tokens']), qp['id']), reverse=True)
    batches = [{'qp_pairs': qp_pairs[i:(i + batch_size)]}
               for i in range(0, len(qp_pairs), batch_size)]
    shuffle(batches)
    return batches


def get_char_input(data, char_dict, max_char_length):
    '''
    Get char input.
    '''
    batch_size = len(data)
    sequence_length = max(len(d) for d in data)
    char_id = np.zeros((max_char_length, sequence_length,
                        batch_size), dtype=np.int32)
    char_lengths = np.zeros((sequence_length, batch_size), dtype=np.float32)
    for batch_idx in range(0, min(len(data), batch_size)):
        batch_data = data[batch_idx]
        for sample_idx in range(0, min(len(batch_data), sequence_length)):
            word = batch_data[sample_idx]['word']
            char_lengths[sample_idx, batch_idx] = min(
                len(word), max_char_length)
            for i in range(0, min(len(word), max_char_length)):
                char_id[i, sample_idx, batch_idx] = get_id(char_dict, word[i])
    return char_id, char_lengths


def get_word_input(data, word_dict, embed, embed_dim):
    '''
    Get word input.
    '''
    batch_size = len(data)
    max_sequence_length = max(len(d) for d in data)
    sequence_length = max_sequence_length
    word_input = np.zeros((max_sequence_length, batch_size,
                           embed_dim), dtype=np.float32)
    ids = np.zeros((sequence_length, batch_size), dtype=np.int32)
    masks = np.zeros((sequence_length, batch_size), dtype=np.float32)
    lengths = np.zeros([batch_size], dtype=np.int32)

    for batch_idx in range(0, min(len(data), batch_size)):
        batch_data = data[batch_idx]

        lengths[batch_idx] = len(batch_data)

        for sample_idx in range(0, min(len(batch_data), sequence_length)):
            word = batch_data[sample_idx]['word'].lower()
            if word in word_dict.keys():
                word_input[sample_idx, batch_idx] = embed[word_dict[word]]
                ids[sample_idx, batch_idx] = word_dict[word]
            masks[sample_idx, batch_idx] = 1

    word_input = np.reshape(word_input, (-1, embed_dim))
    return word_input, ids, masks, lengths


def get_word_index(tokens, char_index):
    '''
    Given word return word index.
    '''
    for (i, token) in enumerate(tokens):
        if token['char_end'] == 0:
            continue
        if token['char_begin'] <= char_index and char_index <= token['char_end']:
            return i
    return 0


def get_answer_begin_end(data):
    '''
    Get answer's index of begin and end.
    '''
    begin = []
    end = []
    for qa_pair in data:
        tokens = qa_pair['passage_tokens']
        char_begin = qa_pair['answer_begin']
        char_end = qa_pair['answer_end']
        word_begin = get_word_index(tokens, char_begin)
        word_end = get_word_index(tokens, char_end)
        begin.append(word_begin)
        end.append(word_end)
    return np.asarray(begin), np.asarray(end)


def get_id(word_dict, word):
    '''
    Given word, return word id.
    '''
    return word_dict.get(word, word_dict['<unk>'])


def get_buckets(min_length, max_length, bucket_count):
    '''
    Get bucket by length.
    '''
    if bucket_count <= 0:
        return [max_length]
    unit_length = int((max_length - min_length) // (bucket_count))
    buckets = [min_length + unit_length *
               (i + 1) for i in range(0, bucket_count)]
    buckets[-1] = max_length
    return buckets


def find_bucket(length, buckets):
    '''
    Find bucket.
    '''
    for bucket in buckets:
        if length <= bucket:
            return bucket
    return buckets[-1]
