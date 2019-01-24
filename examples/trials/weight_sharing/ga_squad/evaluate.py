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
Evaluation scripts for QA model.
'''

from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(str_input):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        '''
        Remove "a|an|the"
        '''
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        '''
        Remove unnessary whitespace
        '''
        return ' '.join(text.split())

    def remove_punc(text):
        '''
        Remove punc
        '''
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        '''
        Change string to lower form.
        '''
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str_input))))


def f1_score(prediction, ground_truth):
    '''
    Calculate the f1 score.
    '''
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    if not prediction_tokens:
        raise ValueError("empty prediction tokens")
    precision = 1.0 * num_same / len(prediction_tokens)

    if not ground_truth_tokens:
        raise ValueError("empty groundtruth tokens")
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1_result = (2 * precision * recall) / (precision + recall + 1e-10)
    return f1_result


def exact_match_score(prediction, ground_truth):
    '''
    Calculate the match score with prediction and ground truth.
    '''
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    '''
    Metric max over the ground truths.
    '''
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def _evaluate(dataset, predictions):
    '''
    Evaluate function.
    '''
    f1_result = exact_match = total = 0
    count = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa_pair in paragraph['qas']:
                total += 1
                if qa_pair['id'] not in predictions:
                    count += 1
                    continue
                ground_truths = list(
                    map(lambda x: x['text'], qa_pair['answers']))
                prediction = predictions[qa_pair['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1_result += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)
    print('total', total, 'exact_match',
          exact_match, 'unanswer_question ', count)
    exact_match = 100.0 * exact_match / total
    f1_result = 100.0 * f1_result / total
    return {'exact_match': exact_match, 'f1': f1_result}


def evaluate(data_file, pred_file):
    '''
    Evaluate.
    '''
    expected_version = '1.1'
    with open(data_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json['version'] != expected_version:
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(pred_file) as prediction_file:
        predictions = json.load(prediction_file)
    # print(json.dumps(evaluate(dataset, predictions)))
    result = _evaluate(dataset, predictions)
    # print('em:', result['exact_match'], 'f1:', result['f1'])
    return result['exact_match']


def evaluate_with_predictions(data_file, predictions):
    '''
    Evalutate with predictions/
    '''
    expected_version = '1.1'
    with open(data_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json['version'] != expected_version:
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    result = _evaluate(dataset, predictions)
    return result['exact_match']


if __name__ == '__main__':
    EXPECT_VERSION = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + EXPECT_VERSION)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    print(evaluate(args.dataset_file, args.prediction_file))
