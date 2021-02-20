import os
import sys
sys.path.insert(0, os.getcwd())
from dngo import DNGO
import random
import argparse
import json
import torch
import numpy as np
from collections import defaultdict
from torch.distributions import Normal
from train_search import Train

def load_arch2vec(embedding_path):
    embedding = torch.load(embedding_path)
    print('load arch2vec from {}'.format(embedding_path))
    ind_list = range(len(embedding))
    features = [embedding[ind]['feature'] for ind in ind_list]
    genotype = [embedding[ind]['genotype'] for ind in ind_list]
    features = torch.stack(features, dim=0)
    print('loading finished. pretrained embeddings shape {}'.format(features.shape))
    return features, genotype


def query(counter, seed, genotype, epochs):
    trainer = Train()
    rewards, rewards_test = trainer.main(counter, seed, genotype, epochs=epochs, train_portion=0.9, save="/home/v-ayanmao")
    val_sum = 0
    for epoch, val_acc in rewards:
        val_sum += val_acc
    val_avg = val_sum / len(rewards)
    return val_avg / 100., rewards_test[-1][-1] / 100.

def get_init_samples(features, genotype, visited):
    count = 0
    # np.random.seed(args.seed)
    np.random.seed(3)

    init_inds = np.random.permutation(list(range(features.shape[0])))[:16]
    init_inds = torch.Tensor(init_inds).long()
    print('init index: {}'.format(init_inds))
    init_feat_samples = features[init_inds]
    init_geno_samples = [genotype[i.item()] for i in init_inds]
    init_valid_label_samples = []
    init_test_label_samples = []

    for geno in init_geno_samples:
        val_acc, test_acc = query(count, 3, geno, 50)
        init_valid_label_samples.append(val_acc)
        init_test_label_samples.append(test_acc)
        count += 1

    init_valid_label_samples = torch.Tensor(init_valid_label_samples)
    init_test_label_samples = torch.Tensor(init_test_label_samples)
    for idx in init_inds:
        visited[idx.item()] = True
    return init_feat_samples, init_geno_samples, init_valid_label_samples, init_test_label_samples, visited


def propose_location(ei, features, genotype, visited, counter):
    count = counter
    k = args.batch_size
    c = 0
    print('remaining length of indices set:', len(features) - len(visited))
    indices = torch.argsort(ei)
    ind_dedup = []
    # remove random sampled indices at each step
    for idx in reversed(indices):
        if c == k:
            break
        if idx.item() not in visited:
            visited[idx.item()] = True
            ind_dedup.append(idx.item())
            c += 1
    ind_dedup = torch.Tensor(ind_dedup).long()
    print('proposed index: {}'.format(ind_dedup))
    proposed_x = features[ind_dedup]
    proposed_geno = [genotype[i.item()] for i in ind_dedup]
    proposed_val_acc = []
    proposed_test_acc = []
    for geno in proposed_geno:
        val_acc, test_acc = query(count, args.seed, geno, args.inner_epochs)
        proposed_val_acc.append(val_acc)
        proposed_test_acc.append(test_acc)
        count += 1

    return proposed_x, proposed_geno, torch.Tensor(proposed_val_acc), torch.Tensor(proposed_test_acc), visited


def expected_improvement_search(features, genotype):
    """ implementation of arch2vec-DNGO on DARTS Search Space """
    CURR_BEST_VALID = 0.
    CURR_BEST_TEST = 0.
    CURR_BEST_GENOTYPE = None
    MAX_BUDGET = args.max_budgets
    window_size = 200
    counter = 0
    visited = {}
    best_trace = defaultdict(list)

    features, genotype = features.cpu().detach(), genotype
    feat_samples, geno_samples, valid_label_samples, test_label_samples, visited = get_init_samples(features, genotype, visited)

    for feat, geno, acc_valid, acc_test in zip(feat_samples, geno_samples, valid_label_samples, test_label_samples):
        counter += 1
        if acc_valid > CURR_BEST_VALID:
            CURR_BEST_VALID = acc_valid
            CURR_BEST_TEST = acc_test
            CURR_BEST_GENOTYPE = geno
        best_trace['validation_acc'].append(float(CURR_BEST_VALID))
        best_trace['test_acc'].append(float(CURR_BEST_TEST))
        best_trace['genotype'].append(CURR_BEST_GENOTYPE)
        best_trace['counter'].append(counter)

    while counter < MAX_BUDGET:
        print("feat_samples:", feat_samples.shape)
        print("length of genotypes:", len(geno_samples))
        print("valid label_samples:", valid_label_samples.shape)
        print("test label samples:", test_label_samples.shape)
        print("current best validation: {}".format(CURR_BEST_VALID))
        print("current best test: {}".format(CURR_BEST_TEST))
        print("counter: {}".format(counter))
        print(feat_samples.shape)
        print(valid_label_samples.shape)
        model = DNGO(num_epochs=100, n_units=128, do_mcmc=False, normalize_output=False)
        model.train(X=feat_samples.numpy(), y=valid_label_samples.view(-1).numpy(), do_optimize=True)
        print(model.network)
        m = []
        v = []
        chunks = int(features.shape[0] / window_size)
        if features.shape[0] % window_size > 0:
            chunks += 1
        features_split = torch.split(features, window_size, dim=0)
        for i in range(chunks):
            m_split, v_split = model.predict(features_split[i].numpy())
            m.extend(list(m_split))
            v.extend(list(v_split))
        mean = torch.Tensor(m)
        sigma = torch.Tensor(v)
        u = (mean - torch.Tensor([args.objective]).expand_as(mean)) / sigma
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        feat_next, geno_next, label_next_valid, label_next_test, visited = propose_location(ei, features, genotype, visited, counter)

        # add proposed networks to the pool
        for feat, geno, acc_valid, acc_test in zip(feat_next, geno_next, label_next_valid, label_next_test):
            feat_samples = torch.cat((feat_samples, feat.view(1, -1)), dim=0)
            geno_samples.append(geno)
            valid_label_samples = torch.cat((valid_label_samples.view(-1, 1), acc_valid.view(1, 1)), dim=0)
            test_label_samples = torch.cat((test_label_samples.view(-1, 1), acc_test.view(1, 1)), dim=0)
            # counter += 1
            # if acc_valid.item() > CURR_BEST_VALID:
            #     CURR_BEST_VALID = acc_valid.item()
            #     CURR_BEST_TEST = acc_test.item()
            #     CURR_BEST_GENOTYPE = geno

            # best_trace['validation_acc'].append(float(CURR_BEST_VALID))
            # best_trace['test_acc'].append(float(CURR_BEST_TEST))
            # best_trace['genotype'].append(CURR_BEST_GENOTYPE)
            # best_trace['counter'].append(counter)

            if counter >= MAX_BUDGET:
                break

    res = dict()
    res['validation_acc'] = best_trace['validation_acc']
    res['test_acc'] = best_trace['test_acc']
    res['genotype'] = best_trace['genotype']
    res['counter'] = best_trace['counter']
    save_path = os.path.join(args.output_path, 'dim{}'.format(args.dim))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print('save to {}'.format(save_path))
    fh = open(os.path.join(save_path, 'run_{}_arch2vec_model_darts.json'.format(args.seed)), 'w')
    json.dump(res, fh)
    fh.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="arch2vec-DNGO")
    parser.add_argument("--seed", type=int, default=3, help="random seed")
    parser.add_argument('--cfg', type=int, default=4, help='configuration (default: 4)')
    parser.add_argument('--dim', type=int, default=16, help='feature dimension')
    parser.add_argument('--objective', type=float, default=0.95, help='ei objective')
    parser.add_argument('--init_size', type=int, default=16, help='init samples')
    parser.add_argument('--batch_size', type=int, default=5, help='acquisition samples')
    parser.add_argument('--inner_epochs', type=int, default=50, help='inner loop epochs')
    parser.add_argument('--train_portion', type=float, default=0.9, help='inner loop train/val split')
    parser.add_argument('--max_budgets', type=int, default=100, help='max number of trials')
    parser.add_argument('--output_path', type=str, default='saved_logs/bo', help='bo')
    parser.add_argument('--logging_path', type=str, default='', help='search logging path')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    embedding_path = 'pretrained/dim-{}/arch2vec-darts.pt'.format(args.dim)
    if not os.path.exists(embedding_path):
        exit()
    features, genotype = load_arch2vec(embedding_path)
    expected_improvement_search(features, genotype)
