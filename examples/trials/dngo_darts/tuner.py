from nni.tuner import Tuner

import torch
import numpy as np
import argparse
from collections import defaultdict

from dngo_darts import load_arch2vec
from dngo_darts import get_init_samples
from dngo_darts import query

from dngo import DNGO
from torch.distributions import Normal






class MayTuner(Tuner):
    
    def __init__(self):
        self.arch0_flag = True
        self.feat_next = []
        self.geno_next = []
        self.proposed_geno = []
        self.i_geno = 0
        self.init_valid_label_samples = []
        self.init_test_label_samples = []
        self.proposed_val_acc = []
        self.proposed_test_acc = []

        self.CURR_BEST_VALID = 0.
        self.CURR_BEST_TEST = 0.
        self.CURR_BEST_GENOTYPE = None
        # self.MAX_BUDGET = args.max_budgets
        self.MAX_BUDGET = 100
        self.window_size = 200
        self.counter = 0
        self.visited = {}
        self.best_trace = defaultdict(list)

        torch.manual_seed(3)
        
        # print(os.getcwd())
        # self.embedding_path = os.getcwd()+"/arch2vec-darts.pt"
        # if not self.embedding_path.exists():
        #     print("path not exist")

        self.embedding_path = "/Users/may/nni/examples/trials/may/arch2vec-darts.pt"
        self.features, self.genotype = load_arch2vec(self.embedding_path)
        self.features, self.genotype = self.features.cpu().detach(), self.genotype



        # part of "def get_init_samples"
        np.random.seed(3)

        init_inds = np.random.permutation(list(range(self.features.shape[0])))[:16]
        init_inds = torch.Tensor(init_inds).long()
        print('init index: {}'.format(init_inds))
        init_feat_samples = self.features[init_inds]
        init_geno_samples = [self.genotype[i.item()] for i in init_inds]
        

        self.feat_samples = init_feat_samples
        self.geno_samples = init_geno_samples
        self.valid_label_samples = []
        self.test_label_samples = []

        self.proposed_geno = init_geno_samples

        for idx in init_inds:
            self.visited[idx.item()] = True



    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        '''
        Receive trial's final result.
        parameter_id: int
        parameters: object created by 'generate_parameters()'
        value: final metrics of the trial, including default metric
        '''
        print("@No.{0} trail result : {1}".format(self.counter, value))

        if self.i_geno == 1:
            self.proposed_val_acc = []
            self.proposed_test_acc = []
        
        val_acc = value["default"]
        test_acc = value["test_acc"]

        # proposed_val_acc.append(val_acc)
        # proposed_test_acc.append(test_acc)

        # self.i_geno == len(self.proposed_geno) means a batch arch queries finish, 
        # now update DNGO_model (to generate next betch genos in "generate_parameters")

        if self.arch0_flag:
            self.init_valid_label_samples.append(val_acc)
            self.init_test_label_samples.append(test_acc)
        else:
            # part of "def propose_location"
            self.proposed_val_acc.append(val_acc)
            self.proposed_test_acc.append(test_acc)
            if self.i_geno == len(self.proposed_geno):
                self.label_next_valid = torch.Tensor(self.proposed_val_acc) 
                self.label_next_test = torch.Tensor(self.proposed_test_acc)
                 # add proposed networks to the pool
                for feat, geno, acc_valid, acc_test in zip(self.feat_next, self.geno_next, self.label_next_valid, self.label_next_test):
                    self.feat_samples = torch.cat((self.feat_samples, feat.view(1, -1)), dim=0)
                    self.geno_samples.append(geno)
                    self.valid_label_samples = torch.cat((self.valid_label_samples.view(-1, 1), acc_valid.view(1, 1)), dim=0)
                    self.test_label_samples = torch.cat((self.test_label_samples.view(-1, 1), acc_test.view(1, 1)), dim=0)
                    # self.counter += 1
                    if acc_valid.item() > self.CURR_BEST_VALID:
                        self.CURR_BEST_VALID = acc_valid
                        self.CURR_BEST_TEST = acc_test
                        self.CURR_BEST_GENOTYPE = geno
                    self.best_trace['validation_acc'].append(float(self.CURR_BEST_VALID))
                    self.best_trace['test_acc'].append(float(self.CURR_BEST_TEST))
                    self.best_trace['genotype'].append(self.CURR_BEST_GENOTYPE)
                    self.best_trace['counter'].append(self.counter)
    

    def generate_parameters(self, parameter_id, **kwargs):
        '''
        Returns a set of trial (hyper-)parameters, as a serializable object
        parameter_id: int
        '''
        self.counter += 1
        print("@len(self.proposed_geno) = ", len(self.proposed_geno))
        print("@self.i_geno = ", self.i_geno)
        # print("@self.proposed_geno", self.proposed_geno)
        if self.i_geno < len(self.proposed_geno):
            next_geno = self.proposed_geno[self.i_geno]
            self.i_geno += 1
        else:
            if self.arch0_flag:
                self.valid_label_samples = torch.Tensor(self.init_valid_label_samples)
                self.test_label_samples = torch.Tensor(self.init_test_label_samples)
                # above: finish "def get_init_samples"
                for feat, geno, acc_valid, acc_test in zip(self.feat_samples, self.geno_samples, self.valid_label_samples, self.test_label_samples):
                    if acc_valid > self.CURR_BEST_VALID:
                        self.CURR_BEST_VALID = acc_valid
                        self.CURR_BEST_TEST = acc_test
                        self.CURR_BEST_GENOTYPE = geno
                    self.best_trace['validation_acc'].append(float(self.CURR_BEST_VALID))
                    self.best_trace['test_acc'].append(float(self.CURR_BEST_TEST))
                    self.best_trace['genotype'].append(self.CURR_BEST_GENOTYPE)
                    self.best_trace['counter'].append(self.counter)

            self.arch0_flag = False

            print("@len(self.proposed_geno) = ", len(self.proposed_geno))
            print("feat_samples:", self.feat_samples.shape)
            print("length of genotypes:", len(self.geno_samples))
            print("valid label_samples:", self.valid_label_samples.shape)
            print("test label samples:", self.test_label_samples.shape)
            print("current best validation: {}".format(self.CURR_BEST_VALID))
            print("current best test: {}".format(self.CURR_BEST_TEST))
            print("counter: {}".format(self.counter))
            print(self.feat_samples.shape)
            print(self.valid_label_samples.shape)

            if __debug__:
                print("__debug__ on, quick, demo walkthrogh")
                model = DNGO(num_epochs=7, n_units=128, do_mcmc=False, normalize_output=False)
            else:
                print("__debug__ off, slow, real tuned")
                model = DNGO(num_epochs=100, n_units=128, do_mcmc=False, normalize_output=False)

            model.train(X=self.feat_samples.numpy(), y=self.valid_label_samples.view(-1).numpy(), do_optimize=True)
            print("@finish model.train()")
            print(model.network)
            m = []
            v = []
            chunks = int(self.features.shape[0] / self.window_size)
            if self.features.shape[0] % self.window_size > 0:
                chunks += 1
            features_split = torch.split(self.features, self.window_size, dim=0)
            for i in range(chunks):
                m_split, v_split = model.predict(features_split[i].numpy())
                m.extend(list(m_split))
                v.extend(list(v_split))
            print("@finish for chunks")
            mean = torch.Tensor(m)
            sigma = torch.Tensor(v)
            # u = (mean - torch.Tensor([args.objective]).expand_as(mean)) / sigma
            u = (mean - torch.Tensor([0.95]).expand_as(mean)) / sigma
            normal = Normal(torch.zeros_like(u), torch.ones_like(u))
            ucdf = normal.cdf(u)
            updf = torch.exp(normal.log_prob(u))
            ei = sigma * (updf + u * ucdf)

            print("@ei = ", ei)
            # feat_next, geno_next, label_next_valid, label_next_test, visited = propose_location(ei, features, genotype, visited, counter)
            count = self.counter
            # k = args.batch_size
            k = 5
            c = 0
            print('remaining length of indices set:', len(self.features) - len(self.visited))
            indices = torch.argsort(ei)
            ind_dedup = []
            # remove random sampled indices at each step
            for idx in reversed(indices):
                if c == k:
                    break
                if idx.item() not in self.visited:
                    self.visited[idx.item()] = True
                    ind_dedup.append(idx.item())
                    c += 1
            ind_dedup = torch.Tensor(ind_dedup).long()
            print('proposed index: {}'.format(ind_dedup))
            proposed_x = self.features[ind_dedup]
            self.proposed_geno = [self.genotype[i.item()] for i in ind_dedup]
            next_geno = self.proposed_geno[0]
            self.i_geno = 1

            self.feat_next = proposed_x
            self.geno_next = self.proposed_geno
            # print("@next_geno==self.proposed_geno[0] : ", next_geno)

        print("@No.{0} geno : {1}".format(self.counter, next_geno))
        return next_geno
    

    def update_search_space(self, search_space):
        '''
        Tuners are advised to support updating search space at run-time.
        If a tuner can only set search space once before generating first hyper-parameters,
        it should explicitly document this behaviour.
        search_space: JSON object created by experiment owner
        '''
        # your code implements here.
